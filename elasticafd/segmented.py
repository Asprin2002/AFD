"""Segmented joint planning over the device-contiguous event pipeline.

This module is the M1-A engine for a *closed static batch* of requests that all
arrive decode-ready at the same clock.  It plans two coupled axes over a shared
decoding-token axis: a global operator timeline (DensePlan per segment) and a
per-request future-KV writer timeline.  Every candidate cost is a weighted
sum of single-iteration event costs (DenseOffloadCost.measure) sampled over
the decode lifetime; historical KV never migrates.

The plan here is deliberately bounded and deterministic, not an unrestricted
oracle: operator segments come from a coarse coordinate split search, and the
KV timeline uses the same epoch-greedy acceptance rule as the existing
offline attention searches.  Independent validation against JointAFScheduler
is provided by ``tools/run_heterogeneous_joint_matrix.py`` (M1-B).

Legality (kept as explicit predicates so a logic error cannot silently pass
the event model):
  - device-contiguous requires full PIM O_proj before any PIM FFN (enforced by
    the ``DensePlan`` menu itself; the event backend raises on violation).
  - PIM QKV is legal only while every active request keeps its *future* KV on
    PIM (no GPU writer under a PIM QKV reader / KV writer).
  - capacity is checked at every sampled block midpoint and end with a
    *running-peak* PIM weight reservation (replica semantics).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import ceil, inf

from .capacity import views_fit_capacity
from .dense_offload import DenseOffloadCost, DensePlan
from .state import Location, RequestView

# Cap on sampled lifetime states so wall time stays flat even at O=8192.
MAX_LIFETIME_STATES = 96
# Coarse operator split grid fractions (mirrors offline_windows grid_points=5).
SPLIT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GIB = float(1024 ** 3)


@dataclass(frozen=True)
class RequestSpec:
    """Immutable decode-ready request description for a static batch."""

    request_id: str
    input_tokens: int
    output_tokens: int
    # ``gpu_replica`` lets the closed batch exercise DP grouping; DP=1 for the
    # canonical GPT-175B/CHIME system, but the field must stay exact.
    gpu_replica: int = 0


@dataclass(frozen=True)
class DCSchedule:
    """A segmented joint schedule for one closed batch.

    ``kv_cuts[i]`` is an ordered tuple of ``(decode_token, Location)`` starting
    the *future*-KV policy: the request's prompt KV stays on PIM, and each cut
    changes the writer for tokens >= boundary.  Distinct cuts always change the
    writer (identical neighbours are dropped by ``normalize_cuts``).

    ``plan_timeline`` is an ordered ``(start_token, DensePlan)`` sequence shared
    by every sub-batch; the first entry must start at zero.  Plans come from
    the device-contiguous legal menu only.
    """

    name: str
    kv_cuts: tuple[tuple[tuple[int, Location], ...], ...]
    plan_timeline: tuple[tuple[int, DensePlan], ...]


def dc_plan_menu() -> tuple[DensePlan, ...]:
    """Plans legal under device_contiguous: FFN requires full PIM O_proj.

    Includes the operator ladder from the trusted F sweep, the FFN pilots, and
    whole-fused QKV combinations.  QKV plans are additionally gated per-state
    by the all-PIM future-KV legality predicate (not enforced here, because the
    predicate depends on the KV schedule).
    """
    plans = {DensePlan()}
    plans.update(DensePlan(f, 0.0) for f in (0.125, 0.25, 0.5, 0.75, 1.0))
    plans.update(DensePlan(1.0, f) for f in (0.03125, 0.0625, 0.125, 0.25))
    plans.update(
        (DensePlan(qkv=1.0),
         DensePlan(1.0, 0.0, 1.0),
         DensePlan(0.5, 0.0, 1.0),
         DensePlan(1.0, 0.0625, 1.0),
         DensePlan(1.0, 0.125, 1.0))
    )
    return tuple(sorted(plans))


def plans_used(plan_timeline: tuple[tuple[int, DensePlan], ...]) -> tuple[DensePlan, ...]:
    return tuple(dict.fromkeys(plan for _, plan in plan_timeline))


def peak_weight_bytes(model, plans: tuple[DensePlan, ...]) -> int:
    """PIM resident weight bytes at the peak fraction across used plans."""
    max_o = max((p.o_proj for p in plans), default=0.0)
    max_f = max((p.ffn for p in plans), default=0.0)
    max_q = max((p.qkv for p in plans), default=0.0)
    return ceil(model.layers * (
        model.qkv_weight_bytes_per_layer * max_q
        + model.output_projection_weight_bytes_per_layer * max_o
        + model.ff_weight_bytes_per_layer * max_f
    ))


def max_output(specs: tuple[RequestSpec, ...]) -> int:
    return max(spec.output_tokens for spec in specs)


def control_schedule(specs: tuple[RequestSpec, ...], name: str = "control"):
    zero = tuple(tuple() for _ in specs)
    return DCSchedule(name, zero, ((0, DensePlan()),))


def active_plan(plan_timeline, token: int) -> DensePlan:
    starts = [start for start, _ in plan_timeline]
    index = bisect_right(starts, token) - 1
    if index < 0:
        raise AssertionError("plan timeline must start at zero")
    return plan_timeline[index][1]


def state_views(
    schedule: DCSchedule,
    specs: tuple[RequestSpec, ...],
    generated: int,
) -> tuple[RequestView, ...]:
    """Placement snapshot at ``generated`` for every active request.

    Generalizes ``tools/attention_strategy.views`` to heterogeneous input and
    output lengths: each request is clipped at its own ``output_tokens`` and
    excluded once finished.  Adjacent same-device runs are coalesced into one
    contiguous run (required for a valid device-contiguous two-cut graph); for
    the alternating cuts every search produces below, the result matches the
    reference view builder.
    """
    views = []
    for index, spec in enumerate(specs):
        # A decode snapshot at ``generated`` tokens models the iteration that
        # produces token ``generated + 1``; a request finished after its own
        # ``output_tokens`` is out of the batch (clipped out, never carried).
        if generated >= spec.output_tokens:
            continue
        active_until = min(generated, spec.output_tokens)
        pim: list[int] = [spec.input_tokens]
        gpu: list[int] = []
        covered = 0
        tail = Location.PIM  # device of the run ending at ``covered``
        for boundary, target in schedule.kv_cuts[index]:
            if boundary > active_until:
                break
            if boundary > covered:
                _append_run(pim, gpu, tail, covered == 0, boundary - covered)
                covered = boundary
            # A cut at ``active_until`` leaves no following width, so it must
            # not decide the tail device: the last actually-written token uses
            # the previous writer, and a zero-width run cannot own "new KV".
            if boundary < active_until:
                tail = target
        if active_until > covered:
            _append_run(pim, gpu, tail, covered == 0, active_until - covered)
        views.append(RequestView(
            spec.request_id,
            tuple(pim),
            tuple(gpu),
            tail,
            spec.gpu_replica,
        ))
    return tuple(views)


def _append_run(pim: list[int], gpu: list[int], tail: Location,
                at_zero: bool, length: int) -> None:
    """Append a contiguous run, matching ``attention_strategy.views``.

    ``at_zero`` marks the prompt prefix, which is always PIM and extends
    ``pim[0]``.  Every other run appends a new physical segment: the builders
    produce strictly alternating writer targets, so two runs on the same device
    are never adjacent (identical neighbours are dropped by the builders and by
    ``normalize_cuts``), exactly as the reference view builder assumes.
    """
    assert length > 0
    if tail is Location.PIM and at_zero:
        assert pim, "prompt prefix requires a PIM run"
        pim[-1] += length
        return
    (pim if tail is Location.PIM else gpu).append(length)


def legal_plan_for_state(plan: DensePlan, views: tuple[RequestView, ...]) -> bool:
    """XOR legality: PIM QKV is legal only while every active future KV is PIM.

    Mirrors exactly the event backend gate (``_transform_tasks``): no GPU run
    anywhere and the last-written token must be on PIM.
    """
    if plan.qkv > 0.0:
        return all(
            view.gpu_tokens == 0 and view.tail_location is Location.PIM
            for view in views
        )
    return True


@dataclass(frozen=True)
class SwitchPricing:
    """M3 timeline-cost hook, defaulting to the 512-run calibrated values.

    ``switch_execution_us`` is charged once per KV writer change and once per
    operator-plan change (matches ``Simulator._apply_switch_decisions`` and the
    ``switch_execution_us=100`` policy knob used by the paired Qwen pilot).
    ``segment_setup_us`` is informational here: the per-layer segment setup of
    the 512 run is already inside each single-iteration event cost, so charging
    it again would double count.  ``weight_copy_enabled`` turns the replica
    delta weight copy (M3) on or off; the zero copy is the sensitivity control.
    """

    switch_execution_us: float = 100.0
    segment_setup_us: float = 1.0
    weight_copy_enabled: bool = True

    @property
    def label(self) -> str:
        return (
            f"sw{self.switch_execution_us}us_copy{int(self.weight_copy_enabled)}"
        )

    def as_dict(self) -> dict:
        return {
            "switch_execution_us": self.switch_execution_us,
            "segment_setup_us": self.segment_setup_us,
            "weight_copy_enabled": self.weight_copy_enabled,
        }


FREE_PRICING = SwitchPricing(0.0, 0.0, False)


def lifetime_stride(horizon: int) -> int:
    return max(1, ceil(horizon / MAX_LIFETIME_STATES))


def normalize_cuts(
    cuts: tuple[tuple[int, Location], ...],
    output_tokens: int,
) -> tuple[tuple[int, Location], ...]:
    """Collapse redundant cuts, keeping the boundary-0 initial-writer decision.

    A cut at boundary 0 sets the request's initial future-KV writer (the
    ``state_views`` builder interprets it as ``previous = 0``).  Cuts landing
    at/after the output end are irrelevant, and a cut whose target equals the
    current writer is a no-op; both are dropped.
    """
    cleaned: list[tuple[int, Location]] = []
    last = Location.PIM  # the prompt/future-KV default writer
    for boundary, target in cuts:
        if boundary < 0 or boundary >= output_tokens:
            continue
        if target is last:
            continue
        cleaned.append((boundary, target))
        last = target
    return tuple(cleaned)


def block_sample_boundaries(
    schedule: DCSchedule,
    specs: tuple[RequestSpec, ...],
    stride: int,
) -> tuple[int, ...]:
    """Block boundaries: sampled grid, request end lines, and all cut tokens."""
    horizon = max_output(specs)
    assert stride > 0
    boundaries = set(range(0, horizon + 1, stride))
    boundaries.update(spec.output_tokens for spec in specs)
    for index, cuts in enumerate(schedule.kv_cuts):
        for boundary, _ in cuts:
            if 1 <= boundary <= specs[index].output_tokens:
                boundaries.add(boundary)
    for boundary, _ in schedule.plan_timeline:
        if 1 <= boundary <= horizon:
            boundaries.add(boundary)
    return tuple(sorted(b for b in boundaries if 0 <= b <= horizon))


def _plan_timeline_key(
    plan_timeline: tuple[tuple[int, DensePlan], ...],
) -> tuple[tuple[int, DensePlan], ...]:
    merged: list[tuple[int, DensePlan]] = []
    for start, plan in plan_timeline:
        if merged and merged[-1][1] == plan:
            continue
        merged.append((start, plan))
    if not merged or merged[0][0] != 0:
        raise AssertionError("operator plan timeline must start at zero")
    return tuple(merged)


def plans_until(
    plan_timeline: tuple[tuple[int, DensePlan], ...], token: int
) -> tuple[DensePlan, ...]:
    return tuple(dict.fromkeys(plan for start, plan in plan_timeline if start <= token))


def running_peak_bytes(engine: DenseOffloadCost, plans: tuple[DensePlan, ...]) -> int:
    return peak_weight_bytes(engine.model, plans)


def fits_running_peak(
    engine: DenseOffloadCost,
    views: tuple[RequestView, ...],
    plans: tuple[DensePlan, ...],
) -> bool:
    """Capacity under the running PIM weight reservation (replica semantics)."""
    return views_fit_capacity(
        views,
        engine.model,
        engine.system,
        pim_capacity_bytes=(
            engine.system.pim.capacity_bytes
            - running_peak_bytes(engine, plans)
            - engine.pim_extra_workspace
        ),
        gpu_replica_capacity_bytes=engine.gpu_replica_kv_capacity_bytes,
    )


@dataclass(frozen=True)
class BlockCost:
    left: int
    right: int
    iterations: int
    plan: DensePlan
    per_iteration_us: float
    iteration_us: float
    peak_pim_resident_gib: float
    kv_switches_in_block: int
    operator_switches_in_block: int
    feasible: bool = True


@dataclass(frozen=True)
class ScheduleCost:
    """Weighted-lifetime event cost with M3 timeline costs kept separate."""

    name: str
    curve: tuple[BlockCost, ...]
    iteration_us: float
    switch_us: float
    weight_copy_us: float
    kv_switches: int
    operator_switches: int
    peak_pim_resident_gib: float
    feasible: bool

    @property
    def total_us(self) -> float:
        if not self.feasible:
            return inf
        return self.iteration_us + self.switch_us + self.weight_copy_us

    def total_without_overhead(self) -> float:
        return self.iteration_us

    @property
    def is_infinite(self) -> bool:
        return not self.feasible


def weight_copy_events(
    plan_timeline: tuple[tuple[int, DensePlan], ...],
    model,
    link,
) -> list[tuple[str, float]]:
    """Delta weight-copy events under replica/reserve-max semantics.

    PIM keeps the peak fraction reached so far; the resident weight only grows,
    and each growth from the previous peak is copied over the shared link.  A
    shrink changes routing, never the resident set.  The first event (at t=0)
    includes the per-layer DMA-launch latency consistent with
    ``DenseOffloadCost.initial_weight_copy_us``; later growth events use one
    link latency each.
    """
    events: list[tuple[str, float]] = []
    peak_o = peak_f = peak_q = 0.0
    for position, (start, plan) in enumerate(plan_timeline):
        delta_o = max(0.0, plan.o_proj - peak_o)
        delta_f = max(0.0, plan.ffn - peak_f)
        delta_q = max(0.0, plan.qkv - peak_q)
        peak_o = max(peak_o, plan.o_proj)
        peak_f = max(peak_f, plan.ffn)
        peak_q = max(peak_q, plan.qkv)
        layer_delta = (
            model.qkv_weight_bytes_per_layer * delta_q
            + model.output_projection_weight_bytes_per_layer * delta_o
            + model.ff_weight_bytes_per_layer * delta_f
        )
        if layer_delta <= 0:
            continue
        transfer_us = layer_delta / link.host_to_pim_bytes_s * 1e6
        latency = model.layers * link.transfer_latency_us if position == 0 else (
            link.transfer_latency_us
        )
        events.append(("grow", transfer_us + latency))
    return events


def timeline_switch_stats(
    schedule: DCSchedule, specs: tuple[RequestSpec, ...]
) -> tuple[int, int]:
    kv_switches = 0
    for index, cuts in enumerate(schedule.kv_cuts):
        previous: Location | None = None
        for boundary, target in cuts:
            if boundary == 0:
                previous = target
                continue
            if boundary < 1 or boundary >= specs[index].output_tokens:
                continue
            if target is previous:
                continue
            kv_switches += 1
            previous = target
    operator_switches = max(0, len(schedule.plan_timeline) - 1)
    return kv_switches, operator_switches


def _switches_in_span(
    schedule: DCSchedule, specs: tuple[RequestSpec, ...], left: int, right: int
) -> int:
    count = 0
    for index, cuts in enumerate(schedule.kv_cuts):
        previous: Location | None = None
        for boundary, target in cuts:
            if boundary == 0:
                previous = target
                continue
            if boundary < left or boundary >= right:
                continue
            if boundary >= specs[index].output_tokens:
                continue
            if target is previous:
                continue
            count += 1
            previous = target
    return count


def eval_schedule(
    engine: DenseOffloadCost,
    schedule: DCSchedule,
    specs: tuple[RequestSpec, ...],
    *,
    stride: int | None = None,
    pricing: SwitchPricing | None = None,
    cost_name: str | None = None,
) -> ScheduleCost:
    """Full-lifetime event cost of a schedule with running-peak capacity.

    Blocks end at every sampled-grid, request-end, KV-cut and plan-change
    boundary.  Each block contributes ``width`` decode iterations billed at the
    event per-iteration cost of the block midpoint.  Capacity is checked at
    both the midpoint and the end of every block, under the *running* peak
    fraction (replica semantics) so that a schedule may front-load KV on PIM
    before dense weights later arrive.
    """
    if pricing is None:
        pricing = SwitchPricing()
    horizon = max_output(specs)
    if stride is None:
        stride = lifetime_stride(horizon)
    plan_timeline = _plan_timeline_key(schedule.plan_timeline)
    schedule = DCSchedule(schedule.name, schedule.kv_cuts, plan_timeline)
    boundaries = block_sample_boundaries(schedule, specs, stride)
    assert boundaries[0] == 0 and boundaries[-1] == horizon
    blocks: list[BlockCost] = []
    peak = 0.0
    kv_total = op_total = 0
    for left, right in zip(boundaries, boundaries[1:]):
        mid = (left + right - 1) // 2
        plan = active_plan(plan_timeline, mid)
        # Replica semantics: resident PIM weight is the *peak reached so far*,
        # so the right-hand capacity check sees plans that start inside the
        # block (weight copy precedes their use).
        plans_mid = plans_until(plan_timeline, mid)
        plans_end = plans_until(plan_timeline, right)
        block_peak_gib = running_peak_bytes(engine, plans_end) / GIB
        views_mid = state_views(schedule, specs, mid)
        feasible = (
            fits_running_peak(engine, views_mid, plans_mid)
            and legal_plan_for_state(plan, views_mid)
            and fits_running_peak(
                engine, state_views(schedule, specs, right), plans_end
            )
        )
        per_iteration = 0.0
        if feasible:
            per_iteration = engine.measure(views_mid, plan).cost.total_us
        block_kv = _switches_in_span(schedule, specs, left, right)
        block_op = int(any(st == left and st > 0 for st, _ in plan_timeline))
        kv_total += block_kv
        op_total += block_op
        peak = max(peak, block_peak_gib)
        blocks.append(BlockCost(
            left, right, right - left, plan, per_iteration,
            per_iteration * (right - left), block_peak_gib,
            block_kv, block_op, feasible,
        ))
        if not feasible:
            return ScheduleCost(schedule.name, tuple(blocks), inf, inf, inf,
                                kv_total, op_total, peak, False)
    kv_switches, operator_switches = timeline_switch_stats(schedule, specs)
    switch_us = pricing.switch_execution_us * (kv_switches + operator_switches)
    copy_events = weight_copy_events(
        plan_timeline, engine.model, engine.system.link
    )
    copy_us = (sum(us for _, us in copy_events)
               if pricing.weight_copy_enabled else 0.0)
    iteration_us = sum(b.iteration_us for b in blocks)
    return ScheduleCost(
        cost_name or schedule.name,
        tuple(blocks),
        iteration_us,
        switch_us,
        copy_us,
        kv_switches,
        operator_switches,
        peak,
        True,
    )


def eval_window(
    engine: DenseOffloadCost,
    schedule: DCSchedule,
    specs: tuple[RequestSpec, ...],
    left: int,
    right: int,
    *,
    stride: int,
) -> float:
    """Short-horizon projection cost for KV move acceptance.

    Uses plans active in the window only (the choice is about marginal
    placement inside this lookahead), bills window-local capacity and legality,
    and returns the weighted iteration microseconds or ``inf`` for infeasible.
    This mirrors the short lookahead used by ``event_horizon_greedy``.
    """
    if right <= left:
        return 0.0
    plan_timeline = _plan_timeline_key(schedule.plan_timeline)
    boundaries = tuple(
        b for b in block_sample_boundaries(schedule, specs, stride)
        if left <= b <= right
    )
    total = 0.0
    for a, b in zip(boundaries, boundaries[1:]):
        mid = (a + b - 1) // 2
        plan = active_plan(plan_timeline, mid)
        # The resident weight set is bounded by plans starting any time in the
        # window (replica peak through the window end).
        running_plans = plans_until(plan_timeline, right)
        views_mid = state_views(schedule, specs, mid)
        if not (
            fits_running_peak(engine, views_mid, running_plans)
            and legal_plan_for_state(plan, views_mid)
        ):
            return inf
        per_iteration = engine.measure(views_mid, plan).cost.total_us
        total += per_iteration * (b - a)
    return total


def _plan_set(plans, *, qkv: bool | None = None, ffn: bool | None = None,
              oproj: bool | None = None) -> tuple[DensePlan, ...]:
    def keep(plan: DensePlan) -> bool:
        if qkv is not None and (plan.qkv > 0) != qkv:
            return False
        if ffn is not None and (plan.ffn > 0) != ffn:
            return False
        if oproj is not None and (plan.o_proj > 0) != oproj:
            return False
        return True
    return tuple(p for p in dc_plan_menu() if keep(p))


def search_operator_timeline(
    engine: DenseOffloadCost,
    specs: tuple[RequestSpec, ...],
    *,
    kv_cuts: tuple[tuple[tuple[int, Location], ...], ...] | None = None,
    plan_menu: tuple[DensePlan, ...] | None = None,
    stride: int | None = None,
    pricing: SwitchPricing = FREE_PRICING,
    max_segments: int = 3,
) -> DCSchedule:
    """Offline global operator-timeline search for fixed KV cuts.

    Deterministic bounded multi-start over the device-contiguous legal menu:
    the constant-plan best is found first, a rise-time search then locates the
    best change point for the top constant plans, and a final pass adds one
    second-tier boundary.  The full-lifetime event score decides each step.
    """
    if plan_menu is None:
        plan_menu = dc_plan_menu()
    horizon = max_output(specs)
    if stride is None:
        stride = lifetime_stride(horizon)
    kv = kv_cuts if kv_cuts is not None else tuple(tuple() for _ in specs)
    control_plan = DensePlan()
    inner = tuple(sorted({round(t) for t in (
        horizon * f for f in (0.25, 0.5, 0.75)
    )}))
    refinement = tuple(sorted({horizon // 8, horizon // 4, horizon // 2,
                               3 * horizon // 4, 7 * horizon // 8}))

    def build(timeline) -> DCSchedule:
        return DCSchedule("op", kv, _plan_timeline_key(tuple(timeline)))

    def cost(timeline) -> tuple[float, DCSchedule]:
        schedule = build(timeline)
        return eval_schedule(engine, schedule, specs, stride=stride,
                             pricing=pricing).total_us, schedule

    control_t = ((0, control_plan),)
    best_cost = cost(control_t)[0]
    best_planning = build(control_t)

    # Phase A: best constant plan over the whole lifetime.
    ranked: list[tuple[float, DensePlan]] = []
    for plan in plan_menu:
        value, _ = cost(((0, plan),))
        ranked.append((value, plan))
    ranked.sort(key=lambda pair: pair[0])
    feasible_ranked = [pair for pair in ranked if pair[0] < inf]

    if feasible_ranked and feasible_ranked[0][0] + 1e-6 < best_cost:
        best_cost = feasible_ranked[0][0]
        best_planning = build(((0, feasible_ranked[0][1]),))

    top_plans = [plan for _, plan in feasible_ranked[:3]]
    if any(plan != control_plan for plan in top_plans):
        # Phase B: single rise/ease at the best change point for top plans.
        change_points = tuple(sorted(set(inner) | set(refinement)))
        for plan in top_plans:
            if plan == control_plan:
                continue
            for boundary in change_points:
                value, planned = cost(((0, control_plan), (boundary, plan)))
                if value + 1e-6 < best_cost:
                    best_cost, best_planning = value, planned

    # Phase C: add a second boundary (bounded, from the current best).
    timeline = best_planning.plan_timeline
    if len(timeline) < max_segments and timeline != control_t:
        attempts: list[tuple[float, DCSchedule]] = []
        for position in range(len(timeline)):
            start = timeline[position][0]
            end = (timeline[position + 1][0]
                   if position + 1 < len(timeline) else horizon)
            for split in tuple(t for t in inner if start < t < end):
                for plan in top_plans:
                    inserted = tuple(
                        sorted(timeline[: position + 1]
                               + ((split, plan),)
                               + timeline[position + 1:])
                    )
                    if len(tuple(plans_used(inserted))) < 2:
                        continue  # no-op: nothing would actually change
                    value, planned = cost(inserted)
                    if value < inf:
                        attempts.append((value, planned))
        attempts.sort(key=lambda pair: pair[0])
        if attempts and attempts[0][0] + 1e-6 < best_cost:
            best_cost, best_planning = attempts[0]
    return best_planning


def search_kv_timeline(
    engine: DenseOffloadCost,
    specs: tuple[RequestSpec, ...],
    *,
    plan_timeline: tuple[tuple[int, DensePlan], ...] = ((0, DensePlan()),),
    stride: int | None = None,
    decision_interval: int = 64,
    lookahead: int = 256,
    min_residence: int = 128,
    threshold: float = 0.002,
    pricing: SwitchPricing = FREE_PRICING,
) -> DCSchedule:
    """Epoch-greedy future-KV writer search with residence guards.

    At each decision boundary the search evaluates moving one eligible group of
    exchangeable requests to the opposite writer and accepts the best move that
    improves the short-lookahead projection by at least ``threshold``.  The
    first epoch is initial placement (free of switch cost, like the creation
    decision in the 512-request run); later moves cost ``switch_execution_us``.
    Capacity bounds every projected state and PIM-QKV legality is enforced.
    """
    if stride is None:
        stride = lifetime_stride(max_output(specs))
    lifetime = max_output(specs)
    cuts: list[list[tuple[int, Location]]] = [[] for _ in specs]
    locations = [Location.PIM for _ in specs]
    last_switch = [-10 ** 9 for _ in specs]
    for token in range(0, lifetime, decision_interval):
        end = min(lifetime, token + lookahead)
        baseline = eval_window(
            engine, _dc_from(cuts, specs, plan_timeline), specs,
            token, end, stride=stride,
        )
        best_move: tuple[float, tuple[int, ...], Location] | None = None
        groups: dict[tuple[object, ...], list[int]] = {}
        for index in range(len(specs)):
            if token - last_switch[index] < min_residence:
                continue
            key = (locations[index], tuple(cuts[index]))
            groups.setdefault(key, []).append(index)
        for key, indices in groups.items():
            counts = {1, len(indices)}
            n = 2
            while n < len(indices):
                counts.add(n)
                n *= 2
            counts.update(max(1, round(len(indices) * p / 4)) for p in (1, 2, 3))
            for count in sorted(counts):
                selected = tuple(indices[:count])
                trial = [list(part) for part in cuts]
                for index in selected:
                    if token == 0:
                        trial[index] = [(0, locations[index].other)]
                    else:
                        trial[index].append((token, locations[index].other))
                value = eval_window(
                    engine, _dc_from(trial, specs, plan_timeline), specs,
                    token, end, stride=stride,
                )
                if value >= inf:
                    continue
                switch_cost = 0.0 if token == 0 else (
                    pricing.switch_execution_us * len(selected)
                )
                relative = (baseline - value - switch_cost) / max(baseline, 1e-9)
                if relative >= threshold and (
                    best_move is None or value < best_move[0]
                ):
                    best_move = (value, selected, locations[selected[0]].other)
        if best_move is not None:
            _, selected, target = best_move
            for index in selected:
                if token == 0:
                    cuts[index] = [(0, target)]
                else:
                    cuts[index].append((token, target))
                locations[index] = target
                if token > 0:
                    last_switch[index] = token
    return _kv_closure(
        engine, specs, _dc_from(cuts, specs, plan_timeline),
        stride=stride, pricing=pricing, decision_interval=decision_interval,
    )


def _kv_closure(
    engine: DenseOffloadCost,
    specs: tuple[RequestSpec, ...],
    schedule: DCSchedule,
    *,
    stride: int,
    pricing: SwitchPricing,
    decision_interval: int,
) -> DCSchedule:
    """Repair a greedy KV timeline that over-accumulated GPU KV.

    ``search_kv_timeline`` validates each move only over its short lookahead;
    GPU KV is monotone until a request finishes, so the greedy can drive GPU
    residency past capacity late in the lifetime at exactly the large-batch
    long-output shapes where bounded future-KV windows are most useful.  This
    closure retracts the GPU writers active at the first infeasible block to a
    common return-to-PIM boundary: feasibility is monotone in that boundary
    while PIM pressure is not binding (the shapes the greedy trips), so a binary
    search finds the *latest* feasible boundary — the bounded-prefix optimum
    with the least PIM relocation.  When nothing is feasible the original
    schedule is returned unchanged and the caller's M1-B guard reports the cell
    honestly instead of projecting a made-up answer.
    """
    initial = eval_schedule(engine, schedule, specs, stride=stride,
                            pricing=pricing)
    if initial.feasible:
        return schedule
    bad = next(b for b in initial.curve if not b.feasible)
    mid = (bad.left + bad.right - 1) // 2
    views = state_views(schedule, specs, mid)
    index_by_id = {spec.request_id: index for index, spec in enumerate(specs)}
    targets = tuple(
        index_by_id[view.request_id]
        for view in views if view.tail_location is Location.GPU
    )
    if not targets:
        return schedule

    def with_boundary(boundary: int) -> DCSchedule:
        parts = []
        for index, part in enumerate(schedule.kv_cuts):
            new_part = list(part)
            if index in targets:
                new_part.append((boundary, Location.PIM))
            parts.append(normalize_cuts(tuple(new_part),
                                        specs[index].output_tokens))
        return DCSchedule(schedule.name + "_closure", tuple(parts),
                          schedule.plan_timeline)

    hi = min(bad.left, max(spec.output_tokens for spec in specs))
    best: DCSchedule | None = None
    best_total = inf
    lo, high = 1, hi
    while lo <= high:
        boundary = (lo + high) // 2
        candidate = with_boundary(boundary)
        cost = eval_schedule(engine, candidate, specs, stride=stride,
                             pricing=pricing)
        if cost.feasible:
            best, best_total = candidate, cost.total_us
            lo = boundary + 1
        else:
            high = boundary - 1
    return best if best is not None else schedule


def _dc_from(cuts, specs, plan_timeline) -> DCSchedule:
    return DCSchedule(
        "kv",
        tuple(tuple(normalize_cuts(tuple(part), spec.output_tokens))
              for part, spec in zip(cuts, specs)),
        _plan_timeline_key(tuple(plan_timeline)),
    )


def search_joint(
    engine: DenseOffloadCost,
    specs: tuple[RequestSpec, ...],
    *,
    stride: int | None = None,
    pricing: SwitchPricing = FREE_PRICING,
    decision_interval: int = 64,
    lookahead: int = 256,
    min_residence: int = 128,
    max_segments: int = 3,
) -> DCSchedule:
    """Two-phase joint search: operator timeline, then KV writers.

    The operator timeline is chosen assuming all-PIM future KV (the legal
    superset that admits every QKV/migration plan), then the KV writer timeline
    is searched under that plan.  Each phase honors M3 costs, and the schedule
    is finally re-scored with real costs.
    """
    op = search_operator_timeline(
        engine, specs,
        plan_menu=dc_plan_menu(),
        stride=stride,
        pricing=pricing,
        max_segments=max_segments,
    )
    kv = search_kv_timeline(
        engine, specs,
        plan_timeline=op.plan_timeline,
        stride=stride,
        decision_interval=decision_interval,
        lookahead=lookahead,
        min_residence=min_residence,
        pricing=pricing,
    )
    return DCSchedule("joint", kv.kv_cuts, op.plan_timeline)
