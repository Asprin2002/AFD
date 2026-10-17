"""Shared logical lifetimes and exact common-prefix forecast replay.

The immutable plan is shared by candidates. Mutable checkpoints belong to
one candidate only and are discarded at return; no physical states are
merged after their request lifetimes diverge.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from math import inf, isfinite, nextafter
from statistics import fmean
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ForecastStep:
    active: tuple[str, ...]
    retiring: frozenset[str]


@dataclass(frozen=True)
class ForecastNode:
    key: int
    indices: tuple[int, ...]
    start: int
    split: int
    children: Mapping[int, ForecastNode]


@dataclass(frozen=True)
class ForecastPlan:
    request_ids: tuple[str, ...]
    context_starts: tuple[int, ...]
    paths: tuple[tuple[ForecastStep, ...], ...]
    root: ForecastNode

    @property
    def nominal_steps(self) -> int:
        return sum(map(len, self.paths))


def build_forecast_plan(world, horizon: int, scenarios: tuple[dict[str, int], ...]) -> ForecastPlan:
    """Record logical exits once, independently of layouts and device time.

    Step records are interned, so long intervals with no exits need only one
    active-request tuple. Context at step k is context_starts + k for active
    requests; physical KV segmentation remains candidate-specific.
    """
    if horizon < 1 or not scenarios:
        raise ValueError('forecast requires a positive horizon and scenarios')
    ids = tuple(world.requests)
    if not ids:
        raise ValueError('forecast requires active requests')
    interned = {}
    paths = []
    for remaining in scenarios:
        if set(remaining) != set(ids) or any(type(remaining[key]) is not int or remaining[key] < 1
                                           for key in ids):
            raise ValueError('scenario must give every active request a positive remaining length')
        active = ids
        steps = []
        for number in range(1, min(horizon, max(remaining.values())) + 1):
            retiring = frozenset(key for key in active if remaining[key] == number)
            key = (active, retiring)
            record = interned.get(key)
            if record is None:
                record = interned[key] = ForecastStep(active, retiring)
            steps.append(record)
            if retiring:
                active = tuple(key for key in active if key not in retiring)
        paths.append(tuple(steps))
    counter = 0

    def branch(indices, start):
        nonlocal counter
        key = counter
        counter += 1
        first = paths[indices[0]]
        split = start
        end = min(len(paths[index]) for index in indices)
        while split < end and all(paths[index][split] is first[split] for index in indices):
            split += 1
        children = {}
        if split < end:
            groups = defaultdict(list)
            for index in indices:
                groups[paths[index][split]].append(index)
            for group in groups.values():
                child = branch(tuple(group), split + 1)
                children.update((index, child) for index in group)
        elif any(len(paths[index]) != end for index in indices):
            raise ValueError('inconsistent forecast exit histories')
        return ForecastNode(key, indices, start, split, MappingProxyType(children))

    return ForecastPlan(ids, tuple(request.source.input_tokens + request.generated
                                   for request in world.requests.values()),
                        tuple(paths), branch(tuple(range(len(paths))), 0))


class ForecastPruned(Exception):
    """A conservative gain ceiling cannot reach this round's incumbent."""

    def __init__(self, upper_gain_us, partial_mean_us, incumbent_gain_us, tie_margin_us):
        super().__init__('remaining nonnegative time cannot make this candidate win')
        self.proof = {'gain_upper_us': upper_gain_us, 'partial_mean_us': partial_mean_us,
                      'incumbent_gain_us': incumbent_gain_us, 'tie_margin_us': tie_margin_us}


@dataclass(frozen=True)
class ForecastResult:
    totals: tuple[float, ...]
    first_step_us: float

    @property
    def mean_us(self):
        return fmean(self.totals)


def evaluate_forecasts(seed, plan: ForecastPlan, *, baseline_us: float | None = None,
                       incumbent_gain_us: float = 0.0, tie_margin_us: float = 0.0,
                       statistics: dict | None = None, snapshot_limit: int = 8,
                       check_interval: int = 8) -> ForecastResult:
    """Consume a private world, replaying scenarios in their original order.

    A branch checkpoint is AFTER its shared step's evaluation/finalization,
    but BEFORE request appends/retirements differ. Every branch still appends
    and retires in original request order, preserving intermediate capacity
    checks. At most snapshot_limit physical checkpoints are retained.
    """
    from . import layout_aware as layout

    if snapshot_limit < 0 or check_interval < 1:
        raise ValueError('invalid forecast checkpoint/check interval')
    if tuple(seed.requests) != plan.request_ids:
        raise ValueError('forecast plan does not match candidate requests')
    if tuple(request.source.input_tokens + request.generated for request in seed.requests.values()) != plan.context_starts:
        raise ValueError('forecast plan does not match observed contexts')
    stats = statistics if statistics is not None else defaultdict(int)

    def add(name, amount=1):
        stats[name] = stats.get(name, 0) + amount

    add('forecast_batches')
    add('forecast_nominal_steps', plan.nominal_steps)
    partials = [0.0] * len(plan.paths)
    totals = [0.0] * len(plan.paths)
    checkpoints: dict[int, tuple[Any, float]] = {}
    completed_nodes: dict[int, float] = {}
    first_step = None
    evaluations = 0
    reused = 0
    @contextmanager
    def accounting(world):
        previous = world._capacity_totals
        world._start_capacity_accounting()
        try:
            yield
        finally:
            world._capacity_totals = previous

    def check_bound():
        if baseline_us is None:
            return
        mean = fmean(partials)
        # Round the mean DOWN before subtracting nearly equal large costs:
        # rounding only the small resulting gain would not cover rounding
        # in the sum/division. Keep an extra ULP for fsum implementations
        # that can double-round an intermediate addition.
        lower_mean = max(0.0, nextafter(nextafter(mean, -inf), -inf))
        upper = nextafter(baseline_us - lower_mean, inf)
        threshold = nextafter(incumbent_gain_us - tie_margin_us, -inf)
        if isfinite(upper) and upper < threshold:
            raise ForecastPruned(upper, mean, incumbent_gain_us, tie_margin_us)

    def evaluate(world, total, indices):
        nonlocal first_step, evaluations
        cost = layout.evaluate_step(world, book_link=True)
        if not isfinite(cost.total_us) or cost.total_us < 0:
            raise ValueError('forecast step cost must be finite and nonnegative')
        world.clock_us += cost.total_us
        total += cost.total_us
        layout.finalize_preparations(world)
        if first_step is None:
            first_step = cost.total_us
        evaluations += 1
        for index in indices:
            partials[index] = max(partials[index], total)
        if evaluations % check_interval == 0:
            check_bound()
        return total

    def advance(world, step):
        for key in step.active:
            world.append_token(world.requests[key])
            if key in step.retiring:
                world.retire(key)

    def walk(node, index, world, total):
        nonlocal reused
        if node.key in completed_nodes:
            reused += node.split - node.start
            return completed_nodes[node.key]
        checkpoint = checkpoints.get(node.key)
        if checkpoint is not None:
            reused += node.split - node.start + 1
            world, total = checkpoint
        else:
            with accounting(world):
                for number in range(node.start, node.split):
                    total = evaluate(world, total, node.indices)
                    advance(world, plan.paths[index][number])
                if not node.children:
                    completed_nodes[node.key] = total
                    return total
                # This token has identical costs, but different exit sets.
                total = evaluate(world, total, node.indices)
                if len(checkpoints) < snapshot_limit:
                    checkpoints[node.key] = (world, total)
                    stats['forecast_peak_checkpoints'] = max(
                        stats.get('forecast_peak_checkpoints', 0), len(checkpoints))
        # Cached states are immutable templates, never advanced in place.
        if node.key in checkpoints:
            child_world = layout.deepcopy(world)
        else:
            child_world = world
        with accounting(child_world):
            advance(child_world, plan.paths[index][node.split])
            return walk(node.children[index], index, child_world, total)

    old_totals = seed._capacity_totals
    try:
        for index in range(len(plan.paths)):
            world = seed if index == 0 else (None if plan.root.key in checkpoints
                                            or plan.root.key in completed_nodes else layout.deepcopy(seed))
            # With no root checkpoint, seed must stay pristine for the next
            # scenario. This fallback is only used when checkpointing is off.
            if snapshot_limit == 0:
                world = layout.deepcopy(seed)
            totals[index] = walk(plan.root, index, world, 0.0)
            partials[index] = totals[index]
            if index + 1 < len(plan.paths):
                check_bound()
        return ForecastResult(tuple(totals), first_step or 0.0)
    except ForecastPruned:
        add('forecast_pruned_candidates')
        add('forecast_pruned_steps', plan.nominal_steps - evaluations - reused)
        raise
    finally:
        add('forecast_step_evaluations', evaluations)
        add('forecast_prefix_reused_steps', reused)
        seed._capacity_totals = old_totals
