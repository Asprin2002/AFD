"""KV-tail placement policies for CHIME and ElasticAFD."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import ceil, inf
from typing import TYPE_CHECKING, Mapping, Sequence

from .capacity import (
    projected_append_views,
    requests_gpu_memory_usage_by_replica,
    views_fit_capacity,
    views_gpu_memory_usage_by_replica,
    views_memory_usage,
)
from .config import PolicyConfig
from .cost import ChimeCostModel
from .state import (
    Location,
    RequestState,
    RequestView,
    make_request_view,
    make_request_view_with_runs,
    views_for_requests,
)
from .trace_profiles import ConditionalRemainingLength

if TYPE_CHECKING:
    from .workload import WorkloadRequest


@dataclass(frozen=True)
class SwitchDecision:
    request_id: str
    old_location: Location
    new_location: Location
    reason: str
    objective_before_us: float | None = None
    objective_after_us: float | None = None


class PlacementPolicy:
    allows_gpu_kv = False

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        self.config = config
        self.cost_model = cost_model

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        return []

    def update_runtime_context(
        self,
        *,
        clock_us: float,
        pending: Sequence["WorkloadRequest"],
        max_batch_size: int,
        initial_location: Location,
    ) -> None:
        """Expose serving state to policies that explicitly request it.

        Online policies intentionally ignore this hook. It exists so an
        offline oracle can include known future arrivals without broadening the
        normal ``decide`` interface or leaking them into other baselines.
        """

    def summary_metadata(self) -> dict[str, object]:
        """Return policy-specific, JSON-serializable experiment metadata."""

        return {}

    def relieve_capacity_pressure(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        pim_free_bytes: float,
        gpu_free_bytes: float,
    ) -> list[SwitchDecision]:
        views = projected_append_views(requests)
        if views_fit_capacity(
            views,
            self.cost_model.model,
            self.cost_model.system,
            pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
            gpu_replica_capacity_bytes=(
                self.cost_model.gpu_replica_kv_capacity_bytes
            ),
        ):
            return []
        raise RuntimeError(
            "the fixed KV tier cannot accept the next token; reduce the "
            "batch/context or choose a placement policy with another KV tier"
        )


class ChimePlacementPolicy(PlacementPolicy):
    """CHIME baseline: every prompt and decode KV remains in DIMM-PIM."""


class ForcedFractionPlacementPolicy(PlacementPolicy):
    """Place a deterministic fraction of request future-KV tails on GPU.

    This is a mechanism probe, not an adaptive scheduler.  It acts once, after
    the initial closed batch is admitted, and never changes the selected set.
    Stable request-ID ordering makes paired A1 sweeps exactly reproducible.
    """

    allows_gpu_kv = True

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        super().__init__(config, cost_model)
        self._planned = False
        self._selected: tuple[str, ...] = ()

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        if self._planned or not requests:
            return []
        if iteration != 0:
            raise RuntimeError(
                "forced_fraction must see the complete closed batch at iteration 0"
            )

        ordered = sorted(requests, key=lambda request: request.request_id)
        count = int(len(ordered) * self.config.forced_gpu_fraction + 0.5)
        selected = {request.request_id for request in ordered[:count]}
        projected = [
            make_request_view(
                request,
                projected_tokens=min(
                    self.config.decision_interval, request.remaining_tokens
                ),
                projected_location=(
                    Location.GPU
                    if request.request_id in selected
                    else Location.PIM
                ),
            )
            for request in requests
        ]
        if not views_fit_capacity(
            projected,
            self.cost_model.model,
            self.cost_model.system,
            pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
            gpu_replica_capacity_bytes=(
                self.cost_model.gpu_replica_kv_capacity_bytes
            ),
        ):
            raise RuntimeError("forced future-KV fraction exceeds KV capacity")

        decisions: list[SwitchDecision] = []
        for request in ordered:
            target = Location.GPU if request.request_id in selected else Location.PIM
            old = request.tail_location
            if request.switch_tail(target, iteration):
                decisions.append(
                    SwitchDecision(
                        request.request_id,
                        old,
                        target,
                        "forced_gpu_fraction",
                    )
                )
        self._selected = tuple(sorted(selected))
        self._planned = True
        return decisions

    def summary_metadata(self) -> dict[str, object]:
        return {
            "forced_gpu_fraction_requested": self.config.forced_gpu_fraction,
            "forced_gpu_request_count": len(self._selected),
            "forced_gpu_request_ids": list(self._selected),
        }


class OnlineStaticFractionPlacementPolicy(PlacementPolicy):
    """Assign each newly arrived request a fixed future-KV tail device.

    The prompt remains where decode admission placed it (PIM for CHIME).  A
    request is assigned once, immediately after it becomes active, and is
    never rebalanced.  This is the online static request-partition control for
    dynamic future-KV placement.
    """

    allows_gpu_kv = True

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        super().__init__(config, cost_model)
        self._assigned: dict[str, Location] = {}
        self._seen = 0
        self._gpu_assigned = 0
        self._gpu_capacity_fallbacks = 0

    def _full_lifetime_fits(
        self,
        requests: Sequence[RequestState],
        candidate_id: str,
    ) -> bool:
        views = []
        for request in requests:
            target = (
                Location.GPU
                if request.request_id == candidate_id
                else self._assigned.get(request.request_id, request.tail_location)
            )
            views.append(
                make_request_view(
                    request,
                    projected_tokens=request.remaining_tokens,
                    projected_location=target,
                )
            )
        return views_fit_capacity(
            views,
            self.cost_model.model,
            self.cost_model.system,
            pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
            gpu_replica_capacity_bytes=(
                self.cost_model.gpu_replica_kv_capacity_bytes
            ),
        )

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        new_requests = sorted(
            (
                request
                for request in requests
                if request.request_id not in self._assigned
            ),
            key=lambda request: (request.arrival_us, request.request_id),
        )
        decisions: list[SwitchDecision] = []
        for request in new_requests:
            self._seen += 1
            desired_gpu = int(
                self._seen * self.config.forced_gpu_fraction + 0.5
            )
            target = (
                Location.GPU
                if self._gpu_assigned < desired_gpu
                else Location.PIM
            )
            if target is Location.GPU and not self._full_lifetime_fits(
                requests, request.request_id
            ):
                target = Location.PIM
                self._gpu_capacity_fallbacks += 1
            self._assigned[request.request_id] = target
            if target is Location.GPU:
                self._gpu_assigned += 1
                old = request.tail_location
                if request.switch_tail(target, iteration):
                    decisions.append(
                        SwitchDecision(
                            request.request_id,
                            old,
                            target,
                            "online_static_fraction",
                        )
                    )
        return decisions

    def summary_metadata(self) -> dict[str, object]:
        return {
            "online_static_fraction_requested": self.config.forced_gpu_fraction,
            "online_static_assigned_requests": len(self._assigned),
            "online_static_gpu_requests": self._gpu_assigned,
            "online_static_gpu_capacity_fallbacks": self._gpu_capacity_fallbacks,
        }


class ElasticPlacementPolicy(PlacementPolicy):
    """Greedy, hardware-cost-aware future-only tail placement.

    Historical KV never moves. At a decision point, the policy projects a short
    horizon and greedily flips request tails while the predicted pipelined
    iteration time improves. The default ``fixed`` predictor deliberately does
    not inspect true output lengths; ``oracle`` is available only as an upper
    bound experiment.
    """

    allows_gpu_kv = True

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        super().__init__(config, cost_model)
        self.length_predictor = (
            ConditionalRemainingLength(
                config.prediction_profile,
                count=config.prediction_samples,
                seed=config.prediction_seed,
            )
            if config.prediction == "profile"
            else None
        )

    def _horizon(self, request: RequestState) -> int:
        horizon = self.config.planning_horizon_tokens
        if self.config.prediction == "fixed":
            return horizon
        if self.config.prediction == "oracle":
            return min(horizon, request.remaining_tokens)
        if self.config.prediction == "profile":
            assert self.length_predictor is not None
            remaining = self.length_predictor.expected_capped_remaining(
                request.generated_tokens, horizon
            )
            if remaining is None:
                return horizon
            return max(1, min(horizon, int(round(remaining))))
        raise ValueError("policy.prediction must be fixed, profile, or oracle")

    def _eligible(self, request: RequestState, iteration: int) -> bool:
        return (
            iteration - request.last_switch_iteration
            >= self.config.min_switch_interval
            and request.tail_segment_tokens >= self.config.min_segment_tokens
        )

    def _objective(
        self,
        requests: Sequence[RequestState],
        overrides: Mapping[str, Location],
    ) -> float:
        views = []
        for request in requests:
            horizon = self._horizon(request)
            if horizon <= 0:
                continue
            location = overrides.get(request.request_id, request.tail_location)
            views.append(
                make_request_view(
                    request,
                    projected_tokens=horizon,
                    projected_location=location,
                )
            )
        if not views:
            return 0.0

        if not views_fit_capacity(
            views,
            self.cost_model.model,
            self.cost_model.system,
            pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
            gpu_replica_capacity_bytes=(
                self.cost_model.gpu_replica_kv_capacity_bytes
            ),
        ):
            return inf

        cost = self.cost_model.iteration_cost(views).total_us
        changed = sum(
            overrides.get(request.request_id, request.tail_location)
            is not request.tail_location
            for request in requests
        )
        amortization = max(1, self.config.planning_horizon_tokens)
        return cost + changed * self.config.switch_penalty_us / amortization

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        if not requests or (
            not force and iteration % self.config.decision_interval
        ):
            return []

        selected: dict[str, Location] = {}
        selected_ids: set[str] = set()
        selected_costs: dict[str, tuple[float, float]] = {}
        current_objective = self._objective(requests, selected)

        for _ in range(self.config.max_switches_per_decision):
            best_request: RequestState | None = None
            best_location: Location | None = None
            best_objective = current_objective
            for request in requests:
                if request.request_id in selected_ids or not self._eligible(
                    request, iteration
                ):
                    continue
                current_location = selected.get(
                    request.request_id, request.tail_location
                )
                candidate_location = current_location.other
                candidate = dict(selected)
                candidate[request.request_id] = candidate_location
                objective = self._objective(requests, candidate)
                if objective < best_objective:
                    best_request = request
                    best_location = candidate_location
                    best_objective = objective

            if best_request is None or best_location is None:
                break
            if current_objective == inf:
                relative_gain = inf if best_objective < inf else 0.0
            else:
                relative_gain = (current_objective - best_objective) / max(
                    current_objective, 1e-12
                )
            if relative_gain < self.config.relative_improvement_threshold:
                break
            selected_costs[best_request.request_id] = (
                current_objective,
                best_objective,
            )
            selected[best_request.request_id] = best_location
            selected_ids.add(best_request.request_id)
            current_objective = best_objective

        decisions: list[SwitchDecision] = []
        by_id = {request.request_id: request for request in requests}
        for request_id, location in selected.items():
            request = by_id[request_id]
            old = request.tail_location
            if request.switch_tail(location, iteration):
                before, after = selected_costs[request_id]
                decisions.append(
                    SwitchDecision(
                        request_id,
                        old,
                        location,
                        "projected_cost",
                        before,
                        after,
                    )
                )
        return decisions

    def relieve_capacity_pressure(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        pim_free_bytes: float,
        gpu_free_bytes: float,
    ) -> list[SwitchDecision]:
        model = self.cost_model.model
        system = self.cost_model.system
        pim_capacity = self.cost_model.pim_kv_capacity_bytes
        overrides: dict[str, Location] = {}
        selected: set[str] = set()

        def usage(
            candidate: Mapping[str, Location],
        ) -> tuple[int, tuple[int, ...]]:
            views = projected_append_views(requests, candidate)
            pim_bytes, _ = views_memory_usage(views, model, system)
            gpu_by_replica = views_gpu_memory_usage_by_replica(
                views, model, system
            )
            return pim_bytes, gpu_by_replica

        def overflow(candidate: Mapping[str, Location]) -> float:
            pim_bytes, gpu_by_replica = usage(candidate)
            return max(0.0, pim_bytes - pim_capacity) + sum(
                max(
                    0.0,
                    gpu_bytes - self.cost_model.gpu_replica_kv_capacity_bytes,
                )
                for gpu_bytes in gpu_by_replica
            )

        current_overflow = overflow(overrides)
        if current_overflow <= 0:
            return []

        for _ in range(len(requests)):
            best_request: RequestState | None = None
            best_overflow = current_overflow
            for request in requests:
                if request.request_id in selected:
                    continue
                candidate = dict(overrides)
                candidate[request.request_id] = request.tail_location.other
                candidate_overflow = overflow(candidate)
                if candidate_overflow < best_overflow:
                    best_overflow = candidate_overflow
                    best_request = request
            if best_request is None:
                break
            overrides[best_request.request_id] = best_request.tail_location.other
            selected.add(best_request.request_id)
            current_overflow = best_overflow
            if current_overflow <= 0:
                break

        if current_overflow > 0:
            raise RuntimeError(
                "PIM plus per-replica GPU KV capacity (including per-segment "
                "block rounding) cannot accept one more token for every active request"
            )

        decisions: list[SwitchDecision] = []
        by_id = {request.request_id: request for request in requests}
        for request_id, target in overrides.items():
            request = by_id[request_id]
            old = request.tail_location
            request.switch_tail(target, iteration)
            decisions.append(
                SwitchDecision(request.request_id, old, target, "capacity")
            )
        return decisions


class LifetimeGreedyPlacementPolicy(ElasticPlacementPolicy):
    """Greedily place the next KV segment using its lifetime reuse value.

    At each decision, the chosen device is committed only for the next decision
    interval. The objective integrates decode cost over every request's known
    remaining lifetime under a continuation assumption: the selected tail
    device remains selected. Capacity is committed and checked only through the
    next decision, where the continuation is recomputed. This exposes the joint
    value of consecutive GPU segments without reserving nonexistent future KV.
    Coarse evaluation intervals make the policy practical on long traces.
    """

    def _block_length(self, request: RequestState) -> int:
        return min(self.config.decision_interval, request.remaining_tokens)

    def _projected_view(
        self,
        request: RequestState,
        target: Location,
        elapsed_tokens: int,
    ):
        return make_request_view_with_runs(
            request,
            ((target, elapsed_tokens),),
        )

    def _committed_block_fits(
        self,
        requests: Sequence[RequestState],
        overrides: Mapping[str, Location],
    ) -> bool:
        """Check only KV created before the next decision.

        Later reference-PIM tokens are not committed by this decision and must
        not make an otherwise feasible current block look infeasible.
        """

        points = {1}
        max_block = 0
        for request in requests:
            block = self._block_length(request)
            max_block = max(max_block, block)
            points.add(block)
            points.add(request.remaining_tokens)
        for elapsed in sorted(point for point in points if 0 < point <= max_block):
            views = [
                self._projected_view(
                    request,
                    overrides.get(request.request_id, request.tail_location),
                    elapsed,
                )
                for request in requests
                # Include a request just after its final append, before release.
                if request.remaining_tokens >= elapsed
            ]
            if not views_fit_capacity(
                views,
                self.cost_model.model,
                self.cost_model.system,
                pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
                gpu_replica_capacity_bytes=(
                    self.cost_model.gpu_replica_kv_capacity_bytes
                ),
            ):
                return False
        return True

    def _evaluation_intervals(
        self, requests: Sequence[RequestState]
    ) -> tuple[tuple[int, int], ...]:
        maximum = max((request.remaining_tokens for request in requests), default=0)
        if maximum <= 0:
            return ()
        stride = self.config.lifetime_evaluation_stride_tokens
        boundaries = {0, maximum}
        boundaries.update(range(stride, maximum, stride))
        for request in requests:
            boundaries.add(request.remaining_tokens)
            boundaries.add(self._block_length(request))
        ordered = sorted(value for value in boundaries if 0 <= value <= maximum)
        return tuple(zip(ordered, ordered[1:]))

    def _objective(
        self,
        requests: Sequence[RequestState],
        overrides: Mapping[str, Location],
    ) -> float:
        if not requests:
            return 0.0
        if not self._committed_block_fits(requests, overrides):
            return inf

        total_us = 0.0
        for start, end in self._evaluation_intervals(requests):
            # Midpoint quadrature; stride=1 evaluates every remaining iteration.
            sample = (start + end - 1) // 2
            views = [
                self._projected_view(
                    request,
                    overrides.get(request.request_id, request.tail_location),
                    sample,
                )
                for request in requests
                if request.remaining_tokens > sample
            ]
            if views:
                total_us += self.cost_model.iteration_cost(views).total_us * (
                    end - start
                )

        changed = sum(
            overrides.get(request.request_id, request.tail_location)
            is not request.tail_location
            for request in requests
        )
        return total_us + changed * self.config.switch_penalty_us

    def _gpu_reuse_byte_iterations(
        self,
        requests: Sequence[RequestState],
        overrides: Mapping[str, Location],
    ) -> int:
        """Logical GPU bytes weighted by future attention reuse iterations."""

        total = 0
        token_bytes = self.cost_model.model.kv_bytes_per_token
        for request in requests:
            target = overrides.get(request.request_id, request.tail_location)
            if target is not Location.GPU:
                continue
            remaining = request.remaining_tokens
            reuse_iterations = sum(
                max(0, remaining - creation_step)
                for creation_step in range(1, remaining + 1)
            )
            total += reuse_iterations * token_bytes
        return total

    @staticmethod
    def _equivalence_signature(request: RequestState) -> tuple[object, ...]:
        """Physical state that makes request-tail flips cost-equivalent.

        Request IDs are deliberately absent.  Replica affinity, immutable KV
        segments, remaining lifetime, and current tail placement are all
        retained, so grouping does not merge requests with different resource
        or capacity consequences.
        """

        return (
            request.gpu_replica,
            request.remaining_tokens,
            request.tail_location.value,
            tuple(
                (segment.location.value, segment.tokens)
                for segment in request.segments
            ),
        )

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        if not requests or (
            not force and iteration % self.config.decision_interval
        ):
            return []

        selected: dict[str, Location] = {}
        selected_ids: set[str] = set()
        selected_costs: dict[str, tuple[float, float]] = {}
        selected_reasons: dict[str, str] = {}
        current_objective = self._objective(requests, selected)
        current_byte_time = self._gpu_reuse_byte_iterations(requests, selected)
        source_bottleneck: Location | None = None
        if self.config.require_source_bottleneck:
            current_cost = self.cost_model.iteration_cost(
                views_for_requests(requests)
            )
            pim_path_us = max(
                current_cost.pim_core_busy_us,
                current_cost.link_busy_us,
            )
            source_bottleneck = (
                Location.PIM
                if pim_path_us > current_cost.gpu_busy_us
                else Location.GPU
            )

        for _ in range(self.config.max_switches_per_decision):
            remaining_budget = (
                self.config.max_switches_per_decision - len(selected_ids)
            )
            if remaining_budget <= 0:
                break
            best_requests: tuple[RequestState, ...] = ()
            best_location: Location | None = None
            best_objective = current_objective
            best_byte_time = current_byte_time
            best_priority: tuple[int, float, float] | None = None

            groups: dict[tuple[object, ...], list[RequestState]] = {}
            for request in requests:
                if request.request_id in selected_ids or not self._eligible(
                    request, iteration
                ):
                    continue
                groups.setdefault(self._equivalence_signature(request), []).append(
                    request
                )

            # A max(TA, TF) objective is generally non-separable: one tail can
            # be below the hysteresis threshold while several equivalent tails
            # jointly cross it.  Evaluate bounded same-state bundles as well as
            # singleton actions.  This keeps the search polynomial and avoids
            # conflating a greedy barrier with a negative mechanism result.
            for group in groups.values():
                current_location = group[0].tail_location
                if (
                    source_bottleneck is not None
                    and current_location is not source_bottleneck
                ):
                    continue
                maximum = min(len(group), remaining_budget)
                for bundle_size in range(1, maximum + 1):
                    bundle = tuple(group[:bundle_size])
                    candidate = dict(selected)
                    for request in bundle:
                        candidate[request.request_id] = current_location.other
                    objective = self._objective(requests, candidate)
                    gain = current_objective - objective
                    if gain <= 0:
                        continue
                    byte_time = self._gpu_reuse_byte_iterations(requests, candidate)
                    added_byte_time = byte_time - current_byte_time
                    if added_byte_time <= 0:
                        priority = (1, inf, gain)
                    else:
                        priority = (0, gain / added_byte_time, gain)
                    if best_priority is None or priority > best_priority:
                        best_priority = priority
                        best_requests = bundle
                        best_location = current_location.other
                        best_objective = objective
                        best_byte_time = byte_time

            if not best_requests or best_location is None:
                break
            if current_objective == inf:
                relative_gain = inf if best_objective < inf else 0.0
            else:
                relative_gain = (current_objective - best_objective) / max(
                    current_objective, 1e-12
                )
            if relative_gain < self.config.relative_improvement_threshold:
                break

            reason = (
                "lifetime_value_per_byte_time_group"
                if len(best_requests) > 1
                else "lifetime_value_per_byte_time"
            )
            for request in best_requests:
                selected_costs[request.request_id] = (
                    current_objective,
                    best_objective,
                )
                selected_reasons[request.request_id] = reason
                selected[request.request_id] = best_location
                selected_ids.add(request.request_id)
            current_objective = best_objective
            current_byte_time = best_byte_time

        decisions: list[SwitchDecision] = []
        by_id = {request.request_id: request for request in requests}
        for request_id, location in selected.items():
            request = by_id[request_id]
            old = request.tail_location
            if request.switch_tail(location, iteration):
                before, after = selected_costs[request_id]
                decisions.append(
                    SwitchDecision(
                        request_id,
                        old,
                        location,
                        selected_reasons[request_id],
                        before,
                        after,
                    )
                )
        return decisions


@dataclass
class _BeamNode:
    cost_us: float
    requests: list[RequestState]
    iteration: int
    first_targets: dict[str, Location]
    score_us: float
    clock_us: float
    pending: tuple["WorkloadRequest", ...]


class RollingBeamPlacementPolicy(ElasticPlacementPolicy):
    """Oracle model-predictive placement over several future KV segments.

    A bounded beam explicitly simulates several decision blocks, retains the
    best partial schedules, executes only the first action, and replans. Future
    arrivals are absent by default. The explicit offline-oracle option can add
    the simulator's known pending trace to the search state.
    """

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        super().__init__(config, cost_model)
        self._runtime_clock_us = 0.0
        self._runtime_pending: tuple["WorkloadRequest", ...] = ()
        self._runtime_max_batch_size = 0
        self._runtime_initial_location = Location.PIM

    def update_runtime_context(
        self,
        *,
        clock_us: float,
        pending: Sequence["WorkloadRequest"],
        max_batch_size: int,
        initial_location: Location,
    ) -> None:
        if not self.config.beam_include_future_arrivals:
            return
        self._runtime_clock_us = clock_us
        self._runtime_pending = tuple(pending)
        self._runtime_max_batch_size = max_batch_size
        self._runtime_initial_location = initial_location

    def _state_signature(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        pending: Sequence["WorkloadRequest"] = (),
        clock_us: float = 0.0,
    ) -> tuple[object, ...]:
        return (
            iteration,
            clock_us if self.config.beam_include_future_arrivals else 0.0,
            tuple(
                (
                    request.request_id,
                    request.generated_tokens,
                    request.tail_location.value,
                    request.last_switch_iteration,
                    tuple(
                        (segment.location.value, segment.tokens)
                        for segment in request.segments
                    ),
                )
                for request in requests
            ),
            tuple(item.request_id for item in pending),
        )

    def _projected_gpu_replica(
        self, requests: Sequence[RequestState]
    ) -> int:
        counts = [0] * self.cost_model.model.data_parallel
        for request in requests:
            counts[request.gpu_replica] += 1
        gpu_bytes = requests_gpu_memory_usage_by_replica(
            requests,
            self.cost_model.model,
            self.cost_model.system,
        )
        return min(
            range(self.cost_model.model.data_parallel),
            key=lambda replica: (counts[replica], gpu_bytes[replica], replica),
        )

    def _admit_projected(
        self,
        requests: list[RequestState],
        pending: Sequence["WorkloadRequest"],
        clock_us: float,
    ) -> tuple[list[RequestState], tuple["WorkloadRequest", ...]]:
        retained: list["WorkloadRequest"] = []
        for item in pending:
            if (
                item.arrival_us > clock_us
                or len(requests) >= self._runtime_max_batch_size
            ):
                retained.append(item)
                continue
            candidate = item.to_state()
            candidate.gpu_replica = self._projected_gpu_replica(requests)
            candidate.admitted_us = clock_us
            if self._runtime_initial_location is Location.GPU:
                candidate.segments[0].location = Location.GPU
                candidate.tail_location = Location.GPU
            if not views_fit_capacity(
                views_for_requests([*requests, candidate]),
                self.cost_model.model,
                self.cost_model.system,
                pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
                gpu_replica_capacity_bytes=(
                    self.cost_model.gpu_replica_kv_capacity_bytes
                ),
            ):
                retained.append(item)
                continue
            requests.append(candidate)
        return requests, tuple(retained)

    @staticmethod
    def _action_key(
        requests: Sequence[RequestState], targets: Mapping[str, Location]
    ) -> tuple[str, ...]:
        return tuple(targets[request.request_id].value for request in requests)

    def _candidate_actions(
        self, requests: Sequence[RequestState], iteration: int
    ) -> tuple[dict[str, Location], ...]:
        stay = {
            request.request_id: request.tail_location for request in requests
        }
        eligible = [
            request for request in requests if self._eligible(request, iteration)
        ]
        if not eligible or self.config.max_switches_per_decision == 0:
            return (stay,)

        # Keep both directions represented: moving PIM->GPU can relieve PIM,
        # while GPU->PIM can avoid future GPU contention or capacity pressure.
        by_location: list[list[RequestState]] = []
        for location in (Location.PIM, Location.GPU):
            by_location.append(
                sorted(
                    (
                        request
                        for request in eligible
                        if request.tail_location is location
                    ),
                    key=lambda request: (
                        request.remaining_tokens,
                        request.context_tokens,
                        request.request_id,
                    ),
                    reverse=True,
                )
            )
        ranked: list[RequestState] = []
        for index in range(max(map(len, by_location), default=0)):
            for group in by_location:
                if index < len(group):
                    ranked.append(group[index])
        top = ranked[: self.config.beam_request_candidates]

        actions: dict[tuple[str, ...], dict[str, Location]] = {}

        def add(switched: Sequence[RequestState]) -> None:
            if len(switched) > self.config.max_switches_per_decision:
                return
            targets = dict(stay)
            for request in switched:
                targets[request.request_id] = request.tail_location.other
            actions.setdefault(self._action_key(requests, targets), targets)

        add(())
        for mask in range(1, 1 << len(top)):
            if bin(mask).count("1") > self.config.max_switches_per_decision:
                continue
            add(
                tuple(
                    request
                    for index, request in enumerate(top)
                    if mask & (1 << index)
                )
            )

        # Prefix actions preserve all switch counts for large homogeneous
        # groups even when beam_request_candidates is smaller than the batch.
        all_ranked = sorted(
            eligible,
            key=lambda request: (
                request.remaining_tokens,
                request.context_tokens,
                request.request_id,
            ),
            reverse=True,
        )
        for count in range(
            1,
            min(len(all_ranked), self.config.max_switches_per_decision) + 1,
        ):
            add(tuple(all_ranked[:count]))
        for group in by_location:
            for count in range(
                1,
                min(len(group), self.config.max_switches_per_decision) + 1,
            ):
                add(tuple(group[:count]))
        return tuple(actions.values())

    def _simulate_block(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        targets: Mapping[str, Location],
        pending: Sequence["WorkloadRequest"] = (),
        clock_us: float = 0.0,
    ) -> tuple[
        float,
        list[RequestState],
        int,
        tuple["WorkloadRequest", ...],
        float,
    ] | None:
        projected = deepcopy(list(requests))
        projected_pending = tuple(pending)
        projected_clock = clock_us
        switches = 0
        for request in projected:
            if request.switch_tail(targets[request.request_id], iteration):
                switches += 1

        total_us = switches * self.config.switch_penalty_us
        steps = 0
        stride = self.config.beam_evaluation_stride_tokens
        while (
            projected or (
                self.config.beam_include_future_arrivals and projected_pending
            )
        ) and steps < self.config.decision_interval:
            if self.config.beam_include_future_arrivals:
                if not projected:
                    next_arrival = projected_pending[0].arrival_us
                    if next_arrival > projected_clock:
                        idle_us = next_arrival - projected_clock
                        total_us += idle_us
                        projected_clock = next_arrival
                projected, projected_pending = self._admit_projected(
                    projected, projected_pending, projected_clock
                )
                if not projected:
                    return None
            span = min(
                stride,
                self.config.decision_interval - steps,
                min(request.remaining_tokens for request in projected),
            )
            midpoint = (span - 1) // 2
            midpoint_views = [
                make_request_view(
                    request,
                    projected_tokens=midpoint,
                    projected_location=request.tail_location,
                )
                for request in projected
            ]
            execution_us = (
                self.cost_model.iteration_cost(midpoint_views).total_us * span
            )
            total_us += execution_us
            projected_clock += execution_us

            for _ in range(span):
                for request in projected:
                    request.append_generated_token()
            appended_views = views_for_requests(projected)
            if not views_fit_capacity(
                appended_views,
                self.cost_model.model,
                self.cost_model.system,
                pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
                gpu_replica_capacity_bytes=(
                    self.cost_model.gpu_replica_kv_capacity_bytes
                ),
            ):
                return None
            projected = [request for request in projected if not request.is_finished]
            steps += span
        return (
            total_us,
            projected,
            iteration + steps,
            projected_pending,
            projected_clock,
        )

    def _terminal_estimate(self, requests: Sequence[RequestState]) -> float:
        if not requests:
            return 0.0
        maximum = max(request.remaining_tokens for request in requests)
        if self.config.beam_terminal_mode == "horizon":
            views = [
                make_request_view(
                    request,
                    projected_tokens=min(
                        self.config.planning_horizon_tokens,
                        request.remaining_tokens,
                    ),
                    projected_location=request.tail_location,
                )
                for request in requests
            ]
            # Scale the projected critical iteration to the remaining decode
            # length so its magnitude is comparable with accumulated block
            # time. This is a terminal heuristic, not simulated execution.
            return self.cost_model.iteration_cost(views).total_us * maximum

        stride = max(1, ceil(maximum / self.config.beam_terminal_samples))
        boundaries = {0, maximum}
        boundaries.update(range(stride, maximum, stride))
        boundaries.update(request.remaining_tokens for request in requests)
        ordered = sorted(boundaries)
        total_us = 0.0
        for start, end in zip(ordered, ordered[1:]):
            sample = (start + end - 1) // 2
            views = [
                make_request_view(
                    request,
                    projected_tokens=sample,
                    projected_location=(
                        Location.PIM
                        if self.config.beam_terminal_mode == "pim_lifetime"
                        else request.tail_location
                    ),
                )
                for request in requests
                if request.remaining_tokens > sample
            ]
            if views:
                total_us += self.cost_model.iteration_cost(views).total_us * (
                    end - start
                )
        return total_us

    def _search_first_action(
        self,
        requests: Sequence[RequestState],
        iteration: int,
    ) -> tuple[dict[str, Location], float, float]:
        before = self._terminal_estimate(requests)
        pending = (
            self._runtime_pending
            if self.config.beam_include_future_arrivals
            else ()
        )
        clock_us = (
            self._runtime_clock_us
            if self.config.beam_include_future_arrivals
            else 0.0
        )
        terminal_cache: dict[tuple[object, ...], float] = {}
        nodes = [
            _BeamNode(
                cost_us=0.0,
                requests=deepcopy(list(requests)),
                iteration=iteration,
                first_targets={},
                score_us=before,
                clock_us=clock_us,
                pending=pending,
            )
        ]
        root_stay_key = tuple(request.tail_location.value for request in requests)

        for depth in range(self.config.beam_depth):
            children: list[_BeamNode] = []
            root_wait: _BeamNode | None = None
            for node in nodes:
                if not node.requests and not node.pending:
                    children.append(node)
                    continue
                actions = (
                    self._candidate_actions(node.requests, node.iteration)
                    if node.requests
                    else ({},)
                )
                for targets in actions:
                    block = self._simulate_block(
                        node.requests,
                        node.iteration,
                        targets,
                        node.pending,
                        node.clock_us,
                    )
                    if block is None:
                        continue
                    (
                        block_cost,
                        survivors,
                        next_iteration,
                        next_pending,
                        next_clock,
                    ) = block
                    cost = node.cost_us + block_cost
                    first_targets = (
                        dict(targets) if depth == 0 else node.first_targets
                    )
                    signature = self._state_signature(
                        survivors,
                        next_iteration,
                        next_pending,
                        next_clock,
                    )
                    terminal = terminal_cache.get(signature)
                    if terminal is None:
                        terminal = self._terminal_estimate(survivors)
                        terminal_cache[signature] = terminal
                    child = _BeamNode(
                        cost,
                        survivors,
                        next_iteration,
                        first_targets,
                        cost + terminal,
                        next_clock,
                        next_pending,
                    )
                    children.append(child)
                    if (
                        self._action_key(requests, first_targets)
                        == root_stay_key
                        and (
                            root_wait is None
                            or (child.score_us, child.cost_us)
                            < (root_wait.score_us, root_wait.cost_us)
                        )
                    ):
                        root_wait = child

            if not children:
                # A later block can become temporarily infeasible under the
                # normal switch-count/eligibility limits even though the first
                # planned block is safe. End this MPC horizon early; the real
                # simulator replans at the next boundary and retains its
                # emergency capacity-relief path before every append.
                break
            children.sort(key=lambda node: (node.score_us, node.cost_us))
            deduplicated: list[_BeamNode] = []
            seen: set[tuple[object, ...]] = set()
            for child in children:
                signature = self._state_signature(
                    child.requests,
                    child.iteration,
                    child.pending,
                    child.clock_us,
                )
                if signature in seen:
                    continue
                seen.add(signature)
                deduplicated.append(child)
                if len(deduplicated) >= self.config.beam_width:
                    break
            if (
                root_wait is not None
                and root_wait not in deduplicated
                and (
                    self.config.beam_include_future_arrivals or depth == 0
                )
            ):
                if len(deduplicated) >= self.config.beam_width:
                    deduplicated[-1] = root_wait
                else:
                    deduplicated.append(root_wait)
            nodes = deduplicated
            if all(not node.requests and not node.pending for node in nodes):
                break

        best = min(nodes, key=lambda node: (node.score_us, node.cost_us))
        if self.config.beam_include_future_arrivals:
            wait_nodes = [
                node
                for node in nodes
                if self._action_key(requests, node.first_targets)
                == root_stay_key
            ]
            if wait_nodes:
                before = min(
                    wait_nodes, key=lambda node: (node.score_us, node.cost_us)
                ).score_us
        return best.first_targets, before, best.score_us

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        if not requests or (
            not force and iteration % self.config.decision_interval
        ):
            return []
        targets, before, after = self._search_first_action(requests, iteration)
        if not targets:
            return []
        relative_gain = (before - after) / max(before, 1e-12)
        if relative_gain < self.config.relative_improvement_threshold:
            return []

        decisions: list[SwitchDecision] = []
        for request in requests:
            target = targets[request.request_id]
            old = request.tail_location
            if request.switch_tail(target, iteration):
                decisions.append(
                    SwitchDecision(
                        request.request_id,
                        old,
                        target,
                        (
                            "rolling_beam_future_arrivals"
                            if self.config.beam_include_future_arrivals
                            else "rolling_beam_makespan"
                        ),
                        before,
                        after,
                    )
                )
        return decisions


@dataclass(frozen=True)
class _HomogeneousMode:
    """One request cohort whose future KV is GPU-written on [start, end)."""

    count: int
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


class HomogeneousModePlacementPolicy(ElasticPlacementPolicy):
    """Offline lifecycle planner for a homogeneous closed batch.

    Instead of assigning a binary device to every request/token pair, this
    planner searches the three parameters of one deployment-friendly mode:
    ``count`` exchangeable requests write future KV to GPU on ``[start, end)``.
    The exact capacity bound reduces to the interval length, while the
    lifecycle objective is integrated by fixed quadrature and then replayed
    exactly for a small set of finalists.  The complete plan is built once at
    iteration zero; later calls only execute its precomputed boundary events.

    This is an intentionally low-dimensional static-batch planner, not a claim
    of an unrestricted global optimum.  It supplies the scalable counterpart
    to :class:`GlobalExactPlacementPolicy`'s combinatorial correctness oracle.
    """

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        super().__init__(config, cost_model)
        self._built = False
        self._mode = _HomogeneousMode(0, 0, 0)
        self._request_ids: tuple[str, ...] = ()
        self._events: dict[int, tuple[Location, ...]] = {}
        self._estimated_objective_us = 0.0
        self._exact_objective_us = 0.0
        self._all_pim_objective_us = 0.0
        self._quadrature_mode_evaluations = 0
        self._exact_mode_evaluations = 0
        self._cost_model_evaluations = 0

    @staticmethod
    def _mode_key(mode: _HomogeneousMode) -> tuple[int, int, int]:
        return mode.count, mode.start, mode.end

    def _validate_homogeneous(
        self, requests: Sequence[RequestState], iteration: int
    ) -> None:
        if iteration != 0:
            raise RuntimeError(
                "homogeneous_modes must construct its offline plan at iteration 0"
            )
        if not requests:
            raise RuntimeError("homogeneous_modes received an empty batch")
        shapes = {
            (request.input_tokens, request.output_tokens, request.generated_tokens)
            for request in requests
        }
        if len(shapes) != 1 or next(iter(shapes))[2] != 0:
            raise RuntimeError(
                "homogeneous_modes requires one decode-ready homogeneous batch"
            )
        if any(request.tail_location is not Location.PIM for request in requests):
            raise RuntimeError(
                "homogeneous_modes requires every prompt segment to start on PIM"
            )

    def _views_at(
        self,
        requests: Sequence[RequestState],
        mode: _HomogeneousMode,
        generated_tokens: int,
    ) -> tuple[RequestView, ...]:
        views: list[RequestView] = []
        for index, request in enumerate(requests):
            if index >= mode.count or mode.duration <= 0:
                views.append(
                    RequestView(
                        request.request_id,
                        (request.input_tokens + generated_tokens,),
                        (),
                        Location.PIM,
                        request.gpu_replica,
                    )
                )
                continue

            if generated_tokens < mode.start:
                pim = (request.input_tokens + generated_tokens,)
                gpu: tuple[int, ...] = ()
                tail = Location.PIM
            elif generated_tokens < mode.end:
                pim = (request.input_tokens + mode.start,)
                gpu = (generated_tokens - mode.start,) if generated_tokens > mode.start else ()
                tail = Location.GPU
            else:
                suffix = generated_tokens - mode.end
                pim = (
                    (request.input_tokens + mode.start, suffix)
                    if suffix
                    else (request.input_tokens + mode.start,)
                )
                gpu = (mode.duration,)
                tail = Location.PIM
            views.append(
                RequestView(
                    request.request_id,
                    pim,
                    gpu,
                    tail,
                    request.gpu_replica,
                )
            )
        return tuple(views)

    def _fits(
        self, requests: Sequence[RequestState], mode: _HomogeneousMode
    ) -> bool:
        views = self._views_at(requests, mode, requests[0].output_tokens)
        return views_fit_capacity(
            views,
            self.cost_model.model,
            self.cost_model.system,
            pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
            gpu_replica_capacity_bytes=(
                self.cost_model.gpu_replica_kv_capacity_bytes
            ),
        )

    def _gpu_interval_fits(
        self, requests: Sequence[RequestState], mode: _HomogeneousMode
    ) -> bool:
        """Check the monotone GPU-only bound used by duration bisection.

        Shared PIM use can have small start-dependent block-rounding effects
        after a return switch, so it remains part of the exact candidate check
        in ``_fits`` rather than the monotone duration search.
        """

        views = self._views_at(requests, mode, requests[0].output_tokens)
        return all(
            used <= self.cost_model.gpu_replica_kv_capacity_bytes
            for used in views_gpu_memory_usage_by_replica(
                views,
                self.cost_model.model,
                self.cost_model.system,
            )
        )

    def _valid_mode(
        self, requests: Sequence[RequestState], mode: _HomogeneousMode
    ) -> bool:
        if mode.count == 0:
            return mode.start == 0 and mode.end == 0
        output_tokens = requests[0].output_tokens
        if not 0 < mode.count <= min(
            len(requests), self.config.max_switches_per_decision
        ):
            return False
        if not 0 <= mode.start < mode.end <= output_tokens:
            return False
        residence = max(
            1,
            self.config.min_segment_tokens,
            self.config.min_switch_interval,
        )
        if mode.duration < residence:
            return False
        if (
            mode.end < output_tokens
            and output_tokens - mode.end < self.config.min_segment_tokens
        ):
            return False
        return self._fits(requests, mode)

    def _quadrature_objective(
        self, requests: Sequence[RequestState], mode: _HomogeneousMode
    ) -> float:
        self._quadrature_mode_evaluations += 1
        output_tokens = requests[0].output_tokens
        samples = self.config.homogeneous_mode_quadrature_samples
        boundaries = {0, output_tokens, mode.start, mode.end}
        boundaries.update(
            round(index * output_tokens / samples)
            for index in range(1, samples)
        )
        ordered = sorted(boundary for boundary in boundaries if 0 <= boundary <= output_tokens)
        total_us = 0.0
        for left, right in zip(ordered, ordered[1:]):
            if right <= left:
                continue
            midpoint = (left + right - 1) // 2
            self._cost_model_evaluations += 1
            total_us += self.cost_model.iteration_cost(
                self._views_at(requests, mode, midpoint)
            ).total_us * (right - left)
        switches = mode.count * (1 + int(mode.end < output_tokens))
        return total_us + switches * self.config.switch_penalty_us

    def _exact_objective(
        self, requests: Sequence[RequestState], mode: _HomogeneousMode
    ) -> float:
        self._exact_mode_evaluations += 1
        output_tokens = requests[0].output_tokens
        total_us = 0.0
        for generated in range(output_tokens):
            self._cost_model_evaluations += 1
            total_us += self.cost_model.iteration_cost(
                self._views_at(requests, mode, generated)
            ).total_us
        switches = mode.count * (1 + int(mode.end < output_tokens))
        return total_us + switches * self.config.switch_penalty_us

    def _coarse_modes(
        self, requests: Sequence[RequestState]
    ) -> tuple[_HomogeneousMode, ...]:
        batch_size = len(requests)
        output_tokens = requests[0].output_tokens
        points = self.config.homogeneous_mode_grid_points
        knots = sorted(
            {
                round(index * output_tokens / (points - 1))
                for index in range(points)
            }
        )
        modes = {_HomogeneousMode(0, 0, 0)}
        max_count = min(batch_size, self.config.max_switches_per_decision)
        for count in range(1, max_count + 1):
            # GPU capacity is a monotone function of interval duration. Find
            # its exact integer bound once for this count instead of
            # discovering overflow during decode. The later candidate check
            # also accounts for PIM segment block rounding.
            low, high = 0, output_tokens
            while low < high:
                midpoint = (low + high + 1) // 2
                if self._gpu_interval_fits(
                    requests, _HomogeneousMode(count, 0, midpoint)
                ):
                    low = midpoint
                else:
                    high = midpoint - 1
            max_duration = low
            if max_duration <= 0:
                continue
            for start in knots[:-1]:
                available = min(output_tokens - start, max_duration)
                durations = {
                    max(1, round(index * available / (points - 1)))
                    for index in range(1, points)
                }
                durations.add(available)
                for duration in durations:
                    mode = _HomogeneousMode(
                        count, start, min(output_tokens, start + duration)
                    )
                    if self._valid_mode(requests, mode):
                        modes.add(mode)
        return tuple(modes)

    def _refine(
        self,
        requests: Sequence[RequestState],
        initial: _HomogeneousMode,
    ) -> tuple[_HomogeneousMode, float]:
        output_tokens = requests[0].output_tokens
        best = initial
        best_cost = self._quadrature_objective(requests, best)
        step = max(
            1,
            output_tokens // max(2, self.config.homogeneous_mode_grid_points - 1) // 2,
        )
        while step >= 1:
            improved = True
            while improved:
                improved = False
                candidates = {best}
                for count_delta in (-1, 0, 1):
                    count = best.count + count_delta
                    if not 0 <= count <= len(requests):
                        continue
                    if count == 0:
                        candidates.add(_HomogeneousMode(0, 0, 0))
                        continue
                    for start_delta in (-step, 0, step):
                        for end_delta in (-step, 0, step):
                            start = min(
                                output_tokens - 1,
                                max(0, best.start + start_delta),
                            )
                            end = min(
                                output_tokens,
                                max(start + 1, best.end + end_delta),
                            )
                            candidate = _HomogeneousMode(count, start, end)
                            if self._valid_mode(requests, candidate):
                                candidates.add(candidate)
                for candidate in candidates:
                    cost = self._quadrature_objective(requests, candidate)
                    if (cost, self._mode_key(candidate)) < (
                        best_cost,
                        self._mode_key(best),
                    ):
                        best, best_cost = candidate, cost
                        improved = True
            step //= 2
        return best, best_cost

    def _build_plan(
        self, requests: Sequence[RequestState], iteration: int
    ) -> None:
        self._validate_homogeneous(requests, iteration)
        coarse = [
            (self._quadrature_objective(requests, mode), mode)
            for mode in self._coarse_modes(requests)
        ]
        coarse.sort(key=lambda item: (item[0], self._mode_key(item[1])))
        seed_count = min(
            len(coarse), self.config.homogeneous_mode_exact_finalists
        )
        refined: dict[tuple[int, int, int], tuple[float, _HomogeneousMode]] = {}
        for _, seed in coarse[:seed_count]:
            mode, estimate = self._refine(requests, seed)
            refined[self._mode_key(mode)] = (estimate, mode)

        # Exact O(output_tokens) integration is used only to rank the handful
        # of low-dimensional finalists.  It never branches on token position.
        exact = [
            (self._exact_objective(requests, mode), estimate, mode)
            for estimate, mode in refined.values()
        ]
        all_pim = _HomogeneousMode(0, 0, 0)
        if self._mode_key(all_pim) not in refined:
            exact.append(
                (
                    self._exact_objective(requests, all_pim),
                    self._quadrature_objective(requests, all_pim),
                    all_pim,
                )
            )
        exact.sort(key=lambda item: (item[0], self._mode_key(item[2])))
        best_exact, best_estimate, best = exact[0]

        self._mode = best
        self._request_ids = tuple(
            request.request_id for request in requests[: best.count]
        )
        self._events = {}
        if best.count:
            self._events[best.start] = tuple(
                Location.GPU for _ in self._request_ids
            )
            if best.end < requests[0].output_tokens:
                self._events[best.end] = tuple(
                    Location.PIM for _ in self._request_ids
                )
        self._estimated_objective_us = best_estimate
        self._exact_objective_us = best_exact
        self._all_pim_objective_us = next(
            cost for cost, _, mode in exact if mode.count == 0
        )
        self._built = True

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        if not self._built:
            self._build_plan(requests, iteration)
        targets = self._events.get(iteration)
        if targets is None:
            return []
        by_id = {request.request_id: request for request in requests}
        decisions: list[SwitchDecision] = []
        for request_id, target in zip(self._request_ids, targets):
            request = by_id.get(request_id)
            if request is None:
                raise RuntimeError(
                    "homogeneous_modes plan diverged from its closed batch"
                )
            old = request.tail_location
            if request.switch_tail(target, iteration):
                decisions.append(
                    SwitchDecision(
                        request_id,
                        old,
                        target,
                        "homogeneous_lifecycle_plan",
                        self._all_pim_objective_us,
                        self._exact_objective_us,
                    )
                )
        return decisions

    def summary_metadata(self) -> dict[str, object]:
        return {
            "homogeneous_mode_count": self._mode.count,
            "homogeneous_mode_start_token": self._mode.start,
            "homogeneous_mode_end_token": self._mode.end,
            "homogeneous_mode_duration_tokens": self._mode.duration,
            "homogeneous_mode_estimated_makespan_s": (
                self._estimated_objective_us / 1e6
            ),
            "homogeneous_mode_exact_makespan_s": (
                self._exact_objective_us / 1e6
            ),
            "homogeneous_mode_all_pim_makespan_s": (
                self._all_pim_objective_us / 1e6
            ),
            "homogeneous_mode_quadrature_evaluations": (
                self._quadrature_mode_evaluations
            ),
            "homogeneous_mode_exact_finalist_evaluations": (
                self._exact_mode_evaluations
            ),
            "homogeneous_mode_cost_model_evaluations": (
                self._cost_model_evaluations
            ),
        }


@dataclass(frozen=True)
class _BalanceCurveGroup:
    request_ids: tuple[str, ...]
    cohort_ids: tuple[str, ...]
    targets: tuple[tuple[int, float], ...]


class BalanceCurvePlacementPolicy(ElasticPlacementPolicy):
    """Closed-form resource-balance planner for a homogeneous static batch.

    For each fixed CHIME sub-batch, PIM attention is an affine decreasing
    function of accumulated GPU KV while GPU attention is an affine increasing
    function.  Their intersection gives the continuous GPU-KV target directly.
    A deterministic tracker then converts that curve to request-tail decisions;
    no schedule tree, beam, or candidate mode search is used.

    The first implementation deliberately accepts only the token-granular,
    zero-segment-overhead CHIME model for which those affine assumptions hold.
    """

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        super().__init__(config, cost_model)
        self._built = False
        self._events: dict[int, dict[str, Location]] = {}
        self._groups: tuple[_BalanceCurveGroup, ...] = ()
        self._planned_switches = 0
        self._coefficient_evaluations = 0
        self._target_peak_gpu_tokens = 0.0
        self._target_final_gpu_tokens = 0.0
        self._target_first_positive_token = 0
        self._realized_peak_gpu_tokens = 0
        self._realized_final_gpu_tokens = 0

    def _validate(self, requests: Sequence[RequestState], iteration: int) -> None:
        if iteration != 0 or not requests:
            raise RuntimeError(
                "balance_curve must construct its offline plan at iteration 0"
            )
        shapes = {
            (request.input_tokens, request.output_tokens, request.generated_tokens)
            for request in requests
        }
        if len(shapes) != 1 or next(iter(shapes))[2] != 0:
            raise RuntimeError(
                "balance_curve requires one decode-ready homogeneous batch"
            )
        if any(request.tail_location is not Location.PIM for request in requests):
            raise RuntimeError(
                "balance_curve requires every prompt segment to start on PIM"
            )
        model = self.cost_model.model
        system = self.cost_model.system
        if model.data_parallel != 1:
            raise RuntimeError(
                "balance_curve v1 currently supports one data-parallel replica"
            )
        if (
            system.pim.token_tile != 1
            or system.gpu.attention_tile_tokens != 1
            or system.pim.kv_block_tokens != 1
            or system.gpu.kv_block_tokens != 1
            or system.pim.segment_setup_us != 0
            or system.gpu.segment_setup_us != 0
        ):
            raise RuntimeError(
                "balance_curve requires token-granular KV and zero segment setup cost"
            )
        if self.config.switch_penalty_us != 0:
            raise RuntimeError(
                "balance_curve v1 does not yet include switch penalty in its "
                "closed-form target"
            )
        if self.config.min_segment_tokens > requests[0].input_tokens:
            raise RuntimeError(
                "balance_curve cannot switch before the configured minimum segment"
            )
        if (
            self.config.min_segment_tokens > self.config.decision_interval
            or self.config.min_switch_interval > self.config.decision_interval
        ):
            raise RuntimeError(
                "balance_curve v1 requires segment and switch minima no larger "
                "than the decision interval"
            )

    @staticmethod
    def _view(
        request: RequestState,
        generated_tokens: int,
        gpu_tokens: int,
    ) -> RequestView:
        context = request.input_tokens + generated_tokens
        if gpu_tokens <= 0:
            return RequestView(
                request.request_id,
                (context,),
                (),
                Location.PIM,
                request.gpu_replica,
            )
        return RequestView(
            request.request_id,
            (context - gpu_tokens,),
            (gpu_tokens,),
            Location.PIM,
            request.gpu_replica,
        )

    def _group_cost(
        self,
        requests: Sequence[RequestState],
        cohort_ids: frozenset[str],
        generated_tokens: int,
        gpu_tokens_per_cohort_request: int,
    ):
        views = tuple(
            self._view(
                request,
                generated_tokens,
                (
                    gpu_tokens_per_cohort_request
                    if request.request_id in cohort_ids
                    else 0
                ),
            )
            for request in requests
        )
        self._coefficient_evaluations += 1
        return self.cost_model.subbatch_layer_cost(views)

    def _closed_form_target(
        self,
        requests: Sequence[RequestState],
        cohort_ids: frozenset[str],
        generated_tokens: int,
        other_dense_us: float,
        single_subbatch: bool,
    ) -> float:
        if generated_tokens < 2 or not cohort_ids:
            return 0.0
        zero = self._group_cost(
            requests, cohort_ids, generated_tokens, 0
        )
        one = self._group_cost(
            requests, cohort_ids, generated_tokens, 1
        )
        two = self._group_cost(
            requests, cohort_ids, generated_tokens, 2
        )

        pim_one = one.pim.total_us
        pim_slope = two.pim.total_us - pim_one
        dense_overlap = 0.0 if single_subbatch else other_dense_us
        gpu_one = one.gpu_attention.total_us + dense_overlap
        gpu_slope = two.gpu_attention.total_us - one.gpu_attention.total_us
        denominator = gpu_slope - pim_slope
        if denominator <= 0:
            return 0.0
        crossover = 1.0 + (pim_one - gpu_one) / denominator
        crossover = min(float(generated_tokens), max(0.0, crossover))
        if crossover <= 0:
            return 0.0

        baseline_phase = max(
            zero.pim.total_us,
            zero.gpu_attention.total_us + dense_overlap,
        ) + zero.merge_us
        pim_at_cross = pim_one + pim_slope * (crossover - 1.0)
        gpu_at_cross = gpu_one + gpu_slope * (crossover - 1.0)
        balanced_phase = max(pim_at_cross, gpu_at_cross) + one.merge_us
        return crossover if balanced_phase < baseline_phase else 0.0

    def _curve_for_cohort(
        self,
        requests: Sequence[RequestState],
        cohort_ids: Sequence[str],
        other_dense_us: float,
        single_subbatch: bool,
    ) -> tuple[tuple[int, float], ...]:
        output_tokens = requests[0].output_tokens
        cohort = frozenset(cohort_ids)
        points = [0]
        points.extend(
            range(
                self.config.decision_interval,
                output_tokens,
                self.config.decision_interval,
            )
        )
        points.append(output_tokens)
        return tuple(
            (
                generated,
                len(cohort)
                * self._closed_form_target(
                    requests,
                    cohort,
                    generated,
                    other_dense_us,
                    single_subbatch,
                ),
            )
            for generated in points
        )

    @staticmethod
    def _required_cohort_size(
        curve: Sequence[tuple[int, float]], maximum: int
    ) -> int:
        required = 0
        previous_t = 0
        previous_target = 0.0
        for generated, target in curve[1:]:
            if generated > 0:
                required = max(required, ceil(target / generated - 1e-12))
            width = generated - previous_t
            if width > 0:
                required = max(
                    required,
                    ceil(max(0.0, target - previous_target) / width - 1e-12),
                )
            previous_t = generated
            previous_target = target
        return min(maximum, required)

    @staticmethod
    def _target_at(
        curve: Sequence[tuple[int, float]], iteration: int
    ) -> float:
        for point, target in curve:
            if point == iteration:
                return target
        raise KeyError(f"balance curve has no point at iteration {iteration}")

    def _capacity_adjusted_counts(
        self,
        desired: list[int],
        current_gpu_tokens: int,
        width: int,
        end_iteration: int,
        input_tokens: int,
        groups: Sequence[_BalanceCurveGroup],
    ) -> list[int]:
        model = self.cost_model.model
        gpu_capacity_tokens = int(
            self.cost_model.gpu_replica_kv_capacity_bytes
            // model.kv_bytes_per_token
        )
        pim_capacity_tokens = int(
            self.cost_model.pim_kv_capacity_bytes
            // model.kv_bytes_per_token
        )
        total_context = sum(len(group.request_ids) for group in groups) * (
            input_tokens + end_iteration
        )

        max_new_writers = max(
            0, (gpu_capacity_tokens - current_gpu_tokens) // width
        )
        while sum(desired) > max_new_writers:
            candidates = [index for index, count in enumerate(desired) if count]
            if not candidates:
                break
            index = max(candidates, key=lambda item: desired[item])
            desired[index] -= 1

        # PIM target capacity is normally loose in the CHIME preset.  Prompt
        # tokens are added by the caller when checking the final plan, so this
        # value only guards decode-token growth here.
        minimum_gpu_tokens = max(0, total_context - pim_capacity_tokens)
        minimum_new_writers = max(
            0,
            ceil((minimum_gpu_tokens - current_gpu_tokens) / width),
        )
        while sum(desired) < minimum_new_writers:
            candidates = [
                index
                for index, group in enumerate(groups)
                if desired[index] < len(group.cohort_ids)
            ]
            if not candidates:
                raise RuntimeError(
                    "balance_curve cannot satisfy PIM capacity with its cohort"
                )
            index = min(candidates, key=lambda item: desired[item])
            desired[index] += 1
        return desired

    def _build_plan(
        self, requests: Sequence[RequestState], iteration: int
    ) -> None:
        self._validate(requests, iteration)
        by_id = {request.request_id: request for request in requests}
        initial_partitions = self.cost_model.partition_views(
            views_for_requests(requests)
        )
        partition_requests = tuple(
            tuple(by_id[view.request_id] for view in partition)
            for partition in initial_partitions
        )
        dense_us = tuple(
            self.cost_model.subbatch_layer_cost(partition).dense.total_us
            for partition in initial_partitions
        )
        single = len(partition_requests) == 1

        groups: list[_BalanceCurveGroup] = []
        for index, group_requests in enumerate(partition_requests):
            other_dense = 0.0 if single else dense_us[1 - index]
            all_ids = tuple(request.request_id for request in group_requests)
            provisional = self._curve_for_cohort(
                group_requests, all_ids, other_dense, single
            )
            cohort_size = self._required_cohort_size(
                provisional, len(group_requests)
            )
            cohort_ids = all_ids[:cohort_size]
            curve = self._curve_for_cohort(
                group_requests, cohort_ids, other_dense, single
            )
            groups.append(_BalanceCurveGroup(all_ids, cohort_ids, curve))
        self._groups = tuple(groups)

        gpu_tokens = {request.request_id: 0 for request in requests}
        tails = {request.request_id: Location.PIM for request in requests}
        tail_run_tokens = {
            request.request_id: request.input_tokens for request in requests
        }
        last_switch = {
            request.request_id: -10**18 for request in requests
        }
        output_tokens = requests[0].output_tokens
        current = 0
        while current < output_tokens:
            end = min(output_tokens, current + self.config.decision_interval)
            width = end - current
            desired: list[int] = []
            for group in self._groups:
                actual = sum(gpu_tokens[request_id] for request_id in group.cohort_ids)
                target = self._target_at(group.targets, end)
                writers = int(max(0.0, (target - actual) / width) + 0.5)
                desired.append(min(len(group.cohort_ids), writers))

            actual_total = sum(gpu_tokens.values())
            desired = self._capacity_adjusted_counts(
                desired,
                actual_total,
                width,
                end,
                requests[0].input_tokens,
                self._groups,
            )

            remaining_switches = self.config.max_switches_per_decision
            targets = dict(tails)
            group_order = sorted(
                range(len(self._groups)),
                key=lambda index: abs(
                    desired[index]
                    - sum(
                        tails[request_id] is Location.GPU
                        for request_id in self._groups[index].cohort_ids
                    )
                ),
                reverse=True,
            )
            for index in group_order:
                group = self._groups[index]
                writer_count = desired[index]
                cohort = set(group.cohort_ids)
                gpu_set = {
                    request_id
                    for request_id in group.cohort_ids
                    if tails[request_id] is Location.GPU
                }

                def eligible(request_id: str) -> bool:
                    return (
                        current - last_switch[request_id]
                        >= self.config.min_switch_interval
                        and tail_run_tokens[request_id]
                        >= self.config.min_segment_tokens
                    )

                if len(gpu_set) > writer_count and remaining_switches:
                    removable = sorted(
                        (request_id for request_id in gpu_set if eligible(request_id)),
                        key=lambda request_id: (
                            gpu_tokens[request_id], request_id
                        ),
                        reverse=True,
                    )
                    remove_count = min(
                        len(gpu_set) - writer_count,
                        len(removable),
                        remaining_switches,
                    )
                    gpu_set.difference_update(removable[:remove_count])
                    remaining_switches -= remove_count
                elif len(gpu_set) < writer_count and remaining_switches:
                    additions = sorted(
                        (
                            request_id
                            for request_id in cohort - gpu_set
                            if eligible(request_id)
                        ),
                        key=lambda request_id: (
                            gpu_tokens[request_id], request_id
                        ),
                    )
                    add_count = min(
                        writer_count - len(gpu_set),
                        len(additions),
                        remaining_switches,
                    )
                    gpu_set.update(additions[:add_count])
                    remaining_switches -= add_count
                for request_id in group.request_ids:
                    targets[request_id] = (
                        Location.GPU if request_id in gpu_set else Location.PIM
                    )

            switches = sum(targets[key] is not tails[key] for key in targets)
            self._planned_switches += switches
            self._events[current] = targets
            for request_id, target in targets.items():
                if target is not tails[request_id]:
                    last_switch[request_id] = current
                    tail_run_tokens[request_id] = width
                else:
                    tail_run_tokens[request_id] += width
                if target is Location.GPU:
                    gpu_tokens[request_id] += width
            total_gpu_tokens = sum(gpu_tokens.values())
            self._realized_peak_gpu_tokens = max(
                self._realized_peak_gpu_tokens, total_gpu_tokens
            )
            total_pim_tokens = len(requests) * (
                requests[0].input_tokens + end
            ) - total_gpu_tokens
            if (
                total_gpu_tokens * self.cost_model.model.kv_bytes_per_token
                > self.cost_model.gpu_replica_kv_capacity_bytes
                or total_pim_tokens * self.cost_model.model.kv_bytes_per_token
                > self.cost_model.pim_kv_capacity_bytes
            ):
                raise RuntimeError(
                    "balance_curve integer tracking violates KV capacity; "
                    "increase max_switches_per_decision or use a coarser grid"
                )
            tails = dict(targets)
            current = end

        target_totals: dict[int, float] = {}
        for group in self._groups:
            for point, target in group.targets:
                target_totals[point] = target_totals.get(point, 0.0) + target
        final_targets = [group.targets[-1][1] for group in self._groups]
        self._target_peak_gpu_tokens = max(target_totals.values(), default=0.0)
        self._target_final_gpu_tokens = sum(final_targets)
        self._target_first_positive_token = next(
            (
                point
                for point, target in sorted(target_totals.items())
                if target > 0
            ),
            0,
        )
        self._realized_final_gpu_tokens = sum(gpu_tokens.values())
        self._built = True

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        if not self._built:
            self._build_plan(requests, iteration)
        targets = self._events.get(iteration)
        if targets is None:
            return []
        decisions: list[SwitchDecision] = []
        for request in requests:
            target = targets[request.request_id]
            old = request.tail_location
            if request.switch_tail(target, iteration):
                decisions.append(
                    SwitchDecision(
                        request.request_id,
                        old,
                        target,
                        "closed_form_balance_curve",
                    )
                )
        return decisions

    def summary_metadata(self) -> dict[str, object]:
        return {
            "balance_curve_solver": (
                "closed_form_crossover_plus_deterministic_tracking"
            ),
            "balance_curve_schedule_searches": 0,
            "balance_curve_coefficient_evaluations": (
                self._coefficient_evaluations
            ),
            "balance_curve_groups": len(self._groups),
            "balance_curve_cohort_size": sum(
                len(group.cohort_ids) for group in self._groups
            ),
            "balance_curve_planned_switches": self._planned_switches,
            "balance_curve_target_peak_gpu_tokens": (
                self._target_peak_gpu_tokens
            ),
            "balance_curve_target_final_gpu_tokens": (
                self._target_final_gpu_tokens
            ),
            "balance_curve_target_first_positive_token": (
                self._target_first_positive_token
            ),
            "balance_curve_realized_peak_gpu_tokens": (
                self._realized_peak_gpu_tokens
            ),
            "balance_curve_realized_final_gpu_tokens": (
                self._realized_final_gpu_tokens
            ),
        }


class GlobalExactPlacementPolicy(ElasticPlacementPolicy):
    """Offline exhaustive oracle for a small closed static batch.

    The search enumerates every eligible tail switch at every decision boundary,
    simulates each intervening decode iteration, and minimizes exact remaining
    makespan under immutable historical segments and physical block-rounded
    capacity. It intentionally ignores future arrivals and is guarded by hard
    candidate/state limits; it is a correctness reference, not a scalable
    serving policy.
    """

    def __init__(self, config: PolicyConfig, cost_model: ChimeCostModel) -> None:
        super().__init__(config, cost_model)
        self._plan: dict[int, tuple[frozenset[str], dict[str, Location]]] = {}
        self._root_objectives: dict[int, tuple[float, float]] = {}

    def _signature(
        self, requests: Sequence[RequestState], iteration: int
    ) -> tuple[object, ...]:
        return (
            iteration,
            tuple(
                (
                    request.request_id,
                    request.generated_tokens,
                    request.tail_location.value,
                    request.last_switch_iteration,
                    tuple(
                        (segment.location.value, segment.tokens)
                        for segment in request.segments
                    ),
                )
                for request in requests
            ),
        )

    def _assignments(
        self, requests: Sequence[RequestState], iteration: int
    ) -> tuple[dict[str, Location], ...]:
        eligible = [
            request for request in requests if self._eligible(request, iteration)
        ]
        if len(eligible) > self.config.exact_max_candidates:
            raise RuntimeError(
                "global_exact has "
                f"{len(eligible)} eligible requests, exceeding "
                f"exact_max_candidates={self.config.exact_max_candidates}"
            )

        # Requests with identical state and replica affinity are exchangeable
        # under the cost model. Enumerating "switch k of n" preserves exactness
        # while reducing a homogeneous batch from 2**n to n+1 candidates.
        groups: dict[tuple[object, ...], list[RequestState]] = {}
        for request in eligible:
            key = (
                request.gpu_replica,
                request.generated_tokens,
                request.remaining_tokens,
                request.tail_location.value,
                request.last_switch_iteration,
                tuple(
                    (segment.location.value, segment.tokens)
                    for segment in request.segments
                ),
            )
            groups.setdefault(key, []).append(request)

        assignments: list[dict[str, Location]] = []
        grouped_requests = tuple(groups.values())
        for switch_counts in product(
            *(range(len(group) + 1) for group in grouped_requests)
        ):
            if sum(switch_counts) > self.config.max_switches_per_decision:
                continue
            targets = {
                request.request_id: request.tail_location for request in requests
            }
            for group, switch_count in zip(grouped_requests, switch_counts):
                for request in group[:switch_count]:
                    targets[request.request_id] = request.tail_location.other
            assignments.append(targets)
        return tuple(assignments)

    def _simulate_block(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        assignment: Mapping[str, Location],
        work: list[int],
    ) -> tuple[float, list[RequestState], int] | None:
        projected = deepcopy(list(requests))
        switches = 0
        for request in projected:
            target = assignment[request.request_id]
            if request.switch_tail(target, iteration):
                switches += 1

        total_us = switches * self.config.switch_penalty_us
        steps = 0
        while projected and steps < self.config.decision_interval:
            work[0] += 1
            if work[0] > self.config.exact_max_cost_evaluations:
                raise RuntimeError(
                    "global_exact exceeded "
                    "exact_max_cost_evaluations="
                    f"{self.config.exact_max_cost_evaluations}; reduce "
                    "batch/output length, increase decision_interval, or "
                    "raise the explicit guard"
                )
            total_us += self.cost_model.iteration_cost(
                views_for_requests(projected)
            ).total_us
            for request in projected:
                request.append_generated_token()
            appended_views = views_for_requests(projected)
            if not views_fit_capacity(
                appended_views,
                self.cost_model.model,
                self.cost_model.system,
                pim_capacity_bytes=self.cost_model.pim_kv_capacity_bytes,
                gpu_replica_capacity_bytes=(
                    self.cost_model.gpu_replica_kv_capacity_bytes
                ),
            ):
                return None
            projected = [request for request in projected if not request.is_finished]
            steps += 1
        return total_us, projected, iteration + steps

    def _search(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        memo: dict[tuple[object, ...], tuple[float, list[tuple[int, frozenset[str], dict[str, Location]]]]],
        work: list[int],
    ) -> tuple[float, list[tuple[int, frozenset[str], dict[str, Location]]]]:
        if not requests:
            return 0.0, []
        signature = self._signature(requests, iteration)
        cached = memo.get(signature)
        if cached is not None:
            return cached
        best_cost = inf
        best_schedule: list[
            tuple[int, frozenset[str], dict[str, Location]]
        ] = []
        active_ids = frozenset(request.request_id for request in requests)
        for assignment in self._assignments(requests, iteration):
            block = self._simulate_block(requests, iteration, assignment, work)
            if block is None:
                continue
            block_cost, survivors, next_iteration = block
            future_cost, future_schedule = self._search(
                survivors, next_iteration, memo, work
            )
            total = block_cost + future_cost
            if total < best_cost:
                best_cost = total
                best_schedule = [
                    (iteration, active_ids, dict(assignment)),
                    *future_schedule,
                ]
        if best_cost == inf:
            raise RuntimeError(
                "global_exact found no capacity-feasible placement schedule"
            )
        memo[signature] = (best_cost, best_schedule)
        return memo[signature]

    def _stay_then_optimal_cost(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        memo: dict[tuple[object, ...], tuple[float, list[tuple[int, frozenset[str], dict[str, Location]]]]],
        work: list[int],
    ) -> float:
        stay = {
            request.request_id: request.tail_location for request in requests
        }
        block = self._simulate_block(requests, iteration, stay, work)
        if block is None:
            return inf
        block_cost, survivors, next_iteration = block
        future_cost, _ = self._search(survivors, next_iteration, memo, work)
        return block_cost + future_cost

    def _build_plan(
        self, requests: Sequence[RequestState], iteration: int
    ) -> None:
        memo: dict[
            tuple[object, ...],
            tuple[
                float,
                list[tuple[int, frozenset[str], dict[str, Location]]],
            ],
        ] = {}
        work = [0]
        best_cost, schedule = self._search(
            deepcopy(list(requests)), iteration, memo, work
        )
        stay_cost = self._stay_then_optimal_cost(
            deepcopy(list(requests)), iteration, memo, work
        )
        self._plan = {
            step_iteration: (active_ids, targets)
            for step_iteration, active_ids, targets in schedule
        }
        self._root_objectives[iteration] = (stay_cost, best_cost)

    def decide(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        force: bool = False,
    ) -> list[SwitchDecision]:
        if not requests or (
            not force and iteration % self.config.decision_interval
        ):
            return []
        active_ids = frozenset(request.request_id for request in requests)
        planned = self._plan.get(iteration)
        if planned is None or planned[0] != active_ids:
            self._build_plan(requests, iteration)
            planned = self._plan[iteration]

        targets = planned[1]
        before, after = self._root_objectives.get(iteration, (None, None))
        decisions: list[SwitchDecision] = []
        for request in requests:
            target = targets[request.request_id]
            old = request.tail_location
            if request.switch_tail(target, iteration):
                decisions.append(
                    SwitchDecision(
                        request.request_id,
                        old,
                        target,
                        "global_exact_makespan",
                        before,
                        after,
                    )
                )
        return decisions

def build_policy(config: PolicyConfig, cost_model: ChimeCostModel) -> PlacementPolicy:
    normalized = config.name.lower().replace("-", "_")
    if normalized in ("chime", "baseline"):
        return ChimePlacementPolicy(config, cost_model)
    if normalized in ("elastic", "elasticafd", "elastic_afd"):
        if config.optimizer == "forced_fraction":
            return ForcedFractionPlacementPolicy(config, cost_model)
        if config.optimizer == "online_static_fraction":
            return OnlineStaticFractionPlacementPolicy(config, cost_model)
        if config.optimizer == "lifetime_greedy":
            return LifetimeGreedyPlacementPolicy(config, cost_model)
        if config.optimizer == "rolling_beam":
            return RollingBeamPlacementPolicy(config, cost_model)
        if config.optimizer == "global_exact":
            return GlobalExactPlacementPolicy(config, cost_model)
        if config.optimizer == "homogeneous_modes":
            return HomogeneousModePlacementPolicy(config, cost_model)
        if config.optimizer == "balance_curve":
            return BalanceCurvePlacementPolicy(config, cost_model)
        return ElasticPlacementPolicy(config, cost_model)
    raise ValueError("policy.name must be chime or elastic")
