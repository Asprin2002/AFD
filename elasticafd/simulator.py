"""Iteration-level discrete-event simulator for CHIME and ElasticAFD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence
from math import ceil
from time import perf_counter

from .capacity import (
    requests_gpu_memory_usage_by_replica,
    requests_memory_usage,
)
from .config import ProjectConfig
from .cost import ChimeCostModel, IterationCost
from .mapping import ExecutionTarget
from .policy import PlacementPolicy, SwitchDecision, build_policy
from .specs import GiB
from .state import Location, RequestState, views_for_requests
from .workload import WorkloadRequest, build_workload


class SimulationCapacityError(RuntimeError):
    pass


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, Any]
    requests: list[dict[str, Any]]
    iterations: list[dict[str, Any]]
    switches: list[dict[str, Any]]
    pipeline_events: list[dict[str, Any]]


class Simulator:
    """Continuous-batching decode simulator.

    Arrival timestamps mean "decode-ready": prompt computation is intentionally
    outside this first version. Input length still determines the initial KV
    allocation and every decode attention scan.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.cost_model = ChimeCostModel(
            config.model,
            config.system,
            config.simulation.fidelity,
            pipeline_model=config.simulation.pipeline_model,
            subbatch_scheduler=config.simulation.subbatch_scheduler,
            subbatch_progression=config.simulation.subbatch_progression,
            operator_mapping=config.simulation.operator_mapping,
            split_overhead_scale=config.simulation.split_overhead_scale,
        )
        policy_pipeline_model = (
            config.simulation.policy_pipeline_model
            or config.simulation.pipeline_model
        )
        self.policy_cost_model = (
            self.cost_model
            if policy_pipeline_model == config.simulation.pipeline_model
            else ChimeCostModel(
                config.model,
                config.system,
                config.simulation.fidelity,
                pipeline_model=policy_pipeline_model,
                subbatch_scheduler=config.simulation.subbatch_scheduler,
                subbatch_progression=config.simulation.subbatch_progression,
                operator_mapping=config.simulation.operator_mapping,
                split_overhead_scale=config.simulation.split_overhead_scale,
            )
        )
        normalized_policy = config.simulation.policy.name.lower().replace("-", "_")
        if (
            self.cost_model.operator_mapping.name != "chime"
            and normalized_policy in ("elastic", "elasticafd", "elastic_afd")
        ):
            raise ValueError(
                "elastic KV-segment placement currently requires operator_mapping=chime"
            )
        if (
            self.cost_model.operator_mapping.name != "chime"
            and config.simulation.subbatch_scheduler == "alignment"
        ):
            raise ValueError(
                "alignment subbatch scheduling is currently defined for "
                "operator_mapping=chime"
            )
        if self.cost_model.pim_kv_capacity_bytes < 0:
            raise ValueError(
                "PIM-resident operator weights exceed the configured PIM capacity"
            )
        self.policy: PlacementPolicy = build_policy(
            config.simulation.policy, self.policy_cost_model
        )
        base_dir = config.source_path.parent if config.source_path else Path.cwd()
        self.workload: list[WorkloadRequest] = build_workload(
            config.workload, base_dir
        )
        if not self.workload:
            raise ValueError("workload is empty")
        if (
            config.workload.type in ("static_batch", "closed_batch")
            and len(self.workload) > config.simulation.max_batch_size
        ):
            raise ValueError(
                "closed workload size exceeds simulation.max_batch_size; that would "
                "silently turn the experiment into continuous batching"
            )

        self.clock_us = 0.0
        self.iteration = 0
        self.pending: list[WorkloadRequest] = list(self.workload)
        self.active: list[RequestState] = []
        self.completed: list[RequestState] = []
        self.pim_used_bytes = 0.0
        self.gpu_used_bytes = 0.0
        self.peak_pim_bytes = 0.0
        self.peak_gpu_bytes = 0.0
        self.peak_gpu_replica_bytes = [0.0] * config.model.data_parallel
        self.peak_pim_logical_bytes = 0.0
        self.peak_gpu_logical_bytes = 0.0
        self.pim_reserved_bytes = 0.0
        self.peak_pim_reserved_bytes = 0.0
        self._pim_reservations: dict[str, float] = {}
        self.iteration_rows: list[dict[str, Any]] = []
        self.switch_rows: list[dict[str, Any]] = []
        self.pipeline_event_rows: list[dict[str, Any]] = []
        self._last_iteration_row: dict[str, Any] | None = None
        self._sum_batch = 0
        self._sum_context = 0
        self._sum_gpu_busy_us = 0.0
        self._sum_pim_busy_us = 0.0
        self._sum_link_busy_us = 0.0
        self._gpu_bottleneck_phases = 0
        self._pim_bottleneck_phases = 0
        self._total_output_tokens = 0
        self._policy_wall_time_s = 0.0
        self._event_policy_decisions = 0
        self._switch_execution_us = 0.0
        self._sum_af_ta_us = 0.0
        self._sum_af_tf_us = 0.0
        self._sum_af_ideal_stage_us = 0.0
        self._sum_af_imbalance_us = 0.0
        self._sum_iteration_cost_us = 0.0
        self._sum_h2d_bytes = 0
        self._sum_d2h_bytes = 0
        self._sum_pim_scan_bytes = 0
        self._sum_gpu_scan_bytes = 0

    @property
    def token_bytes(self) -> int:
        return self.config.model.kv_bytes_per_token

    @property
    def pim_free_bytes(self) -> float:
        return self.cost_model.pim_kv_capacity_bytes - self.pim_used_bytes

    @property
    def gpu_capacity_bytes(self) -> float:
        return self.cost_model.gpu_kv_capacity_bytes

    @property
    def gpu_replica_capacity_bytes(self) -> float:
        return self.cost_model.gpu_replica_kv_capacity_bytes

    @property
    def gpu_free_bytes(self) -> float:
        return self.gpu_capacity_bytes - self.gpu_used_bytes

    def _record_switches(self, decisions: Sequence[SwitchDecision]) -> None:
        by_id = {request.request_id: request for request in self.active}
        for decision in decisions:
            request = by_id.get(decision.request_id)
            self.switch_rows.append(
                {
                    "iteration": self.iteration,
                    "time_ms": self.clock_us / 1e3,
                    "request_id": decision.request_id,
                    "old_location": decision.old_location.value,
                    "new_location": decision.new_location.value,
                    "reason": decision.reason,
                    "context_tokens": request.context_tokens if request else None,
                    "pim_tokens": (
                        request.tokens_at(Location.PIM) if request else None
                    ),
                    "gpu_tokens": (
                        request.tokens_at(Location.GPU) if request else None
                    ),
                    "segments_before_append": (
                        len(request.segments) if request else None
                    ),
                    "objective_before_us": decision.objective_before_us,
                    "objective_after_us": decision.objective_after_us,
                    "predicted_gain_us": (
                        decision.objective_before_us - decision.objective_after_us
                        if decision.objective_before_us is not None
                        and decision.objective_after_us is not None
                        else None
                    ),
                    "execution_overhead_us": (
                        self.config.simulation.policy.switch_execution_us
                    ),
                }
            )

    def _apply_switch_decisions(
        self, decisions: Sequence[SwitchDecision]
    ) -> None:
        """Record and charge runtime control cost for tail-location changes."""

        self._record_switches(decisions)
        overhead = (
            len(decisions)
            * self.config.simulation.policy.switch_execution_us
        )
        self.clock_us += overhead
        self._switch_execution_us += overhead

    def _select_gpu_replica(self, requests: Sequence[RequestState]) -> int:
        counts = [0] * self.config.model.data_parallel
        for request in requests:
            counts[request.gpu_replica] += 1
        gpu_bytes = requests_gpu_memory_usage_by_replica(
            requests, self.config.model, self.config.system
        )
        return min(
            range(self.config.model.data_parallel),
            key=lambda replica: (counts[replica], gpu_bytes[replica], replica),
        )

    def _make_state(
        self,
        item: WorkloadRequest,
        initial: Location,
        colocated_with: Sequence[RequestState] = (),
    ) -> RequestState:
        request = item.to_state()
        request.gpu_replica = self._select_gpu_replica(colocated_with)
        if initial is Location.GPU:
            request.segments[0].location = Location.GPU
            request.tail_location = Location.GPU
        request.admitted_us = self.clock_us
        return request

    @property
    def _initial_kv_location(self) -> Location:
        target = self.cost_model.operator_mapping.attention
        return Location.GPU if target is ExecutionTarget.GPU else Location.PIM

    def _target_pim_reservation_bytes(self, item: WorkloadRequest) -> int:
        block = self.config.system.pim.kv_block_tokens
        target_tokens = item.input_tokens + item.output_tokens
        return ceil(target_tokens / block) * block * self.token_bytes

    def _alignment_admission_order(
        self, ready: Sequence[WorkloadRequest]
    ) -> list[WorkloadRequest]:
        """Greedily order decode-ready requests by projected token throughput.

        This is a decode-only approximation of CHIME's request-selection
        problem.  It cannot model the paper's prefill/chunk alignment until a
        prefill execution path exists.
        """

        if len(ready) <= 1:
            return list(ready)
        window_size = min(
            len(ready), self.config.simulation.admission_lookahead
        )
        candidates = list(ready[:window_size])
        deferred = list(ready[window_size:])
        selected: list[WorkloadRequest] = []
        selected_states: list[RequestState] = []
        slots = max(0, self.config.simulation.max_batch_size - len(self.active))
        starvation_us = self.config.simulation.admission_starvation_ms * 1e3

        while candidates and len(selected) < slots:
            starving = [
                item
                for item in candidates
                if self.clock_us - item.arrival_us >= starvation_us
            ]
            if starving:
                choice = min(starving, key=lambda item: item.arrival_us)
            else:
                scored: list[tuple[float, float, str, WorkloadRequest]] = []
                for item in candidates:
                    state = self._make_state(
                        item,
                        self._initial_kv_location,
                        [*self.active, *selected_states],
                    )
                    trial = [*self.active, *selected_states, state]
                    if not self._requests_fit_capacity(trial):
                        objective = float("inf")
                    else:
                        cost = self.cost_model.iteration_cost(
                            views_for_requests(trial)
                        )
                        objective = cost.total_us / len(trial)
                    scored.append(
                        (objective, item.arrival_us, item.request_id, item)
                    )
                choice = min(scored, key=lambda value: value[:3])[3]
            selected.append(choice)
            selected_states.append(
                self._make_state(
                    choice,
                    self._initial_kv_location,
                    [*self.active, *selected_states],
                )
            )
            candidates.remove(choice)

        return [*selected, *candidates, *deferred]

    def _admit_ready(self) -> int:
        if len(self.active) >= self.config.simulation.max_batch_size:
            return 0
        admitted = 0
        ready = [item for item in self.pending if item.arrival_us <= self.clock_us]
        future = [item for item in self.pending if item.arrival_us > self.clock_us]
        if self.config.simulation.admission_scheduler == "alignment":
            ready = self._alignment_admission_order(ready)
        ordered_pending = [*ready, *future]
        retained: list[WorkloadRequest] = []
        for item in ordered_pending:
            if item.arrival_us > self.clock_us:
                retained.append(item)
                continue
            if len(self.active) >= self.config.simulation.max_batch_size:
                retained.append(item)
                continue

            primary_candidate = self._make_state(
                item, self._initial_kv_location, self.active
            )
            pim_bytes, gpu_bytes = requests_memory_usage(
                [*self.active, primary_candidate], self.config.model, self.config.system
            )
            reservation = 0
            reservation_fits = True
            if self.config.simulation.admission_reservation == "pim_target":
                reservation = self._target_pim_reservation_bytes(item)
                reservation_fits = (
                    self.pim_reserved_bytes + reservation
                    <= self.cost_model.pim_kv_capacity_bytes
                )
            if (
                self._requests_fit_capacity([*self.active, primary_candidate])
                and reservation_fits
            ):
                request = primary_candidate
                self.pim_used_bytes = pim_bytes
                self.gpu_used_bytes = gpu_bytes
                if reservation:
                    self._pim_reservations[request.request_id] = reservation
                    self.pim_reserved_bytes += reservation
            elif (
                self.policy.allows_gpu_kv
                and self.config.simulation.admission_reservation == "none"
                and self.config.simulation.allow_gpu_prompt_fallback
            ):
                # Capacity fallback: prefill has just run on the GPU, so retaining
                # its KV is cheaper than moving an oversized prompt to PIM.
                gpu_candidate = self._make_state(item, Location.GPU, self.active)
                pim_bytes, gpu_bytes = requests_memory_usage(
                    [*self.active, gpu_candidate],
                    self.config.model,
                    self.config.system,
                )
                if not self._requests_fit_capacity([*self.active, gpu_candidate]):
                    retained.append(item)
                    continue
                request = gpu_candidate
                self.pim_used_bytes = pim_bytes
                self.gpu_used_bytes = gpu_bytes
            else:
                retained.append(item)
                continue
            self.active.append(request)
            admitted += 1

        self.pending = sorted(retained, key=lambda item: item.arrival_us)
        self._update_peaks()
        if (
            self.config.workload.type in ("static_batch", "closed_batch")
            and self.iteration == 0
            and self.clock_us == 0.0
            and admitted != len(ready)
        ):
            raise SimulationCapacityError(
                "a closed decode-ready batch must be admitted atomically; "
                "its complete prompt-KV working set does not fit"
            )
        return admitted

    def _requests_fit_capacity(
        self, requests: Sequence[RequestState]
    ) -> bool:
        pim_bytes, _ = requests_memory_usage(
            requests, self.config.model, self.config.system
        )
        if pim_bytes > self.cost_model.pim_kv_capacity_bytes:
            return False
        return all(
            used <= self.gpu_replica_capacity_bytes
            for used in requests_gpu_memory_usage_by_replica(
                requests, self.config.model, self.config.system
            )
        )

    def _update_peaks(self) -> None:
        self.peak_pim_bytes = max(self.peak_pim_bytes, self.pim_used_bytes)
        self.peak_gpu_bytes = max(self.peak_gpu_bytes, self.gpu_used_bytes)
        gpu_replica_bytes = requests_gpu_memory_usage_by_replica(
            self.active, self.config.model, self.config.system
        )
        self.peak_gpu_replica_bytes = [
            max(peak, used)
            for peak, used in zip(self.peak_gpu_replica_bytes, gpu_replica_bytes)
        ]
        pim_logical = (
            sum(request.tokens_at(Location.PIM) for request in self.active)
            * self.token_bytes
        )
        gpu_logical = (
            sum(request.tokens_at(Location.GPU) for request in self.active)
            * self.token_bytes
        )
        self.peak_pim_logical_bytes = max(
            self.peak_pim_logical_bytes, pim_logical
        )
        self.peak_gpu_logical_bytes = max(
            self.peak_gpu_logical_bytes, gpu_logical
        )
        self.peak_pim_reserved_bytes = max(
            self.peak_pim_reserved_bytes, self.pim_reserved_bytes
        )

    def _ensure_progress_capacity(self) -> None:
        try:
            decisions = self.policy.relieve_capacity_pressure(
                self.active,
                self.iteration,
                self.pim_free_bytes,
                self.gpu_free_bytes,
            )
        except RuntimeError as error:
            raise SimulationCapacityError(str(error)) from error
        self._apply_switch_decisions(decisions)

    def _append_and_complete(self) -> int:
        for request in self.active:
            if request.generated_tokens == 0:
                request.first_token_us = self.clock_us
            request.append_generated_token()
            self._total_output_tokens += 1

        # Capture block-rounded occupancy before requests finishing on this step
        # release their KV.
        self.pim_used_bytes, self.gpu_used_bytes = requests_memory_usage(
            self.active, self.config.model, self.config.system
        )
        self._update_peaks()

        survivors: list[RequestState] = []
        for request in self.active:
            if request.is_finished:
                request.finish_us = self.clock_us
                self.pim_reserved_bytes -= self._pim_reservations.pop(
                    request.request_id, 0.0
                )
                self.completed.append(request)
            else:
                survivors.append(request)
        completed_count = len(self.active) - len(survivors)
        self.active = survivors
        self.pim_used_bytes, self.gpu_used_bytes = requests_memory_usage(
            self.active, self.config.model, self.config.system
        )
        return completed_count

    def _iteration_row(
        self,
        start_us: float,
        cost: IterationCost,
        switch_count: int,
        batch_size: int,
        avg_context: float,
        pim_before: float,
        gpu_before: float,
    ) -> dict[str, Any]:
        ta_us, tf_us, ideal_stage_us, imbalance_us = self._af_components(cost)
        layers = self.config.model.layers
        h2d_bytes = sum(batch.pim.h2d_bytes for batch in cost.subbatches) * layers
        d2h_bytes = sum(batch.pim.d2h_bytes for batch in cost.subbatches) * layers
        pim_scan_bytes = (
            sum(batch.pim.physical_kv_bytes for batch in cost.subbatches) * layers
        )
        gpu_scan_bytes = (
            sum(
                batch.gpu_attention.physical_kv_bytes
                for batch in cost.subbatches
            )
            * layers
        )
        return {
            "iteration": self.iteration,
            "start_ms": start_us / 1e3,
            "duration_ms": cost.total_us / 1e3,
            "batch_size": batch_size,
            "avg_context_tokens": avg_context,
            "pim_kv_gib": pim_before / GiB,
            "gpu_kv_gib": gpu_before / GiB,
            "pim_reserved_gib": self.pim_reserved_bytes / GiB,
            "switches": switch_count,
            "per_layer_us": cost.per_layer_us,
            "dense_busy_ms": cost.dense_us / 1e3,
            "gpu_attention_busy_ms": cost.gpu_attention_us / 1e3,
            "pim_core_busy_ms": cost.pim_core_busy_us / 1e3,
            "link_busy_ms": cost.link_busy_us / 1e3,
            "merge_busy_ms": cost.merge_us / 1e3,
            "split_requests": sum(batch.split_requests for batch in cost.subbatches),
            "gpu_bottleneck_phases": cost.phase_gpu_bottlenecks,
            "pim_bottleneck_phases": cost.phase_pim_bottlenecks,
            "pipeline_model": cost.pipeline_model,
            "af_ta_work_ms": ta_us / 1e3,
            "af_tf_work_ms": tf_us / 1e3,
            "af_ideal_stage_ms": ideal_stage_us / 1e3,
            "af_imbalance_ms": imbalance_us / 1e3,
            "af_actual_minus_ideal_ms": (cost.total_us - ideal_stage_us) / 1e3,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "pim_kv_scan_bytes": pim_scan_bytes,
            "gpu_kv_scan_bytes": gpu_scan_bytes,
        }

    def _af_components(self, cost: IterationCost) -> tuple[float, float, float, float]:
        """Return an ideal resource lower bound for one decode iteration.

        GPU busy time contains dense work, GPU-local Attention and merge.  The
        A side is limited independently by its busiest PIM rankset and the
        GPU-PIM link.  Taking the larger resource demand on each side preserves
        CHIME's layer/rankset interleaving; summing per-layer PIM latency would
        incorrectly serialize ranksets and can exceed an event-DAG makespan.
        """

        if len(cost.subbatches) not in (1, 2):
            raise AssertionError("CHIME A/F accounting expects one or two subbatches")
        ta_us = max(cost.pim_core_busy_us, cost.link_busy_us)
        tf_us = cost.gpu_busy_us
        ideal_stage_us = max(ta_us, tf_us)
        imbalance_us = abs(ta_us - tf_us)
        return ta_us, tf_us, ideal_stage_us, imbalance_us

    def _run_iteration(self, *, force_policy_decision: bool = False) -> int:
        switch_start = len(self.switch_rows)
        self.policy.update_runtime_context(
            clock_us=self.clock_us,
            pending=self.pending,
            max_batch_size=self.config.simulation.max_batch_size,
            initial_location=self._initial_kv_location,
        )
        policy_start = perf_counter()
        decisions = self.policy.decide(
            self.active,
            self.iteration,
            force=force_policy_decision,
        )
        if force_policy_decision:
            self._event_policy_decisions += 1
        self._policy_wall_time_s += perf_counter() - policy_start
        self._apply_switch_decisions(decisions)
        self._ensure_progress_capacity()

        views = views_for_requests(self.active)
        cost = self.cost_model.iteration_cost(views)
        start_us = self.clock_us
        batch_size = len(self.active)
        avg_context = fmean(request.context_tokens for request in self.active)
        pim_before = self.pim_used_bytes
        gpu_before = self.gpu_used_bytes
        self.clock_us += cost.total_us

        completed = self._append_and_complete()

        self._sum_batch += batch_size
        self._sum_context += sum(view.context_tokens for view in views)
        self._sum_gpu_busy_us += cost.gpu_busy_us
        self._sum_pim_busy_us += cost.pim_core_busy_us
        self._sum_link_busy_us += cost.link_busy_us
        ta_us, tf_us, ideal_stage_us, imbalance_us = self._af_components(cost)
        self._sum_af_ta_us += ta_us
        self._sum_af_tf_us += tf_us
        self._sum_af_ideal_stage_us += ideal_stage_us
        self._sum_af_imbalance_us += imbalance_us
        self._sum_iteration_cost_us += cost.total_us
        layers = self.config.model.layers
        self._sum_h2d_bytes += (
            sum(batch.pim.h2d_bytes for batch in cost.subbatches) * layers
        )
        self._sum_d2h_bytes += (
            sum(batch.pim.d2h_bytes for batch in cost.subbatches) * layers
        )
        self._sum_pim_scan_bytes += (
            sum(batch.pim.physical_kv_bytes for batch in cost.subbatches) * layers
        )
        self._sum_gpu_scan_bytes += (
            sum(
                batch.gpu_attention.physical_kv_bytes
                for batch in cost.subbatches
            )
            * layers
        )
        self._gpu_bottleneck_phases += cost.phase_gpu_bottlenecks
        self._pim_bottleneck_phases += cost.phase_pim_bottlenecks
        row = self._iteration_row(
            start_us,
            cost,
            len(self.switch_rows) - switch_start,
            batch_size,
            avg_context,
            pim_before,
            gpu_before,
        )
        self._last_iteration_row = row
        if self.iteration % self.config.simulation.record_every == 0:
            self.iteration_rows.append(row)
            if self.config.simulation.record_pipeline_events:
                for event in cost.events:
                    self.pipeline_event_rows.append(
                        {
                            "iteration": self.iteration,
                            "task_id": event.task_id,
                            "kind": event.kind,
                            "layer": event.layer,
                            "subbatch": event.subbatch,
                            "start_us_in_iteration": event.start_us,
                            "end_us_in_iteration": event.end_us,
                            "start_ms_global": (start_us + event.start_us) / 1e3,
                            "end_ms_global": (start_us + event.end_us) / 1e3,
                            "duration_us": event.duration_us,
                            "resources": "|".join(event.resources),
                        }
                    )
        self.iteration += 1
        return completed

    def run(self) -> SimulationResult:
        status = "completed"
        completed_last_iteration = 0
        while self.pending or self.active:
            if self.config.simulation.max_iterations is not None and (
                self.iteration >= self.config.simulation.max_iterations
            ):
                status = "max_iterations"
                break

            if not self.active:
                if self.pending[0].arrival_us > self.clock_us:
                    self.clock_us = self.pending[0].arrival_us
                admitted = self._admit_ready()
                if admitted == 0:
                    first = self.pending[0]
                    raise SimulationCapacityError(
                        f"request {first.request_id} has a {first.input_tokens}-token "
                        "prompt that cannot be admitted into the available KV tiers"
                    )
            else:
                admitted = self._admit_ready()

            event_replan = (
                self.config.simulation.policy.event_driven_replan
                and (admitted > 0 or completed_last_iteration > 0)
            )
            completed_last_iteration = self._run_iteration(
                force_policy_decision=event_replan
            )

        if (
            self._last_iteration_row is not None
            and (
                not self.iteration_rows
                or self.iteration_rows[-1]["iteration"]
                != self._last_iteration_row["iteration"]
            )
        ):
            self.iteration_rows.append(self._last_iteration_row)
        return self._result(status)

    def _request_rows(self) -> list[dict[str, Any]]:
        states = self.completed + self.active
        rows: list[dict[str, Any]] = []
        for request in sorted(states, key=lambda item: item.request_id):
            admitted = request.admitted_us if request.admitted_us is not None else 0.0
            first = request.first_token_us if request.first_token_us is not None else self.clock_us
            finish = request.finish_us if request.finish_us is not None else self.clock_us
            queue_ms = (admitted - request.arrival_us) / 1e3
            e2e_ms = (finish - request.arrival_us) / 1e3
            mean_tpot_ms = (finish - admitted) / max(
                1, request.generated_tokens
            ) / 1e3
            slo_checks = []
            if self.config.simulation.slo_queue_ms is not None:
                slo_checks.append(queue_ms <= self.config.simulation.slo_queue_ms)
            if self.config.simulation.slo_tpot_ms is not None:
                slo_checks.append(
                    mean_tpot_ms <= self.config.simulation.slo_tpot_ms
                )
            if self.config.simulation.slo_e2e_ms is not None:
                slo_checks.append(e2e_ms <= self.config.simulation.slo_e2e_ms)
            rows.append(
                {
                    "request_id": request.request_id,
                    "gpu_replica": request.gpu_replica,
                    "arrival_ms": request.arrival_us / 1e3,
                    "admitted_ms": admitted / 1e3,
                    "first_token_ms": first / 1e3,
                    "finish_ms": finish / 1e3,
                    "queue_ms": queue_ms,
                    "first_token_latency_ms": (first - request.arrival_us) / 1e3,
                    "decode_latency_ms": (finish - admitted) / 1e3,
                    "e2e_from_decode_ready_ms": e2e_ms,
                    "mean_tpot_ms": mean_tpot_ms,
                    "meets_slo": (
                        request.is_finished
                        and bool(slo_checks)
                        and all(slo_checks)
                    ),
                    "input_tokens": request.input_tokens,
                    "target_output_tokens": request.output_tokens,
                    "generated_tokens": request.generated_tokens,
                    "switches": request.switch_count,
                    "segments": len(request.segments),
                    "pim_tokens_final": request.tokens_at(Location.PIM),
                    "gpu_tokens_final": request.tokens_at(Location.GPU),
                    "segment_layout": "|".join(
                        f"{segment.location.value}:{segment.start_token}-{segment.end_token}"
                        for segment in request.segments
                    ),
                    "completed": request.is_finished,
                }
            )
        return rows

    def _result(self, status: str) -> SimulationResult:
        request_rows = self._request_rows()
        completed_rows = [row for row in request_rows if row["completed"]]
        e2e = [float(row["e2e_from_decode_ready_ms"]) for row in completed_rows]
        tpot = [float(row["mean_tpot_ms"]) for row in completed_rows]
        first_token = [float(row["first_token_latency_ms"]) for row in completed_rows]
        queue = [float(row["queue_ms"]) for row in completed_rows]
        slo_evaluated = any(
            threshold is not None
            for threshold in (
                self.config.simulation.slo_queue_ms,
                self.config.simulation.slo_tpot_ms,
                self.config.simulation.slo_e2e_ms,
            )
        )
        slo_compliant = (
            sum(bool(row["meets_slo"]) for row in completed_rows)
            if slo_evaluated
            else 0
        )
        total_s = self.clock_us / 1e6
        arrivals = [request.arrival_us for request in self.workload]
        arrival_span_s = (max(arrivals) - min(arrivals)) / 1e6
        trace_offered_rate = (
            (len(arrivals) - 1) / arrival_span_s
            if len(arrivals) > 1 and arrival_span_s > 0
            else 0.0
        )
        slo_attainment = (
            slo_compliant / len(completed_rows)
            if slo_evaluated and completed_rows
            else None
        )
        phase_count = self._gpu_bottleneck_phases + self._pim_bottleneck_phases
        summary = {
            "status": status,
            "cost_model_revision": "v0.7.1-event-cache",
            "parameter_policy": (
                "paper-derived parameters plus Fig.17 dense/link effective-bandwidth "
                "calibration; see PARAMETER_PROVENANCE.md"
            ),
            "model": self.config.model.name,
            "policy": self.config.simulation.policy.name,
            "policy_optimizer": (
                self.config.simulation.policy.optimizer
                if self.config.simulation.policy.name.lower().replace("-", "_")
                in ("elastic", "elasticafd", "elastic_afd")
                else "none"
            ),
            "policy_prediction": self.config.simulation.policy.prediction,
            "policy_decision_interval": (
                self.config.simulation.policy.decision_interval
            ),
            "policy_event_driven_replan": (
                self.config.simulation.policy.event_driven_replan
            ),
            "event_policy_decisions": self._event_policy_decisions,
            "policy_beam_depth": self.config.simulation.policy.beam_depth,
            "policy_beam_width": self.config.simulation.policy.beam_width,
            "policy_beam_terminal_mode": (
                self.config.simulation.policy.beam_terminal_mode
            ),
            "policy_beam_include_future_arrivals": (
                self.config.simulation.policy.beam_include_future_arrivals
            ),
            "fidelity": self.config.simulation.fidelity,
            "pipeline_model": (
                self.config.simulation.pipeline_model
                if self.cost_model.operator_mapping.name == "chime"
                else "operator_event"
            ),
            "pipeline_model_requested": self.config.simulation.pipeline_model,
            "policy_pipeline_model": self.policy_cost_model.pipeline_model,
            "subbatch_scheduler": self.config.simulation.subbatch_scheduler,
            "subbatch_progression": self.config.simulation.subbatch_progression,
            "event_cost_cache_hits": self.cost_model.event_cost_cache_stats["hits"],
            "event_cost_cache_misses": self.cost_model.event_cost_cache_stats["misses"],
            "event_cost_cache_entries": self.cost_model.event_cost_cache_stats["entries"],
            "operator_mapping": self.cost_model.operator_mapping.name,
            "operator_mapping_label": self.cost_model.operator_mapping.label,
            "operator_mapping_provenance": (
                self.cost_model.operator_mapping.provenance
            ),
            "workload": self.config.workload.type,
            "arrival_semantics": "decode_ready; prefill latency is not modeled",
            "admission_reservation": self.config.simulation.admission_reservation,
            "admission_scheduler": self.config.simulation.admission_scheduler,
            "allow_gpu_prompt_fallback": (
                self.config.simulation.allow_gpu_prompt_fallback
            ),
            "admission_semantics": (
                "decode-ready cost-guided approximation; prefill is not modeled"
                if self.config.simulation.admission_scheduler == "alignment"
                else "FCFS"
            ),
            "requests": len(self.workload),
            "completed_requests": len(completed_rows),
            "iterations": self.iteration,
            "output_tokens": self._total_output_tokens,
            "simulated_time_s": total_s,
            "throughput_tokens_s": self._total_output_tokens / max(total_s, 1e-12),
            "request_completion_rate_rps": len(completed_rows)
            / max(total_s, 1e-12),
            "trace_arrival_span_s": arrival_span_s,
            "trace_offered_rate_rps": trace_offered_rate,
            "drain_time_after_last_arrival_s": (
                max(0.0, self.clock_us - max(arrivals))
            )
            / 1e6,
            "average_batch_size": self._sum_batch / max(self.iteration, 1),
            "average_context_tokens": self._sum_context
            / max(self._sum_batch, 1),
            "gpu_path_utilization": self._sum_gpu_busy_us
            / max(self.clock_us, 1e-12),
            "pim_core_path_utilization": self._sum_pim_busy_us
            / max(self.clock_us, 1e-12),
            "link_path_utilization": self._sum_link_busy_us
            / max(self.clock_us, 1e-12),
            "gpu_bottleneck_phase_fraction": self._gpu_bottleneck_phases
            / max(phase_count, 1),
            "pim_bottleneck_phase_fraction": self._pim_bottleneck_phases
            / max(phase_count, 1),
            "pim_total_capacity_gib": self.config.system.pim.capacity_bytes / GiB,
            "pim_persistent_weight_gib": (
                self.cost_model.pim_persistent_weight_bytes / GiB
            ),
            "pim_kv_capacity_gib": self.cost_model.pim_kv_capacity_bytes / GiB,
            "gpu_kv_capacity_gib": self.gpu_capacity_bytes / GiB,
            "gpu_replica_kv_capacity_gib": (
                self.gpu_replica_capacity_bytes / GiB
            ),
            "peak_gpu_replica_kv_gib": [
                value / GiB for value in self.peak_gpu_replica_bytes
            ],
            "gpu_offloaded_weight_gib": (
                self.cost_model.gpu_offloaded_weight_bytes / GiB
            ),
            "peak_pim_kv_gib": self.peak_pim_bytes / GiB,
            "peak_gpu_kv_gib": self.peak_gpu_bytes / GiB,
            "peak_pim_logical_kv_gib": self.peak_pim_logical_bytes / GiB,
            "peak_gpu_logical_kv_gib": self.peak_gpu_logical_bytes / GiB,
            "peak_pim_allocation_overhead_gib": (
                self.peak_pim_bytes - self.peak_pim_logical_bytes
            )
            / GiB,
            "peak_gpu_allocation_overhead_gib": (
                self.peak_gpu_bytes - self.peak_gpu_logical_bytes
            )
            / GiB,
            "peak_pim_reserved_gib": self.peak_pim_reserved_bytes / GiB,
            "tail_switches": len(self.switch_rows),
            "switch_execution_us": (
                self.config.simulation.policy.switch_execution_us
            ),
            "switch_execution_time_s": self._switch_execution_us / 1e6,
            "policy_switch_penalty_us": (
                self.config.simulation.policy.switch_penalty_us
            ),
            "gpu_segment_setup_us": self.config.system.gpu.segment_setup_us,
            "pim_segment_setup_us": self.config.system.pim.segment_setup_us,
            "recorded_pipeline_events": len(self.pipeline_event_rows),
            "policy_search_wall_time_s": self._policy_wall_time_s,
            "split_overhead_scale": self.config.simulation.split_overhead_scale,
            "af_ta_work_s": self._sum_af_ta_us / 1e6,
            "af_tf_work_s": self._sum_af_tf_us / 1e6,
            "af_ideal_stage_s": self._sum_af_ideal_stage_us / 1e6,
            "af_iteration_cost_s": self._sum_iteration_cost_us / 1e6,
            "af_actual_minus_ideal_s": (
                self._sum_iteration_cost_us - self._sum_af_ideal_stage_us
            ) / 1e6,
            "af_imbalance_s": self._sum_af_imbalance_us / 1e6,
            "af_balance_ratio": min(self._sum_af_ta_us, self._sum_af_tf_us)
            / max(self._sum_af_ta_us, self._sum_af_tf_us, 1e-12),
            "h2d_transfer_gib": self._sum_h2d_bytes / GiB,
            "d2h_transfer_gib": self._sum_d2h_bytes / GiB,
            "pim_kv_scan_gib": self._sum_pim_scan_bytes / GiB,
            "gpu_kv_scan_gib": self._sum_gpu_scan_bytes / GiB,
            "slo": {
                "evaluated": slo_evaluated,
                "queue_ms": self.config.simulation.slo_queue_ms,
                "tpot_ms": self.config.simulation.slo_tpot_ms,
                "e2e_ms": self.config.simulation.slo_e2e_ms,
                "attainment_target": (
                    self.config.simulation.slo_attainment_target
                ),
                "compliant_requests": slo_compliant,
                "attainment": slo_attainment,
                "goodput_rps": (
                    slo_compliant / max(total_s, 1e-12)
                    if slo_evaluated
                    else None
                ),
                "target_met": (
                    slo_attainment >= self.config.simulation.slo_attainment_target
                    if slo_attainment is not None
                    else None
                ),
            },
            "latency_ms": {
                "first_token_mean": fmean(first_token) if first_token else 0.0,
                "first_token_p95": _percentile(first_token, 95),
                "e2e_mean": fmean(e2e) if e2e else 0.0,
                "e2e_p50": _percentile(e2e, 50),
                "e2e_p95": _percentile(e2e, 95),
                "e2e_p99": _percentile(e2e, 99),
                "mean_tpot": fmean(tpot) if tpot else 0.0,
                "p95_request_mean_tpot": _percentile(tpot, 95),
                "queue_mean": fmean(queue) if queue else 0.0,
                "queue_p50": _percentile(queue, 50),
                "queue_p95": _percentile(queue, 95),
                "queue_p99": _percentile(queue, 99),
            },
        }
        summary.update(self.policy.summary_metadata())
        return SimulationResult(
            summary=summary,
            requests=request_rows,
            iterations=self.iteration_rows,
            switches=self.switch_rows,
            pipeline_events=self.pipeline_event_rows,
        )


def run_simulation(config: ProjectConfig) -> SimulationResult:
    if config.simulation.cost_backend == "layout_aware":
        from .layout_aware import run_layout_simulation
        return run_layout_simulation(config)
    return Simulator(config).run()
