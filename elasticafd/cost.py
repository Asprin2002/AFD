"""Tile/resource roofline for CHIME-style decode execution."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from math import ceil
from typing import Iterable, Literal, Sequence

from .pipeline import (
    PipelineEvent, PipelineTask, order_two_subbatch_device_segments,
    order_two_subbatch_stages, schedule_pipeline,
)
from .mapping import ExecutionTarget, OperatorMapping, get_operator_mapping
from .specs import ModelSpec, SystemSpec
from .state import Location, RequestView


Fidelity = Literal["roofline", "tile"]


def _us(seconds: float) -> float:
    return seconds * 1e6


def _round_up(value: int, quantum: int) -> int:
    return ceil(value / quantum) * quantum


@dataclass(frozen=True)
class DenseLayerCost:
    qkv_us: float
    proj_ff_us: float
    qkv_memory_us: float
    qkv_compute_us: float
    proj_ff_memory_us: float
    proj_ff_compute_us: float

    @property
    def total_us(self) -> float:
        return self.qkv_us + self.proj_ff_us


@dataclass(frozen=True)
class MatrixLayerCost:
    total_us: float
    memory_us: float
    compute_us: float
    weight_bytes: int


@dataclass(frozen=True)
class PimLayerCost:
    total_us: float
    core_us: float
    bank_us: float
    cross_chip_us: float
    softmax_us: float
    setup_us: float
    relayout_us: float
    link_us: float
    exposed_link_us: float
    h2d_us: float
    d2h_us: float
    logical_kv_bytes: int
    physical_kv_bytes: int
    h2d_bytes: int
    d2h_bytes: int
    active_requests: int
    segments: int


@dataclass(frozen=True)
class GpuAttentionLayerCost:
    total_us: float
    memory_us: float
    compute_us: float
    setup_us: float
    logical_kv_bytes: int
    physical_kv_bytes: int
    active_requests: int
    segments: int


@dataclass(frozen=True)
class SubBatchLayerCost:
    size: int
    dense: DenseLayerCost
    pim: PimLayerCost
    gpu_attention: GpuAttentionLayerCost
    merge_us: float
    split_requests: int


@dataclass(frozen=True)
class IterationCost:
    total_us: float
    per_layer_us: float
    gpu_busy_us: float
    pim_core_busy_us: float
    link_busy_us: float
    gpu_attention_us: float
    dense_us: float
    merge_us: float
    phase_gpu_bottlenecks: int
    phase_pim_bottlenecks: int
    subbatches: tuple[SubBatchLayerCost, ...]
    pipeline_model: str = "analytic"
    events: tuple[PipelineEvent, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ChimeCostModel:
    """Cost model for one CHIME system and one LLM.

    ``roofline`` uses logical bytes only. ``tile`` additionally exposes row/tile
    padding, segment setup, cross-chip traffic, rank-side softmax, and CHIME's
    rankset-granular communication overlap.
    """

    def __init__(
        self,
        model: ModelSpec,
        system: SystemSpec,
        fidelity: Fidelity = "tile",
        pipeline_model: str = "analytic",
        subbatch_scheduler: str = "lpt",
        subbatch_progression: str = "strict_ordered",
        operator_mapping: str = "chime",
        split_overhead_scale: float = 1.0,
    ) -> None:
        if fidelity not in ("roofline", "tile"):
            raise ValueError("fidelity must be 'roofline' or 'tile'")
        self.model = model
        self.system = system
        self.fidelity = fidelity
        if pipeline_model not in ("analytic", "rankset_event"):
            raise ValueError("pipeline_model must be analytic or rankset_event")
        if subbatch_scheduler not in ("lpt", "alignment", "fixed"):
            raise ValueError("subbatch_scheduler must be lpt, alignment, or fixed")
        if subbatch_progression not in (
            "strict_ordered", "free_wavefront", "device_contiguous"
        ):
            raise ValueError("invalid subbatch progression")
        if subbatch_progression == "device_contiguous" and pipeline_model != "rankset_event":
            raise ValueError("device-contiguous progression requires the event model")
        self.pipeline_model = pipeline_model
        self.subbatch_scheduler = subbatch_scheduler
        self.subbatch_progression = subbatch_progression
        if split_overhead_scale < 0.0:
            raise ValueError("split_overhead_scale cannot be negative")
        self.split_overhead_scale = split_overhead_scale
        self.operator_mapping: OperatorMapping = get_operator_mapping(
            operator_mapping
        )
        # Event-level policy search repeatedly evaluates physically identical
        # placements that differ only in request ID.  Keep a small bounded
        # cache so those equivalent 96-layer DAGs are scheduled once.  A
        # bounded cache is important because completed simulation iterations
        # continuously introduce new context lengths.
        self._event_cost_cache: OrderedDict[
            tuple[tuple[object, ...], ...], IterationCost
        ] = OrderedDict()
        self._event_cost_cache_max_entries = 512
        self._event_cost_cache_hits = 0
        self._event_cost_cache_misses = 0

    @staticmethod
    def _physical_view_key(view: RequestView) -> tuple[object, ...]:
        """Request-ID-independent state consumed by the cost model."""

        return (
            view.gpu_replica,
            view.pim_segment_lengths,
            view.gpu_segment_lengths,
            view.tail_location.value,
        )

    def _canonical_views(
        self, views: Sequence[RequestView]
    ) -> tuple[RequestView, ...]:
        """Make sub-batch construction invariant to request list ordering.

        Request identity has no performance meaning in this model.  Sorting by
        physical placement also makes equivalent greedy candidates share the
        event-cost cache without changing any tensor size or resource demand.
        """

        return tuple(
            sorted(
                views,
                key=lambda view: (self._physical_view_key(view), view.request_id),
            )
        )

    @property
    def event_cost_cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._event_cost_cache_hits,
            "misses": self._event_cost_cache_misses,
            "entries": len(self._event_cost_cache),
            "max_entries": self._event_cost_cache_max_entries,
        }

    def _operator_weight_elements(self, operator: str) -> int:
        if operator == "qkv":
            return self.model.qkv_weight_elements_per_layer
        if operator == "o_proj":
            return self.model.output_projection_weight_elements_per_layer
        if operator == "ffn":
            return self.model.ff_weight_elements_per_layer
        raise ValueError(f"{operator} is not a dense operator")

    @property
    def pim_persistent_weight_bytes(self) -> int:
        per_copy = sum(
            self._operator_weight_elements(operator) * self.model.dtype_bytes
            for operator in self.operator_mapping.pim_dense_operators
        ) * self.model.layers
        # Keep the same aggregate-DP accounting convention as GPU weights.
        return per_copy * self.model.data_parallel

    @property
    def gpu_offloaded_weight_bytes(self) -> int:
        return self.pim_persistent_weight_bytes

    @property
    def pim_kv_capacity_bytes(self) -> float:
        return self.system.pim.capacity_bytes - self.pim_persistent_weight_bytes

    @property
    def gpu_kv_capacity_bytes(self) -> float:
        return self.system.gpu_kv_capacity_bytes(
            self.model,
            reclaimed_weight_bytes=self.gpu_offloaded_weight_bytes,
        )

    @property
    def gpu_replica_kv_capacity_bytes(self) -> float:
        """KV capacity usable by one fixed data-parallel replica."""

        return self.gpu_kv_capacity_bytes / self.model.data_parallel

    def gpu_matrix_layer_cost(
        self, operator: str, batch_size: int
    ) -> MatrixLayerCost:
        if batch_size <= 0:
            return MatrixLayerCost(0.0, 0.0, 0.0, 0)
        elements = self._operator_weight_elements(operator)
        weight_bytes = elements * self.model.dtype_bytes
        local_batch = ceil(batch_size / self.model.data_parallel)
        group_bandwidth = self.system.gpu.bandwidth_bytes_s / self.model.data_parallel
        group_flops = self.system.gpu.flops_s / self.model.data_parallel
        memory_us = _us(
            weight_bytes
            / (group_bandwidth * self.system.gpu.dense_hbm_efficiency)
        )
        compute_us = _us(
            2
            * elements
            * local_batch
            / (group_flops * self.system.gpu.compute_efficiency)
        )
        proj_elements = self.model.proj_ff_weight_elements_per_layer
        o_fraction = (
            self.model.output_projection_weight_elements_per_layer / proj_elements
        )
        launch_us = {
            "qkv": self.system.gpu.qkv_launch_us,
            "o_proj": self.system.gpu.proj_ff_launch_us * o_fraction,
            "ffn": self.system.gpu.proj_ff_launch_us * (1.0 - o_fraction),
        }[operator]
        return MatrixLayerCost(
            max(memory_us, compute_us) + launch_us,
            memory_us,
            compute_us,
            weight_bytes,
        )

    def pim_matrix_layer_cost(
        self, operator: str, batch_size: int
    ) -> MatrixLayerCost:
        """Optimistic DIMM-PIM dense roofline used by operator mappings.

        It assumes weights are persistent and that a bank-PU weight read can
        feed one FP16 MAC.  It is intentionally a replaceable bound, not a
        claim that CHIME implements these dense kernels.
        """

        if batch_size <= 0:
            return MatrixLayerCost(0.0, 0.0, 0.0, 0)
        elements = self._operator_weight_elements(operator)
        weight_bytes = elements * self.model.dtype_bytes
        bandwidth = (
            self.system.pim.layer_bandwidth_bytes_s
            * self.system.pim.bank_hbm_efficiency
        )
        memory_us = _us(weight_bytes / bandwidth)
        dense_flops_s = (
            self.system.pim.layer_bandwidth_bytes_s
            * self.system.pim.dense_mac_flops_per_byte
            * self.system.pim.dense_compute_efficiency
        )
        compute_us = _us(2 * elements * batch_size / dense_flops_s)
        return MatrixLayerCost(
            max(memory_us, compute_us), memory_us, compute_us, weight_bytes
        )

    def dense_layer_cost(
        self,
        batch_size: int,
        replica_batch_sizes: Sequence[int] | None = None,
    ) -> DenseLayerCost:
        if batch_size <= 0:
            return DenseLayerCost(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        gpu = self.system.gpu
        model = self.model
        dp = model.data_parallel
        if replica_batch_sizes is None:
            local_batch = ceil(batch_size / dp)
        else:
            if len(replica_batch_sizes) != dp:
                raise ValueError("replica_batch_sizes must have one entry per DP replica")
            if sum(replica_batch_sizes) != batch_size:
                raise ValueError("replica_batch_sizes must sum to batch_size")
            local_batch = max(replica_batch_sizes)
        group_bandwidth = gpu.bandwidth_bytes_s / dp
        group_flops = gpu.flops_s / dp

        qkv_memory_us = _us(
            model.qkv_weight_bytes_per_layer
            / (group_bandwidth * gpu.dense_hbm_efficiency)
        )
        qkv_compute_us = _us(
            2
            * model.qkv_weight_elements_per_layer
            * local_batch
            / (group_flops * gpu.compute_efficiency)
        )
        proj_memory_us = _us(
            model.proj_ff_weight_bytes_per_layer
            / (group_bandwidth * gpu.dense_hbm_efficiency)
        )
        proj_compute_us = _us(
            2
            * model.proj_ff_weight_elements_per_layer
            * local_batch
            / (group_flops * gpu.compute_efficiency)
        )

        qkv_us = max(qkv_memory_us, qkv_compute_us) + gpu.qkv_launch_us
        proj_us = max(proj_memory_us, proj_compute_us) + gpu.proj_ff_launch_us
        return DenseLayerCost(
            qkv_us=qkv_us,
            proj_ff_us=proj_us,
            qkv_memory_us=qkv_memory_us,
            qkv_compute_us=qkv_compute_us,
            proj_ff_memory_us=proj_memory_us,
            proj_ff_compute_us=proj_compute_us,
        )

    def pim_layer_cost(self, views: Sequence[RequestView]) -> PimLayerCost:
        model = self.model
        pim = self.system.pim
        link = self.system.link
        active = [view for view in views if view.pim_tokens > 0]
        if not active:
            return PimLayerCost(
                total_us=0.0,
                core_us=0.0,
                bank_us=0.0,
                cross_chip_us=0.0,
                softmax_us=0.0,
                setup_us=0.0,
                relayout_us=0.0,
                link_us=0.0,
                exposed_link_us=0.0,
                h2d_us=0.0,
                d2h_us=0.0,
                logical_kv_bytes=0,
                physical_kv_bytes=0,
                h2d_bytes=0,
                d2h_bytes=0,
                active_requests=0,
                segments=0,
            )

        logical_tokens = sum(view.pim_tokens for view in active)
        segments = sum(len(view.pim_segment_lengths) for view in active)
        if self.fidelity == "tile":
            physical_tokens = sum(
                _round_up(length, pim.token_tile)
                for view in active
                for length in view.pim_segment_lengths
            )
        else:
            physical_tokens = logical_tokens

        logical_bytes = logical_tokens * model.kv_bytes_per_token_per_layer
        physical_bytes = physical_tokens * model.kv_bytes_per_token_per_layer
        effective_bank_bandwidth = (
            pim.layer_bandwidth_bytes_s
            * pim.bank_hbm_efficiency
            * pim.relayout_efficiency
        )
        bank_us = _us(physical_bytes / effective_bank_bandwidth)

        if self.fidelity == "tile":
            # CHIME maps one MHA head over all 8 chips (Nhc=8), while a GQA-8
            # head remains on one chip (Nhc=1). The score transfer is pipelined
            # with bank-PU KV reads, hence it enters a max rather than a sum.
            chips_per_head = (
                pim.chips_per_rank
                if (not pim.optimized_head_mapping or model.gqa_group_size == 1)
                else 1
            )
            cross_chip_bytes = (
                logical_tokens
                * model.num_attention_heads
                * model.dtype_bytes
                * chips_per_head
                * pim.cross_chip_traffic_multiplier
            )
            cross_chip_us = _us(
                cross_chip_bytes
                / (
                    pim.rank_bus_per_rankset_bytes_s
                    * pim.compute_ranksets_per_layer
                )
            )
            softmax_elements = logical_tokens * model.num_attention_heads
            softmax_us = (
                _us(
                    softmax_elements
                    / (pim.softmax_elements_s * pim.compute_ranksets_per_layer)
                )
                if pim.softmax_elements_s > 0
                else 0.0
            )
            setup_us = (
                len(active) * pim.request_setup_us + segments * pim.segment_setup_us
            )
        else:
            cross_chip_us = 0.0
            softmax_us = 0.0
            setup_us = 0.0

        if self.fidelity == "tile" and not pim.bubble_free_pipeline:
            core_us = bank_us + cross_chip_us + softmax_us
        else:
            core_us = max(bank_us, cross_chip_us, softmax_us)

        relayout_us = 0.0
        if self.fidelity == "tile" and not pim.hardware_relayout:
            qkv_bytes = (
                model.q_bytes_per_token_per_layer
                + model.new_kv_bytes_per_token_per_layer
            )
            relayout_us = len(active) * (
                pim.software_relayout_fixed_us
                + _us(
                    2
                    * qkv_bytes
                    / (pim.software_relayout_bandwidth_gbps * 1e9)
                )
            )

        h2d_bytes = 0
        d2h_bytes = 0
        for view in active:
            h2d_bytes += model.q_bytes_per_token_per_layer
            if view.tail_location is Location.PIM:
                h2d_bytes += model.new_kv_bytes_per_token_per_layer
            d2h_bytes += model.output_bytes_per_token_per_layer
            if view.is_split:
                # Online-softmax merge needs FP32 max and denominator per head.
                d2h_bytes += round(
                    model.num_attention_heads * 2 * 4 * self.split_overhead_scale
                )

        h2d_us = _us(h2d_bytes / link.host_to_pim_bytes_s) + link.transfer_latency_us
        d2h_us = _us(d2h_bytes / link.pim_to_host_bytes_s) + link.transfer_latency_us
        link_us = max(h2d_us, d2h_us) if link.full_duplex else h2d_us + d2h_us
        exposed_fraction = (
            pim.exposed_communication_fraction if self.fidelity == "tile" else 1.0
        )
        exposed_link_us = link_us * exposed_fraction
        total_us = core_us + setup_us + relayout_us + exposed_link_us

        return PimLayerCost(
            total_us=total_us,
            core_us=core_us,
            bank_us=bank_us,
            cross_chip_us=cross_chip_us,
            softmax_us=softmax_us,
            setup_us=setup_us,
            relayout_us=relayout_us,
            link_us=link_us,
            exposed_link_us=exposed_link_us,
            h2d_us=h2d_us,
            d2h_us=d2h_us,
            logical_kv_bytes=logical_bytes,
            physical_kv_bytes=physical_bytes,
            h2d_bytes=h2d_bytes,
            d2h_bytes=d2h_bytes,
            active_requests=len(active),
            segments=segments,
        )

    def gpu_attention_layer_cost(
        self, views: Sequence[RequestView]
    ) -> GpuAttentionLayerCost:
        model = self.model
        gpu = self.system.gpu
        active = [view for view in views if view.gpu_tokens > 0]
        if not active:
            return GpuAttentionLayerCost(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

        logical_bytes_per_request: list[int] = []
        physical_bytes_per_request: list[int] = []
        flops_per_request: list[int] = []
        for view in active:
            logical_bytes = view.gpu_tokens * model.kv_bytes_per_token_per_layer
            if self.fidelity == "tile":
                physical_tokens = sum(
                    _round_up(length, gpu.attention_tile_tokens)
                    for length in view.gpu_segment_lengths
                )
            else:
                physical_tokens = view.gpu_tokens
            physical_bytes = physical_tokens * model.kv_bytes_per_token_per_layer
            flops = (
                4
                * view.gpu_tokens
                * model.num_attention_heads
                * model.head_dim
            )
            logical_bytes_per_request.append(logical_bytes)
            physical_bytes_per_request.append(physical_bytes)
            flops_per_request.append(flops)

        # Requests remain on the DP replica selected at admission. This avoids
        # treating another replica's idle HBM capacity or bandwidth as fungible.
        loads = [[0, 0] for _ in range(model.data_parallel)]  # bytes, FLOPs
        for index, view in enumerate(active):
            if not 0 <= view.gpu_replica < model.data_parallel:
                raise ValueError(
                    f"request {view.request_id}: invalid GPU replica {view.gpu_replica}"
                )
            loads[view.gpu_replica][0] += physical_bytes_per_request[index]
            loads[view.gpu_replica][1] += flops_per_request[index]

        group_bandwidth = gpu.bandwidth_bytes_s / model.data_parallel
        group_flops = gpu.flops_s / model.data_parallel
        memory_us = max(
            _us(load[0] / (group_bandwidth * gpu.attention_hbm_efficiency))
            for load in loads
        )
        compute_us = max(
            _us(load[1] / (group_flops * gpu.attention_compute_efficiency))
            for load in loads
        )
        segments = sum(len(view.gpu_segment_lengths) for view in active)
        setup_us = 0.0
        if self.fidelity == "tile":
            setup_us = gpu.attention_launch_us + segments * gpu.segment_setup_us
        total_us = max(memory_us, compute_us) + setup_us
        return GpuAttentionLayerCost(
            total_us=total_us,
            memory_us=memory_us,
            compute_us=compute_us,
            setup_us=setup_us,
            logical_kv_bytes=sum(logical_bytes_per_request),
            physical_kv_bytes=sum(physical_bytes_per_request),
            active_requests=len(active),
            segments=segments,
        )

    def merge_layer_cost(self, views: Sequence[RequestView]) -> tuple[float, int]:
        split_views = [view for view in views if view.is_split]
        split = len(split_views)
        if not split_views:
            return 0.0, 0
        gpu = self.system.gpu
        model = self.model
        merge_bytes_per_request = (
            3 * model.output_bytes_per_token_per_layer
            + 2 * model.num_attention_heads * 4
        )
        counts = [0] * model.data_parallel
        for view in split_views:
            if not 0 <= view.gpu_replica < model.data_parallel:
                raise ValueError(
                    f"request {view.request_id}: invalid GPU replica {view.gpu_replica}"
                )
            counts[view.gpu_replica] += 1
        group_bandwidth = gpu.bandwidth_bytes_s / model.data_parallel
        replica_times = (
            gpu.merge_launch_us
            + count * gpu.merge_request_us
            + _us(
                count
                * merge_bytes_per_request
                / (group_bandwidth * gpu.attention_hbm_efficiency)
            )
            for count in counts
            if count
        )
        return max(replica_times) * self.split_overhead_scale, split

    def subbatch_layer_cost(
        self, views: Sequence[RequestView]
    ) -> SubBatchLayerCost:
        merge_us, split = self.merge_layer_cost(views)
        replica_batch_sizes = [0] * self.model.data_parallel
        for view in views:
            if not 0 <= view.gpu_replica < self.model.data_parallel:
                raise ValueError(
                    f"request {view.request_id}: invalid GPU replica {view.gpu_replica}"
                )
            replica_batch_sizes[view.gpu_replica] += 1
        return SubBatchLayerCost(
            size=len(views),
            dense=self.dense_layer_cost(len(views), replica_batch_sizes),
            pim=self.pim_layer_cost(views),
            gpu_attention=self.gpu_attention_layer_cost(views),
            merge_us=merge_us,
            split_requests=split,
        )

    def _partition_views_lpt(
        self, views: Sequence[RequestView]
    ) -> tuple[tuple[RequestView, ...], ...]:
        """Create CHIME's two balanced decode sub-batches.

        Bins are kept balanced both globally and within each fixed DP replica.
        Within those constraints LPT balances attention work, including
        GPU-resident segments at their relative bandwidth cost.
        """

        if len(views) <= 1:
            return (tuple(views),)
        max_size = ceil(len(views) / 2)
        bins: list[list[RequestView]] = [[], []]
        loads = [0.0, 0.0]
        replica_totals = [0] * self.model.data_parallel
        for view in views:
            if not 0 <= view.gpu_replica < self.model.data_parallel:
                raise ValueError(
                    f"request {view.request_id}: invalid GPU replica {view.gpu_replica}"
                )
            replica_totals[view.gpu_replica] += 1
        replica_max = [ceil(total / 2) for total in replica_totals]
        replica_counts = [
            [0] * self.model.data_parallel,
            [0] * self.model.data_parallel,
        ]
        gpu_to_pim_weight = (
            self.system.pim.layer_bandwidth_bytes_s
            / self.system.gpu.bandwidth_bytes_s
        )
        ordered = sorted(
            views,
            key=lambda view: view.pim_tokens + gpu_to_pim_weight * view.gpu_tokens,
            reverse=True,
        )
        for view in ordered:
            candidates = [
                index
                for index in range(2)
                if len(bins[index]) < max_size
                and replica_counts[index][view.gpu_replica]
                < replica_max[view.gpu_replica]
            ]
            if not candidates:
                raise AssertionError("failed to construct a DP-balanced sub-batch")
            target = min(candidates, key=lambda i: (loads[i], len(bins[i])))
            bins[target].append(view)
            replica_counts[target][view.gpu_replica] += 1
            loads[target] += view.pim_tokens + gpu_to_pim_weight * view.gpu_tokens
        return tuple(tuple(batch) for batch in bins if batch)

    def _partition_score(
        self, partitions: Sequence[Sequence[RequestView]]
    ) -> float:
        costs = [self.subbatch_layer_cost(batch) for batch in partitions]
        if len(costs) == 1:
            cost = costs[0]
            return (
                cost.dense.qkv_us
                + max(cost.pim.total_us, cost.gpu_attention.total_us)
                + cost.merge_us
                + cost.dense.proj_ff_us
            )
        first, second = costs
        return (
            max(
                first.pim.total_us,
                first.gpu_attention.total_us + second.dense.total_us,
            )
            + first.merge_us
            + max(
                second.pim.total_us,
                second.gpu_attention.total_us + first.dense.total_us,
            )
            + second.merge_us
        )

    def _partition_views_alignment(
        self, views: Sequence[RequestView]
    ) -> tuple[tuple[RequestView, ...], ...]:
        """Local-search CHIME sub-batches for the actual pipeline objective."""

        initial = self._partition_views_lpt(views)
        if len(initial) < 2 or len(views) <= 2:
            return initial
        bins = [list(initial[0]), list(initial[1])]
        current = self._partition_score(bins)
        # Equal cardinality is retained; it preserves dense batching while
        # swaps search for PIM/GPU phase alignment.
        for _ in range(max(1, len(views))):
            best: tuple[int, int] | None = None
            best_score = current
            for left in range(len(bins[0])):
                for right in range(len(bins[1])):
                    if (
                        bins[0][left].gpu_replica
                        != bins[1][right].gpu_replica
                    ):
                        continue
                    candidate = [list(bins[0]), list(bins[1])]
                    candidate[0][left], candidate[1][right] = (
                        candidate[1][right],
                        candidate[0][left],
                    )
                    score = self._partition_score(candidate)
                    if score + 1e-9 < best_score:
                        best_score = score
                        best = (left, right)
            if best is None:
                break
            left, right = best
            bins[0][left], bins[1][right] = bins[1][right], bins[0][left]
            current = best_score
        return tuple(tuple(batch) for batch in bins)

    def partition_views(
        self, views: Sequence[RequestView]
    ) -> tuple[tuple[RequestView, ...], ...]:
        if self.subbatch_scheduler == "fixed":
            # Closed-batch experiments retain the same request membership at
            # every decode age. Do not rebalance by changing KV lengths.
            ordered = sorted(views, key=lambda view: (view.gpu_replica,
                                                      view.request_id))
            if len(ordered) <= 1:
                return (tuple(ordered),)
            return (tuple(ordered[::2]), tuple(ordered[1::2]))
        if self.subbatch_scheduler == "alignment":
            return self._partition_views_alignment(views)
        return self._partition_views_lpt(views)

    def iteration_cost(self, views: Sequence[RequestView]) -> IterationCost:
        if self.operator_mapping.name != "chime":
            if self.subbatch_progression == "device_contiguous":
                raise ValueError(
                    "static operator mappings do not support device-contiguous "
                    "placement; use the contiguous dense event model"
                )
            return self._iteration_cost_static_operator_event(views)
        if self.pipeline_model == "rankset_event":
            canonical = self._canonical_views(views)
            cache_key = tuple(
                self._physical_view_key(view) for view in canonical
            )
            cached = self._event_cost_cache.get(cache_key)
            if cached is not None:
                self._event_cost_cache_hits += 1
                self._event_cost_cache.move_to_end(cache_key)
                return cached
            self._event_cost_cache_misses += 1
            cost = self._iteration_cost_rankset_event(canonical)
            self._event_cost_cache[cache_key] = cost
            self._event_cost_cache.move_to_end(cache_key)
            if len(self._event_cost_cache) > self._event_cost_cache_max_entries:
                self._event_cost_cache.popitem(last=False)
            return cost
        return self._iteration_cost_analytic(views)

    def _force_attention_target(
        self,
        views: Sequence[RequestView],
        target: ExecutionTarget,
    ) -> tuple[RequestView, ...]:
        if target not in (ExecutionTarget.GPU, ExecutionTarget.PIM):
            raise ValueError("a static operator mapping needs a fixed attention target")
        forced: list[RequestView] = []
        for view in views:
            context = view.context_tokens
            forced.append(
                RequestView(
                    request_id=view.request_id,
                    pim_segment_lengths=(context,) if target is ExecutionTarget.PIM else (),
                    gpu_segment_lengths=(context,) if target is ExecutionTarget.GPU else (),
                    tail_location=(
                        Location.PIM
                        if target is ExecutionTarget.PIM
                        else Location.GPU
                    ),
                    gpu_replica=view.gpu_replica,
                )
            )
        return tuple(forced)

    def _iteration_cost_static_operator_event(
        self, views: Sequence[RequestView]
    ) -> IterationCost:
        """Schedule a fixed four-stage mapping on the shared resource model."""

        if not views:
            raise ValueError("cannot cost an empty decode iteration")
        mapping = self.operator_mapping
        attention_views = self._force_attention_target(views, mapping.attention)
        # LPT remains useful for attention balance.  CHIME's alignment search
        # has a crossed two-device objective and is intentionally not reused
        # for a different static mapping.
        partitions = self._partition_views_lpt(attention_views)
        subcosts = tuple(self.subbatch_layer_cost(batch) for batch in partitions)
        pim = self.system.pim
        link = self.system.link
        tasks: list[PipelineTask] = []
        # task id, device, and layer containing the produced activation.
        previous: dict[int, tuple[str, ExecutionTarget, int]] = {}

        def rankset_resources(layer: int) -> tuple[str, ...]:
            return tuple(
                f"pim:{(layer + offset) % pim.ranksets}"
                for offset in range(pim.compute_ranksets_per_layer)
            )

        def link_resource(direction: str) -> str:
            return f"link:{direction}" if link.full_duplex else "link"

        def communication_rank_resources(
            source: ExecutionTarget,
            destination: ExecutionTarget,
            source_layer: int,
            destination_layer: int,
        ) -> tuple[str, ...]:
            layer = source_layer if source is ExecutionTarget.PIM else destination_layer
            local = rankset_resources(max(0, layer))
            if pim.rankset_comm_overlap:
                return local
            return tuple(f"pim:{index}" for index in range(pim.ranksets))

        def transfer_duration(
            source: ExecutionTarget, destination: ExecutionTarget, byte_count: int
        ) -> tuple[float, str]:
            if source is ExecutionTarget.GPU and destination is ExecutionTarget.PIM:
                return (
                    _us(byte_count / link.host_to_pim_bytes_s)
                    + link.transfer_latency_us,
                    "h2d",
                )
            if source is ExecutionTarget.PIM and destination is ExecutionTarget.GPU:
                return (
                    _us(byte_count / link.pim_to_host_bytes_s)
                    + link.transfer_latency_us,
                    "d2h",
                )
            raise ValueError("external transfer must cross GPU and PIM")

        stages = ("qkv", "attention", "o_proj", "ffn")
        for layer_index in range(self.model.layers):
            local_ranksets = rankset_resources(layer_index)
            for subbatch, batch_views in enumerate(partitions):
                batch_size = len(batch_views)
                for stage in stages:
                    target = mapping.target(stage)
                    if target is ExecutionTarget.KV:
                        raise AssertionError("KV target is only valid for CHIME mapping")
                    prefix = f"l{layer_index}:s{subbatch}:{stage}"
                    predecessor: tuple[str, ...] = ()
                    if subbatch in previous:
                        previous_id, previous_target, previous_layer = previous[subbatch]
                        predecessor = (previous_id,)
                    else:
                        previous_target = ExecutionTarget.GPU
                        previous_layer = layer_index

                    transfer_bytes = 0
                    if stage == "qkv":
                        transfer_bytes = (
                            batch_size * self.model.output_bytes_per_token_per_layer
                        )
                    elif stage == "attention":
                        transfer_bytes = batch_size * (
                            self.model.q_bytes_per_token_per_layer
                            + self.model.new_kv_bytes_per_token_per_layer
                        )
                    else:
                        transfer_bytes = (
                            batch_size * self.model.output_bytes_per_token_per_layer
                        )

                    # A cross-device edge is explicit.  PIM-to-PIM layer
                    # changes also move the activation between layer-interleaved
                    # ranksets rather than assuming a free teleport.
                    if previous_target is not target:
                        transfer_us, direction = transfer_duration(
                            previous_target, target, transfer_bytes
                        )
                        transfer_id = f"{prefix}:transfer"
                        comm_ranksets = communication_rank_resources(
                            previous_target,
                            target,
                            previous_layer,
                            layer_index,
                        )
                        tasks.append(
                            PipelineTask(
                                transfer_id,
                                transfer_us,
                                (link_resource(direction), *comm_ranksets),
                                predecessor,
                                direction,
                                layer_index,
                                subbatch,
                                priority=0,
                            )
                        )
                        predecessor = (transfer_id,)
                    elif (
                        target is ExecutionTarget.PIM
                        and previous_layer != layer_index
                        and rankset_resources(previous_layer) != local_ranksets
                    ):
                        transfer_id = f"{prefix}:rankset_move"
                        source_ranksets = rankset_resources(previous_layer)
                        internal_us = _us(
                            transfer_bytes / pim.rank_bus_per_rankset_bytes_s
                        )
                        tasks.append(
                            PipelineTask(
                                transfer_id,
                                internal_us,
                                tuple(dict.fromkeys((*source_ranksets, *local_ranksets))),
                                predecessor,
                                "pim_internal",
                                layer_index,
                                subbatch,
                                priority=0,
                            )
                        )
                        predecessor = (transfer_id,)

                    if stage == "attention":
                        if target is ExecutionTarget.PIM:
                            attention_cost = self.pim_layer_cost(batch_views)
                            duration = (
                                attention_cost.core_us
                                + attention_cost.setup_us
                                + attention_cost.relayout_us
                            )
                            kind = "pim_attention"
                        else:
                            attention_cost = self.gpu_attention_layer_cost(batch_views)
                            duration = attention_cost.total_us
                            kind = "gpu_attention"
                    elif target is ExecutionTarget.PIM:
                        duration = self.pim_matrix_layer_cost(stage, batch_size).total_us
                        kind = f"pim_{stage}"
                    else:
                        duration = self.gpu_matrix_layer_cost(stage, batch_size).total_us
                        kind = f"gpu_{stage}"

                    task_id = f"{prefix}:compute"
                    resources = local_ranksets if target is ExecutionTarget.PIM else ("gpu",)
                    tasks.append(
                        PipelineTask(
                            task_id,
                            duration,
                            resources,
                            predecessor,
                            kind,
                            layer_index,
                            subbatch,
                            priority=2,
                        )
                    )
                    if (
                        stage == "attention"
                        and len(partitions) == 2
                        and self.subbatch_progression == "strict_ordered"
                    ):
                        tasks.append(
                            PipelineTask(
                                f"l{layer_index}:s{subbatch}:attention_done",
                                0.0,
                                (),
                                (task_id,),
                                "attention_done",
                                layer_index,
                                subbatch,
                            )
                        )
                    previous[subbatch] = (task_id, target, layer_index)

        # Logits/sampling remain GPU-side in this abstraction.
        for subbatch, (previous_id, previous_target, previous_layer) in previous.items():
            if previous_target is not ExecutionTarget.PIM:
                continue
            batch_size = len(partitions[subbatch])
            byte_count = batch_size * self.model.output_bytes_per_token_per_layer
            transfer_us, direction = transfer_duration(
                ExecutionTarget.PIM, ExecutionTarget.GPU, byte_count
            )
            tasks.append(
                PipelineTask(
                    f"final:s{subbatch}:d2h",
                    transfer_us,
                    (
                        link_resource(direction),
                        *communication_rank_resources(
                            ExecutionTarget.PIM,
                            ExecutionTarget.GPU,
                            previous_layer,
                            previous_layer,
                        ),
                    ),
                    (previous_id,),
                    "d2h",
                    self.model.layers,
                    subbatch,
                    priority=0,
                )
            )

        schedule = schedule_pipeline(
            order_two_subbatch_stages(tasks)
            if self.subbatch_progression == "strict_ordered" else tasks
        )
        events = schedule.events
        dense_kinds = {
            "gpu_qkv",
            "pim_qkv",
            "gpu_o_proj",
            "pim_o_proj",
            "gpu_ffn",
            "pim_ffn",
        }
        dense_us = sum(
            event.duration_us for event in events if event.kind in dense_kinds
        )
        gpu_attention_us = sum(
            event.duration_us for event in events if event.kind == "gpu_attention"
        )
        gpu_busy_us = schedule.resource_busy_us.get("gpu", 0.0)
        rankset_busy = [
            schedule.resource_busy_us.get(f"pim:{index}", 0.0)
            for index in range(pim.ranksets)
        ]
        pim_busy_us = max(rankset_busy, default=0.0)
        if link.full_duplex:
            link_busy_us = max(
                schedule.resource_busy_us.get("link:h2d", 0.0),
                schedule.resource_busy_us.get("link:d2h", 0.0),
            )
        else:
            link_busy_us = schedule.resource_busy_us.get("link", 0.0)
        gpu_bottlenecks = 2 if gpu_busy_us >= pim_busy_us else 0
        return IterationCost(
            total_us=schedule.makespan_us,
            per_layer_us=schedule.makespan_us / self.model.layers,
            gpu_busy_us=gpu_busy_us,
            pim_core_busy_us=pim_busy_us,
            link_busy_us=link_busy_us,
            gpu_attention_us=gpu_attention_us,
            dense_us=dense_us,
            merge_us=0.0,
            phase_gpu_bottlenecks=gpu_bottlenecks,
            phase_pim_bottlenecks=2 - gpu_bottlenecks,
            subbatches=subcosts,
            pipeline_model="operator_event",
            events=events,
        )

    def _iteration_cost_analytic(
        self, views: Sequence[RequestView]
    ) -> IterationCost:
        if not views:
            raise ValueError("cannot cost an empty decode iteration")
        partitions = self.partition_views(views)
        subcosts = tuple(self.subbatch_layer_cost(batch) for batch in partitions)
        layers = self.model.layers

        if len(subcosts) == 1:
            cost = subcosts[0]
            attention_us = max(cost.pim.total_us, cost.gpu_attention.total_us)
            per_layer_us = (
                cost.dense.qkv_us
                + attention_us
                + cost.merge_us
                + cost.dense.proj_ff_us
            )
            gpu_bottlenecks = int(cost.gpu_attention.total_us >= cost.pim.total_us)
            pim_bottlenecks = 1 - gpu_bottlenecks
        else:
            first, second = subcosts
            # In steady state, attention of one sub-batch overlaps the complete
            # dense path of the other. GPU-local attention shares that GPU path.
            gpu_path_first = first.gpu_attention.total_us + second.dense.total_us
            gpu_path_second = second.gpu_attention.total_us + first.dense.total_us
            phase_first = max(first.pim.total_us, gpu_path_first) + first.merge_us
            phase_second = max(second.pim.total_us, gpu_path_second) + second.merge_us
            per_layer_us = phase_first + phase_second
            gpu_bottlenecks = int(gpu_path_first >= first.pim.total_us) + int(
                gpu_path_second >= second.pim.total_us
            )
            pim_bottlenecks = 2 - gpu_bottlenecks

        dense_us = sum(cost.dense.total_us for cost in subcosts) * layers
        gpu_attention_us = (
            sum(cost.gpu_attention.total_us for cost in subcosts) * layers
        )
        merge_us = sum(cost.merge_us for cost in subcosts) * layers
        gpu_busy_us = dense_us + gpu_attention_us + merge_us
        pim_busy_us = sum(cost.pim.core_us for cost in subcosts) * layers
        link_busy_us = sum(cost.pim.link_us for cost in subcosts) * layers
        return IterationCost(
            total_us=per_layer_us * layers,
            per_layer_us=per_layer_us,
            gpu_busy_us=gpu_busy_us,
            pim_core_busy_us=pim_busy_us,
            link_busy_us=link_busy_us,
            gpu_attention_us=gpu_attention_us,
            dense_us=dense_us,
            merge_us=merge_us,
            phase_gpu_bottlenecks=gpu_bottlenecks,
            phase_pim_bottlenecks=pim_bottlenecks,
            subbatches=subcosts,
            pipeline_model="analytic",
        )

    def _iteration_cost_rankset_event(
        self, views: Sequence[RequestView]
    ) -> IterationCost:
        if not views:
            raise ValueError("cannot cost an empty decode iteration")
        partitions = self.partition_views(views)
        subcosts = tuple(self.subbatch_layer_cost(batch) for batch in partitions)
        pim = self.system.pim
        link = self.system.link
        tasks: list[PipelineTask] = []
        previous_proj: dict[int, str] = {}

        def rankset_resources(layer: int) -> tuple[str, ...]:
            return tuple(
                f"pim:{(layer + offset) % pim.ranksets}"
                for offset in range(pim.compute_ranksets_per_layer)
            )

        def link_resource(direction: str) -> str:
            return f"link:{direction}" if link.full_duplex else "link"

        for layer in range(self.model.layers):
            rank_resources = rankset_resources(layer)
            for subbatch, cost in enumerate(subcosts):
                prefix = f"l{layer}:s{subbatch}"
                qkv_id = f"{prefix}:qkv"
                qkv_predecessors = (
                    (previous_proj[subbatch],) if subbatch in previous_proj else ()
                )
                tasks.append(
                    PipelineTask(
                        qkv_id,
                        cost.dense.qkv_us,
                        ("gpu",),
                        qkv_predecessors,
                        "qkv",
                        layer,
                        subbatch,
                        priority=4,
                    )
                )

                attention_done: list[str] = []
                if cost.pim.active_requests:
                    h2d_id = f"{prefix}:h2d"
                    split_partial_bytes = (
                        cost.split_requests
                        * (self.model.output_bytes_per_token_per_layer
                           + self.model.num_attention_heads * 2 * 4)
                    )
                    comm_rank_resources = (
                        rank_resources
                        if pim.rankset_comm_overlap
                        else tuple(f"pim:{index}" for index in range(pim.ranksets))
                    )
                    tasks.append(
                        PipelineTask(
                            h2d_id,
                            cost.pim.h2d_us + (
                                split_partial_bytes
                                / link.host_to_pim_bytes_s * 1e6
                                if self.subbatch_progression == "device_contiguous"
                                else 0.0
                            ),
                            (link_resource("h2d"), *comm_rank_resources),
                            (
                                (qkv_id, f"{prefix}:gpu_attention")
                                if cost.split_requests
                                and self.subbatch_progression == "device_contiguous"
                                else (qkv_id,)
                            ),
                            "h2d",
                            layer,
                            subbatch,
                            priority=0,
                        )
                    )
                    pim_id = f"{prefix}:pim_attention"
                    tasks.append(
                        PipelineTask(
                            pim_id,
                            cost.pim.core_us
                            + cost.pim.setup_us
                            + cost.pim.relayout_us,
                            rank_resources,
                            (h2d_id,),
                            "pim_attention",
                            layer,
                            subbatch,
                            priority=1,
                        )
                    )
                    d2h_predecessor = pim_id
                    if cost.split_requests and self.subbatch_progression == "device_contiguous":
                        pim_merge_id = f"{prefix}:pim_merge"
                        tasks.append(PipelineTask(
                            pim_merge_id, cost.merge_us, rank_resources,
                            (pim_id,), "pim_merge", layer, subbatch, priority=1,
                        ))
                        d2h_predecessor = pim_merge_id
                    d2h_id = f"{prefix}:d2h"
                    tasks.append(
                        PipelineTask(
                            d2h_id,
                            (
                                self.model.output_bytes_per_token_per_layer
                                * cost.pim.active_requests
                                / link.pim_to_host_bytes_s * 1e6
                                + link.transfer_latency_us
                                if cost.split_requests
                                and self.subbatch_progression == "device_contiguous"
                                else cost.pim.d2h_us
                            ),
                            (link_resource("d2h"), *comm_rank_resources),
                            (d2h_predecessor,),
                            "d2h",
                            layer,
                            subbatch,
                            priority=0,
                        )
                    )
                    attention_done.append(d2h_id)

                if cost.gpu_attention.active_requests:
                    gpu_attention_id = f"{prefix}:gpu_attention"
                    tasks.append(
                        PipelineTask(
                            gpu_attention_id,
                            cost.gpu_attention.total_us,
                            ("gpu",),
                            (qkv_id,),
                            "gpu_attention",
                            layer,
                            subbatch,
                            priority=2,
                        )
                    )
                    attention_done.append(gpu_attention_id)

                if not attention_done:
                    raise AssertionError("a sub-batch has no attention placement")
                if cost.split_requests and self.subbatch_progression != "device_contiguous":
                    merge_id = f"{prefix}:merge"
                    tasks.append(
                        PipelineTask(
                            merge_id,
                            cost.merge_us,
                            ("gpu",),
                            tuple(attention_done),
                            "merge",
                            layer,
                            subbatch,
                            priority=2,
                        )
                    )
                    proj_predecessor = merge_id
                else:
                    # Even without an intra-request split, a mixed sub-batch
                    # may contain some all-GPU and some all-PIM requests.
                    # Batched projection must wait for both groups.
                    proj_predecessor = tuple(attention_done)

                if (
                    len(partitions) == 2
                    and self.subbatch_progression == "strict_ordered"
                ):
                    attention_predecessors = (
                        (proj_predecessor,)
                        if isinstance(proj_predecessor, str)
                        else proj_predecessor
                    )
                    tasks.append(
                        PipelineTask(
                            f"{prefix}:attention_done",
                            0.0,
                            (),
                            attention_predecessors,
                            "attention_done",
                            layer,
                            subbatch,
                        )
                    )

                proj_id = f"{prefix}:proj_ff"
                tasks.append(
                    PipelineTask(
                        proj_id,
                        cost.dense.proj_ff_us,
                        ("gpu",),
                        (
                            (proj_predecessor,)
                            if isinstance(proj_predecessor, str)
                            else proj_predecessor
                        ),
                        "proj_ff",
                        layer,
                        subbatch,
                        priority=3,
                    )
                )
                previous_proj[subbatch] = proj_id

        if self.subbatch_progression == "strict_ordered":
            tasks = order_two_subbatch_stages(tasks)
        elif self.subbatch_progression == "device_contiguous":
            tasks = order_two_subbatch_device_segments(tasks)
        schedule = schedule_pipeline(tasks)
        events = schedule.events
        dense_us = sum(
            event.duration_us for event in events if event.kind in ("qkv", "proj_ff")
        )
        gpu_attention_us = sum(
            event.duration_us for event in events if event.kind == "gpu_attention"
        )
        merge_us = sum(
            event.duration_us for event in events
            if event.kind in ("merge", "pim_merge")
        )
        gpu_busy_us = schedule.resource_busy_us.get("gpu", 0.0)
        rankset_busy = [
            schedule.resource_busy_us.get(f"pim:{index}", 0.0)
            for index in range(pim.ranksets)
        ]
        pim_busy_us = max(rankset_busy, default=0.0)
        if link.full_duplex:
            link_busy_us = max(
                schedule.resource_busy_us.get("link:h2d", 0.0),
                schedule.resource_busy_us.get("link:d2h", 0.0),
            )
        else:
            link_busy_us = schedule.resource_busy_us.get("link", 0.0)
        pim_path = max(pim_busy_us, link_busy_us)
        gpu_bottlenecks = 2 if gpu_busy_us >= pim_path else 0
        pim_bottlenecks = 2 - gpu_bottlenecks
        return IterationCost(
            total_us=schedule.makespan_us,
            per_layer_us=schedule.makespan_us / self.model.layers,
            gpu_busy_us=gpu_busy_us,
            pim_core_busy_us=pim_busy_us,
            link_busy_us=link_busy_us,
            gpu_attention_us=gpu_attention_us,
            dense_us=dense_us,
            merge_us=merge_us,
            phase_gpu_bottlenecks=gpu_bottlenecks,
            phase_pim_bottlenecks=pim_bottlenecks,
            subbatches=subcosts,
            pipeline_model="rankset_event",
            events=events,
        )

    def physical_pim_lower_bound_us(self, views: Iterable[RequestView]) -> float:
        logical_bytes = sum(
            view.pim_tokens * self.model.kv_bytes_per_token_per_layer
            for view in views
        )
        return _us(logical_bytes / self.system.pim.layer_bandwidth_bytes_s)
