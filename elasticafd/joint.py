"""Joint future-KV and dynamic O-projection scheduling upper bound.

This module is deliberately separate from the calibrated CHIME timing path.
It reuses the simulator's immutable :class:`RequestState` segments and physical
capacity checks, but evaluates a joint GPU/PIM/link resource bound.  The bound
is suitable for screening whether the two action families are complementary;
it is not yet a calibrated implementation of a PIM FP16 dense kernel.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import product
from math import ceil, inf, isclose
from typing import Iterable, Mapping, Sequence

from .capacity import views_fit_capacity
from .cost import ChimeCostModel
from .pipeline import (
    PipelineEvent, PipelineTask, order_two_subbatch_device_segments,
    order_two_subbatch_stages, schedule_pipeline,
)
from .specs import GiB, ModelSpec, SystemSpec
from .state import (
    Location,
    RequestState,
    RequestView,
    make_request_view,
    views_for_requests,
)


@dataclass(frozen=True)
class JointAction:
    """One decision block's future-only KV targets and O_proj row fraction."""

    tail_targets: tuple[tuple[str, Location], ...]
    o_proj_pim_fraction: float

    @classmethod
    def from_mapping(
        cls,
        targets: Mapping[str, Location],
        o_proj_pim_fraction: float,
    ) -> "JointAction":
        return cls(tuple(sorted(targets.items())), o_proj_pim_fraction)

    def target_mapping(self) -> dict[str, Location]:
        return dict(self.tail_targets)


@dataclass(frozen=True)
class JointIterationCost:
    """Per-iteration resource-bound cost used by the joint schedulers."""

    total_us: float
    per_layer_us: float
    gpu_busy_us: float
    pim_busy_us: float
    link_busy_us: float
    gpu_attention_us: float
    pim_attention_us: float
    gpu_o_proj_us: float
    pim_o_proj_us: float
    extra_operator_link_us: float
    o_proj_pim_fraction: float
    pipeline_model: str = "resource_bound"
    events: tuple[PipelineEvent, ...] = ()


@dataclass(frozen=True)
class JointRunResult:
    mode: str
    total_us: float
    schedule: tuple[tuple[int, JointAction], ...]
    cost_evaluations: int
    kv_switches: int
    operator_switches: int


@dataclass(frozen=True)
class _BlockResult:
    total_us: float
    survivors: list[RequestState]
    next_iteration: int
    o_proj_pim_fraction: float
    last_operator_switch_iteration: int
    kv_switches: int
    operator_switches: int


class JointAFResourceModel:
    """Optimistic shared-resource model for one decode iteration.

    A single PIM-side O_proj weight replica is shared by the GPU data-parallel
    groups.  GPU weights remain resident, so this is replication rather than
    migration and does not reclaim GPU memory.  The maximum allowed fraction is
    reserved for the whole experiment; changing the active fraction moves no
    weights and only changes routing for later activations.
    """

    def __init__(
        self,
        model: ModelSpec,
        system: SystemSpec,
        *,
        max_o_proj_pim_fraction: float,
        fidelity: str = "tile",
    ) -> None:
        if not 0.0 <= max_o_proj_pim_fraction <= 1.0:
            raise ValueError("max_o_proj_pim_fraction must lie in [0, 1]")
        self.model = model
        self.system = system
        self.max_o_proj_pim_fraction = max_o_proj_pim_fraction
        self.base = ChimeCostModel(model, system, fidelity)  # type: ignore[arg-type]

    @property
    def pim_o_proj_weight_bytes(self) -> int:
        return ceil(
            self.model.output_projection_weight_bytes_per_layer
            * self.model.layers
            * self.max_o_proj_pim_fraction
        )

    @property
    def pim_o_proj_weight_gib(self) -> float:
        return self.pim_o_proj_weight_bytes / GiB

    @property
    def pim_kv_capacity_bytes(self) -> float:
        return self.system.pim.capacity_bytes - self.pim_o_proj_weight_bytes

    @property
    def gpu_replica_kv_capacity_bytes(self) -> float:
        # Dynamic O_proj keeps the GPU copy, so no GPU weight capacity is
        # reclaimed by enabling this scheduler.
        return self.system.gpu_replica_kv_capacity_bytes(self.model)

    @property
    def initial_weight_copy_us(self) -> float:
        return (
            self.pim_o_proj_weight_bytes
            / self.system.link.host_to_pim_bytes_s
            * 1e6
        )

    def views_fit(self, views: Sequence[RequestView]) -> bool:
        if self.pim_kv_capacity_bytes < 0:
            return False
        return views_fit_capacity(
            views,
            self.model,
            self.system,
            pim_capacity_bytes=self.pim_kv_capacity_bytes,
            gpu_replica_capacity_bytes=self.gpu_replica_kv_capacity_bytes,
        )

    def _gpu_o_proj_us(self, batch_size: int, fraction: float) -> float:
        remaining = 1.0 - fraction
        if batch_size <= 0 or remaining <= 0:
            return 0.0
        full = self.base.gpu_matrix_layer_cost("o_proj", batch_size)
        launch_us = full.total_us - max(full.memory_us, full.compute_us)
        return max(
            full.memory_us * remaining,
            full.compute_us * remaining,
        ) + launch_us

    def _pim_o_proj_us(self, batch_size: int, fraction: float) -> float:
        if batch_size <= 0 or fraction <= 0:
            return 0.0
        full = self.base.pim_matrix_layer_cost("o_proj", batch_size)
        return max(full.memory_us * fraction, full.compute_us * fraction)

    def _extra_operator_link_us(
        self, views: Sequence[RequestView], fraction: float
    ) -> float:
        if fraction <= 0:
            return 0.0
        vector_bytes = self.model.output_bytes_per_token_per_layer
        # Any GPU-resident attention contribution means the merged attention
        # vector exists on GPU and must be sent to PIM before its O_proj rows.
        h2d_bytes = sum(
            vector_bytes for view in views if view.gpu_tokens > 0
        )
        d2h_bytes = 0.0
        for view in views:
            if view.gpu_tokens > 0:
                d2h_bytes += vector_bytes * fraction
            elif fraction < 1.0:
                # The normal PIM-attention output still feeds the GPU rows;
                # only the PIM-produced output slice is additional traffic.
                d2h_bytes += vector_bytes * fraction
            # For an all-PIM request at fraction=1, the normal attention output
            # transfer is replaced by an equal-sized O_proj output transfer.

        link = self.system.link
        h2d_us = (
            h2d_bytes / link.host_to_pim_bytes_s * 1e6
            + (link.transfer_latency_us if h2d_bytes else 0.0)
        )
        d2h_us = (
            d2h_bytes / link.pim_to_host_bytes_s * 1e6
            + (link.transfer_latency_us if d2h_bytes else 0.0)
        )
        return max(h2d_us, d2h_us) if link.full_duplex else h2d_us + d2h_us

    def iteration_cost(
        self,
        views: Sequence[RequestView],
        o_proj_pim_fraction: float,
    ) -> JointIterationCost:
        if not views:
            raise ValueError("cannot cost an empty decode iteration")
        if not 0.0 <= o_proj_pim_fraction <= self.max_o_proj_pim_fraction:
            raise ValueError(
                "o_proj_pim_fraction exceeds the persistent replica"
            )

        partitions = self.base.partition_views(views)
        gpu_per_layer = 0.0
        pim_per_layer = 0.0
        link_per_layer = 0.0
        gpu_attention_per_layer = 0.0
        pim_attention_per_layer = 0.0
        gpu_o_proj_per_layer = 0.0
        pim_o_proj_per_layer = 0.0
        extra_link_per_layer = 0.0

        for subbatch in partitions:
            batch_size = len(subbatch)
            qkv = self.base.gpu_matrix_layer_cost("qkv", batch_size)
            ffn = self.base.gpu_matrix_layer_cost("ffn", batch_size)
            gpu_attention = self.base.gpu_attention_layer_cost(subbatch)
            pim_attention = self.base.pim_layer_cost(subbatch)
            merge_us, _ = self.base.merge_layer_cost(subbatch)
            gpu_o_proj_us = self._gpu_o_proj_us(
                batch_size, o_proj_pim_fraction
            )
            pim_o_proj_us = self._pim_o_proj_us(
                batch_size, o_proj_pim_fraction
            )
            extra_link_us = self._extra_operator_link_us(
                subbatch, o_proj_pim_fraction
            )

            gpu_per_layer += (
                qkv.total_us
                + gpu_attention.total_us
                + merge_us
                + gpu_o_proj_us
                + ffn.total_us
            )
            pim_per_layer += (
                pim_attention.core_us
                + pim_attention.setup_us
                + pim_attention.relayout_us
                + pim_o_proj_us
            )
            link_per_layer += pim_attention.link_us + extra_link_us
            gpu_attention_per_layer += gpu_attention.total_us
            pim_attention_per_layer += pim_attention.total_us
            gpu_o_proj_per_layer += gpu_o_proj_us
            pim_o_proj_per_layer += pim_o_proj_us
            extra_link_per_layer += extra_link_us

        # This is a resource lower bound: dependencies and bank-command
        # conflicts are intentionally deferred to the calibrated joint event
        # model in a later stage.
        per_layer_us = max(gpu_per_layer, pim_per_layer, link_per_layer)
        layers = self.model.layers
        return JointIterationCost(
            total_us=per_layer_us * layers,
            per_layer_us=per_layer_us,
            gpu_busy_us=gpu_per_layer * layers,
            pim_busy_us=pim_per_layer * layers,
            link_busy_us=link_per_layer * layers,
            gpu_attention_us=gpu_attention_per_layer * layers,
            pim_attention_us=pim_attention_per_layer * layers,
            gpu_o_proj_us=gpu_o_proj_per_layer * layers,
            pim_o_proj_us=pim_o_proj_per_layer * layers,
            extra_operator_link_us=extra_link_per_layer * layers,
            o_proj_pim_fraction=o_proj_pim_fraction,
        )


class JointAFEventModel(JointAFResourceModel):
    """Dependency-aware joint GPU/PIM/link event model.

    CHIME's calibrated attention and GPU dense costs are preserved. PIM
    O-projection remains the optimistic dense roofline from the Stage 3A
    screen, but it now conflicts with attention on the same rankset and all
    cross-device activation edges are explicit in the task DAG.
    """

    def __init__(
        self,
        model: ModelSpec,
        system: SystemSpec,
        *,
        max_o_proj_pim_fraction: float,
        fidelity: str = "tile",
        o_proj_routing: str = "uniform_rows",
        split_merge_owner: str = "gpu",
        pim_merge_cost_scale: float = 0.0,
        subbatch_progression: str = "strict_ordered",
        subbatch_scheduler: str = "lpt",
    ) -> None:
        if o_proj_routing not in ("uniform_rows", "split_aware_requests"):
            raise ValueError(
                "o_proj_routing must be uniform_rows or split_aware_requests"
            )
        if split_merge_owner not in ("gpu", "pim_when_pim_oproj", "pim"):
            raise ValueError(
                "split_merge_owner must be gpu, pim_when_pim_oproj, or pim"
            )
        if (
            split_merge_owner in ("pim_when_pim_oproj", "pim")
            and o_proj_routing != "uniform_rows"
        ):
            raise ValueError(
                "PIM-owned merge currently requires uniform-row O_proj routing"
            )
        if pim_merge_cost_scale < 0.0:
            raise ValueError("pim_merge_cost_scale cannot be negative")
        if subbatch_progression not in (
            "strict_ordered", "free_wavefront", "device_contiguous"
        ):
            raise ValueError("invalid subbatch progression")
        if subbatch_progression == "device_contiguous" and split_merge_owner != "pim":
            raise ValueError("device-contiguous Attention requires PIM-owned merge")
        super().__init__(
            model,
            system,
            max_o_proj_pim_fraction=max_o_proj_pim_fraction,
            fidelity=fidelity,
        )
        self.o_proj_routing = o_proj_routing
        self.split_merge_owner = split_merge_owner
        self.subbatch_progression = subbatch_progression
        self.base.subbatch_scheduler = subbatch_scheduler
        # CHIME does not publish a PIM partial-softmax merge kernel.  The
        # owner-coupled experiment therefore expresses its duration as an
        # explicit sensitivity multiple of the existing GPU merge estimate.
        # A scale of zero is an optimistic dependency/communication lower
        # bound; it must not be presented as a calibrated PIM kernel.
        self.pim_merge_cost_scale = pim_merge_cost_scale
        self._event_cost_cache: dict[
            tuple[tuple[RequestView, ...], float], JointIterationCost
        ] = {}

    @property
    def cached_iteration_costs(self) -> int:
        return len(self._event_cost_cache)

    def _transfer_us(self, byte_count: float, direction: str) -> float:
        if byte_count <= 0:
            return 0.0
        link = self.system.link
        bandwidth = (
            link.host_to_pim_bytes_s
            if direction == "h2d"
            else link.pim_to_host_bytes_s
        )
        return byte_count / bandwidth * 1e6 + link.transfer_latency_us

    def _transform_tasks(
        self,
        tasks: list[PipelineTask],
        partitions: Sequence[Sequence[RequestView]],
        o_proj_pim_fraction: float,
    ) -> list[PipelineTask]:
        """Extension point; the established O_proj event graph is unchanged."""
        return tasks

    def iteration_cost(
        self,
        views: Sequence[RequestView],
        o_proj_pim_fraction: float,
    ) -> JointIterationCost:
        if not views:
            raise ValueError("cannot cost an empty decode iteration")
        if not 0.0 <= o_proj_pim_fraction <= self.max_o_proj_pim_fraction:
            raise ValueError(
                "o_proj_pim_fraction exceeds the persistent replica"
            )
        cache_key = (tuple(views), o_proj_pim_fraction)
        cached = self._event_cost_cache.get(cache_key)
        if cached is not None:
            return cached

        partitions = self.base.partition_views(views)
        pim = self.system.pim
        link = self.system.link
        model = self.model
        tasks: list[PipelineTask] = []
        previous_ffn: dict[int, str] = {}

        def rankset_resources(layer: int) -> tuple[str, ...]:
            return tuple(
                f"pim:{(layer + offset) % pim.ranksets}"
                for offset in range(pim.compute_ranksets_per_layer)
            )

        def link_resource(direction: str) -> str:
            return f"link:{direction}" if link.full_duplex else "link"

        def communication_resources(
            layer: int, direction: str
        ) -> tuple[str, ...]:
            ranksets = rankset_resources(layer)
            if not pim.rankset_comm_overlap:
                ranksets = tuple(
                    f"pim:{index}" for index in range(pim.ranksets)
                )
            return (link_resource(direction), *ranksets)

        for layer in range(model.layers):
            ranksets = rankset_resources(layer)
            for subbatch, batch_views in enumerate(partitions):
                prefix = f"l{layer}:s{subbatch}"
                batch_size = len(batch_views)
                pim_views = [view for view in batch_views if view.pim_tokens > 0]
                gpu_views = [view for view in batch_views if view.gpu_tokens > 0]
                split_views = [view for view in batch_views if view.is_split]
                pim_only_views = [
                    view
                    for view in batch_views
                    if view.pim_tokens > 0 and view.gpu_tokens == 0
                ]
                gpu_only_views = [
                    view
                    for view in batch_views
                    if view.gpu_tokens > 0 and view.pim_tokens == 0
                ]
                merge_on_pim = bool(
                    split_views
                    and (
                        self.split_merge_owner == "pim"
                        or (
                            self.split_merge_owner == "pim_when_pim_oproj"
                            and o_proj_pim_fraction > 0.0
                        )
                    )
                )

                qkv_id = f"{prefix}:qkv"
                tasks.append(
                    PipelineTask(
                        qkv_id,
                        self.base.gpu_matrix_layer_cost(
                            "qkv", batch_size
                        ).total_us,
                        ("gpu",),
                        (
                            (previous_ffn[subbatch],)
                            if subbatch in previous_ffn
                            else ()
                        ),
                        "gpu_qkv",
                        layer,
                        subbatch,
                        priority=4,
                    )
                )
                attention_task_start = len(tasks)

                pim_attention_id: str | None = None
                pim_to_gpu_id: str | None = None
                if pim_views:
                    pim_cost = self.base.pim_layer_cost(batch_views)
                    pim_input_id = f"{prefix}:pim_attention_h2d"
                    tasks.append(
                        PipelineTask(
                            pim_input_id,
                            pim_cost.h2d_us + (
                                len(split_views)
                                * (model.output_bytes_per_token_per_layer
                                   + model.num_attention_heads * 2 * 4)
                                / link.host_to_pim_bytes_s * 1e6
                                if merge_on_pim
                                and self.subbatch_progression == "device_contiguous"
                                else 0.0
                            ),
                            communication_resources(layer, "h2d"),
                            (
                                (qkv_id, f"{prefix}:gpu_attention")
                                if merge_on_pim
                                and self.subbatch_progression == "device_contiguous"
                                else (qkv_id,)
                            ),
                            "pim_attention_h2d",
                            layer,
                            subbatch,
                            priority=0,
                        )
                    )
                    pim_attention_id = f"{prefix}:pim_attention"
                    tasks.append(
                        PipelineTask(
                            pim_attention_id,
                            pim_cost.core_us
                            + pim_cost.setup_us
                            + pim_cost.relayout_us,
                            ranksets,
                            (pim_input_id,),
                            "pim_attention",
                            layer,
                            subbatch,
                            priority=1,
                        )
                    )

                    # If merge is GPU-owned, every split PIM partial goes to
                    # GPU; a partial GPU O_proj also needs PIM-only outputs.
                    # If merge is PIM-owned, split partials stay local and only
                    # PIM-only requests needed by a GPU O_proj slice leave at
                    # this point. Split merged outputs are transferred later.
                    if merge_on_pim:
                        to_gpu = (
                            pim_only_views
                            if o_proj_pim_fraction < 1.0
                            else []
                        )
                    else:
                        to_gpu = (
                            pim_views
                            if o_proj_pim_fraction < 1.0
                            else split_views
                        )
                    if to_gpu:
                        bytes_to_gpu = sum(
                            model.output_bytes_per_token_per_layer
                            + (
                                model.num_attention_heads * 2 * 4
                                if view.is_split and not merge_on_pim
                                else 0
                            )
                            for view in to_gpu
                        )
                        pim_to_gpu_id = f"{prefix}:pim_attention_d2h"
                        tasks.append(
                            PipelineTask(
                                pim_to_gpu_id,
                                self._transfer_us(bytes_to_gpu, "d2h"),
                                communication_resources(layer, "d2h"),
                                (pim_attention_id,),
                                "pim_attention_d2h",
                                layer,
                                subbatch,
                                priority=0,
                            )
                        )

                gpu_attention_id: str | None = None
                if gpu_views:
                    gpu_attention_id = f"{prefix}:gpu_attention"
                    tasks.append(
                        PipelineTask(
                            gpu_attention_id,
                            self.base.gpu_attention_layer_cost(
                                batch_views
                            ).total_us,
                            ("gpu",),
                            (qkv_id,),
                            "gpu_attention",
                            layer,
                            subbatch,
                            priority=2,
                        )
                    )

                merge_id: str | None = None
                pim_merge_d2h_id: str | None = None
                if split_views:
                    merge_us, _ = self.base.merge_layer_cost(batch_views)
                    if merge_on_pim:
                        if pim_attention_id is None or gpu_attention_id is None:
                            raise AssertionError(
                                "split attention requires both partial results"
                            )
                        if self.subbatch_progression == "device_contiguous":
                            gpu_partial_h2d_id = pim_input_id
                        else:
                            gpu_partial_h2d_id = f"{prefix}:gpu_partial_h2d"
                            partial_bytes = len(split_views) * (
                                model.output_bytes_per_token_per_layer
                                + model.num_attention_heads * 2 * 4
                            )
                            tasks.append(
                                PipelineTask(
                                    gpu_partial_h2d_id,
                                    self._transfer_us(partial_bytes, "h2d"),
                                    communication_resources(layer, "h2d"),
                                    (gpu_attention_id,),
                                    "gpu_partial_h2d",
                                    layer,
                                    subbatch,
                                    priority=0,
                                )
                            )
                        merge_id = f"{prefix}:pim_merge"
                        tasks.append(
                            PipelineTask(
                                merge_id,
                                merge_us * self.pim_merge_cost_scale,
                                ranksets,
                                (pim_attention_id, gpu_partial_h2d_id),
                                "pim_merge",
                                layer,
                                subbatch,
                                priority=1,
                            )
                        )
                        if o_proj_pim_fraction < 1.0:
                            pim_merge_d2h_id = f"{prefix}:pim_merge_d2h"
                            tasks.append(
                                PipelineTask(
                                    pim_merge_d2h_id,
                                    self._transfer_us(
                                        len(split_views)
                                        * model.output_bytes_per_token_per_layer,
                                        "d2h",
                                    ),
                                    communication_resources(layer, "d2h"),
                                    (merge_id,),
                                    "pim_merge_d2h",
                                    layer,
                                    subbatch,
                                    priority=0,
                                )
                            )
                    else:
                        if pim_to_gpu_id is None or gpu_attention_id is None:
                            raise AssertionError(
                                "split attention requires both partial results"
                            )
                        merge_id = f"{prefix}:merge"
                        tasks.append(
                            PipelineTask(
                                merge_id,
                                merge_us,
                                ("gpu",),
                                (pim_to_gpu_id, gpu_attention_id),
                                "merge",
                                layer,
                                subbatch,
                                priority=2,
                            )
                        )

                if (
                    len(partitions) == 2
                    and self.subbatch_progression == "strict_ordered"
                ):
                    attention_ids = tuple(
                        task.task_id for task in tasks[attention_task_start:]
                    )
                    if not attention_ids:
                        raise AssertionError("a sub-batch has no Attention tasks")
                    tasks.append(
                        PipelineTask(
                            f"{prefix}:attention_done",
                            0.0,
                            (),
                            attention_ids,
                            "attention_done",
                            layer,
                            subbatch,
                        )
                    )

                gpu_o_proj_ids: list[str] = []
                if (
                    self.o_proj_routing == "uniform_rows"
                    or o_proj_pim_fraction == 0.0
                ) and o_proj_pim_fraction < 1.0:
                    gpu_ready: list[str] = []
                    if pim_only_views:
                        if pim_to_gpu_id is None:
                            raise AssertionError("PIM output missing GPU edge")
                        gpu_ready.append(pim_to_gpu_id)
                    if gpu_only_views:
                        if gpu_attention_id is None:
                            raise AssertionError("GPU attention task missing")
                        gpu_ready.append(gpu_attention_id)
                    if split_views:
                        if merge_on_pim:
                            if pim_merge_d2h_id is None:
                                raise AssertionError(
                                    "PIM merged output missing GPU edge"
                                )
                            gpu_ready.append(pim_merge_d2h_id)
                        else:
                            if merge_id is None:
                                raise AssertionError("split merge task missing")
                            gpu_ready.append(merge_id)
                    gpu_o_proj_id = f"{prefix}:gpu_o_proj"
                    tasks.append(
                        PipelineTask(
                            gpu_o_proj_id,
                            self._gpu_o_proj_us(
                                batch_size, o_proj_pim_fraction
                            ),
                            ("gpu",),
                            tuple(dict.fromkeys(gpu_ready)),
                            "gpu_o_proj",
                            layer,
                            subbatch,
                            priority=3,
                        )
                    )
                    gpu_o_proj_ids.append(gpu_o_proj_id)
                elif self.o_proj_routing == "split_aware_requests":
                    # Requests whose attention output is assembled on GPU run
                    # their full O_proj there.  PIM-only requests use the
                    # selected row fraction on PIM and only the complementary
                    # rows on GPU.  Separate tasks conservatively charge the
                    # extra masked/grouped GPU kernel instead of assuming free
                    # fusion across the two request classes.
                    gpu_origin_count = len(gpu_only_views) + len(split_views)
                    if gpu_origin_count:
                        origin_ready: list[str] = []
                        if gpu_only_views:
                            if gpu_attention_id is None:
                                raise AssertionError("GPU attention task missing")
                            origin_ready.append(gpu_attention_id)
                        if split_views:
                            if merge_id is None:
                                raise AssertionError("split merge task missing")
                            origin_ready.append(merge_id)
                        origin_id = f"{prefix}:gpu_o_proj_origin"
                        tasks.append(
                            PipelineTask(
                                origin_id,
                                self._gpu_o_proj_us(gpu_origin_count, 0.0),
                                ("gpu",),
                                tuple(dict.fromkeys(origin_ready)),
                                "gpu_o_proj",
                                layer,
                                subbatch,
                                priority=3,
                            )
                        )
                        gpu_o_proj_ids.append(origin_id)
                    if pim_only_views and o_proj_pim_fraction < 1.0:
                        if pim_to_gpu_id is None:
                            raise AssertionError("PIM output missing GPU edge")
                        pim_only_id = f"{prefix}:gpu_o_proj_pim_only"
                        tasks.append(
                            PipelineTask(
                                pim_only_id,
                                self._gpu_o_proj_us(
                                    len(pim_only_views), o_proj_pim_fraction
                                ),
                                ("gpu",),
                                (pim_to_gpu_id,),
                                "gpu_o_proj",
                                layer,
                                subbatch,
                                priority=3,
                            )
                        )
                        gpu_o_proj_ids.append(pim_only_id)

                pim_o_proj_output_id: str | None = None
                pim_o_proj_batch_size = (
                    batch_size
                    if self.o_proj_routing == "uniform_rows"
                    else len(pim_only_views)
                )
                if o_proj_pim_fraction > 0.0 and pim_o_proj_batch_size:
                    pim_ready: list[str] = []
                    if pim_only_views:
                        if pim_attention_id is None:
                            raise AssertionError("PIM attention task missing")
                        pim_ready.append(pim_attention_id)
                    if split_views and merge_on_pim:
                        if merge_id is None:
                            raise AssertionError("PIM split merge task missing")
                        pim_ready.append(merge_id)

                    # Uniform row splitting sends GPU-origin activations back
                    # to PIM. Split-aware routing deliberately keeps their
                    # O_proj on GPU and therefore has no such edge.
                    gpu_origin_count = 0
                    if self.o_proj_routing == "uniform_rows":
                        gpu_origin_count = len(gpu_only_views)
                        if not merge_on_pim:
                            gpu_origin_count += len(split_views)
                    if gpu_origin_count:
                        origin_predecessors: list[str] = []
                        if gpu_only_views:
                            if gpu_attention_id is None:
                                raise AssertionError("GPU attention task missing")
                            origin_predecessors.append(gpu_attention_id)
                        if split_views and not merge_on_pim:
                            if merge_id is None:
                                raise AssertionError("split merge task missing")
                            origin_predecessors.append(merge_id)
                        op_input_id = f"{prefix}:o_proj_h2d"
                        tasks.append(
                            PipelineTask(
                                op_input_id,
                                self._transfer_us(
                                    gpu_origin_count
                                    * model.output_bytes_per_token_per_layer,
                                    "h2d",
                                ),
                                communication_resources(layer, "h2d"),
                                tuple(dict.fromkeys(origin_predecessors)),
                                "o_proj_h2d",
                                layer,
                                subbatch,
                                priority=0,
                            )
                        )
                        pim_ready.append(op_input_id)

                    pim_o_proj_id = f"{prefix}:pim_o_proj"
                    tasks.append(
                        PipelineTask(
                            pim_o_proj_id,
                            self._pim_o_proj_us(
                                pim_o_proj_batch_size, o_proj_pim_fraction
                            ),
                            ranksets,
                            tuple(dict.fromkeys(pim_ready)),
                            "pim_o_proj",
                            layer,
                            subbatch,
                            priority=1,
                        )
                    )
                    pim_o_proj_output_id = f"{prefix}:o_proj_d2h"
                    tasks.append(
                        PipelineTask(
                            pim_o_proj_output_id,
                            self._transfer_us(
                                pim_o_proj_batch_size
                                * model.output_bytes_per_token_per_layer
                                * o_proj_pim_fraction,
                                "d2h",
                            ),
                            communication_resources(layer, "d2h"),
                            (pim_o_proj_id,),
                            "o_proj_d2h",
                            layer,
                            subbatch,
                            priority=0,
                        )
                    )

                output_parts = tuple(
                    task_id
                    for task_id in (*gpu_o_proj_ids, pim_o_proj_output_id)
                    if task_id is not None
                )
                if not output_parts:
                    raise AssertionError("O_proj produced no output")
                combine_id = f"{prefix}:o_proj_join"
                tasks.append(
                    PipelineTask(
                        combine_id,
                        0.0,
                        (),
                        output_parts,
                        "o_proj_join",
                        layer,
                        subbatch,
                        priority=3,
                    )
                )
                ffn_id = f"{prefix}:ffn"
                tasks.append(
                    PipelineTask(
                        ffn_id,
                        self.base.gpu_matrix_layer_cost(
                            "ffn", batch_size
                        ).total_us,
                        ("gpu",),
                        (combine_id,),
                        "gpu_ffn",
                        layer,
                        subbatch,
                        priority=4,
                    )
                )
                previous_ffn[subbatch] = ffn_id

        if (
            self.subbatch_progression == "device_contiguous"
            and 0.0 < o_proj_pim_fraction < 1.0
        ):
            # The GPU continuation needs both the merged Attention vector
            # and the PIM O_proj shard. They cross at the *same* PIM->GPU cut:
            # coalesce their payload into one DMA rather than charging a
            # second fixed latency or permitting an early GPU continuation.
            coalesced: list[PipelineTask] = []
            for task in tasks:
                if task.kind == "o_proj_d2h":
                    part = partitions[task.subbatch]
                    extra = sum(
                        model.output_bytes_per_token_per_layer
                        for view in part if view.pim_tokens > 0
                    )
                    byte_count = (
                        len(part) * model.output_bytes_per_token_per_layer
                        * o_proj_pim_fraction + extra
                    )
                    coalesced.append(replace(
                        task, duration_us=self._transfer_us(byte_count, "d2h"),
                    ))
                elif task.kind in ("pim_attention_d2h", "pim_merge_d2h"):
                    coalesced.append(replace(
                        task, duration_us=0.0, resources=(),
                        predecessors=(
                            f"l{task.layer}:s{task.subbatch}:o_proj_d2h",
                        ),
                    ))
                else:
                    coalesced.append(task)
            tasks = coalesced

        scheduled_tasks = self._transform_tasks(
            tasks, partitions, o_proj_pim_fraction,
        )
        if self.subbatch_progression == "strict_ordered":
            scheduled_tasks = order_two_subbatch_stages(scheduled_tasks)
        elif self.subbatch_progression == "device_contiguous":
            scheduled_tasks = order_two_subbatch_device_segments(scheduled_tasks)
        schedule = schedule_pipeline(scheduled_tasks)
        events = schedule.events
        gpu_attention_us = sum(
            event.duration_us
            for event in events
            if event.kind == "gpu_attention"
        )
        pim_attention_us = sum(
            event.duration_us
            for event in events
            if event.kind == "pim_attention"
        )
        gpu_o_proj_us = sum(
            event.duration_us
            for event in events
            if event.kind == "gpu_o_proj"
        )
        pim_o_proj_us = sum(
            event.duration_us
            for event in events
            if event.kind == "pim_o_proj"
        )
        extra_operator_link_us = sum(
            event.duration_us
            for event in events
            if event.kind
            in (
                "gpu_partial_h2d",
                "pim_merge_d2h",
                "o_proj_h2d",
                "o_proj_d2h",
            )
        )
        rankset_busy = [
            schedule.resource_busy_us.get(f"pim:{index}", 0.0)
            for index in range(pim.ranksets)
        ]
        if link.full_duplex:
            link_busy_us = max(
                schedule.resource_busy_us.get("link:h2d", 0.0),
                schedule.resource_busy_us.get("link:d2h", 0.0),
            )
        else:
            link_busy_us = schedule.resource_busy_us.get("link", 0.0)
        result = JointIterationCost(
            total_us=schedule.makespan_us,
            per_layer_us=schedule.makespan_us / model.layers,
            gpu_busy_us=schedule.resource_busy_us.get("gpu", 0.0),
            pim_busy_us=max(rankset_busy, default=0.0),
            link_busy_us=link_busy_us,
            gpu_attention_us=gpu_attention_us,
            pim_attention_us=pim_attention_us,
            gpu_o_proj_us=gpu_o_proj_us,
            pim_o_proj_us=pim_o_proj_us,
            extra_operator_link_us=extra_operator_link_us,
            o_proj_pim_fraction=o_proj_pim_fraction,
            pipeline_model=(
                "device_contiguous_pim_merge"
                if self.subbatch_progression == "device_contiguous"
                else
                "dependency_event_split_aware_oproj"
                if self.o_proj_routing == "split_aware_requests"
                else (
                    "dependency_event_owner_coupled_merge"
                    if self.split_merge_owner == "pim_when_pim_oproj"
                    else "dependency_event"
                )
            ),
            events=events,
        )
        self._event_cost_cache[cache_key] = result
        return result


class JointAFScheduler:
    """Small exact oracle plus a bounded coordinate-greedy policy."""

    def __init__(
        self,
        cost_model: JointAFResourceModel,
        *,
        fractions: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
        decision_interval: int = 64,
        min_kv_segment_tokens: int = 64,
        min_kv_switch_interval: int = 128,
        min_operator_residence_tokens: int = 128,
        max_kv_switches_per_decision: int = 8,
        kv_switch_penalty_us: float = 0.0,
        operator_switch_penalty_us: float = 0.0,
        max_exact_candidates: int = 8,
        max_cost_evaluations: int = 200_000,
    ) -> None:
        values = tuple(sorted(dict.fromkeys(float(value) for value in fractions)))
        if not values or values[0] != 0.0:
            raise ValueError("joint fractions must include 0")
        if values[-1] > cost_model.max_o_proj_pim_fraction:
            raise ValueError("fraction exceeds the cost model's weight replica")
        if decision_interval <= 0:
            raise ValueError("decision_interval must be positive")
        if min_kv_segment_tokens < 0 or min_kv_switch_interval < 0:
            raise ValueError("KV residence constraints cannot be negative")
        if min_operator_residence_tokens < 0:
            raise ValueError("operator residence cannot be negative")
        if max_kv_switches_per_decision < 0:
            raise ValueError("max_kv_switches_per_decision cannot be negative")
        if kv_switch_penalty_us < 0 or operator_switch_penalty_us < 0:
            raise ValueError("switch penalties cannot be negative")
        self.cost_model = cost_model
        self.fractions = values
        self.decision_interval = decision_interval
        self.min_kv_segment_tokens = min_kv_segment_tokens
        self.min_kv_switch_interval = min_kv_switch_interval
        self.min_operator_residence_tokens = min_operator_residence_tokens
        self.max_kv_switches_per_decision = max_kv_switches_per_decision
        self.kv_switch_penalty_us = kv_switch_penalty_us
        self.operator_switch_penalty_us = operator_switch_penalty_us
        self.max_exact_candidates = max_exact_candidates
        self.max_cost_evaluations = max_cost_evaluations

    def _request_eligible(self, request: RequestState, iteration: int) -> bool:
        return (
            iteration - request.last_switch_iteration
            >= self.min_kv_switch_interval
            and request.tail_segment_tokens >= self.min_kv_segment_tokens
        )

    def _kv_target_assignments(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        *,
        allow_gpu_kv: bool,
        coordinate_only: bool,
        enforce_switch_limit: bool = True,
        enforce_residence: bool = True,
    ) -> tuple[dict[str, Location], ...]:
        stay = {
            request.request_id: request.tail_location for request in requests
        }
        if not allow_gpu_kv or (
            enforce_switch_limit
            and self.max_kv_switches_per_decision == 0
        ):
            return (stay,)
        eligible = (
            [
                request
                for request in requests
                if self._request_eligible(request, iteration)
            ]
            if enforce_residence
            else list(requests)
        )
        if len(eligible) > self.max_exact_candidates and not coordinate_only:
            raise RuntimeError(
                f"joint exact has {len(eligible)} eligible requests, exceeding "
                f"max_exact_candidates={self.max_exact_candidates}"
            )
        if not eligible:
            return (stay,)

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
        grouped = tuple(groups.values())
        switch_limit = (
            self.max_kv_switches_per_decision
            if enforce_switch_limit
            else len(eligible)
        )

        switch_vectors: list[tuple[int, ...]] = []
        if coordinate_only:
            switch_vectors.append((0,) * len(grouped))
            for group_index, group in enumerate(grouped):
                for count in range(
                    1, min(len(group), switch_limit) + 1,
                ):
                    vector = [0] * len(grouped)
                    vector[group_index] = count
                    switch_vectors.append(tuple(vector))
        else:
            for counts in product(*(range(len(group) + 1) for group in grouped)):
                if sum(counts) <= switch_limit:
                    switch_vectors.append(tuple(counts))

        assignments: list[dict[str, Location]] = []
        for counts in switch_vectors:
            targets = dict(stay)
            for group, count in zip(grouped, counts):
                for request in group[:count]:
                    targets[request.request_id] = request.tail_location.other
            assignments.append(targets)
        return tuple(assignments)

    def _actions(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        current_fraction: float,
        last_operator_switch_iteration: int,
        *,
        allow_gpu_kv: bool,
        allow_operator: bool,
        coordinate_only: bool,
    ) -> tuple[JointAction, ...]:
        creation_decision = iteration == 0 and all(
            request.generated_tokens == 0 and request.switch_count == 0
            for request in requests
        )
        targets = self._kv_target_assignments(
            requests,
            iteration,
            allow_gpu_kv=allow_gpu_kv,
            coordinate_only=coordinate_only,
            enforce_switch_limit=not creation_decision,
            enforce_residence=not creation_decision,
        )
        operator_eligible = (
            allow_operator
            and iteration - last_operator_switch_iteration
            >= self.min_operator_residence_tokens
        )
        fractions = self.fractions if operator_eligible else (current_fraction,)
        return tuple(
            JointAction.from_mapping(assignment, fraction)
            for assignment in targets
            for fraction in fractions
        )

    def _state_signature(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        current_fraction: float,
        last_operator_switch_iteration: int,
    ) -> tuple[object, ...]:
        return (
            iteration,
            current_fraction,
            last_operator_switch_iteration,
            tuple(
                (
                    request.request_id,
                    request.gpu_replica,
                    request.generated_tokens,
                    request.tail_location.value,
                    request.last_switch_iteration,
                    tuple(
                        (segment.location.value, segment.start_token, segment.end_token)
                        for segment in request.segments
                    ),
                )
                for request in requests
            ),
        )

    def _simulate_block(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        current_fraction: float,
        last_operator_switch_iteration: int,
        action: JointAction,
        work: list[int] | None = None,
    ) -> _BlockResult | None:
        projected = deepcopy(list(requests))
        targets = action.target_mapping()
        kv_switches = 0
        creation_decision = iteration == 0 and all(
            request.generated_tokens == 0 and request.switch_count == 0
            for request in projected
        )
        for request in projected:
            if creation_decision:
                # The prompt stays on PIM, but the first generated token can be
                # assigned freely. This is initial placement, not migration or
                # a runtime switch subject to the per-decision change cap.
                request.tail_location = targets[request.request_id]
                request.last_switch_iteration = iteration
            elif request.switch_tail(targets[request.request_id], iteration):
                kv_switches += 1
        operator_switches = int(
            action.o_proj_pim_fraction != current_fraction
        )
        if operator_switches:
            last_operator_switch_iteration = iteration

        total_us = (
            kv_switches * self.kv_switch_penalty_us
            + operator_switches * self.operator_switch_penalty_us
        )
        steps = 0
        while projected and steps < self.decision_interval:
            if work is not None:
                work[0] += 1
                if work[0] > self.max_cost_evaluations:
                    raise RuntimeError(
                        "joint exact exceeded max_cost_evaluations="
                        f"{self.max_cost_evaluations}; reduce batch/output, "
                        "increase decision_interval, or raise the guard"
                    )
            views = views_for_requests(projected)
            if not self.cost_model.views_fit(views):
                return None
            total_us += self.cost_model.iteration_cost(
                views, action.o_proj_pim_fraction
            ).total_us
            for request in projected:
                request.append_generated_token()
            appended_views = views_for_requests(projected)
            if not self.cost_model.views_fit(appended_views):
                return None
            projected = [
                request for request in projected if not request.is_finished
            ]
            steps += 1
        return _BlockResult(
            total_us,
            projected,
            iteration + steps,
            action.o_proj_pim_fraction,
            last_operator_switch_iteration,
            kv_switches,
            operator_switches,
        )

    def _search(
        self,
        requests: Sequence[RequestState],
        iteration: int,
        current_fraction: float,
        last_operator_switch_iteration: int,
        *,
        allow_gpu_kv: bool,
        allow_operator: bool,
        memo: dict[
            tuple[object, ...],
            tuple[float, tuple[tuple[int, JointAction], ...], int, int],
        ],
        work: list[int],
    ) -> tuple[float, tuple[tuple[int, JointAction], ...], int, int]:
        if not requests:
            return 0.0, (), 0, 0
        signature = self._state_signature(
            requests,
            iteration,
            current_fraction,
            last_operator_switch_iteration,
        )
        cached = memo.get(signature)
        if cached is not None:
            return cached

        best = (inf, (), 0, 0)
        actions = self._actions(
            requests,
            iteration,
            current_fraction,
            last_operator_switch_iteration,
            allow_gpu_kv=allow_gpu_kv,
            allow_operator=allow_operator,
            coordinate_only=False,
        )
        for action in actions:
            block = self._simulate_block(
                requests,
                iteration,
                current_fraction,
                last_operator_switch_iteration,
                action,
                work,
            )
            if block is None:
                continue
            future_us, future_schedule, future_kv, future_operator = self._search(
                block.survivors,
                block.next_iteration,
                block.o_proj_pim_fraction,
                block.last_operator_switch_iteration,
                allow_gpu_kv=allow_gpu_kv,
                allow_operator=allow_operator,
                memo=memo,
                work=work,
            )
            candidate = (
                block.total_us + future_us,
                ((iteration, action), *future_schedule),
                block.kv_switches + future_kv,
                block.operator_switches + future_operator,
            )
            if candidate[0] < best[0] - 1e-9:
                best = candidate
        if best[0] == inf:
            raise RuntimeError("joint exact found no capacity-feasible schedule")
        memo[signature] = best
        return best

    def run_exact(
        self,
        requests: Sequence[RequestState],
        *,
        allow_gpu_kv: bool = True,
        allow_operator: bool = True,
    ) -> JointRunResult:
        initial = deepcopy(list(requests))
        if not self.cost_model.views_fit(views_for_requests(initial)):
            raise RuntimeError("initial prompt plus weight replica exceeds capacity")
        work = [0]
        total_us, schedule, kv_switches, operator_switches = self._search(
            initial,
            0,
            0.0,
            -10**18,
            allow_gpu_kv=allow_gpu_kv,
            allow_operator=allow_operator,
            memo={},
            work=work,
        )
        return JointRunResult(
            "joint_exact",
            total_us,
            schedule,
            work[0],
            kv_switches,
            operator_switches,
        )

    def _simulate_fixed(
        self,
        requests: Sequence[RequestState],
        action: JointAction,
    ) -> float | None:
        projected = deepcopy(list(requests))
        targets = action.target_mapping()
        for request in projected:
            # This is a creation-time static choice for all future KV, not a
            # migration of the prompt segment and not a charged runtime switch.
            request.tail_location = targets[request.request_id]
        total_us = 0.0
        while projected:
            views = views_for_requests(projected)
            if not self.cost_model.views_fit(views):
                return None
            total_us += self.cost_model.iteration_cost(
                views, action.o_proj_pim_fraction
            ).total_us
            for request in projected:
                request.append_generated_token()
            if not self.cost_model.views_fit(views_for_requests(projected)):
                return None
            projected = [
                request for request in projected if not request.is_finished
            ]
        return total_us

    def run_best_static(
        self,
        requests: Sequence[RequestState],
        *,
        allow_gpu_kv: bool = True,
        allow_operator: bool = True,
    ) -> JointRunResult:
        fractions = self.fractions if allow_operator else (0.0,)
        # At creation time all initially PIM requests are exchangeable only if
        # their lengths and replica affinities match. Reusing the grouped exact
        # generator avoids redundant permutations while retaining each count.
        assignments = self._kv_target_assignments(
            requests,
            0,
            allow_gpu_kv=allow_gpu_kv,
            coordinate_only=False,
            enforce_switch_limit=False,
            enforce_residence=False,
        )
        best_us = inf
        best_action: JointAction | None = None
        evaluations = 0
        # The exact assignment space is exponential in batch size; the budget
        # parameter the scheduler already carries bounds it, returning the best
        # static action found within budget.  Small batches stay exhaustive
        # (the default budget is far above 2^batch x fractions for the trusted
        # grid); callers that need the full oracle raise the budget.
        for assignment in assignments:
            for fraction in fractions:
                if evaluations >= self.max_cost_evaluations:
                    break
                action = JointAction.from_mapping(assignment, fraction)
                cost = self._simulate_fixed(requests, action)
                evaluations += 1
                if cost is not None and cost < best_us:
                    best_us = cost
                    best_action = action
        if best_action is None:
            raise RuntimeError("joint static search found no feasible action")
        return JointRunResult(
            "joint_best_static",
            best_us,
            ((0, best_action),),
            evaluations,
            0,
            0,
        )

    def _hold_to_completion_estimate(
        self,
        requests: Sequence[RequestState],
        o_proj_pim_fraction: float,
        *,
        samples: int,
    ) -> tuple[float, int] | None:
        """Estimate remaining cost when the current action is held.

        Output lengths are assumed to be known exactly.  Request-completion
        boundaries are always evaluated, while long intervals use at most
        ``samples`` evenly spaced context samples.  This keeps the online
        heuristic bounded without pretending to be the exact oracle.
        """

        if samples <= 0:
            raise ValueError("greedy terminal samples must be positive")
        if not requests:
            return 0.0, 0

        max_remaining = max(request.remaining_tokens for request in requests)
        stride = max(1, ceil(max_remaining / samples))
        boundaries = {0, max_remaining}
        boundaries.update(
            request.remaining_tokens for request in requests
        )
        boundaries.update(range(stride, max_remaining, stride))
        ordered = sorted(boundaries)
        total_us = 0.0
        evaluations = 0

        for start, end in zip(ordered, ordered[1:]):
            active = [
                request
                for request in requests
                if request.remaining_tokens > start
            ]
            if not active:
                continue
            midpoint = start + (end - start - 1) // 2
            views = [
                make_request_view(request, projected_tokens=midpoint)
                for request in active
            ]
            if not self.cost_model.views_fit(views):
                return None
            total_us += self.cost_model.iteration_cost(
                views, o_proj_pim_fraction
            ).total_us * (end - start)
            evaluations += 1

            # The simulator checks capacity after appending each token and
            # before dropping requests that just finished, so check that same
            # peak at every completion/sample boundary.
            peak_views = [
                make_request_view(request, projected_tokens=end)
                for request in requests
                if request.remaining_tokens >= end
            ]
            if peak_views and not self.cost_model.views_fit(peak_views):
                return None

        return total_us, evaluations

    def run_greedy(
        self,
        requests: Sequence[RequestState],
        *,
        allow_gpu_kv: bool = True,
        allow_operator: bool = True,
        relative_improvement_threshold: float = 0.001,
        terminal_samples: int = 8,
    ) -> JointRunResult:
        """Run a bounded, exact-length-aware coordinate heuristic.

        Each decision scores one simulated decision block plus a sampled
        estimate of the remaining lifetime if that action is held.  Replanning
        still happens every block, but the look-ahead prevents a locally useful
        KV split that cannot amortize itself over the request's known lifetime.
        """

        projected = deepcopy(list(requests))
        if not self.cost_model.views_fit(views_for_requests(projected)):
            raise RuntimeError("initial prompt plus weight replica exceeds capacity")
        iteration = 0
        fraction = 0.0
        last_operator_switch = -10**18
        total_us = 0.0
        evaluations = 0
        kv_switches = 0
        operator_switches = 0
        schedule: list[tuple[int, JointAction]] = []

        while projected:
            actions = self._actions(
                projected,
                iteration,
                fraction,
                last_operator_switch,
                allow_gpu_kv=allow_gpu_kv,
                allow_operator=allow_operator,
                coordinate_only=True,
            )
            evaluated: list[
                tuple[float, int, int, JointAction, _BlockResult]
            ] = []
            stay_result: _BlockResult | None = None
            stay_score: float | None = None
            for action in actions:
                block_work = [0]
                block = self._simulate_block(
                    projected,
                    iteration,
                    fraction,
                    last_operator_switch,
                    action,
                    block_work,
                )
                evaluations += block_work[0]
                if block is None:
                    continue
                terminal = self._hold_to_completion_estimate(
                    block.survivors,
                    block.o_proj_pim_fraction,
                    samples=terminal_samples,
                )
                if terminal is None:
                    continue
                terminal_us, terminal_evaluations = terminal
                evaluations += terminal_evaluations
                score = block.total_us + terminal_us
                changes = block.kv_switches + block.operator_switches
                evaluated.append(
                    (score, changes, block.operator_switches, action, block)
                )
                if changes == 0:
                    stay_result = block
                    stay_score = score
            if not evaluated:
                raise RuntimeError("joint greedy found no capacity-feasible action")
            evaluated.sort(key=lambda item: (item[0], item[1], item[2]))
            selected_score, _, _, action, selected = evaluated[0]
            if stay_result is not None and selected is not stay_result:
                if stay_score is None:
                    raise AssertionError("stay score missing for stay action")
                relative_gain = (
                    stay_score - selected_score
                ) / max(stay_score, 1e-12)
                if relative_gain < relative_improvement_threshold:
                    selected = stay_result
                    action = JointAction.from_mapping(
                        {
                            request.request_id: request.tail_location
                            for request in projected
                        },
                        fraction,
                    )
            schedule.append((iteration, action))
            total_us += selected.total_us
            kv_switches += selected.kv_switches
            operator_switches += selected.operator_switches
            projected = selected.survivors
            iteration = selected.next_iteration
            fraction = selected.o_proj_pim_fraction
            last_operator_switch = selected.last_operator_switch_iteration

        return JointRunResult(
            "joint_lifetime_greedy",
            total_us,
            tuple(schedule),
            evaluations,
            kv_switches,
            operator_switches,
        )

    def replay_sampled(
        self,
        requests: Sequence[RequestState],
        schedule: Sequence[tuple[int, JointAction]],
        *,
        evaluation_stride_tokens: int = 64,
    ) -> JointRunResult:
        """Replay a fixed schedule with midpoint-sampled iteration costs.

        This is intended for expensive dependency-event timing models. State,
        completion, immutable segment, and capacity transitions remain exact;
        only the smoothly changing cost inside each stride-sized token interval
        is approximated by its midpoint value.
        """

        if evaluation_stride_tokens <= 0:
            raise ValueError("evaluation_stride_tokens must be positive")
        if not schedule or schedule[0][0] != 0:
            raise ValueError("a replay schedule must start at iteration 0")
        starts = [iteration for iteration, _ in schedule]
        if starts != sorted(set(starts)):
            raise ValueError("replay schedule iterations must be unique and sorted")

        projected = deepcopy(list(requests))
        if not self.cost_model.views_fit(views_for_requests(projected)):
            raise RuntimeError("initial prompt plus weight replica exceeds capacity")
        total_us = 0.0
        evaluations = 0
        kv_switches = 0
        operator_switches = 0
        current_fraction = 0.0
        iteration = 0

        for action_index, (action_iteration, action) in enumerate(schedule):
            if action_iteration != iteration:
                raise ValueError(
                    f"schedule jumps from iteration {iteration} to "
                    f"{action_iteration}"
                )
            targets = action.target_mapping()
            for request in projected:
                try:
                    target = targets[request.request_id]
                except KeyError as error:
                    raise ValueError(
                        f"schedule action at {iteration} omits active request "
                        f"{request.request_id}"
                    ) from error
                if iteration == 0:
                    request.tail_location = target
                    request.last_switch_iteration = 0
                elif request.switch_tail(target, iteration):
                    kv_switches += 1
            if action.o_proj_pim_fraction != current_fraction:
                operator_switches += 1
            current_fraction = action.o_proj_pim_fraction

            next_action_iteration = (
                schedule[action_index + 1][0]
                if action_index + 1 < len(schedule)
                else inf
            )
            while projected and iteration < next_action_iteration:
                until_completion = min(
                    request.remaining_tokens for request in projected
                )
                width = min(
                    evaluation_stride_tokens,
                    until_completion,
                    int(next_action_iteration - iteration)
                    if next_action_iteration != inf
                    else evaluation_stride_tokens,
                )
                if width <= 0:
                    raise AssertionError("sampled replay made no progress")
                midpoint = (width - 1) // 2
                midpoint_views = [
                    make_request_view(request, projected_tokens=midpoint)
                    for request in projected
                ]
                if not self.cost_model.views_fit(midpoint_views):
                    raise RuntimeError("sampled replay exceeds capacity")
                total_us += self.cost_model.iteration_cost(
                    midpoint_views, current_fraction
                ).total_us * width
                evaluations += 1

                endpoint_views = [
                    make_request_view(request, projected_tokens=width)
                    for request in projected
                ]
                if not self.cost_model.views_fit(endpoint_views):
                    raise RuntimeError("sampled replay endpoint exceeds capacity")
                for _ in range(width):
                    for request in projected:
                        request.append_generated_token()
                projected = [
                    request for request in projected if not request.is_finished
                ]
                iteration += width

        if projected:
            raise ValueError("replay schedule ends before all requests finish")
        return JointRunResult(
            "joint_sampled_replay",
            total_us,
            tuple(schedule),
            evaluations,
            kv_switches,
            operator_switches,
        )


def format_joint_schedule(
    schedule: Sequence[tuple[int, JointAction]],
) -> str:
    """Compact human-readable schedule for CSV output."""

    entries: list[str] = []
    previous: tuple[float, tuple[str, ...]] | None = None
    for iteration, action in schedule:
        gpu_ids = tuple(
            request_id
            for request_id, target in action.tail_targets
            if target is Location.GPU
        )
        signature = (action.o_proj_pim_fraction, gpu_ids)
        if signature == previous:
            continue
        gpu_label = "+".join(gpu_ids) if gpu_ids else "-"
        entries.append(
            f"{iteration}:f={action.o_proj_pim_fraction:g},gpu={gpu_label}"
        )
        previous = signature
    return "|".join(entries)


def parse_joint_schedule(
    value: str,
    request_ids: Sequence[str],
) -> tuple[tuple[int, JointAction], ...]:
    """Parse :func:`format_joint_schedule` output for experiment replay."""

    entries: list[tuple[int, JointAction]] = []
    for raw_entry in value.split("|"):
        raw_entry = raw_entry.strip()
        if not raw_entry:
            continue
        try:
            iteration_text, fields = raw_entry.split(":", 1)
            fraction_text, gpu_text = fields.split(",gpu=", 1)
        except ValueError as error:
            raise ValueError(f"invalid joint schedule entry: {raw_entry!r}") from error
        if not fraction_text.startswith("f="):
            raise ValueError(f"invalid joint fraction: {fraction_text!r}")
        gpu_ids = set() if gpu_text == "-" else set(gpu_text.split("+"))
        unknown = gpu_ids - set(request_ids)
        if unknown:
            raise ValueError(f"schedule contains unknown requests: {sorted(unknown)}")
        targets = {
            request_id: (
                Location.GPU if request_id in gpu_ids else Location.PIM
            )
            for request_id in request_ids
        }
        entries.append(
            (
                int(iteration_text),
                JointAction.from_mapping(
                    targets, float(fraction_text.removeprefix("f="))
                ),
            )
        )
    if not entries:
        raise ValueError("joint schedule is empty")
    return tuple(entries)


def assert_exact_not_slower(
    exact: JointRunResult, comparison: JointRunResult
) -> None:
    """Small-oracle invariant used by tests and experiment wrappers."""

    if exact.total_us > comparison.total_us and not isclose(
        exact.total_us, comparison.total_us, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise AssertionError(
            f"exact cost {exact.total_us} exceeds {comparison.mode} "
            f"cost {comparison.total_us}"
        )
