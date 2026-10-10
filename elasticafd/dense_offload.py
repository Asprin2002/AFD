"""Conditional dense-offload pilot; not a claim of CHIME dense/GELU support.

The existing continuous-batching engine is reused. Only the operator DAG and
the online action selector are extended. All executions are token-exact;
two-point lookahead is used ONLY for choosing actions, never for execution.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from math import ceil
from typing import Sequence

from .capacity import views_fit_capacity
from .config import PolicyConfig, ProjectConfig
from .cost import ChimeCostModel, IterationCost, MatrixLayerCost
from .joint import JointAFEventModel
from .pipeline import PipelineTask
from .policy import PlacementPolicy, SwitchDecision
from .simulator import Simulator
from .specs import GiB, ModelSpec, SystemSpec
from .state import Location, RequestState, RequestView, make_request_view


@dataclass(frozen=True, order=True)
class DensePlan:
    o_proj: float = 0.0
    ffn: float = 0.0
    # QKV is initially opened only as a whole fused operator. Keeping a
    # fraction-valued field makes the plan representation uniform without
    # claiming a calibrated head/row-sharded PIM QKV kernel.
    qkv: float = 0.0


def plan_menu(mode: str) -> tuple[DensePlan, ...]:
    """A bounded menu; joint contains every ablation and 2^3 whole mapping."""
    plans = {DensePlan()}
    if mode in ("oproj", "f_all", "joint"):
        plans.update(DensePlan(f, 0) for f in (0.125, 0.25, 0.5, 0.75, 1))
    if mode in ("ffn", "f_all", "joint"):
        plans.update(DensePlan(0, f) for f in (0.03125, 0.0625, 0.125, 0.25, 0.5))
    if mode in ("joint", "f_all"):
        plans.update(DensePlan(o, f) for o, f in ((0.5, 0.0625), (1, 0.125)))
    if mode in ("qkv", "joint", "f_all"):
        plans.add(DensePlan(qkv=1.0))
    if mode in ("joint", "f_all"):
        plans.update(
            DensePlan(o_proj=o, ffn=f, qkv=q)
            for o in (0.0, 1.0)
            for f in (0.0, 1.0)
            for q in (0.0, 1.0)
        )
        plans.update(
            (
                DensePlan(0.5, 0.0625, 1.0),
                DensePlan(1.0, 0.125, 1.0),
            )
        )
    if mode not in (
        "control",
        "attention",
        "qkv",
        "oproj",
        "ffn",
        "f_all",
        "joint",
    ):
        raise ValueError(f"unknown dense pilot mode: {mode}")
    return tuple(sorted(plans))


class FFNEventModel(JointAFEventModel):
    """Conditional QKV/FFN extensions to the split-Attention/O_proj graph.

    QKV is moved only as a whole fused operator. Its input activation is sent
    to PIM once per layer; Q and any GPU-target new K/V are returned only when
    the GPU Attention branch needs them. W1 columns and the matching W2 rows
    stay together, GELU is local, and the GPU reduces FFN partials.
    """

    def __init__(self, model: ModelSpec, system: SystemSpec, *, ffn_fraction: float,
                 max_o_proj_fraction: float, allow_conditional_ffn: bool,
                 qkv_fraction: float = 0.0,
                 dense_bandwidth_scale: float = 1.0,
                 subbatch_progression: str = "strict_ordered",
                 subbatch_scheduler: str = "lpt") -> None:
        if dense_bandwidth_scale <= 0:
            raise ValueError("dense bandwidth scale must be positive")
        if (ffn_fraction or qkv_fraction) and not allow_conditional_ffn:
            raise ValueError("dense offload requires explicit conditional capability")
        if model.ffn_matrices != 2 or model.dtype_bytes != 2 or model.data_parallel != 1:
            raise ValueError("pilot supports FP16 two-matrix FFN and DP=1 only")
        if not 0 <= ffn_fraction <= 1:
            raise ValueError("ffn_fraction must be in [0, 1]")
        if qkv_fraction not in (0.0, 1.0):
            raise ValueError("QKV is supported only as a whole fused operator")
        if subbatch_progression == "device_contiguous":
            if ffn_fraction and max_o_proj_fraction < 1.0:
                raise ValueError("PIM FFN requires full PIM O_proj in this pipeline")
        neurons = model.ffn_hidden_size * ffn_fraction
        if neurons != int(neurons):
            raise ValueError("FFN fraction must select an integer number of neurons")
        super().__init__(model, system, max_o_proj_pim_fraction=max_o_proj_fraction,
                         split_merge_owner=(
                             "pim" if subbatch_progression == "device_contiguous"
                             else "gpu"
                         ),
                         pim_merge_cost_scale=(
                             1.0 if subbatch_progression == "device_contiguous"
                             else 0.0
                         ),
                         subbatch_progression=subbatch_progression,
                         subbatch_scheduler=subbatch_scheduler)
        self.ffn_fraction = ffn_fraction
        self.qkv_fraction = qkv_fraction
        self.dense_bandwidth_scale = dense_bandwidth_scale

    def _pim_matrix_layer_cost(self, operator: str, batch_size: int) -> MatrixLayerCost:
        full = self.base.pim_matrix_layer_cost(operator, batch_size)
        memory_us = full.memory_us / self.dense_bandwidth_scale
        return MatrixLayerCost(
            max(memory_us, full.compute_us), memory_us,
            full.compute_us, full.weight_bytes,
        )

    def _pim_o_proj_us(self, batch_size: int, fraction: float) -> float:
        if batch_size <= 0 or fraction <= 0:
            return 0.0
        full = self._pim_matrix_layer_cost("o_proj", batch_size)
        return max(full.memory_us * fraction, full.compute_us * fraction)

    def _transform_tasks(self, tasks, partitions, o_proj_pim_fraction):
        f = self.ffn_fraction
        if (
            self.subbatch_progression == "device_contiguous"
            and self.qkv_fraction
            and any(
                view.gpu_tokens > 0 or view.tail_location is Location.GPU
                for part in partitions for view in part
            )
        ):
            raise ValueError(
                "PIM QKV in the two-cut pipeline requires all Attention "
                "and new KV to remain on PIM"
            )
        local_ffn = (
            self.subbatch_progression == "device_contiguous" and f > 0
            and o_proj_pim_fraction == 1.0
        )
        if self.subbatch_progression == "device_contiguous" and f and not local_ffn:
            raise ValueError("PIM FFN requires the complete O_proj on PIM")
        result = []
        pim, gpu, link, model = self.system.pim, self.system.gpu, self.system.link, self.model
        for task in tasks:
            if task.kind != "gpu_ffn" or not f:
                result.append(task)
                continue
            b = len(partitions[task.subbatch])
            ranksets = tuple(f"pim:{(task.layer + i) % pim.ranksets}"
                            for i in range(pim.compute_ranksets_per_layer))
            comm_ranks = ranksets if pim.rankset_comm_overlap else tuple(
                f"pim:{i}" for i in range(pim.ranksets))
            prefix = task.task_id

            def add(suffix, duration, resources, predecessors, kind, priority=1):
                task_id = prefix + suffix
                result.append(PipelineTask(task_id, duration, resources, predecessors,
                                           kind, task.layer, task.subbatch, priority))
                return task_id

            h2d_bytes = b * model.hidden_size * model.dtype_bytes
            d2h_bytes = b * model.hidden_size * 4  # NOT multiplied by f
            if local_ffn:
                h2d = f"l{task.layer}:s{task.subbatch}:pim_o_proj"
            else:
                h2d = add(":h2d", self._transfer_us(h2d_bytes, "h2d"),
                          (("link:h2d" if link.full_duplex else "link"), *comm_ranks),
                          task.predecessors, "ffn_h2d", 0)
            full_pim = self._pim_matrix_layer_cost("ffn", b)
            matrix_us = max(full_pim.memory_us, full_pim.compute_us) * f / 2
            up = add(":up", matrix_us, ranksets, (h2d,), "pim_ffn_up")
            elements = b * model.ffn_hidden_size * f
            # Conditional vector unit: 16 equivalent operations per GELU and
            # one read + write. This uncalibrated assumption is recorded.
            pim_flops = (pim.layer_bandwidth_bytes_s * pim.dense_mac_flops_per_byte
                         * pim.dense_compute_efficiency)
            gelu_us = max(elements * 16 / pim_flops,
                          elements * 2 * model.dtype_bytes / pim.layer_bandwidth_bytes_s) * 1e6
            act = add(":gelu", gelu_us, ranksets, (up,), "pim_ffn_gelu")
            down = add(":down", matrix_us, ranksets, (act,), "pim_ffn_down")
            output = add(":d2h", self._transfer_us(d2h_bytes, "d2h"),
                         (("link:d2h" if link.full_duplex else "link"), *comm_ranks),
                         (down,), "ffn_d2h", 0)
            predecessors = [output]
            if f < 1:
                full_gpu = self.base.gpu_matrix_layer_cost("ffn", b)
                launch = full_gpu.total_us - max(full_gpu.memory_us, full_gpu.compute_us)
                gpu_us = max(full_gpu.memory_us, full_gpu.compute_us) * (1 - f) + launch
                local = add(":gpu", gpu_us, ("gpu",), task.predecessors, "gpu_ffn", 4)
                predecessors.append(local)
            # FP32 PIM output + GPU partial -> FP16 continuation. The full-f
            # case only converts the returned vector. Launch is not free.
            bytes_per_element = 10 if f < 1 else 6
            reduce_us = max(b * model.hidden_size * bytes_per_element /
                            (gpu.bandwidth_bytes_s * gpu.dense_hbm_efficiency),
                            b * model.hidden_size / (gpu.flops_s * gpu.compute_efficiency)) * 1e6
            reduce_us += gpu.merge_launch_us
            # Preserve the old completion ID so next-layer QKV waits on both
            # FFN branches, the return DMA and the reduction.
            result.append(replace(task, duration_us=reduce_us,
                                  predecessors=tuple(predecessors), kind="ffn_reduce"))
        if local_ffn and f == 1.0:
            result = [
                replace(task, duration_us=0.0, resources=())
                if task.kind == "o_proj_d2h" else task
                for task in result
            ]
        if not self.qkv_fraction:
            return result

        # Rewrite the inherited GPU QKV nodes in a second pass. Keeping each
        # original qkv task_id as PIM-compute completion preserves the existing
        # Attention dependency names. Device-crossing consumers are redirected
        # to an explicit Q/new-KV return transfer below.
        rewritten: list[PipelineTask] = []
        qkv_output_ids: dict[tuple[int, int], str | None] = {}
        for task in result:
            if task.kind != "gpu_qkv":
                continue
            batch_views = partitions[task.subbatch]
            batch_size = len(batch_views)
            ranksets = tuple(
                f"pim:{(task.layer + i) % pim.ranksets}"
                for i in range(pim.compute_ranksets_per_layer)
            )
            comm_ranks = ranksets if pim.rankset_comm_overlap else tuple(
                f"pim:{i}" for i in range(pim.ranksets)
            )
            h2d_resource = "link:h2d" if link.full_duplex else "link"
            d2h_resource = "link:d2h" if link.full_duplex else "link"
            input_id = task.task_id + ":input_h2d"
            input_bytes = batch_size * model.output_bytes_per_token_per_layer
            rewritten.append(
                PipelineTask(
                    input_id,
                    self._transfer_us(input_bytes, "h2d"),
                    (h2d_resource, *comm_ranks),
                    task.predecessors,
                    "qkv_input_h2d",
                    task.layer,
                    task.subbatch,
                    0,
                )
            )
            rewritten.append(
                replace(
                    task,
                    duration_us=self._pim_matrix_layer_cost(
                        "qkv", batch_size
                    ).total_us,
                    resources=ranksets,
                    predecessors=(input_id,),
                    kind="pim_qkv",
                    priority=1,
                )
            )

            output_bytes = (
                sum(view.gpu_tokens > 0 for view in batch_views)
                * model.q_bytes_per_token_per_layer
                + sum(view.tail_location is Location.GPU for view in batch_views)
                * model.new_kv_bytes_per_token_per_layer
            )
            output_id: str | None = None
            if output_bytes:
                output_id = task.task_id + ":output_d2h"
                rewritten.append(
                    PipelineTask(
                        output_id,
                        self._transfer_us(output_bytes, "d2h"),
                        (d2h_resource, *comm_ranks),
                        (task.task_id,),
                        "qkv_output_d2h",
                        task.layer,
                        task.subbatch,
                        0,
                    )
                )
            qkv_output_ids[(task.layer, task.subbatch)] = output_id

        for task in result:
            if task.kind == "gpu_qkv":
                continue
            key = (task.layer, task.subbatch)
            qkv_id = f"l{task.layer}:s{task.subbatch}:qkv"
            if task.kind == "pim_attention_h2d":
                rewritten.append(
                    replace(
                        task,
                        duration_us=0.0,
                        resources=(),
                        predecessors=(qkv_id,),
                        kind="pim_attention_input_local",
                    )
                )
                continue
            if task.kind == "gpu_attention" and qkv_id in task.predecessors:
                output_id = qkv_output_ids.get(key)
                if output_id is None:
                    raise AssertionError("GPU Attention needs a QKV D2H edge")
                rewritten.append(
                    replace(
                        task,
                        predecessors=tuple(
                            output_id if predecessor == qkv_id else predecessor
                            for predecessor in task.predecessors
                        ),
                    )
                )
                continue
            rewritten.append(task)
        return rewritten


@dataclass(frozen=True)
class Measurement:
    cost: IterationCost
    h2d_bytes: int
    d2h_bytes: int
    gpu_ffn_us: float
    pim_ffn_us: float
    gpu_oproj_us: float
    pim_oproj_us: float
    operator_link_us: float
    gpu_qkv_us: float = 0.0
    pim_qkv_us: float = 0.0


class DenseOffloadCost(ChimeCostModel):
    """Simulator adapter with an exact-duration, bounded event-cost cache."""

    def __init__(self, config: ProjectConfig, plans: Sequence[DensePlan], *,
                 dense_compute_scale: float = 1,
                 dense_bandwidth_scale: float = 1,
                 allow_conditional_ffn: bool = False,
                 event_cost_cache_max_entries: int = 256):
        if dense_compute_scale <= 0:
            raise ValueError("dense compute scale must be positive")
        if dense_bandwidth_scale <= 0:
            raise ValueError("dense bandwidth scale must be positive")
        if event_cost_cache_max_entries <= 0:
            raise ValueError("event cost cache size must be positive")
        if config.simulation.subbatch_progression == "device_contiguous":
            for plan in plans:
                if plan.ffn and plan.o_proj != 1.0:
                    raise ValueError(
                        "device-contiguous plans require full PIM O_proj "
                        "before any PIM FFN"
                    )
        system = replace(config.system, pim=replace(
            config.system.pim, dense_compute_efficiency=(
                config.system.pim.dense_compute_efficiency * dense_compute_scale)))
        super().__init__(config.model, system, "tile", pipeline_model="rankset_event",
                         subbatch_progression=config.simulation.subbatch_progression,
                         subbatch_scheduler=config.simulation.subbatch_scheduler)
        self.dense_bandwidth_scale = dense_bandwidth_scale
        self.plans = tuple(plans)
        self.plan = DensePlan()
        max_o, max_f = max(p.o_proj for p in plans), max(p.ffn for p in plans)
        max_q = max(p.qkv for p in plans)
        self.weight_bytes = ceil(config.model.layers * (
            config.model.qkv_weight_bytes_per_layer * max_q +
            config.model.output_projection_weight_bytes_per_layer * max_o +
            config.model.ff_weight_bytes_per_layer * max_f))
        b, m = config.simulation.max_batch_size, config.model
        # At most one FFN per subbatch is live. The two subbatches together
        # have <= max_batch_size requests. Reserve full return buffers plus
        # the largest local FFN intermediate; no new unbounded activation pool.
        self.gpu_extra_workspace = (
            (b * m.hidden_size * 4 if max_f else 0)
            + (
                b
                * (m.q_bytes_per_token_per_layer + m.new_kv_bytes_per_token_per_layer)
                if max_q and config.simulation.subbatch_progression != "device_contiguous"
                else 0
            )
        )
        self.pim_extra_workspace = ceil(b * (
            m.ffn_hidden_size * max_f * m.dtype_bytes +
            (m.hidden_size * (m.dtype_bytes + 4) if max_f else 0) +
            (
                m.q_bytes_per_token_per_layer + m.new_kv_bytes_per_token_per_layer
                if max_q
                else 0
            )))
        self.backends = {
            (p.qkv, p.ffn): FFNEventModel(
                m,
                system,
                ffn_fraction=p.ffn,
                qkv_fraction=p.qkv,
                max_o_proj_fraction=max_o,
                allow_conditional_ffn=allow_conditional_ffn,
                dense_bandwidth_scale=dense_bandwidth_scale,
                subbatch_progression=config.simulation.subbatch_progression,
                subbatch_scheduler=config.simulation.subbatch_scheduler,
            )
            for p in plans
        }
        # Preserve the pre-QKV pilot's public lookup used by diagnostics.
        self.backends.update(
            {
                ffn: backend
                for (qkv, ffn), backend in self.backends.items()
                if qkv == 0.0
            }
        )
        self.measurements: OrderedDict[tuple, Measurement] = OrderedDict()
        self.event_cost_cache_max_entries = event_cost_cache_max_entries
        self.record_pipeline_events = config.simulation.record_pipeline_events
        self.cache_hits = self.cache_misses = 0
        self.last_measurement: Measurement | None = None

    @property
    def pim_persistent_weight_bytes(self):
        return self.weight_bytes

    def pim_matrix_layer_cost(self, operator: str, batch_size: int) -> MatrixLayerCost:
        full = super().pim_matrix_layer_cost(operator, batch_size)
        memory_us = full.memory_us / self.dense_bandwidth_scale
        return MatrixLayerCost(
            max(memory_us, full.compute_us), memory_us,
            full.compute_us, full.weight_bytes,
        )

    @property
    def pim_kv_capacity_bytes(self):
        return self.system.pim.capacity_bytes - self.weight_bytes - self.pim_extra_workspace

    @property
    def gpu_kv_capacity_bytes(self):
        return self.system.gpu_kv_capacity_bytes(self.model) - self.gpu_extra_workspace

    @property
    def gpu_replica_kv_capacity_bytes(self):
        return self.gpu_kv_capacity_bytes  # DP=1 enforced by backend

    @property
    def event_cost_cache_stats(self):
        return dict(hits=self.cache_hits, misses=self.cache_misses,
                    entries=len(self.measurements),
                    max_entries=self.event_cost_cache_max_entries)

    @property
    def initial_weight_copy_us(self):
        if not self.weight_bytes:
            return 0.0
        return self.weight_bytes / self.system.link.host_to_pim_bytes_s * 1e6 + (
            self.model.layers * self.system.link.transfer_latency_us)

    def fits(self, views):
        return views_fit_capacity(views, self.model, self.system,
                                 pim_capacity_bytes=self.pim_kv_capacity_bytes,
                                 gpu_replica_capacity_bytes=self.gpu_replica_kv_capacity_bytes)

    def measure(self, views: Sequence[RequestView], plan: DensePlan, *, events=False):
        if plan not in self.plans:
            raise ValueError("plan not covered by resident weight reservation")
        partitions = self.partition_views(views)
        subcosts = tuple(self.subbatch_layer_cost(part) for part in partitions)
        # Key uses exact durations and every graph-relevant count, not context
        # rounding or iteration sampling. Logical KV/byte metadata is refreshed
        # on hits. Distinct states can share an identical physical task DAG.
        signature = tuple((len(part), sum(v.pim_tokens > 0 for v in part),
                           sum(v.gpu_tokens > 0 for v in part), sum(v.is_split for v in part),
                           c.pim.core_us, c.pim.setup_us, c.pim.relayout_us, c.pim.h2d_us,
                           c.gpu_attention.total_us, c.merge_us)
                          for part, c in zip(partitions, subcosts))
        key = (plan, signature)
        if key in self.measurements and not events:
            self.cache_hits += 1
            self.measurements.move_to_end(key)
            m = self.measurements[key]
            return replace(m, cost=replace(m.cost, subbatches=subcosts))
        self.cache_misses += 1
        backend = self.backends[(plan.qkv, plan.ffn)]
        result = backend.iteration_cost(views, plan.o_proj)
        # The historical closed-batch oracle cache is unbounded; this adapter
        # owns a bounded cache and does not accumulate per-token event graphs.
        backend._event_cost_cache.clear()
        ev = result.events
        def total(*kinds):
            return sum(e.duration_us for e in ev if e.kind in kinds)
        dense = total(
            "gpu_qkv",
            "pim_qkv",
            "gpu_o_proj",
            "pim_o_proj",
            "gpu_ffn",
            "pim_ffn_up",
            "pim_ffn_down",
        )
        merge = total("merge", "pim_merge", "ffn_reduce")
        # Read transfers from the actually executed graph, including transfers
        # replaced by PIM O_proj, not from the baseline subbatch metadata.
        h2d = d2h = 0
        for event in ev:
            if "h2d" in event.kind:
                h2d += round(max(0, event.duration_us - self.system.link.transfer_latency_us)
                             * self.system.link.host_to_pim_bytes_s / 1e6)
            elif "d2h" in event.kind:
                d2h += round(max(0, event.duration_us - self.system.link.transfer_latency_us)
                             * self.system.link.pim_to_host_bytes_s / 1e6)
        a = max(result.pim_busy_us, result.link_busy_us)
        cost = IterationCost(result.total_us, result.per_layer_us, result.gpu_busy_us,
                             result.pim_busy_us, result.link_busy_us,
                             result.gpu_attention_us, dense, merge, int(result.gpu_busy_us >= a),
                             int(result.gpu_busy_us < a), subcosts,
                             "dense_offload_event", result.events if events else ())
        measured = Measurement(cost, h2d, d2h, total("gpu_ffn"),
                               total("pim_ffn_up", "pim_ffn_gelu", "pim_ffn_down"),
                               result.gpu_o_proj_us, result.pim_o_proj_us,
                               result.extra_operator_link_us + total(
                                   "ffn_h2d", "ffn_d2h", "qkv_input_h2d", "qkv_output_d2h"
                               ),
                               total("gpu_qkv"), total("pim_qkv"))
        self.measurements[key] = replace(measured, cost=replace(cost, events=()))
        self.measurements.move_to_end(key)
        if len(self.measurements) > self.event_cost_cache_max_entries:
            self.measurements.popitem(last=False)
        return measured

    def iteration_cost(self, views):
        self.last_measurement = self.measure(
            views, self.plan, events=self.record_pipeline_events,
        )
        return self.last_measurement.cost

    def resource_score(self, views, plan):
        """Cheap shortlist ranking ONLY; acceptance/execution use the event DAG."""
        gpu_us = pim_us = link_us = 0.0
        pim_owned_merge = self.subbatch_progression == "device_contiguous"
        for part in self.partition_views(views):
            b = len(part)
            c = self.subbatch_layer_cost(part)
            gpu_us += (self.gpu_matrix_layer_cost("qkv", b).total_us * (1-plan.qkv) +
                       self.gpu_matrix_layer_cost("o_proj", b).total_us * (1-plan.o_proj) +
                       self.gpu_matrix_layer_cost("ffn", b).total_us * (1-plan.ffn) +
                       c.gpu_attention.total_us +
                       (0.0 if pim_owned_merge else c.merge_us))
            pim_us += (c.pim.core_us + c.pim.setup_us +
                       self.pim_matrix_layer_cost("qkv", b).total_us * plan.qkv +
                       self.pim_matrix_layer_cost("o_proj", b).total_us * plan.o_proj +
                       self.pim_matrix_layer_cost("ffn", b).total_us * plan.ffn +
                       (c.merge_us if pim_owned_merge else 0.0))
            link_us += c.pim.link_us
            if pim_owned_merge:
                split_count = sum(view.is_split for view in part)
                link_us += (
                    split_count
                    * (self.model.output_bytes_per_token_per_layer
                       + self.model.num_attention_heads * 2 * 4)
                    / self.system.link.host_to_pim_bytes_s * 1e6
                )
            if plan.qkv:
                link_us += self._rough_qkv_extra_link_us(part)
        return max(gpu_us, pim_us / self.system.pim.ranksets, link_us)

    def _rough_qkv_extra_link_us(self, views: Sequence[RequestView]) -> float:
        """Conservative analytic shortlist charge; event acceptance is exact."""

        model = self.model
        h2d_bytes = len(views) * model.output_bytes_per_token_per_layer
        d2h_bytes = (
            sum(view.gpu_tokens > 0 for view in views)
            * model.q_bytes_per_token_per_layer
            + sum(view.tail_location is Location.GPU for view in views)
            * model.new_kv_bytes_per_token_per_layer
        )
        link = self.system.link
        h2d_us = (
            h2d_bytes / link.host_to_pim_bytes_s * 1e6
            + link.transfer_latency_us
        )
        d2h_us = (
            d2h_bytes / link.pim_to_host_bytes_s * 1e6
            + link.transfer_latency_us
            if d2h_bytes
            else 0.0
        )
        return max(h2d_us, d2h_us) if link.full_duplex else h2d_us + d2h_us


class OnlineDensePolicy(PlacementPolicy):
    """Bounded causal joint search, no future arrivals and no global oracle claim."""

    def __init__(self, config: PolicyConfig, cost_model: DenseOffloadCost, mode: str,
                 *, kv_marginal_gain_threshold: float = 0.0,
                 lifetime_objective: bool = False,
                 exhaustive_operator_menu: bool = False):
        super().__init__(config, cost_model)
        if not 0 <= kv_marginal_gain_threshold < 1:
            raise ValueError("KV marginal threshold must be in [0, 1)")
        self.mode = mode
        self.allows_gpu_kv = mode in ("attention", "joint")
        self.last_operator_switch = -10**18
        self.pending_operator_cost_us = 0.0
        self.operator_switches = 0
        self.decisions = []
        self.kv_marginal_gain_threshold = kv_marginal_gain_threshold
        self.lifetime_objective = lifetime_objective
        self.exhaustive_operator_menu = exhaustive_operator_menu
        self.kv_marginal_guard_vetoes = 0

    def _targets(self, requests, iteration):
        keep = tuple(r.tail_location for r in requests)
        options = {keep}
        if not self.allows_gpu_kv:
            return [keep]
        eligible = [i for i, r in enumerate(requests)
                    if iteration-r.last_switch_iteration >= self.config.min_switch_interval
                    and (r.generated_tokens == 0 or r.segments[-1].tokens >= self.config.min_segment_tokens)]
        ordered = sorted(eligible, key=lambda i: (-requests[i].remaining_tokens, requests[i].request_id))
        for frac in (0, 0.25, 0.5, 1):
            targets = list(keep)
            selected = set(ordered[:ceil(len(ordered)*frac)])
            for i in eligible:
                targets[i] = Location.GPU if i in selected else Location.PIM
            if sum(a != b for a, b in zip(targets, keep)) <= self.config.max_switches_per_decision:
                options.add(tuple(targets))
        return sorted(options, key=lambda x: tuple(v.value for v in x))

    def decide(self, requests, iteration, *, force=False):
        if self.mode == "control" or (not force and iteration % self.config.decision_interval):
            return []
        cost = self.cost_model
        keep = tuple(r.tail_location for r in requests)
        known_lifetime = min(r.remaining_tokens for r in requests)
        horizon = max(
            1,
            known_lifetime
            if self.lifetime_objective
            else min(self.config.planning_horizon_tokens, known_lifetime),
        )
        plans = cost.plans if iteration-self.last_operator_switch >= self.config.min_switch_interval else (cost.plan,)
        candidates = []
        for targets in self._targets(requests, iteration):
            end = [make_request_view(r, projected_tokens=horizon, projected_location=t)
                   for r, t in zip(requests, targets)]
            if not cost.fits(end):
                continue
            start = [make_request_view(r, projected_location=t) for r, t in zip(requests, targets)]
            ranked = sorted(plans, key=lambda p: (cost.resource_score(end, p), p))
            # Keep current/no-offload plus best O-only, FFN-only, mixed plans.
            # These are screening candidates, not an exhaustive runtime oracle.
            selected = set(ranked[:2]) | ({cost.plan, DensePlan()} & set(plans))
            # Every single-F ablation is cheap enough to retain for every KV
            # assignment. This prevents an approximate resource ranking from
            # deleting the useful Attention+one-F combinations seen in the
            # exact B1/B2 validation, while keeping the online menu bounded.
            selected.update(
                plan
                for plan in plans
                if sum(
                    value > 0 for value in (plan.qkv, plan.o_proj, plan.ffn)
                )
                <= 1
            )
            if targets == keep:
                # Exact operator-menu search at unchanged KV. In particular,
                # all single-family heuristic actions are joint candidates.
                selected.update(plans)
            elif self.exhaustive_operator_menu:
                # Formal joint validation enumerates the complete resident
                # operator menu for every future-KV assignment. This is slower
                # than the online shortlist but guarantees that a profitable
                # Attention+F combination cannot be screened out by the
                # optimistic resource score.
                selected.update(plans)
            for family in (
                lambda p: p.qkv > 0 and p.o_proj == 0 and p.ffn == 0,
                lambda p: p.ffn == 0 and p.qkv == 0,
                lambda p: p.o_proj == 0 and p.qkv == 0,
                lambda p: p.qkv == 0 and p.ffn > 0 and p.o_proj > 0,
                lambda p: p.qkv > 0 and (p.ffn > 0 or p.o_proj > 0),
            ):
                family_plans = [p for p in ranked if family(p)]
                if family_plans:
                    selected.add(family_plans[0])
            for plan in sorted(selected):
                before = cost.measure(start, plan).cost.total_us
                after = cost.measure(end, plan).cost.total_us
                switch_count = sum(a != b for a, b in zip(targets, keep)) + (plan != cost.plan)
                objective = (before + after) * horizon / 2 + switch_count * self.config.switch_execution_us
                candidates.append((objective, switch_count, plan, targets, before))
        current = next((c for c in candidates if c[2] == cost.plan and c[3] == keep), None)
        if current is None:
            raise RuntimeError("current plan exceeds lookahead capacity; reduce pilot load")
        best = min(candidates, key=lambda c: (c[0], c[1], c[2], tuple(t.value for t in c[3])))
        if best[3] != keep and self.kv_marginal_gain_threshold:
            # A profitable reversible dense action must not subsidize a nearly
            # indifferent irreversible KV commitment. This paired candidate
            # is always present: unchanged-KV enumerates the full dense menu.
            same_dense = next(c for c in candidates if c[2] == best[2] and c[3] == keep)
            if best[0] >= same_dense[0] * (1-self.kv_marginal_gain_threshold):
                best = same_dense
                self.kv_marginal_guard_vetoes += 1
        if best[0] >= current[0] * (1-self.config.relative_improvement_threshold):
            return []
        _, _, plan, targets, _ = best
        old_plan = cost.plan
        changes = []
        for r, target in zip(requests, targets):
            old = r.tail_location
            if r.switch_tail(target, iteration):
                changes.append(SwitchDecision(r.request_id, old, target, "dense_joint_event",
                                               current[0], best[0]))
        if plan != old_plan:
            self.operator_switches += 1
            self.last_operator_switch = iteration
            self.pending_operator_cost_us += self.config.switch_execution_us
            cost.plan = plan
        self.decisions.append(dict(iteration=iteration, horizon=horizon,
                                   old_o=old_plan.o_proj, old_f=old_plan.ffn,
                                   old_q=old_plan.qkv,
                                   new_o=plan.o_proj, new_f=plan.ffn,
                                   new_q=plan.qkv,
                                   objective_before_us=current[0], objective_after_us=best[0],
                                   kv_switches=len(changes), candidate_count=len(candidates)))
        return changes

    def summary_metadata(self):
        return dict(dense_policy_mode=self.mode, operator_switches=self.operator_switches,
                    dense_policy_decisions=self.decisions, future_arrivals_visible=False,
                    kv_marginal_gain_threshold=self.kv_marginal_gain_threshold,
                    kv_marginal_guard_vetoes=self.kv_marginal_guard_vetoes,
                    lifetime_objective=self.lifetime_objective,
                    exhaustive_operator_menu=self.exhaustive_operator_menu,
                    objective=(
                        "two_endpoint_event_cost_until_next_known_completion"
                        if self.lifetime_objective
                        else "two_endpoint_event_cost_over_bounded_horizon"
                    ),
                    search="resource_shortlist_then_event_acceptance; not global oracle")


class ScriptedDensePolicy(PlacementPolicy):
    """Execute a precomputed closed-workload event-plan without re-searching.

    This policy is intentionally an offline replay mechanism. It makes no
    oracle claim for online arrivals; it exists to validate a sampled global
    plan with token-exact KV growth and the normal simulator accounting.
    """

    allows_gpu_kv = True

    def __init__(
        self,
        config: PolicyConfig,
        cost_model: DenseOffloadCost,
        timeline: Sequence[dict[str, object]],
    ) -> None:
        super().__init__(config, cost_model)
        self.timeline = tuple(dict(entry) for entry in timeline)
        if not self.timeline or int(self.timeline[0]["start"]) != 0:
            raise ValueError("scripted timeline must start at iteration zero")
        previous_end = 0
        for entry in self.timeline:
            start, end = int(entry["start"]), int(entry["end"])
            if start != previous_end or end <= start:
                raise ValueError("scripted timeline must be contiguous and positive")
            plan = self._plan(entry)
            if plan not in cost_model.plans:
                raise ValueError("scripted plan is not covered by resident weights")
            previous_end = end
        self.by_start = {int(entry["start"]): entry for entry in self.timeline}
        self.pending_operator_cost_us = 0.0
        self.operator_switches = 0
        self.decisions: list[dict[str, object]] = []

    @staticmethod
    def _plan(entry: dict[str, object]) -> DensePlan:
        return DensePlan(
            o_proj=float(entry["o_proj"]),
            ffn=float(entry["ffn"]),
            qkv=float(entry["qkv"]),
        )

    def decide(self, requests, iteration, *, force=False):
        entry = self.by_start.get(iteration)
        if entry is None:
            return []
        ordered = sorted(requests, key=lambda request: request.request_id)
        gpu_writers = int(entry["gpu_writers"])
        if not 0 <= gpu_writers <= len(ordered):
            raise ValueError("scripted gpu_writers exceeds active batch")
        selected = {request.request_id for request in ordered[:gpu_writers]}
        horizon = int(entry["end"]) - int(entry["start"])
        projected = [
            make_request_view(
                request,
                projected_tokens=min(horizon, request.remaining_tokens),
                projected_location=(
                    Location.GPU
                    if request.request_id in selected
                    else Location.PIM
                ),
            )
            for request in ordered
        ]
        if not self.cost_model.fits(projected):
            raise RuntimeError("scripted action exceeds KV capacity")

        changes = []
        for request in ordered:
            target = (
                Location.GPU
                if request.request_id in selected
                else Location.PIM
            )
            old = request.tail_location
            if request.switch_tail(target, iteration):
                changes.append(
                    SwitchDecision(
                        request.request_id,
                        old,
                        target,
                        "offline_sampled_event_plan",
                    )
                )
        plan = self._plan(entry)
        old_plan = self.cost_model.plan
        if plan != old_plan:
            self.operator_switches += 1
            self.pending_operator_cost_us += self.config.switch_execution_us
            self.cost_model.plan = plan
        self.decisions.append(
            {
                "iteration": iteration,
                "horizon": horizon,
                "old_q": old_plan.qkv,
                "old_o": old_plan.o_proj,
                "old_f": old_plan.ffn,
                "new_q": plan.qkv,
                "new_o": plan.o_proj,
                "new_f": plan.ffn,
                "gpu_writers": gpu_writers,
                "kv_switches": len(changes),
            }
        )
        return changes

    def summary_metadata(self):
        return {
            "dense_policy_mode": "scripted_offline_event_plan",
            "operator_switches": self.operator_switches,
            "dense_policy_decisions": self.decisions,
            "future_arrivals_visible": True,
            "objective": "offline_precomputed_timeline",
            "search": "precomputed legal static or history-carrying timeline; token-exact execution only",
        }


class DenseOffloadSimulator(Simulator):
    def __init__(self, config: ProjectConfig, *, mode: str, dense_compute_scale=1.0,
                 allow_conditional_ffn=False, kv_marginal_gain_threshold=0.0,
                 lifetime_objective=False, exhaustive_operator_menu=False,
                 scripted_timeline: Sequence[dict[str, object]] | None = None,
                 event_cost_cache_max_entries=256):
        # Fail on semantics this narrow pilot does not implement.
        if config.simulation.admission_scheduler != "fcfs" or config.simulation.allow_gpu_prompt_fallback:
            raise ValueError("dense pilot requires FCFS and PIM prompt placement")
        super().__init__(config)
        menu_mode = "joint" if scripted_timeline is not None else mode
        self.cost_model = DenseOffloadCost(config, plan_menu(menu_mode),
                                           dense_compute_scale=dense_compute_scale,
                                           allow_conditional_ffn=allow_conditional_ffn,
                                           event_cost_cache_max_entries=event_cost_cache_max_entries)
        self.policy_cost_model = self.cost_model
        if scripted_timeline is None:
            self.policy = OnlineDensePolicy(
                config.simulation.policy,
                self.cost_model,
                mode,
                kv_marginal_gain_threshold=kv_marginal_gain_threshold,
                lifetime_objective=lifetime_objective,
                exhaustive_operator_menu=exhaustive_operator_menu,
            )
        else:
            self.policy = ScriptedDensePolicy(
                config.simulation.policy, self.cost_model, scripted_timeline
            )
        self.dense_compute_scale = dense_compute_scale
        self.operator_control_us = 0.0
        self.mechanism_totals = dict(gpu_dense_removed_same_state_us=0.0,
                                     pim_added_same_state_us=0.0,
                                     gpu_qkv_us=0.0, pim_qkv_us=0.0,
                                     gpu_ffn_us=0.0, pim_ffn_us=0.0,
                                     gpu_oproj_us=0.0, pim_oproj_us=0.0,
                                     operator_link_us=0.0)

    def _apply_switch_decisions(self, decisions):
        super()._apply_switch_decisions(decisions)
        overhead = self.policy.pending_operator_cost_us
        self.clock_us += overhead
        self.operator_control_us += overhead
        self.policy.pending_operator_cost_us = 0.0

    def _iteration_row(self, *args, **kwargs):
        row = super()._iteration_row(*args, **kwargs)
        measurement = self.cost_model.last_measurement
        assert measurement is not None
        row.update(o_proj_pim_fraction=self.cost_model.plan.o_proj,
                   ffn_pim_fraction=self.cost_model.plan.ffn,
                   qkv_pim_fraction=self.cost_model.plan.qkv,
                   h2d_bytes=measurement.h2d_bytes, d2h_bytes=measurement.d2h_bytes)
        return row

    def _run_iteration(self, **kwargs):
        # Counterfactual diagnostic at the identical KV/batch state before
        # this step; not a separate workload-level latency estimate.
        before_h2d, before_d2h = self._sum_h2d_bytes, self._sum_d2h_bytes
        completed = super()._run_iteration(**kwargs)
        m = self.cost_model.last_measurement
        assert m is not None
        self._sum_h2d_bytes = before_h2d + m.h2d_bytes
        self._sum_d2h_bytes = before_d2h + m.d2h_bytes
        for name in (
            "gpu_qkv_us",
            "pim_qkv_us",
            "gpu_ffn_us",
            "pim_ffn_us",
            "gpu_oproj_us",
            "pim_oproj_us",
            "operator_link_us",
        ):
            self.mechanism_totals[name] += getattr(m, name)
        # Removing Q/O/FFN GPU work is exact within the selected shard model.
        full_q = full_o = full_f = 0.0
        for part in m.cost.subbatches:
            full_q += self.cost_model.gpu_matrix_layer_cost("qkv", part.size).total_us * self.config.model.layers
            full_o += self.cost_model.gpu_matrix_layer_cost("o_proj", part.size).total_us * self.config.model.layers
            full_f += self.cost_model.gpu_matrix_layer_cost("ffn", part.size).total_us * self.config.model.layers
        self.mechanism_totals["gpu_dense_removed_same_state_us"] += (
            full_q + full_o + full_f
            - m.gpu_qkv_us - m.gpu_oproj_us - m.gpu_ffn_us
        )
        self.mechanism_totals["pim_added_same_state_us"] += (
            m.pim_qkv_us + m.pim_oproj_us + m.pim_ffn_us
        )
        return completed

    def _result(self, status):
        result = super()._result(status)
        result.summary.update(pipeline_model="dense_offload_event",
                              operator_mapping="conditional_dense_offload",
                              dense_compute_scale=self.dense_compute_scale,
                              conditional_pim_dense_kernel=True,
                              conditional_pim_qkv_kernel=True,
                              qkv_granularity="whole_fused_operator",
                              conditional_pim_gelu_ops_per_element=16,
                              initial_weight_copy_s=self.cost_model.initial_weight_copy_us/1e6,
                              weight_copy_in_online_latency=False,
                              gpu_extra_workspace_bytes=self.cost_model.gpu_extra_workspace,
                              pim_extra_workspace_bytes=self.cost_model.pim_extra_workspace,
                              operator_control_time_s=self.operator_control_us/1e6,
                              mechanism_totals=self.mechanism_totals,
                              historical_kv_migration=False)
        return result
