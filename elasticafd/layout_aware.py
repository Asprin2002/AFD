"""Opt-in, layout-aware AF decode model.

This is a mechanism model, not a calibrated PIM dense-kernel prediction.  Its
state is physical: a shared logical KV block has one reference set and each
device copy is charged once.  The legacy simulator does not import this module.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import ceil, exp, expm1, gamma, log, log1p
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter
from typing import Any, Iterable

from .config import ProjectConfig
from .layout_cost_cache import current_cost_cache, reuse_costs
from .layout_forecast import build_forecast_plan, evaluate_forecasts, ForecastPruned
from .partial_softmax import PartialSoftmaxState, merge_partial_softmax
from .specs import GiB
from .workload import WorkloadRequest, build_workload, load_prefix_blocks


GPU = "gpu"
PIM = "pim"
OPS = ("qkv", "attention", "merge", "output_projection", "ffn")


def merge_attention_partials(states: Iterable[PartialSoftmaxState]) -> PartialSoftmaxState:
    """Exact Merge semantics used by split Attention's timing-only operator."""
    return merge_partial_softmax(states)


@dataclass(frozen=True)
class PrefixNode:
    block_id: str
    parent_id: str | None
    tokens: int


class PrefixCatalog:
    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.nodes: dict[str, PrefixNode] = {}
        for row in rows:
            key = str(row["id"])
            if key in self.nodes:
                raise ValueError(f"duplicate prefix block {key}")
            tokens = int(row["tokens"])
            if tokens <= 0:
                raise ValueError("prefix block tokens must be positive")
            parent = row.get("parent_id")
            self.nodes[key] = PrefixNode(key, str(parent) if parent is not None else None, tokens)
        for key in self.nodes:
            self.path(key)

    def path(self, leaf_id: str | None) -> tuple[PrefixNode, ...]:
        if leaf_id is None:
            return ()
        nodes: list[PrefixNode] = []
        seen: set[str] = set()
        key: str | None = leaf_id
        while key is not None:
            if key in seen:
                raise ValueError("prefix catalog has a cycle")
            if key not in self.nodes:
                raise ValueError(f"unknown prefix block {key}")
            seen.add(key)
            node = self.nodes[key]
            nodes.append(node)
            key = node.parent_id
        return tuple(reversed(nodes))


@dataclass(frozen=True)
class PimInterval:
    """One repeated PIM interval around Attention in real operator order.

    Coordinates are relative to qkv(layer)=0, attention=1, merge=2,
    output projection=3, FFN=4.  -1 is the preceding FFN.  Adjacent repeats
    have at least one GPU operator between intervals.
    """

    left: int = 1
    right: int = 1

    def __post_init__(self) -> None:
        if self.left not in (-1, 0, 1) or self.right not in (1, 2, 3, 4):
            raise ValueError("invalid PIM interval boundary")
        if self.right - self.left > 3:
            raise ValueError("PIM interval would consume the whole cycle")

    def contains(self, layer: int, op: str) -> bool:
        # Intervals repeat every five operators. Legal edges can reach the
        # preceding FFN, but never the following layer's QKV.
        position = OPS.index(op)
        return (self.left <= position <= self.right
                or self.left <= position - 5 <= self.right)

    def neighbors(self) -> tuple[PimInterval, ...]:
        values = []
        for left, right in ((self.left - 1, self.right),
                            (self.left + 1, self.right),
                            (self.left, self.right - 1),
                            (self.left, self.right + 1)):
            try:
                values.append(PimInterval(left, right))
            except ValueError:
                pass
        return tuple(values)

    def alternatives(self) -> tuple[PimInterval, ...]:
        """Legal complete intervals, including multi-operator extensions."""
        values = []
        for left in (-1, 0, 1):
            for right in (1, 2, 3, 4):
                try:
                    interval = PimInterval(left, right)
                except ValueError:
                    continue
                if interval != self:
                    values.append(interval)
        return tuple(values)


@dataclass
class PhysicalCopy:
    location: str
    rankset: int | None
    replica: int
    ready_us: float
    layout: str = "kv_token_major"
    preparation_job_id: int | None = None


@dataclass
class KVBlock:
    block_id: str
    layer: int
    tokens: float
    replica: int
    shared: bool
    refs: set[str]
    copies: dict[str, PhysicalCopy]
    owner: str = PIM
    pending_owner: str | None = None
    owner_ready_us: float = 0.0
    cached_at_us: float | None = None
    start_token: int = 0

    @property
    def end_token(self) -> float:
        return self.start_token + self.tokens

    def owner_at(self, now_us: float) -> str:
        if self.pending_owner is not None and now_us >= self.owner_ready_us:
            return self.pending_owner
        return self.owner


@dataclass
class WeightCopy:
    replica: int
    layer: int
    operator: str
    ranksets: tuple[int, ...]
    shard_bytes: tuple[int, ...]
    ready_us: float
    layout: str = "pim_fixed_shard"
    preparation_job_id: int | None = None

    @property
    def bytes(self) -> int:
        return sum(self.shard_bytes)


@dataclass
class LinkEvent:
    start_us: float
    end_us: float
    direction: str
    kind: str
    job_id: int | None = None
    bytes: int = 0
    startup_us: float = 0.0


@dataclass
class PreparationJob:
    job_id: int
    direction: str
    size: int
    earliest_us: float
    conversion_us: float
    ready_us: float = 0.0
    completed_bytes: int = 0
    completed_end_us: float = 0.0


@dataclass
class LinkCalendar:
    events: list[LinkEvent] = field(default_factory=list)
    jobs: dict[int, PreparationJob] = field(default_factory=dict)
    completed_preparation_busy_us: float = 0.0
    next_job_id: int = 0
    _pending_jobs: dict[int, PreparationJob] | None = field(default=None, repr=False, compare=False)
    _known_jobs: int = field(default=0, repr=False, compare=False)
    _pruned_through_us: float = field(default=float("-inf"), repr=False, compare=False)


@dataclass
class _CapacityTotals:
    """Incremental physical allocation counts, scoped to a forecast replay."""

    gpu_kv: list[float]
    pim_kv: list[float]
    pim_weights: list[float]
    gpu_writers: list[int]
    pim_writers: int


@dataclass
class LayoutRequest:
    source: WorkloadRequest
    replica: int
    block_ids: list[list[str]]
    admitted_us: float
    generated: int = 0
    first_token_us: float | None = None
    finished_us: float | None = None
    interval: PimInterval = field(default_factory=PimInterval)
    pending_interval: PimInterval | None = None
    pending_interval_block_ids: tuple[str, ...] = ()
    interval_ready_us: float = 0.0
    future_writer: str = PIM
    switches: int = 0

    def interval_at(self, now_us: float) -> PimInterval:
        if self.pending_interval is not None and now_us >= self.interval_ready_us:
            return self.pending_interval
        return self.interval


@dataclass
class LayoutWorld:
    config: ProjectConfig
    catalog: PrefixCatalog
    blocks: dict[str, KVBlock] = field(default_factory=dict)
    requests: dict[str, LayoutRequest] = field(default_factory=dict)
    weights: dict[tuple[int, int, str], WeightCopy] = field(default_factory=dict)
    clock_us: float = 0.0
    sequence: int = 0
    peak_gpu_bytes: float = 0.0
    peak_pim_bytes: float = 0.0
    peak_pim_kv_bytes: float = 0.0
    peak_gpu_kv_bytes: float = 0.0
    peak_shared_refs: int = 0
    prefix_hits: int = 0
    preparation_bytes: int = 0
    preparation_us: float = 0.0
    link_calendar: LinkCalendar = field(default_factory=LinkCalendar)
    link_busy_us: float = 0.0
    foreground_wait_us: float = 0.0
    replan_index: int = 0
    _capacity_totals: _CapacityTotals | None = field(default=None, repr=False, compare=False)

    @property
    def kv_bytes(self) -> int:
        return self.config.model.kv_bytes_per_token_per_layer

    def rankset(self, layer: int) -> int:
        return layer % self.config.system.pim.ranksets

    def planning_reserve_tokens(self) -> int:
        policy = self.config.simulation.policy
        return max(policy.planning_horizon_tokens, policy.decision_interval,
                   self.config.simulation.layout_growth_reserve_tokens)

    def add_copy(self, block: KVBlock, location: str, ready_us: float,
                 preparation_job_id: int | None = None) -> None:
        if location not in block.copies:
            block.copies[location] = PhysicalCopy(
                location, self.rankset(block.layer) if location == PIM else None,
                block.replica, ready_us,
                preparation_job_id=preparation_job_id)
            self._account_copy(block, block.copies[location], 1)

    def _account_copy(self, block: KVBlock, copy: PhysicalCopy, sign: int) -> None:
        totals = self._capacity_totals
        if totals is None:
            return
        tile = (self.config.system.gpu.kv_block_tokens if copy.location == GPU
                else self.config.system.pim.kv_block_tokens)
        size = sign * ceil(block.tokens / tile) * tile * self.kv_bytes
        if copy.location == GPU:
            totals.gpu_kv[copy.replica] += size
        else:
            assert copy.rankset is not None
            totals.pim_kv[copy.rankset] += size

    def _account_writer(self, request: LayoutRequest, sign: int) -> None:
        totals = self._capacity_totals
        if totals is not None:
            if request.future_writer == GPU:
                totals.gpu_writers[request.replica] += sign
            else:
                totals.pim_writers += sign

    def _remove_block(self, key: str) -> None:
        block = self.blocks.pop(key)
        for copy in block.copies.values():
            self._account_copy(block, copy, -1)

    def _start_capacity_accounting(self) -> None:
        # Real worlds stay directly editable. Forecasts mutate only through
        # the methods below, so their counters can safely replace rescans.
        if self._capacity_totals is not None:
            return
        gpu, pim = self.kv_usage()
        weights = [0.0] * self.config.system.pim.ranksets
        for weight in self.weights.values():
            for rankset, size in zip(weight.ranksets, weight.shard_bytes):
                weights[rankset] += size
        self._capacity_totals = _CapacityTotals(
            gpu, pim, weights, [0] * self.config.model.data_parallel, 0)
        for request in self.requests.values():
            self._account_writer(request, 1)

    def admit(self, source: WorkloadRequest, replica: int) -> None:
        path = self.catalog.path(source.prefix_leaf_id)
        prefix_tokens = sum(node.tokens for node in path)
        if prefix_tokens > source.input_tokens:
            raise ValueError(f"prefix exceeds prompt for {source.request_id}")
        if any(f"prefix:{replica}:0:{node.block_id}" in self.blocks for node in path):
            self.prefix_hits += 1
        rows: list[list[str]] = []
        for layer in range(self.config.model.layers):
            ids: list[str] = []
            start_token = 0
            for node in path:
                key = f"prefix:{replica}:{layer}:{node.block_id}"
                block = self.blocks.get(key)
                if block is None:
                    block = KVBlock(key, layer, node.tokens, replica, True, set(), {},
                                    start_token=start_token)
                    self.add_copy(block, PIM, self.clock_us)
                    self.blocks[key] = block
                block.refs.add(source.request_id)
                self.peak_shared_refs = max(self.peak_shared_refs, len(block.refs))
                block.cached_at_us = None
                ids.append(key)
                start_token += node.tokens
            remaining = source.input_tokens - prefix_tokens
            if remaining:
                key = f"private:{source.request_id}:{layer}:prompt"
                block = KVBlock(key, layer, remaining, replica, False,
                                {source.request_id}, {}, start_token=prefix_tokens)
                self.add_copy(block, PIM, self.clock_us)
                self.blocks[key] = block
                ids.append(key)
            rows.append(ids)
        self.requests[source.request_id] = LayoutRequest(
            source, replica, rows, self.clock_us)
        self._account_writer(self.requests[source.request_id], 1)
        try:
            self.check_capacity(self.planning_reserve_tokens())
        except MemoryError:
            # Retired prefixes are an optional cache, never a reason to reject
            # a decode-ready request while reclaimable capacity exists.
            cached = sorted((block for block in self.blocks.values()
                             if block.shared and not block.refs),
                            key=lambda block: block.cached_at_us or 0)
            for block in cached:
                self._remove_block(block.block_id)
                try:
                    self.check_capacity(self.planning_reserve_tokens())
                    break
                except MemoryError:
                    continue
            else:
                self.check_capacity(self.planning_reserve_tokens())

    def append_token(self, request: LayoutRequest) -> None:
        for layer, ids in enumerate(request.block_ids):
            # A private tail is mutable only on its current writer.  A writer
            # change starts a new physical segment without copying history.
            tail = self.blocks[ids[-1]] if ids else None
            if (tail is not None and not tail.shared and len(tail.copies) == 1
                    and tail.owner == request.future_writer):
                if self._capacity_totals is not None:
                    copy = next(iter(tail.copies.values()))
                    tile = (self.config.system.gpu.kv_block_tokens if copy.location == GPU
                            else self.config.system.pim.kv_block_tokens)
                    delta = (ceil((tail.tokens + 1) / tile)
                             - ceil(tail.tokens / tile)) * tile * self.kv_bytes
                    if copy.location == GPU:
                        self._capacity_totals.gpu_kv[copy.replica] += delta
                    else:
                        self._capacity_totals.pim_kv[copy.rankset] += delta
                tail.tokens += 1
            else:
                self.sequence += 1
                key = f"private:{request.source.request_id}:{layer}:{self.sequence}"
                block = KVBlock(key, layer, 1, request.replica, False,
                                {request.source.request_id}, {}, request.future_writer,
                                start_token=request.source.input_tokens + request.generated)
                self.add_copy(block, request.future_writer, self.clock_us)
                self.blocks[key] = block
                ids.append(key)
        request.generated += 1
        self.check_capacity()

    def retire(self, request_id: str) -> None:
        request = self.requests.pop(request_id)
        self._account_writer(request, -1)
        for ids in request.block_ids:
            for key in ids:
                block = self.blocks[key]
                block.refs.discard(request_id)
                if not block.refs:
                    if (block.shared and block.owner == PIM
                            and block.pending_owner is None
                            and len(block.copies) == 1):
                        block.cached_at_us = self.clock_us
                    else:
                        self._remove_block(key)
        self.evict_cache()

    def evict_cache(self) -> None:
        budget = self.config.simulation.prefix_cache_budget_gib * GiB
        cached = [block for block in self.blocks.values()
                  if block.shared and not block.refs]
        cached_bytes = sum(block.tokens * self.kv_bytes * len(block.copies)
                           for block in cached)
        for block in sorted(cached, key=lambda item: item.cached_at_us or 0):
            if cached_bytes <= budget:
                break
            cached_bytes -= block.tokens * self.kv_bytes * len(block.copies)
            self._remove_block(block.block_id)

    def usage(self) -> tuple[list[float], list[float]]:
        gpu, pim = self.kv_usage()
        if self._capacity_totals is not None:
            return gpu, [value + size for value, size in
                         zip(pim, self._capacity_totals.pim_weights)]
        for weight in self.weights.values():
            for rankset, size in zip(weight.ranksets, weight.shard_bytes):
                pim[rankset] += size
        return gpu, pim

    def kv_usage(self) -> tuple[list[float], list[float]]:
        if self._capacity_totals is not None:
            return (self._capacity_totals.gpu_kv.copy(),
                    self._capacity_totals.pim_kv.copy())
        gpu = [0.0] * self.config.model.data_parallel
        pim = [0.0] * self.config.system.pim.ranksets
        for block in self.blocks.values():
            for copy in block.copies.values():
                block_tokens = (self.config.system.gpu.kv_block_tokens
                                if copy.location == GPU else
                                self.config.system.pim.kv_block_tokens)
                size = ceil(block.tokens / block_tokens) * block_tokens * self.kv_bytes
                if copy.location == GPU:
                    gpu[copy.replica] += size
                else:
                    assert copy.rankset is not None
                    pim[copy.rankset] += size
        return gpu, pim

    def check_capacity(self, growth_tokens: int | None = None) -> None:
        kv_gpu, kv_pim = self.kv_usage()
        gpu, pim = kv_gpu.copy(), kv_pim.copy()
        if self._capacity_totals is not None:
            pim = [value + size for value, size in
                   zip(pim, self._capacity_totals.pim_weights)]
        else:
            for weight in self.weights.values():
                for rankset, size in zip(weight.ranksets, weight.shard_bytes):
                    pim[rankset] += size
        reserve_tokens = (self.config.simulation.layout_growth_reserve_tokens
                          if growth_tokens is None else
                          max(growth_tokens, self.config.simulation.layout_growth_reserve_tokens))
        reserve = reserve_tokens * self.kv_bytes
        temporary = self.config.simulation.layout_temporary_gib * GiB
        gpu = [value + temporary for value in gpu]
        pim = [value + temporary for value in pim]
        if self._capacity_totals is not None:
            for replica, count in enumerate(self._capacity_totals.gpu_writers):
                gpu[replica] += count * reserve * self.config.model.layers
            layers, extra = divmod(self.config.model.layers, len(pim))
            for rankset in range(len(pim)):
                pim[rankset] += (self._capacity_totals.pim_writers * reserve
                                 * (layers + (rankset < extra)))
        else:
            for request in self.requests.values():
                if request.future_writer == GPU:
                    gpu[request.replica] += reserve * self.config.model.layers
                else:
                    for layer in range(self.config.model.layers):
                        pim[self.rankset(layer)] += reserve
        gpu_limit = self.config.system.gpu_replica_kv_capacity_bytes(self.config.model)
        pim_limit = self.config.system.pim.capacity_bytes / self.config.system.pim.ranksets
        if max(gpu, default=0) > gpu_limit + 1e-9:
            raise MemoryError("GPU replica KV capacity exceeded")
        if max(pim, default=0) > pim_limit + 1e-9:
            raise MemoryError("PIM rankset KV/weight capacity exceeded")
        self.peak_gpu_bytes = max(self.peak_gpu_bytes, sum(gpu))
        self.peak_pim_bytes = max(self.peak_pim_bytes, sum(pim))
        self.peak_gpu_kv_bytes = max(self.peak_gpu_kv_bytes, sum(kv_gpu))
        self.peak_pim_kv_bytes = max(self.peak_pim_kv_bytes, sum(kv_pim))


@dataclass(frozen=True)
class StageCost:
    gpu_us: float
    pim_us: float
    gpu_to_pim_us: float
    pim_to_gpu_us: float
    attention_flops: float
    pim_scan_bytes: float
    gpu_scan_bytes: float
    dense_us: float
    merge_us: float
    cross_group_bytes: float = 0.0
    cross_group_us: float = 0.0


_EMPTY_STAGE_COST = StageCost(*(0.0 for _ in range(11)))


@dataclass(frozen=True)
class StepCost:
    total_us: float
    stage0_us: float
    stage1_us: float
    boundary_us: float
    gpu_busy_us: float
    pim_busy_us: float
    dense_us: float
    attention_flops: float
    merge_us: float
    pim_scan_bytes: float
    gpu_scan_bytes: float
    stage0_bottleneck: str
    stage1_bottleneck: str
    cross_group_bytes: float = 0.0
    cross_group_us: float = 0.0
    stage0_gpu_layers: int = 0
    stage0_pim_layers: int = 0
    stage1_gpu_layers: int = 0
    stage1_pim_layers: int = 0
    gpu_critical_us: float = 0.0
    pim_critical_us: float = 0.0
    link_busy_us: float = 0.0
    foreground_wait_us: float = 0.0


def _us(seconds: float) -> float:
    return seconds * 1e6


@lru_cache(maxsize=262144)
def _conditional_survival(generated: int, offset: int,
                          mean: float, shape: float) -> float:
    """P(output survives offset more tokens | it survived generated tokens).

    The total output-length prior is Weibull(mean, shape).  The default
    shape=1 is exponential, so its remaining mean stays at ``mean`` after
    observing a long-running request instead of collapsing to one token.
    """
    if offset <= 0:
        return 1.0
    scale = mean / gamma(1.0 + 1.0 / shape)
    delta = (((generated + offset) / scale) ** shape
             - (generated / scale) ** shape)
    return exp(-max(0.0, delta))


@lru_cache(maxsize=16384)
def _expected_active_tokens(generated: int, horizon: int,
                            mean: float, shape: float) -> float:
    return sum(_conditional_survival(generated, offset, mean, shape)
               for offset in range(horizon))


def _linear_us(config: ProjectConfig, device: str, operator: str,
               m: float, *, layer: int, presence: float = 1.0) -> float:
    cache = current_cost_cache()
    if cache is None:
        return _compute_linear_us(config, device, operator, m, layer=layer, presence=presence)
    key = (cache.config_token(config), device, operator, m, presence)
    result = cache.get("linear", key)
    if result is None:
        result = _compute_linear_us(config, device, operator, m, layer=layer, presence=presence)
        cache.put("linear", key, result)
    return result


def _compute_linear_us(config: ProjectConfig, device: str, operator: str,
                       m: float, *, layer: int, presence: float = 1.0) -> float:
    if m <= 0:
        return 0.0
    model, system, sim = config.model, config.system, config.simulation
    f = model.ffn_hidden_size
    matrices = _dense_matrices(config, operator)
    if device == GPU:
        bandwidth = (system.gpu.bandwidth_bytes_s / model.data_parallel
                     * system.gpu.dense_hbm_efficiency)
        throughput = (system.gpu.flops_s / model.data_parallel
                      * system.gpu.compute_efficiency)
        launch = system.gpu.proj_ff_launch_us
    else:
        return max(_linear_shard_us(config, operator, m, presence))
    total = 0.0
    for k, n in matrices:
        flops = 2 * m * k * n
        physical_bytes = (presence * k * n + m * k + m * n) * model.dtype_bytes
        total += _us(max(flops / max(throughput, 1),
                         physical_bytes / max(bandwidth, 1))) + launch * presence * sim.layout_launch_scale
    if operator == "ffn":
        total += _us(m * f * model.dtype_bytes / max(bandwidth, 1))
    return total


def _dense_matrices(config: ProjectConfig, operator: str) -> tuple[tuple[int, int], ...]:
    model = config.model
    h, kv, f = model.hidden_size, model.kv_width, model.ffn_hidden_size
    if operator == "qkv":
        return ((h, h + 2 * kv),)
    if operator == "output_projection":
        return ((h, h),)
    if operator == "ffn":
        return ((h, f), (h, f), (f, h)) if model.ffn_matrices == 3 else ((h, f), (f, h))
    raise ValueError(operator)


def _shard_widths(width: int, count: int) -> tuple[int, ...]:
    quotient, remainder = divmod(width, count)
    return tuple(quotient + (index < remainder) for index in range(count))


def _dense_groups(world: LayoutWorld, layer: int) -> tuple[int, ...]:
    count = world.config.simulation.layout_dense_ranksets
    if type(count) is not int or not 1 <= count <= world.config.system.pim.ranksets:
        raise ValueError("layout_dense_ranksets must be between 1 and PIM ranksets")
    return _cyclic_dense_groups(world.rankset(layer), count, world.config.system.pim.ranksets)


@lru_cache(maxsize=1024)
def _cyclic_dense_groups(home: int, count: int, ranksets: int) -> tuple[int, ...]:
    # Fixed integer layout arithmetic, independent of all physical state.
    return tuple((home + index) % ranksets for index in range(count))


def _linear_shard_us(config: ProjectConfig, operator: str, m: float,
                     presence: float = 1.0) -> tuple[float, ...]:
    cache = current_cost_cache()
    if cache is None:
        return _compute_linear_shard_us(config, operator, m, presence)
    key = (cache.config_token(config), operator, m, presence)
    result = cache.get("shards", key)
    if result is None:
        result = _compute_linear_shard_us(config, operator, m, presence)
        cache.put("shards", key, result)
    return result


def _compute_linear_shard_us(config: ProjectConfig, operator: str, m: float,
                            presence: float = 1.0) -> tuple[float, ...]:
    count = config.simulation.layout_dense_ranksets
    if m <= 0:
        return (0.0,) * count
    model, system, sim = config.model, config.system, config.simulation
    bandwidth = system.pim.bandwidth_per_rankset_bytes_s * system.pim.bank_hbm_efficiency
    throughput = (bandwidth * system.pim.dense_mac_flops_per_byte
                  * system.pim.dense_compute_efficiency)
    launch = system.pim.request_setup_us * presence * sim.layout_launch_scale
    costs = [0.0] * count
    for k, n in _dense_matrices(config, operator):
        for index, n_shard in enumerate(_shard_widths(n, count)):
            # Every shard reads the complete input; weights and output are N-sharded.
            physical_bytes = (presence * k * n_shard + m * k + m * n_shard) * model.dtype_bytes
            flops = 2 * m * k * n_shard
            costs[index] += (_us(max(flops / max(throughput, 1),
                                     physical_bytes / max(bandwidth, 1))) + launch)
    if operator == "ffn":
        # Local intermediate activation traffic is sharded; the subsequent
        # all-gather needed by the down projection is charged on the rank bus.
        for index, n_shard in enumerate(_shard_widths(model.ffn_hidden_size, count)):
            costs[index] += _us(m * n_shard * model.dtype_bytes / max(bandwidth, 1))
    return tuple(cost * sim.layout_pim_dense_scale for cost in costs)


def _selected_blocks(world: LayoutWorld, request: LayoutRequest, layer: int,
                     offset: int = 0) -> list[KVBlock]:
    blocks = [world.blocks[key] for key in request.block_ids[layer]]
    if world.config.simulation.attention_selector == "dense":
        return blocks
    # Physical migration segments can be much larger than one sparse logical
    # block.  Select by token position, not by the count of physical segments.
    window = (world.config.simulation.recent_blocks
              * world.config.system.pim.kv_block_tokens)
    current_end = request.source.input_tokens + request.generated
    lower = max(0, current_end + offset - window)
    selected = []
    for block in blocks:
        start = max(block.start_token, lower)
        end = min(block.end_token, current_end)
        if end > start:
            selected.append(block if (start == block.start_token
                                      and end == block.end_token) else
                            replace(block, start_token=start, tokens=end - start))
    return selected


def _subbatches(world: LayoutWorld) -> tuple[list[LayoutRequest], list[LayoutRequest]]:
    result: tuple[list[LayoutRequest], list[LayoutRequest]] = ([], [])
    loads = [0, 0]
    ordered = sorted(world.requests.values(),
                     key=lambda req: (-(req.source.input_tokens + req.generated),
                                      req.source.request_id))
    for request in ordered:
        side = 0 if loads[0] <= loads[1] else 1
        result[side].append(request)
        loads[side] += request.source.input_tokens + request.generated
    return result


def _attention_us(world: LayoutWorld, location: str, m: float,
                  length: float, presence: float = 1.0) -> tuple[float, float, float]:
    cache = current_cost_cache()
    if cache is None:
        return _compute_attention_us(world, location, m, length, presence)
    key = (cache.config_token(world.config), location, m, length, presence)
    result = cache.get("attention", key)
    if result is None:
        result = _compute_attention_us(world, location, m, length, presence)
        cache.put("attention", key, result)
    return result


def _compute_attention_us(world: LayoutWorld, location: str, m: float,
                          length: float, presence: float = 1.0) -> tuple[float, float, float]:
    model, system, sim = world.config.model, world.config.system, world.config.simulation
    flops = 4 * model.num_attention_heads * m * length * model.head_dim
    kv_bytes = length * model.kv_bytes_per_token_per_layer
    # One physical KV read per shared segment plus query/output traffic.  It is
    # an optimistic reuse bound and is explicitly exposed to sensitivity scans.
    physical_kv_read = presence * kv_bytes
    bytes_read = physical_kv_read + 2 * m * model.hidden_size * model.dtype_bytes
    if location == GPU:
        bandwidth = (system.gpu.bandwidth_bytes_s / model.data_parallel
                     * system.gpu.attention_hbm_efficiency
                     * sim.layout_attention_bandwidth_scale)
        throughput = (system.gpu.flops_s / model.data_parallel
                      * system.gpu.attention_compute_efficiency)
        softmax = 0.0
    else:
        bandwidth = (system.pim.bandwidth_per_rankset_bytes_s
                     * system.pim.bank_hbm_efficiency
                     * sim.layout_attention_bandwidth_scale)
        throughput = (bandwidth * system.pim.dense_mac_flops_per_byte
                      * system.pim.dense_compute_efficiency)
        softmax_rate = (system.pim.softmax_elements_s
                        if system.pim.softmax_elements_s > 0 else
                        sim.layout_pim_softmax_gelements_s_per_rankset * 1e9)
        softmax = _us(m * model.num_attention_heads * length / softmax_rate)
        softmax *= sim.layout_pim_softmax_scale
    latency = _us(max(bytes_read / max(bandwidth, 1),
                      flops / max(throughput, 1))) + softmax
    return latency, flops, physical_kv_read


def _partition_attention_batch(
        batch: dict[tuple[Any, ...], tuple[float, float, float, int | None]],
        ) -> dict[tuple[Any, ...], tuple[float, float, float, int | None]]:
    """Read each physical KV interval once per subbatch, even for partial overlap.

    Each input range already aggregates requests selecting the exact same slice.
    Sweep their endpoints so an overlapping slice gets the sum of its query
    weights and the probability that at least one query accesses it.
    """
    grouped: dict[tuple[str, str, int | None], list[tuple[float, float, float, float]]] = defaultdict(list)
    result = {}
    for key, (m, length, absence, group) in batch.items():
        if len(key) != 4:  # Forecast-only private KV has no physical block interval.
            result[key] = (m, length, absence, group)
            continue
        side, block_id, start, _ = key
        grouped[(side, block_id, group)].append((start, start + length, m, absence))

    for (side, block_id, group), ranges in grouped.items():
        events: dict[float, list[tuple[int, float, float]]] = defaultdict(list)
        for start, end, m, absence in ranges:
            events[start].append((1, m, absence))
            events[end].append((-1, m, absence))
        points = sorted(events)
        active_m = 0.0
        zero_absences = 0
        log_absence = 0.0
        for index, start in enumerate(points[:-1]):
            for sign, m, absence in events[start]:
                active_m += sign * m
                if absence == 0.0:
                    zero_absences += sign
                else:
                    log_absence += sign * log(absence)
            length = points[index + 1] - start
            if length <= 0 or active_m <= 0:
                continue
            presence = 1.0 if zero_absences else -expm1(log_absence)
            result[(side, block_id, start, length)] = (
                active_m, length, 1.0 - presence, group)
    return result


def _stage_request_inputs(world: LayoutWorld, requests: list[LayoutRequest],
                          at_us: float, offset: int) -> tuple:
    """Inputs shared by all layers of this single, synchronous evaluation.

    Nothing here survives evaluate_step: append, retirement, or a readiness
    change on the next evaluation must build fresh inputs.
    """
    sim = world.config.simulation
    virtual = (min(offset, sim.recent_blocks * world.config.system.pim.kv_block_tokens)
               if sim.attention_selector == "recent_blocks" else offset)
    future_identities = {}
    rows = []
    for request in requests:
        weight = _conditional_survival(request.generated, offset,
                                       sim.layout_survival_mean_tokens,
                                       sim.layout_survival_shape)
        future_identity = (future_identities.setdefault(
            request.source.request_id, len(future_identities)) if virtual else None)
        interval = request.interval_at(at_us)
        rows.append((request.replica, (interval.left, interval.right), request.future_writer,
                     weight, future_identity))
    return virtual, tuple(rows)


def _stage_cache_key(world: LayoutWorld, requests: list[LayoutRequest],
                     at_us: float, offset: int, layer: int, config_token: int,
                     request_inputs: tuple | None = None,
                     layer_input: tuple | None = None) -> tuple:
    """Describe exactly the inputs read by one layer's pure stage cost.

    Local physical identities encode sharing, not request names or equal
    lengths. Preserve request/block order so floating-point accumulation also
    stays identical. Readiness resolves at this call's time, on each layer;
    future calls rebuild the key on their own candidate timeline.
    """
    identities: dict[str, int] = {}
    rows = []
    virtual, inputs = (request_inputs if request_inputs is not None else
                       _stage_request_inputs(world, requests, at_us, offset))
    for request, common in zip(requests, inputs):
        blocks = []
        for block in _selected_blocks(world, request, layer, offset):
            identity = identities.setdefault(block.block_id, len(identities))
            side = block.owner_at(at_us)
            group = block.copies[PIM].rankset if side == PIM else block.replica
            blocks.append((identity, block.start_token, block.tokens, side, group))
        rows.append((common, tuple(blocks)))
    if layer_input is None:
        layer_input = _stage_layer_input(world, layer, config_token)
    return layer_input, virtual, tuple(rows)


def _stage_layer_input(world: LayoutWorld, layer: int, config_token: int) -> tuple:
    last = layer + 1 == world.config.model.layers
    return (config_token, layer == 0, last, world.rankset(layer),
            _dense_groups(world, layer), () if last else _dense_groups(world, layer + 1))


def _stage_cost(world: LayoutWorld, requests: list[LayoutRequest],
                at_us: float, offset: int = 0, *, layer: int | None = None,
                request_inputs: tuple | None = None,
                config_token: int | None = None) -> StageCost:
    cache = current_cost_cache()
    if not requests:
        if cache is not None:
            cache.counts["empty_stage_hits"] += 1
        return _EMPTY_STAGE_COST
    if cache is None or layer is None:
        return _compute_stage_cost(world, requests, at_us, offset, layer=layer)
    if config_token is None:
        config_token = cache.config_token(world.config)
    layouts = cache.layer_layouts.get(config_token)
    if layouts is None:
        layouts = tuple(_stage_layer_input(world, index, config_token)
                        for index in range(world.config.model.layers))
        cache.layer_layouts[config_token] = layouts
    layer_input = layouts[layer] if 0 <= layer < len(layouts) else None
    key = _stage_cache_key(world, requests, at_us, offset, layer, config_token,
                           request_inputs, layer_input)
    result = cache.get("stage", key)
    if result is None:
        result = _compute_stage_cost(world, requests, at_us, offset, layer=layer)
        cache.put("stage", key, result)
    return result


def _compute_stage_cost(world: LayoutWorld, requests: list[LayoutRequest],
                        at_us: float, offset: int = 0, *, layer: int | None = None) -> StageCost:
    if not requests:
        return _EMPTY_STAGE_COST
    if layer is None:
        # A request cannot start the next layer before the current layer has
        # produced its output.  Ranksets overlap only *within* a layer call.
        costs = (_stage_cost(world, requests, at_us, offset, layer=index)
                 for index in range(world.config.model.layers))
        totals = [0.0] * 11
        for cost in costs:
            for index, value in enumerate(vars(cost).values()):
                totals[index] += value
        return StageCost(*totals)
    model, system, sim = world.config.model, world.config.system, world.config.simulation
    gpu_replica_time = [0.0] * model.data_parallel
    # Dependency order applies between operators.  Only independently
    # executable groups within the *same* operator call take a maximum.
    pim_rank_time = {op: [0.0] * system.pim.ranksets for op in OPS}
    pim_reduction_us = pim_sync_us = 0.0
    dense_us = merge_us = attention_flops = pim_scan = gpu_scan = 0.0
    # One packed transfer for requests at the same operator boundary and layer.
    transfers: dict[tuple[int, int, str], float] = defaultdict(float)
    transfer_absence: dict[tuple[int, int, str], float] = defaultdict(lambda: 1.0)
    cross_group: dict[tuple[int, str, int], float] = defaultdict(float)
    cross_group_absence: dict[tuple[int, str, int], float] = defaultdict(lambda: 1.0)
    dense_groups = _dense_groups(world, layer)
    shard_count = len(dense_groups)

    def group_transfer(name: str, replica: int, payload_bytes: float,
                       probability: float, mode: str) -> None:
        if shard_count == 1 or not payload_bytes:
            return
        # The bus is shared by the groups; messages in different operator
        # boundaries serialize. The caller supplies exact bytes for a direct
        # transfer and one full tensor for broadcast or all-gather.
        factor = 1 if mode == "direct" else shard_count - 1
        key = (layer, name, replica)
        cross_group[key] += probability * payload_bytes * factor
        cross_group_absence[key] *= 1 - probability
    for layer in (layer,):
        operator_batch: dict[tuple[str, str, int], tuple[float, float]] = {}
        attention_batch: dict[tuple[Any, ...], tuple[float, float, float, int | None]] = {}
        gpu_attention_absence = [1.0] * model.data_parallel
        pim_group_absence = [1.0] * system.pim.ranksets
        for request in requests:
            weight = _conditional_survival(
                request.generated, offset, sim.layout_survival_mean_tokens,
                sim.layout_survival_shape)
            interval = request.interval_at(at_us)
            selected = _selected_blocks(world, request, layer, offset)
            sides = {block.owner_at(at_us) for block in selected}
            # Predicted future tokens are private and use the selected writer;
            # they do not acquire a fictitious shared-prefix identity.
            virtual_tokens = (min(offset, sim.recent_blocks
                                  * system.pim.kv_block_tokens)
                              if sim.attention_selector == "recent_blocks" else offset)
            virtual_side = request.future_writer if virtual_tokens else None
            if virtual_side is not None:
                sides.add(virtual_side)
            split = GPU in sides and PIM in sides
            if split and interval.contains(layer, "qkv") and interval.contains(layer, "merge"):
                raise ValueError("PIM QKV cannot feed GPU Attention then PIM Merge")
            merge_side = PIM if interval.contains(layer, "merge") else GPU
            if split:
                key = (merge_side, "merge", request.replica)
                m, absence = operator_batch.get(key, (0.0, 1.0))
                operator_batch[key] = (m + weight, absence * (1 - weight))
            for operator in ("qkv", "output_projection", "ffn"):
                side = PIM if interval.contains(layer, operator) else GPU
                key = (side, operator, request.replica)
                m, absence = operator_batch.get(key, (0.0, 1.0))
                operator_batch[key] = (m + weight, absence * (1 - weight))
            if GPU in sides:
                gpu_attention_absence[request.replica] *= 1 - weight
            pim_groups = {block.copies[PIM].rankset for block in selected
                          if block.owner_at(at_us) == PIM}
            if virtual_tokens and request.future_writer == PIM:
                pim_groups.add(world.rankset(layer))
            for group in pim_groups:
                assert group is not None
                pim_group_absence[group] *= 1 - weight
            if len(pim_groups) > 1:
                # The default layout keeps a layer on one rankset. A custom
                # physical placement can span groups and pays a separate
                # (currently uncalibrated) synchronization term.
                pim_sync_us += weight * sim.layout_rankset_sync_us
            for block in selected:
                side = block.owner_at(at_us)
                key = (side, block.block_id, block.start_token, block.tokens)
                group = (block.copies[side].rankset if side == PIM
                         else block.replica)
                old_m, old_length, absence, _ = attention_batch.get(
                    key, (0.0, block.tokens, 1.0, group))
                attention_batch[key] = (old_m + weight, old_length,
                                        absence * (1 - weight), group)
            local_counts = {side: (sum(block.owner_at(at_us) == side
                                       for block in selected)
                                   + (1 if virtual_tokens and request.future_writer == side else 0))
                            for side in (GPU, PIM)}
            for side, count in local_counts.items():
                if count <= 1:
                    continue
                bytes_to_reduce = ((count - 1) * (model.head_dim + 2)
                                   * model.num_attention_heads * model.dtype_bytes)
                bandwidth = (system.gpu.bandwidth_bytes_s / model.data_parallel
                             if side == GPU
                             else system.pim.bandwidth_per_rankset_bytes_s)
                reduction = _us(weight * bytes_to_reduce / bandwidth)
                if side == GPU:
                    gpu_replica_time[request.replica] += reduction
                else:
                    pim_reduction_us += reduction
            if virtual_tokens:
                key = (request.future_writer, f"future:{request.source.request_id}:{layer}")
                attention_batch[key] = (weight, float(virtual_tokens), 1 - weight,
                                        world.rankset(layer) if request.future_writer == PIM
                                        else request.replica)

            # Main-path transitions include cross-layer FFN -> QKV.  Branch
            # statistics share that boundary packet; no extra round trip.
            path = [PIM if interval.contains(layer, "qkv") else GPU,
                    PIM if PIM in sides else GPU,
                    merge_side if split else (PIM if PIM in sides else GPU),
                    PIM if interval.contains(layer, "output_projection") else GPU,
                    PIM if interval.contains(layer, "ffn") else GPU]
            if shard_count > 1:
                hbytes = model.hidden_size * model.dtype_bytes
                # A preceding sharded FFN leaves its output replicated when
                # the next QKV is PIM. Otherwise QKV receives one full input
                # at the home group and broadcasts it to the other shards.
                prior_pim_ffn = (layer > 0 and
                                 interval.contains(layer - 1, "ffn"))
                if path[0] == PIM and not prior_pim_ffn:
                    group_transfer("qkv_input", request.replica, hbytes,
                                   weight, "broadcast")
                if path[0] == PIM:
                    local_width = _shard_widths(
                        model.hidden_size + 2 * model.kv_width, shard_count)[0]
                    gather_width = (max(0, model.hidden_size - local_width)
                                    if PIM in sides else 0)
                    if request.future_writer == PIM:
                        local_kv = max(0, local_width - model.hidden_size)
                        gather_width += max(0, 2 * model.kv_width - local_kv)
                    group_transfer("qkv_to_home", request.replica,
                                   gather_width * model.dtype_bytes, weight, "direct")
                if path[3] == PIM:
                    group_transfer("output_input", request.replica, hbytes,
                                   weight, "broadcast")
                    if path[4] == PIM:
                        # The following FFN needs the full O projection on
                        # every group, so gather rather than send only to home.
                        group_transfer("output_to_ffn", request.replica, hbytes,
                                       weight, "allgather")
                if path[4] == PIM:
                    if path[3] != PIM:
                        group_transfer("ffn_input", request.replica, hbytes,
                                       weight, "broadcast")
                    group_transfer("ffn_middle", request.replica,
                                   model.ffn_hidden_size * model.dtype_bytes,
                                   weight, "allgather")
                    if (layer + 1 < model.layers
                            and interval.contains(layer + 1, "qkv")):
                        group_transfer("ffn_to_next_qkv", request.replica,
                                       hbytes, weight, "allgather")
                        missing = len(set(_dense_groups(world, layer + 1))
                                      - set(dense_groups))
                        group_transfer("ffn_to_next_qkv_extra", request.replica,
                                       missing * hbytes, weight, "direct")
            elif (layer + 1 < model.layers and path[4] == PIM
                  and interval.contains(layer + 1, "qkv")
                  and world.rankset(layer) != world.rankset(layer + 1)):
                # A one-group layout still moves a PIM FFN output when the
                # following layer's QKV resides on another rankset.
                key = (layer, "ffn_to_next_qkv", request.replica)
                cross_group[key] += weight * model.hidden_size * model.dtype_bytes
                cross_group_absence[key] *= 1 - weight
            crossings: list[tuple[int, str]] = []
            packet: dict[tuple[int, int, str], float] = defaultdict(float)
            for position in range(4):
                if path[position] == path[position + 1]:
                    continue
                payload = model.hidden_size * model.dtype_bytes
                if position == 0:
                    payload = model.q_bytes_per_token_per_layer
                elif position == 1:
                    payload = ((model.head_dim + 2) * model.num_attention_heads
                               * model.dtype_bytes)
                    if split and path[0] == PIM and merge_side == GPU:
                        payload += model.q_bytes_per_token_per_layer
                if split and position == 0 and merge_side == PIM:
                    payload += ((model.head_dim + 2) * model.num_attention_heads
                                * model.dtype_bytes)
                direction = "ga" if path[position] == GPU else "ag"
                packet[(layer, position, direction)] += payload
                crossings.append((position, direction))
            qkv_side = path[0]
            if qkv_side != request.future_writer:
                direction = "ga" if qkv_side == GPU else "ag"
                # New KV is produced by QKV and packed into an already
                # required transfer in the same direction when possible.
                boundary_position = next((position for position, side in crossings
                                          if side == direction), 0)
                packet[(layer, boundary_position, direction)] += (
                    model.new_kv_bytes_per_token_per_layer)
            if layer + 1 < model.layers:
                next_qkv = PIM if interval.contains(layer + 1, "qkv") else GPU
                if path[-1] != next_qkv:
                    direction = "ga" if path[-1] == GPU else "ag"
                    packet[(layer, 4, direction)] += model.hidden_size * model.dtype_bytes
            if layer == 0 and path[0] == PIM:
                # The first decode input starts on GPU after prefill. This
                # boundary is absent from the inter-layer transition loop.
                packet[(layer, -1, "ga")] += model.hidden_size * model.dtype_bytes
            for key, payload in packet.items():
                transfers[key] += weight * payload
                transfer_absence[key] *= 1 - weight

        for (side, operator, replica), (m, absence) in operator_batch.items():
            presence = 1 - absence
            if operator == "merge":
                cost = ((system.gpu.merge_launch_us * presence * sim.layout_launch_scale
                         + system.gpu.merge_request_us * m)
                        if side == GPU else
                        (system.pim.request_setup_us * presence * sim.layout_launch_scale + _us(
                            m * model.hidden_size * model.dtype_bytes
                            / max(system.pim.bandwidth_per_rankset_bytes_s, 1))))
                merge_us += cost
            else:
                if side == PIM:
                    shards = _linear_shard_us(world.config, operator, m, presence)
                    cost = max(shards)
                else:
                    cost = _linear_us(world.config, side, operator, m, layer=layer,
                                      presence=presence)
                dense_us += cost
            if side == GPU:
                gpu_replica_time[replica] += cost
            else:
                if operator == "merge":
                    pim_rank_time[operator][world.rankset(layer)] += cost
                else:
                    for group, shard_cost in zip(dense_groups, shards):
                        pim_rank_time[operator][group] += shard_cost
        for key, (m, length, absence, group) in _partition_attention_batch(attention_batch).items():
            side = key[0]
            cost, flops, kv_bytes = _attention_us(world, side, m, length,
                                                  presence=1 - absence)
            attention_flops += flops
            if side == GPU:
                assert group is not None
                gpu_replica_time[group] += cost
                gpu_scan += kv_bytes
            else:
                # The physical block, rather than a hypothetical free DIMM,
                # selects the execution group.
                assert group is not None
                pim_rank_time["attention"][group] += cost
                pim_scan += kv_bytes
        for replica, absence in enumerate(gpu_attention_absence):
            gpu_replica_time[replica] += (
                system.gpu.attention_launch_us * sim.layout_launch_scale
                * (1 - absence))
        for group, absence in enumerate(pim_group_absence):
            pim_rank_time["attention"][group] += (
                system.pim.request_setup_us * sim.layout_launch_scale
                * (1 - absence))

    ga = ag = 0.0
    for key, size in transfers.items():
        direction = key[2]
        presence = 1 - transfer_absence[key]
        if direction == "ga":
            ga += (system.link.transfer_latency_us * sim.layout_launch_scale * presence
                   + _us(size / system.link.host_to_pim_bytes_s))
        else:
            ag += (system.link.transfer_latency_us * sim.layout_launch_scale * presence
                   + _us(size / system.link.pim_to_host_bytes_s))
    cross_bytes = sum(cross_group.values())
    cross_us = sum(_us(size / system.pim.rank_bus_per_rankset_bytes_s)
                   + sim.layout_rankset_sync_us * sim.layout_launch_scale
                   * (1 - cross_group_absence[key])
                   for key, size in cross_group.items())
    pim_time = (sum(max(pim_rank_time[op]) for op in OPS)
                + pim_reduction_us + pim_sync_us + cross_us)
    return StageCost(max(gpu_replica_time), pim_time, ga, ag,
                     attention_flops, pim_scan, gpu_scan, dense_us, merge_us,
                     cross_bytes, cross_us)


def _prune_link(calendar: LinkCalendar, now: float) -> None:
    calendar._pruned_through_us = max(calendar._pruned_through_us, now)
    retained = []
    for event in calendar.events:
        if event.end_us <= now:
            if event.kind == "preparation":
                calendar.completed_preparation_busy_us += event.end_us - event.start_us
                job = calendar.jobs[event.job_id]
                job.completed_bytes += event.bytes
                job.completed_end_us = max(job.completed_end_us, event.end_us)
                if job.completed_bytes == job.size:
                    job.ready_us = job.completed_end_us + job.conversion_us
                    if calendar._pending_jobs is not None:
                        calendar._pending_jobs.pop(job.job_id, None)
        else:
            retained.append(event)
    calendar.events = retained


def _pending_preparations(calendar: LinkCalendar) -> dict[int, PreparationJob]:
    if calendar._pending_jobs is None or calendar._known_jobs != len(calendar.jobs):
        calendar._pending_jobs = {
            key: job for key, job in calendar.jobs.items()
            if job.completed_bytes != job.size}
        calendar._known_jobs = len(calendar.jobs)
    return calendar._pending_jobs


def _reserve_link(calendar: LinkCalendar, full_duplex: bool, direction: str,
                  duration_us: float, earliest_us: float, kind: str,
                  job_id: int | None = None, size: int = 0, *,
                  measure_wait: bool = True) -> tuple[float, float]:
    """Place a packet in the first gap on its physical link timeline."""
    if duration_us <= 0:
        return earliest_us, 0.0
    start = earliest_us
    # The common half-duplex preparation stream appends packets in time
    # order. Avoid sorting the entire calendar for every 4 MiB packet.
    if (not calendar.events or (not full_duplex
                               and calendar.events[-1].end_us <= earliest_us)):
        calendar.events.append(LinkEvent(start, start + duration_us, direction,
                                         kind, job_id, size))
        return start, 0.0
    for event in sorted(calendar.events, key=lambda item: item.start_us):
        if full_duplex and event.direction != direction:
            continue
        if event.end_us <= start:
            continue
        if event.start_us >= start + duration_us:
            break
        start = event.end_us
    wait = (sum(max(0.0, min(start, event.end_us) - max(earliest_us, event.start_us))
                for event in calendar.events if event.kind == "preparation"
                and (not full_duplex or event.direction == direction)) if measure_wait else 0.0)
    calendar.events.append(LinkEvent(start, start + duration_us, direction,
                                     kind, job_id, size))
    return start, wait


def _reflow_preparations(calendar: LinkCalendar, config: ProjectConfig,
                         cutoff_us: float) -> None:
    """Foreground keeps priority over unsent background packets."""
    preserved = [event for event in calendar.events
                 if event.kind == "foreground" or event.start_us < cutoff_us]
    calendar.events = preserved
    retained_bytes: dict[int, int] = {}
    retained_end: dict[int, float] = {}
    for event in preserved:
        if event.kind == "preparation":
            job_id = event.job_id
            retained_bytes[job_id] = retained_bytes.get(job_id, 0) + event.bytes
            previous = retained_end.get(job_id)
            retained_end[job_id] = (event.end_us if previous is None else
                                    max(previous, event.end_us))
    prior_end: dict[str, float] = defaultdict(lambda: cutoff_us)
    full_duplex = config.system.link.full_duplex
    ga_bandwidth = config.system.link.host_to_pim_bytes_s
    ag_bandwidth = config.system.link.pim_to_host_bytes_s
    launch = config.system.link.transfer_latency_us * config.simulation.layout_launch_scale
    # Completed jobs remain available for copy readiness and statistics, but
    # cannot delay packets starting at or after the pruning time. Preserve
    # the original all-job traversal for an explicitly retrospective query.
    jobs = (calendar.jobs if cutoff_us < calendar._pruned_through_us
            else _pending_preparations(calendar))
    for job in jobs.values():
        lane = job.direction if full_duplex else "shared"
        sent = job.completed_bytes + retained_bytes.get(job.job_id, 0)
        last_end = retained_end.get(job.job_id, job.completed_end_us)
        if sent == job.size:
            job.ready_us = last_end + job.conversion_us
            prior_end[lane] = max(prior_end[lane], last_end)
            continue
        earliest = max(cutoff_us, job.earliest_us, prior_end[lane], last_end)
        size = job.size - sent
        bandwidth = ga_bandwidth if job.direction == "ga" else ag_bandwidth
        startup = launch if sent == 0 else 0.0
        duration = startup + _us(size / bandwidth)
        start, _ = _reserve_link(calendar, full_duplex,
                                 job.direction, duration, earliest,
                                 "preparation", job.job_id, size, measure_wait=False)
        calendar.events[-1].startup_us = startup
        earliest = start + duration
        job.ready_us = earliest + job.conversion_us
        prior_end[lane] = earliest


def _append_preparation(calendar: LinkCalendar, config: ProjectConfig,
                        job: PreparationJob) -> None:
    """Append a newly queued job without rebuilding earlier reservations."""
    if calendar._pending_jobs is not None:
        calendar._pending_jobs[job.job_id] = job
        calendar._known_jobs = len(calendar.jobs)
    full_duplex = config.system.link.full_duplex
    previous = max((event.end_us for event in calendar.events
                    if event.kind == "preparation"
                    and (not full_duplex or event.direction == job.direction)),
                   default=job.earliest_us)
    earliest = max(job.earliest_us, previous)
    bandwidth = (config.system.link.host_to_pim_bytes_s
                 if job.direction == "ga" else
                 config.system.link.pim_to_host_bytes_s)
    startup = (config.system.link.transfer_latency_us
               * config.simulation.layout_launch_scale)
    duration = startup + _us(job.size / bandwidth)
    start, _ = _reserve_link(calendar, full_duplex, job.direction,
                             duration, earliest, "preparation", job.job_id,
                             job.size, measure_wait=False)
    calendar.events[-1].startup_us = startup
    earliest = start + duration
    job.ready_us = earliest + job.conversion_us


def _truncate_inflight_preparation(calendar: LinkCalendar,
                                   config: ProjectConfig, cutoff_us: float,
                                   direction: str) -> None:
    """Keep only the active DMA through its next preemption boundary."""
    for event in calendar.events:
        if (event.kind != "preparation" or
                (config.system.link.full_duplex and event.direction != direction) or not
                event.start_us < cutoff_us < event.end_us):
            continue
        bandwidth = (config.system.link.host_to_pim_bytes_s
                     if event.direction == "ga" else
                     config.system.link.pim_to_host_bytes_s)
        quantum = config.simulation.layout_preparation_packet_bytes
        active_bytes = max(0.0, (cutoff_us - event.start_us - event.startup_us)
                           * bandwidth / 1e6)
        completed_chunks = max(1, ceil(active_bytes / quantum))
        kept_bytes = min(event.bytes, completed_chunks * quantum)
        event.bytes = kept_bytes
        event.end_us = event.start_us + event.startup_us + _us(kept_bytes / bandwidth)


def _sync_preparation_ready(world: LayoutWorld) -> None:
    jobs = world.link_calendar.jobs
    for block in world.blocks.values():
        for copy in block.copies.values():
            if copy.preparation_job_id is not None:
                copy.ready_us = jobs[copy.preparation_job_id].ready_us
        if block.pending_owner is not None:
            block.owner_ready_us = block.copies[block.pending_owner].ready_us
    for weight in world.weights.values():
        if weight.preparation_job_id is not None:
            weight.ready_us = jobs[weight.preparation_job_id].ready_us
    for request in world.requests.values():
        if request.pending_interval is None:
            continue
        required = [world.weights[(request.replica, layer, operator)].ready_us
                    for layer in range(world.config.model.layers)
                    for operator in ("qkv", "output_projection", "ffn")
                    if request.pending_interval.contains(layer, operator)]
        required.extend(world.blocks[key].owner_ready_us
                        for key in request.pending_interval_block_ids
                        if key in world.blocks)
        request.interval_ready_us = max([world.clock_us, *required], default=world.clock_us)


def _book_boundary(calendar: LinkCalendar, config: ProjectConfig,
                   start_us: float, ga_us: float, ag_us: float
                   ) -> tuple[float, float, float]:
    finish = start_us
    busy = wait = 0.0
    if (not _pending_preparations(calendar)
            and start_us >= calendar._pruned_through_us):
        # No background stream exists, so neither preemption nor rebuilding
        # its queue can change anything. Preserve the same packet reservations.
        if (not calendar.events or (not config.system.link.full_duplex
                                   and calendar.events[-1].end_us <= start_us)):
            if ga_us > 0:
                calendar.events.append(LinkEvent(start_us, start_us + ga_us, "ga", "foreground"))
                finish = max(finish, start_us + ga_us)
                busy += ga_us
            if ag_us > 0:
                earliest = start_us if config.system.link.full_duplex else finish
                calendar.events.append(LinkEvent(earliest, earliest + ag_us, "ag", "foreground"))
                finish = max(finish, earliest + ag_us)
                busy += ag_us
            return finish, busy, wait
        for direction, duration in (("ga", ga_us), ("ag", ag_us)):
            if duration <= 0:
                continue
            earliest = start_us if config.system.link.full_duplex else finish
            packet_start, packet_wait = _reserve_link(
                calendar, config.system.link.full_duplex, direction, duration,
                earliest, "foreground")
            finish = max(finish, packet_start + duration)
            busy += duration
            wait += packet_wait
        return finish, busy, wait
    # The stage model knows direction-level aggregate payloads, not the
    # individual operator packet release times. GA then AG is a fixed tie-break.
    for direction, duration in (("ga", ga_us), ("ag", ag_us)):
        if duration <= 0:
            continue
        earliest = start_us if config.system.link.full_duplex else finish
        cutoff = earliest
        _truncate_inflight_preparation(calendar, config, cutoff, direction)
        calendar.events = [event for event in calendar.events
                           if event.kind == "foreground" or event.start_us < cutoff]
        packet_start, packet_wait = _reserve_link(
            calendar, config.system.link.full_duplex, direction, duration,
            earliest, "foreground")
        finish = max(finish, packet_start + duration)
        busy += duration
        wait += packet_wait
        # On a shared lane AG arrives exactly when GA finishes. Any unsent
        # preparation rebuilt here would be discarded again before AG. Keep
        # the active packet, and rebuild once after both foreground packets.
        if not (direction == "ga" and ag_us > 0 and not config.system.link.full_duplex):
            _reflow_preparations(calendar, config, cutoff)
    return finish, busy, wait


def evaluate_step(world: LayoutWorld, at_us: float | None = None,
                  offset: int = 0, *, book_link: bool = False) -> StepCost:
    now = world.clock_us if at_us is None else at_us
    calendar = world.link_calendar if book_link else deepcopy(world.link_calendar)
    _prune_link(calendar, now)
    subbatch0, subbatch1 = _subbatches(world)
    stage0 = stage1 = boundary = 0.0
    gpu_busy = pim_busy = dense = flops = merge = pim_scan = gpu_scan = 0.0
    cross_bytes = cross_us = 0.0
    stage0_gpu = stage0_pim = stage1_gpu = stage1_pim = 0.0
    critical_gpu = critical_pim = 0.0
    stage0_gpu_critical = stage0_pim_critical = 0.0
    stage1_gpu_critical = stage1_pim_critical = 0.0
    stage0_gpu_layers = stage0_pim_layers = stage1_gpu_layers = stage1_pim_layers = 0
    link_busy = foreground_wait = 0.0
    elapsed = 0.0
    cache = current_cost_cache()
    config_token = cache.config_token(world.config) if cache is not None else None
    inputs0 = _stage_request_inputs(world, subbatch0, now, offset) if cache is not None else None
    inputs1 = _stage_request_inputs(world, subbatch1, now, offset) if cache is not None else None
    # Equation (2) sums the dependent layer cycles.  Equation (8)'s group
    # maximum has already been applied inside each individual layer call.
    for layer in range(world.config.model.layers):
        first = _stage_cost(world, subbatch0, now, offset, layer=layer,
                            request_inputs=inputs0, config_token=config_token)
        second = _stage_cost(world, subbatch1, now, offset, layer=layer,
                             request_inputs=inputs1, config_token=config_token)
        overlap = world.config.simulation.layout_boundary_overlap_fraction
        scale = world.config.simulation.layout_boundary_scale
        for phase, gpu_time, pim_time, ga, ag in (
            (0, first.gpu_us, second.pim_us,
             first.gpu_to_pim_us, second.pim_to_gpu_us),
            (1, second.gpu_us, first.pim_us,
             second.gpu_to_pim_us, first.pim_to_gpu_us),
        ):
            phase_start = now + elapsed
            finish, occupied, waited = _book_boundary(
                calendar, world.config, phase_start, ga, ag)
            link_busy += occupied
            foreground_wait += waited
            exposed = max(0.0, finish - phase_start - overlap * abs(gpu_time - pim_time))
            base = max(gpu_time, pim_time)
            # A sensitivity scale must not allow a phase to end before its
            # actual boundary packet has arrived.
            duration = max(base + scale * exposed, finish - phase_start)
            boundary += duration - base
            elapsed += duration
            if phase == 0:
                stage0 += duration
                stage0_gpu_layers += gpu_time > pim_time
                stage0_pim_layers += pim_time > gpu_time
            else:
                stage1 += duration
                stage1_gpu_layers += gpu_time > pim_time
                stage1_pim_layers += pim_time > gpu_time
            if gpu_time > pim_time:
                critical_gpu += gpu_time
                if phase == 0:
                    stage0_gpu_critical += gpu_time
                else:
                    stage1_gpu_critical += gpu_time
            elif pim_time > gpu_time:
                critical_pim += pim_time
                if phase == 0:
                    stage0_pim_critical += pim_time
                else:
                    stage1_pim_critical += pim_time
            else:
                critical_gpu += base / 2
                critical_pim += base / 2
                if phase == 0:
                    stage0_gpu_critical += base / 2
                    stage0_pim_critical += base / 2
                else:
                    stage1_gpu_critical += base / 2
                    stage1_pim_critical += base / 2
        stage0_gpu += first.gpu_us
        stage0_pim += second.pim_us
        stage1_gpu += second.gpu_us
        stage1_pim += first.pim_us
        gpu_busy += first.gpu_us + second.gpu_us
        pim_busy += first.pim_us + second.pim_us
        dense += first.dense_us + second.dense_us
        flops += first.attention_flops + second.attention_flops
        merge += first.merge_us + second.merge_us
        pim_scan += first.pim_scan_bytes + second.pim_scan_bytes
        gpu_scan += first.gpu_scan_bytes + second.gpu_scan_bytes
        cross_bytes += first.cross_group_bytes + second.cross_group_bytes
        cross_us += first.cross_group_us + second.cross_group_us
    if book_link:
        world.link_busy_us += link_busy
        world.foreground_wait_us += foreground_wait
        _sync_preparation_ready(world)
    total = stage0 + stage1
    stage0_label = (GPU if stage0_gpu_critical > stage0_pim_critical else
                    PIM if stage0_pim_critical > stage0_gpu_critical else "mixed")
    stage1_label = (GPU if stage1_gpu_critical > stage1_pim_critical else
                    PIM if stage1_pim_critical > stage1_gpu_critical else "mixed")
    return StepCost(total, stage0, stage1, boundary,
                    gpu_busy, pim_busy, dense, flops, merge, pim_scan, gpu_scan,
                    stage0_label, stage1_label, cross_bytes, cross_us,
                    stage0_gpu_layers, stage0_pim_layers,
                    stage1_gpu_layers, stage1_pim_layers,
                    critical_gpu, critical_pim, link_busy, foreground_wait)


@dataclass(frozen=True)
class LayoutAction:
    kind: str
    request_ids: tuple[str, ...] = ()
    block_ids: tuple[str, ...] = ()
    target: str | None = None
    interval: PimInterval | None = None
    steps: tuple[LayoutAction, ...] = ()


def _weight_bytes(world: LayoutWorld, operator: str) -> int:
    model = world.config.model
    if operator == "qkv":
        return model.qkv_weight_bytes_per_layer
    if operator == "output_projection":
        return model.output_projection_weight_bytes_per_layer
    if operator == "ffn":
        return model.ff_weight_bytes_per_layer
    raise ValueError(operator)


def _schedule_preparation(world: LayoutWorld, size: int,
                          destination: str, conversion_us: float = 0.0
                          ) -> tuple[float, float, int]:
    """Queue a packetized DMA; conversion follows transfer off the link."""
    direction = "ga" if destination == PIM else "ag"
    bandwidth = (world.config.system.link.host_to_pim_bytes_s
                 if destination == PIM else
                 world.config.system.link.pim_to_host_bytes_s)
    transfer = (world.config.system.link.transfer_latency_us
                * world.config.simulation.layout_launch_scale
                + _us(size / bandwidth))
    calendar = world.link_calendar
    job_id = calendar.next_job_id
    calendar.next_job_id += 1
    calendar.jobs[job_id] = PreparationJob(
        job_id, direction, size, world.clock_us, conversion_us)
    _prune_link(calendar, world.clock_us)
    _append_preparation(calendar, world.config, calendar.jobs[job_id])
    _sync_preparation_ready(world)
    return transfer + conversion_us, calendar.jobs[job_id].ready_us, job_id


def _prepare_interval(world: LayoutWorld, interval: PimInterval,
                      replicas: set[int]) -> float:
    transfer_us = 0.0
    for replica in sorted(replicas):
        for layer in range(world.config.model.layers):
            for operator in ("qkv", "output_projection", "ffn"):
                if not interval.contains(layer, operator):
                    continue
                key = (replica, layer, operator)
                if key in world.weights:
                    continue
                size = _weight_bytes(world, operator)
                world.preparation_bytes += size
                duration, ready, job_id = _schedule_preparation(
                    world, size, PIM, size / GiB
                    * world.config.simulation.layout_weight_conversion_us_per_gib)
                transfer_us += duration
                groups = _dense_groups(world, layer)
                # Every matrix is output-sharded. Sum integer shard bytes
                # across matrices so no byte is lost when a width is uneven.
                matrices = _dense_matrices(world.config, operator)
                widths = tuple(_shard_widths(n, len(groups)) for _, n in matrices)
                shard_bytes = tuple(
                    sum(k * width[index] * world.config.model.dtype_bytes
                        for (k, _), width in zip(matrices, widths))
                    for index in range(len(groups)))
                assert sum(shard_bytes) == size
                world.weights[key] = WeightCopy(replica, layer, operator,
                                                groups, shard_bytes, ready,
                                                preparation_job_id=job_id)
                if world._capacity_totals is not None:
                    for rankset, size in zip(groups, shard_bytes):
                        world._capacity_totals.pim_weights[rankset] += size
    return transfer_us


def apply_action(world: LayoutWorld, action: LayoutAction,
                 *, baseline_us: float | None = None) -> float:
    """Apply a complete action; return physical preparation time in microseconds.

    The caller applies this to a copy when checking feasibility.  A pending
    physical copy counts toward capacity before it becomes a compute owner.
    """
    if baseline_us is None:
        baseline_us = evaluate_step(world).total_us
    if action.kind == "bundle":
        prepare = sum(apply_action(world, part, baseline_us=baseline_us)
                      for part in action.steps)
        world.check_capacity(world.planning_reserve_tokens())
        return prepare
    prepare_us = 0.0
    if action.kind in ("future", "combined"):
        if action.target not in (GPU, PIM):
            raise ValueError("future writer requires GPU or PIM target")
        for request_id in action.request_ids:
            request = world.requests[request_id]
            if request.future_writer != action.target:
                world._account_writer(request, -1)
                request.future_writer = action.target
                world._account_writer(request, 1)
                request.switches += 1
    if action.kind in ("copy", "combined"):
        if action.target not in (GPU, PIM):
            raise ValueError("copy requires GPU or PIM target")
        affected_refs: set[str] = set()
        for key in dict.fromkeys(action.block_ids):
            block = world.blocks[key]
            if not block.refs:
                continue
            if (block.pending_owner or block.owner) != action.target:
                affected_refs.update(block.refs)
            if action.target not in block.copies:
                tile = (world.config.system.gpu.kv_block_tokens
                        if action.target == GPU else
                        world.config.system.pim.kv_block_tokens)
                size = ceil(block.tokens / tile) * tile * world.kv_bytes
                world.preparation_bytes += size
                duration, ready, job_id = _schedule_preparation(world, size, action.target)
                prepare_us += duration
                world.add_copy(block, action.target, ready, job_id)
            block.pending_owner = action.target
            block.owner_ready_us = block.copies[action.target].ready_us
        for request_id in affected_refs:
            world.requests[request_id].switches += 1
    if action.kind in ("interval", "combined"):
        if action.interval is None:
            raise ValueError("interval action requires a PIM interval")
        replicas = {world.requests[key].replica for key in action.request_ids}
        prepare_us += _prepare_interval(world, action.interval, replicas)
        for request_id in action.request_ids:
            request = world.requests[request_id]
            changed = (request.pending_interval or request.interval) != action.interval
            request.pending_interval = action.interval
            request.pending_interval_block_ids = tuple(action.block_ids)
            needed = (world.weights[(request.replica, layer, operator)].ready_us
                      for layer in range(world.config.model.layers)
                      for operator in ("qkv", "output_projection", "ffn")
                      if action.interval.contains(layer, operator))
            copy_ready = (world.blocks[key].owner_ready_us
                          for key in action.block_ids)
            request.interval_ready_us = max(
                world.clock_us, max(needed, default=world.clock_us),
                max(copy_ready, default=world.clock_us))
            request.switches += changed
    if action.kind not in ("future", "copy", "interval", "combined"):
        raise ValueError(action.kind)
    world.preparation_us += prepare_us
    world.check_capacity(world.planning_reserve_tokens())
    return prepare_us


def finalize_preparations(world: LayoutWorld) -> None:
    for block in world.blocks.values():
        if block.pending_owner is not None and world.clock_us >= block.owner_ready_us:
            old = block.owner
            block.owner = block.pending_owner
            block.pending_owner = None
            if old != block.owner:
                copy = block.copies.pop(old, None)
                if copy is not None:
                    world._account_copy(block, copy, -1)
    for request in world.requests.values():
        if request.pending_interval is not None and world.clock_us >= request.interval_ready_us:
            request.interval = request.pending_interval
            request.pending_interval = None
            request.pending_interval_block_ids = ()
    world.check_capacity()


def _candidate_groups(world: LayoutWorld) -> list[tuple[str, ...]]:
    # Shared-prefix requests move as a natural reference group; unrelated
    # requests never enter an arbitrary subset search.
    groups: dict[tuple[int, str, str], list[str]] = defaultdict(list)
    for request in world.requests.values():
        identity = ((request.replica, "prefix", request.source.prefix_leaf_id)
                    if request.source.prefix_leaf_id is not None else
                    (request.replica, "private", request.source.request_id))
        groups[identity].append(request.source.request_id)
    return sorted(tuple(sorted(peers)) for peers in groups.values())


def _candidate_priority(world: LayoutWorld, action: LayoutAction,
                        subbatch: dict[str, int],
                        stages: tuple[StageCost, StageCost],
                        hot_rankset: dict[int, int | None]
                        ) -> tuple[int, float]:
    """Cheap phase-aware screen; full scheduling decides the actual gain."""
    first, second = stages
    phase_loads = ((first.gpu_us, second.pim_us),
                   (second.gpu_us, first.pim_us))
    horizon = world.config.simulation.policy.planning_horizon_tokens
    impact = [0.0, 0.0]
    for request_id in action.request_ids:
        request = world.requests[request_id]
        batch = subbatch[request_id]
        remaining = _expected_active_tokens(
            request.generated, horizon,
            world.config.simulation.layout_survival_mean_tokens,
            world.config.simulation.layout_survival_shape)
        survival = remaining / horizon
        moves: list[tuple[str, str, float]] = []
        if action.kind in ("future", "combined") and request.future_writer != action.target:
            moves.append((request.future_writer, action.target or GPU,
                          remaining / 2))
        if action.kind in ("copy", "combined") and action.block_ids:
            block = world.blocks[action.block_ids[0]]
            if block.owner_at(world.clock_us) != action.target:
                moved_ranks = {world.blocks[key].copies[PIM].rankset
                               for key in action.block_ids
                               if world.blocks[key].owner_at(world.clock_us) == PIM}
                rank_factor = (1.0 if hot_rankset[batch] in moved_ranks
                               or hot_rankset[batch] is None else 0.25)
                moves.append((block.owner_at(world.clock_us), action.target or GPU,
                              block.tokens * rank_factor))
        if action.kind in ("interval", "combined") and action.interval is not None:
            old = request.pending_interval or request.interval
            for op in ("qkv", "merge", "output_projection", "ffn"):
                if old.contains(0, op) != action.interval.contains(0, op):
                    source = PIM if old.contains(0, op) else GPU
                    target = PIM if action.interval.contains(0, op) else GPU
                    moves.append((source, target,
                                  max(1.0, request.source.input_tokens / 16)))
        for source, target, magnitude in moves:
            phase = batch if source == GPU else 1 - batch
            gpu_load, pim_load = phase_loads[phase]
            source_load = gpu_load if source == GPU else pim_load
            target_load = gpu_load if target == GPU else pim_load
            pressure = (source_load - target_load) / max(source_load, target_load, 1)
            impact[phase] += magnitude * survival * max(0.1, 1 + pressure)
    phase = 0 if impact[0] >= impact[1] else 1
    preparation_bytes = 0
    if action.kind in ("copy", "combined"):
        for key in action.block_ids:
            block = world.blocks[key]
            if action.target not in block.copies:
                tile = (world.config.system.gpu.kv_block_tokens if action.target == GPU
                        else world.config.system.pim.kv_block_tokens)
                preparation_bytes += ceil(block.tokens / tile) * tile * world.kv_bytes
    if action.kind in ("interval", "combined") and action.interval is not None:
        for replica in {world.requests[key].replica for key in action.request_ids}:
            for layer in range(world.config.model.layers):
                for operator in ("qkv", "output_projection", "ffn"):
                    if (action.interval.contains(layer, operator)
                            and (replica, layer, operator) not in world.weights):
                        preparation_bytes += _weight_bytes(world, operator)
    score = sum(impact) / (1 + preparation_bytes / max(
        sum(impact) * world.kv_bytes, 1))
    return phase, score


def generate_candidates(world: LayoutWorld,
                        *, allow_pending_extension: bool = False,
                        screening: dict[str, Any] | None = None) -> list[LayoutAction]:
    if not world.requests:
        return []
    batch0, batch1 = _subbatches(world)
    a = _stage_cost(world, batch0, world.clock_us)
    b = _stage_cost(world, batch1, world.clock_us)
    bottlenecks = (PIM if b.pim_us > a.gpu_us else GPU,
                   PIM if a.pim_us > b.gpu_us else GPU)
    preferred = GPU if bottlenecks.count(PIM) >= bottlenecks.count(GPU) else PIM
    groups = _candidate_groups(world)
    proposals: list[LayoutAction] = []
    seen_copies: set[tuple[tuple[str, ...], str]] = set()
    for group in groups:
        candidates: list[LayoutAction] = []
        for target in (preferred, PIM if preferred == GPU else GPU):
            if any(world.requests[key].future_writer != target for key in group):
                candidates.append(LayoutAction("future", group, target=target))
        request = world.requests[group[0]]
        current_interval = request.pending_interval or request.interval
        if allow_pending_extension or all(world.requests[key].pending_interval is None
                                          for key in group):
            for interval in current_interval.alternatives():
                if all((world.requests[key].pending_interval or world.requests[key].interval)
                       == interval for key in group):
                    continue
                candidates.append(LayoutAction("interval", group, interval=interval))
        # Move high-impact physical blocks, including every active ref of a
        # shared prefix, before considering more expensive combined moves.
        blocks = {block.block_id: block for key in group
                  for block in _selected_blocks(world, world.requests[key], 0)
                  if block.refs}
        shared = sorted((block for block in blocks.values() if block.shared),
                        key=lambda block: (-block.tokens * len(block.refs),
                                           -block.start_token))
        private = sorted((block for block in blocks.values() if not block.shared),
                         key=lambda block: (-block.tokens, -block.start_token))
        for block in (*shared, *private[:2]):
            if block.pending_owner is not None:
                continue
            target = GPU if block.owner_at(world.clock_us) == PIM else PIM
            owner = world.requests[min(block.refs)]
            position = owner.block_ids[0].index(block.block_id)
            keys = tuple(owner.block_ids[layer][position]
                         for layer in range(world.config.model.layers)
                         if position < len(owner.block_ids[layer]))
            physical = (keys, target)
            if keys and physical not in seen_copies:
                seen_copies.add(physical)
                refs = tuple(sorted(block.refs))
                candidates.append(LayoutAction("copy", refs, keys, target))
                intervals = {world.requests[key].pending_interval
                             or world.requests[key].interval for key in refs}
                if (len(intervals) == 1 and
                        any(world.requests[key].future_writer != target for key in refs)):
                    candidates.append(LayoutAction("combined", refs, keys, target,
                                                   intervals.pop()))
        proposals.extend(candidates)

    subbatch = {request.source.request_id: index
                for index, batch in enumerate((batch0, batch1)) for request in batch}
    hot_rankset: dict[int, int | None] = {}
    for index, batch in enumerate((batch0, batch1)):
        rank_load: dict[int, float] = defaultdict(float)
        for request in batch:
            for layer in range(world.config.model.layers):
                for block in _selected_blocks(world, request, layer):
                    if block.owner_at(world.clock_us) == PIM:
                        rank = block.copies[PIM].rankset
                        assert rank is not None
                        rank_load[rank] += block.tokens
        hot_rankset[index] = max(rank_load, key=rank_load.get) if rank_load else None
    lanes: dict[tuple[int, str], dict[tuple[str, ...], list[tuple[float, LayoutAction]]]] = (
        defaultdict(lambda: defaultdict(list)))
    for action in proposals:
        phase, priority = _candidate_priority(
            world, action, subbatch, (a, b), hot_rankset)
        lanes[phase, action.kind][action.request_ids].append((priority, action))
    identities = sorted({action.request_ids for action in proposals})
    rotation = ((world.replan_index
                 * max(1, world.config.simulation.layout_max_candidates // 2))
                % len(identities)) if identities else 0
    rotated = identities[rotation:] + identities[:rotation]
    tie_rank = {identity: index for index, identity in enumerate(rotated)}
    queues: dict[tuple[int, str], list[LayoutAction]] = {}
    for lane, by_group in lanes.items():
        for actions in by_group.values():
            actions.sort(key=lambda item: -item[0])
        order = sorted(by_group, key=lambda identity: (
            -by_group[identity][0][0], tie_rank[identity]))
        queue: list[LayoutAction] = []
        while any(by_group.values()):
            for identity in order:
                if by_group[identity]:
                    queue.append(by_group[identity].pop(0)[1])
        queues[lane] = queue
    ordered: list[LayoutAction] = []
    lane_order = tuple((phase, kind) for kind in
                       ("future", "interval", "copy", "combined")
                       for phase in (0, 1))
    while len(ordered) < world.config.simulation.layout_max_candidates and any(queues.values()):
        for lane in lane_order:
            if queues.get(lane) and len(ordered) < world.config.simulation.layout_max_candidates:
                ordered.append(queues[lane].pop(0))
    if screening is not None:
        eligible = set(groups) | set(identities)
        evaluated = {action.request_ids for action in ordered}
        screening.update({"eligible_groups": len(eligible),
                          "evaluated_groups": len(evaluated),
                          "truncated_groups": len(eligible - evaluated),
                          "proposed_actions": len(proposals),
                          "evaluated_actions": len(ordered),
                          "evaluated_request_ids": sorted({key for action in ordered
                                                           for key in action.request_ids})})
    return ordered


def _forecast_scenarios(world: LayoutWorld, horizon: int) -> tuple[dict[str, int], ...]:
    """Draw coherent remaining lengths without consulting target output tokens.

    Stratified conditional Weibull quantiles give every active request short
    and long outcomes. All candidates reuse the same sampled paths.
    """
    sim = world.config.simulation
    count = sim.layout_forecast_scenarios
    rng = Random(world.config.simulation.policy.prediction_seed
                 + 1000003 * world.replan_index)
    scale = sim.layout_survival_mean_tokens / gamma(1 + 1 / sim.layout_survival_shape)
    scenarios: tuple[dict[str, int], ...] = tuple({} for _ in range(count))
    for request_id in sorted(world.requests):
        generated = world.requests[request_id].generated
        quantiles = [(index + rng.random()) / count for index in range(count)]
        rng.shuffle(quantiles)
        for scenario, quantile in zip(scenarios, quantiles):
            hazard = ((generated / scale) ** sim.layout_survival_shape
                      - log1p(-quantile))
            total_length = scale * hazard ** (1 / sim.layout_survival_shape)
            scenario[request_id] = min(horizon, max(1, ceil(total_length - generated)))
    return scenarios


def _forecast_total(world: LayoutWorld, horizon: int,
                    remaining: dict[str, int]) -> float:
    """Replay one complete future path using real request and KV lifetimes."""
    total = 0.0
    emitted = {request_id: 0 for request_id in world.requests}
    previous_totals = world._capacity_totals
    world._start_capacity_accounting()
    try:
        for _ in range(horizon):
            if not world.requests:
                break
            active = tuple(world.requests.values())
            cost = evaluate_step(world, book_link=True)
            world.clock_us += cost.total_us
            total += cost.total_us
            finalize_preparations(world)
            for request in active:
                request_id = request.source.request_id
                world.append_token(request)
                emitted[request_id] += 1
                if emitted[request_id] >= remaining[request_id]:
                    world.retire(request_id)
    finally:
        # Keep this detail out of real states and let callers edit public
        # block objects after a forecast returns, including on failure.
        world._capacity_totals = previous_totals
    return total


def _respects_service_guard(world: LayoutWorld, baseline_us: float,
                            candidate_us: float, horizon: int,
                            baseline_first_us: float,
                            candidate_first_us: float,
                            evaluated_steps: float | None = None) -> bool:
    sim = world.config.simulation
    # A complete scenario stops after its last request finishes, possibly
    # before H. Both layouts replay the same lengths, hence the same divisor.
    steps = max(evaluated_steps if evaluated_steps is not None else horizon, 1)
    baseline_step_ms = baseline_us / steps / 1e3
    candidate_step_ms = candidate_us / steps / 1e3
    if (sim.slo_tpot_ms is not None
            and baseline_step_ms <= sim.slo_tpot_ms
            and candidate_step_ms > sim.slo_tpot_ms):
        return False
    if (sim.slo_tpot_ms is not None
            and baseline_first_us / 1e3 <= sim.slo_tpot_ms
            and candidate_first_us / 1e3 > sim.slo_tpot_ms):
        return False
    if sim.slo_e2e_ms is not None:
        for request in world.requests.values():
            elapsed_ms = (world.clock_us - request.source.arrival_us) / 1e3
            if (elapsed_ms + baseline_first_us / 1e3 <= sim.slo_e2e_ms
                    < elapsed_ms + candidate_first_us / 1e3):
                return False
            residual = _expected_active_tokens(
                request.generated,
                max(horizon, ceil(8 * sim.layout_survival_mean_tokens)),
                sim.layout_survival_mean_tokens, sim.layout_survival_shape)
            baseline_e2e = elapsed_ms + residual * baseline_step_ms
            candidate_e2e = elapsed_ms + residual * candidate_step_ms
            if baseline_e2e <= sim.slo_e2e_ms < candidate_e2e:
                return False
    return True


def _capacity_recovery_action(world: LayoutWorld) -> LayoutAction | None:
    """Relieve GPU growth pressure with an atomic PIM writer switch.

    A single switch can leave the reserve infeasible even when several
    switches made at the same decision point would be legal.  Existing KV
    copies stay at their locations; PIM pressure still needs another remedy.
    """
    trial = deepcopy(world)
    moved: list[str] = []
    while True:
        try:
            trial.check_capacity(trial.planning_reserve_tokens())
            break
        except MemoryError as exc:
            overloaded_gpu = "GPU replica" in str(exc)
        if not overloaded_gpu:
            return None
        gpu, _ = trial.usage()
        reserve = (trial.planning_reserve_tokens() * trial.kv_bytes
                   * trial.config.model.layers)
        projected = [value + trial.config.simulation.layout_temporary_gib * GiB
                     for value in gpu]
        for request in trial.requests.values():
            if request.future_writer == GPU:
                projected[request.replica] += reserve
        replica = max(range(len(projected)), key=projected.__getitem__)
        choices = sorted(request_id for request_id, request in trial.requests.items()
                         if request.replica == replica and request.future_writer == GPU)
        if not choices:
            return None
        request_id = choices[0]
        trial._account_writer(trial.requests[request_id], -1)
        trial.requests[request_id].future_writer = PIM
        trial._account_writer(trial.requests[request_id], 1)
        moved.append(request_id)
    if not moved:
        return None
    return LayoutAction("future", tuple(sorted(moved)), target=PIM)


def choose_action(world: LayoutWorld, *,
                  screening: list[dict[str, Any]] | None = None,
                  reuse_statistics: dict | None = None,
                  forecast_mode: str = "bounded"
                  ) -> tuple[LayoutAction | None, float, float]:
    with reuse_costs(reuse_statistics):
        return _choose_action(world, screening=screening, forecast_mode=forecast_mode)


def _choose_action(world: LayoutWorld, *,
                   screening: list[dict[str, Any]] | None = None,
                   forecast_mode: str = "bounded"
                   ) -> tuple[LayoutAction | None, float, float]:
    if forecast_mode not in ("reference", "shared", "bounded"):
        raise ValueError("forecast_mode must be reference, shared or bounded")
    if not world.requests:
        return None, 0.0, 0.0
    baseline_step = evaluate_step(world).total_us
    # Replanning cadence (Delta) and forecast window (H) are independent.
    # Capacity checks reserve the configured H as well, so do not silently
    # shorten only the benefit side of the comparison.
    horizon = world.config.simulation.policy.planning_horizon_tokens
    try:
        deepcopy(world).check_capacity(world.planning_reserve_tokens())
        baseline_feasible = True
    except MemoryError:
        baseline_feasible = False
    if not baseline_feasible:
        # An overflowing baseline has no valid future latency to compare.
        # Repair physical capacity before stochastic benefit optimization.
        recovery = _capacity_recovery_action(world)
        if recovery is None:
            raise MemoryError("no candidate can reserve growth until the next replan")
        candidate = deepcopy(world)
        apply_action(candidate, recovery, baseline_us=baseline_step)
        if screening is not None:
            screening.append({"greedy_round": "capacity_recovery",
                              "candidate_outcomes": [{
                                  "kind": "future", "request_ids": list(recovery.request_ids),
                                  "status": "selected", "gain_us": None,
                                  "preparation_bytes": 0}]})
        return recovery, 0.0, baseline_step
    scenarios = _forecast_scenarios(world, horizon)
    evaluated_steps = fmean(max(remaining.values()) for remaining in scenarios)
    plan = None
    cache = current_cost_cache()
    statistics = cache.counts if cache is not None else None
    if forecast_mode == "reference":
        baseline = fmean(_forecast_total(deepcopy(world), horizon, remaining)
                         for remaining in scenarios)
    else:
        plan = build_forecast_plan(world, horizon, scenarios)
        if statistics is not None:
            statistics['forecast_plan_builds'] += 1
        baseline = evaluate_forecasts(deepcopy(world), plan, statistics=statistics).mean_us
    selected: list[LayoutAction] = []
    best_score = 0.0
    # Two greedy rounds suffice to expose copy+interval interaction while
    # bounding the number of full two-stage reevaluations per decision.
    for _round in range(2):
        current = deepcopy(world)
        if selected:
            try:
                apply_action(current, LayoutAction("bundle", steps=tuple(selected)),
                             baseline_us=baseline_step)
            except (MemoryError, ValueError, KeyError):
                break
        improvement: LayoutAction | None = None
        improvement_score = best_score
        improvement_bytes = float("inf")
        tie_margin = max(1e-6, world.config.simulation.layout_error_margin_us)
        round_screen: dict[str, Any] = {"greedy_round": _round + 1}
        outcomes: list[dict[str, Any]] = []
        round_screen["candidate_outcomes"] = outcomes
        actions = generate_candidates(current,
                                      allow_pending_extension=bool(selected),
                                      screening=round_screen)
        if screening is not None:
            screening.append(round_screen)
        for action in actions:
            candidate = deepcopy(world)
            bundle = LayoutAction("bundle", steps=(*selected, action))
            try:
                apply_action(candidate, bundle, baseline_us=baseline_step)
                if forecast_mode == "reference":
                    first_step = evaluate_step(candidate).total_us
                    future = fmean(_forecast_total(deepcopy(candidate), horizon, remaining)
                                   for remaining in scenarios)
                else:
                    result = evaluate_forecasts(candidate, plan,
                        baseline_us=baseline if forecast_mode == "bounded" else None,
                        incumbent_gain_us=improvement_score, tie_margin_us=tie_margin,
                        statistics=statistics)
                    first_step, future = result.first_step_us, result.mean_us
                if not _respects_service_guard(world, baseline, future, horizon,
                                               baseline_step, first_step,
                                               evaluated_steps):
                    outcomes.append({"kind": action.kind,
                                     "request_ids": list(action.request_ids),
                                     "status": "service_guard"})
                    continue
                score = baseline - future
            except ForecastPruned as exc:
                outcomes.append({"kind": action.kind,
                                 "request_ids": list(action.request_ids),
                                 "status": "bound_pruned", **exc.proof})
                continue
            except (MemoryError, ValueError, KeyError) as exc:
                outcomes.append({"kind": action.kind,
                                 "request_ids": list(action.request_ids),
                                 "status": type(exc).__name__,
                                 "reason": str(exc)})
                continue
            prepared = candidate.preparation_bytes - world.preparation_bytes
            outcomes.append({"kind": action.kind,
                             "request_ids": list(action.request_ids),
                             "status": "scored", "gain_us": score,
                             "preparation_bytes": prepared})
            if (score > best_score + 1e-9 and
                    (score > improvement_score + tie_margin
                     or (abs(score - improvement_score) <= tie_margin
                         and prepared < improvement_bytes))):
                improvement, improvement_score = action, score
                improvement_bytes = prepared
        if improvement is None:
            break
        selected.append(improvement)
        best_score = improvement_score
        round_screen["best_gain_us"] = best_score
    threshold = (world.config.simulation.layout_error_margin_us
                 + world.config.simulation.policy.relative_improvement_threshold
                 * baseline)
    if best_score <= threshold:
        return None, best_score, baseline
    return (selected[0] if len(selected) == 1 else
            LayoutAction("bundle", steps=tuple(selected))), best_score, baseline


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def run_layout_simulation(config: ProjectConfig):
    """Run one online layout-aware decode trace through the standard CLI result."""
    from .simulator import SimulationResult

    if config.simulation.policy.name not in ("layout_fixed", "layout_greedy"):
        raise ValueError("layout_aware requires layout_fixed or layout_greedy policy")
    if (config.simulation.policy.prediction != "fixed"
            or config.simulation.policy.optimizer != "horizon_greedy"):
        raise ValueError("layout_aware requires fixed-prior scenario forecasting")
    if config.simulation.attention_selector not in ("dense", "recent_blocks"):
        raise ValueError("unknown layout attention selector")
    base_dir = config.source_path.parent if config.source_path else Path.cwd()
    workload = build_workload(config.workload, base_dir)
    if not workload:
        raise ValueError("workload is empty")
    if (config.workload.type in ("static_batch", "closed_batch")
            and len(workload) > config.simulation.max_batch_size):
        raise ValueError("closed workload exceeds max_batch_size")
    catalog = PrefixCatalog(load_prefix_blocks(config.workload, base_dir))
    for source in workload:
        path = catalog.path(source.prefix_leaf_id)
        if sum(node.tokens for node in path) > source.input_tokens:
            raise ValueError(f"prefix exceeds prompt for {source.request_id}")
    world = LayoutWorld(config, catalog)
    pending = list(workload)
    completed: list[LayoutRequest] = []
    iterations: list[dict[str, Any]] = []
    switches: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    step_count = 0
    output_tokens = 0
    sum_batch = sum_gpu = sum_pim = sum_boundary = 0.0
    sum_dense = sum_merge = sum_flops = sum_pim_scan = sum_gpu_scan = 0.0
    sum_cross_bytes = sum_cross_us = 0.0
    gpu_bottlenecks = pim_bottlenecks = 0
    gpu_layers = pim_layers = 0
    gpu_critical = pim_critical = 0.0
    copy_actions = future_actions = interval_actions = 0
    policy_wall_s = 0.0
    policy_decisions = 0
    candidate_screening: list[dict[str, Any]] = []
    last_decision_step = -config.simulation.layout_min_replan_steps
    completed_event = False
    pending_replan_event = False
    while pending or world.requests:
        if not world.requests and pending and pending[0].arrival_us > world.clock_us:
            world.clock_us = pending[0].arrival_us
        admitted = False
        while (pending and pending[0].arrival_us <= world.clock_us
               and len(world.requests) < config.simulation.max_batch_size):
            source = pending[0]
            # Prefer an existing shared prefix on a feasible replica, then
            # lower KV usage. Replicas cannot borrow each other's HBM.
            gpu_usage, _ = world.usage()
            path = catalog.path(source.prefix_leaf_id)
            replicas = sorted(range(config.model.data_parallel),
                              key=lambda replica: (
                                  -sum(f"prefix:{replica}:0:{node.block_id}" in world.blocks
                                       for node in path), gpu_usage[replica], replica))
            accepted = None
            for replica in replicas:
                trial = deepcopy(world)
                try:
                    trial.admit(source, replica)
                except MemoryError:
                    continue
                accepted = trial
                break
            if accepted is None:
                if not world.requests:
                    raise MemoryError(f"request {source.request_id} cannot fit")
                break
            world = accepted
            pending.pop(0)
            admitted = True
        if not world.requests:
            continue
        periodic = step_count % config.simulation.policy.decision_interval == 0
        pending_replan_event |= admitted or (
            completed_event and config.simulation.policy.event_driven_replan)
        should_replan = (periodic or pending_replan_event) and (
            step_count - last_decision_step >= config.simulation.layout_min_replan_steps)
        completed_event = False
        if config.simulation.policy.name == "layout_greedy" and should_replan:
            decision_start = perf_counter()
            screening: list[dict[str, Any]] = []
            action, gain, before = choose_action(world, screening=screening)
            policy_wall_s += perf_counter() - decision_start
            policy_decisions += 1
            world.replan_index += 1
            candidate_screening.append({"iteration": step_count,
                                        "rounds": screening})
            last_decision_step = step_count
            pending_replan_event = False
            if action is not None:
                prep = apply_action(world, action)
                parts = action.steps if action.kind == "bundle" else (action,)
                for part_index, part in enumerate(parts):
                    switches.append({
                        "iteration": step_count,
                        "time_ms": world.clock_us / 1e3,
                        "kind": part.kind,
                        "request_ids": "|".join(part.request_ids),
                        "block_ids": "|".join(part.block_ids),
                        "target": part.target,
                        "interval": (f"{part.interval.left}:{part.interval.right}"
                                     if part.interval else None),
                        "preparation_us": prep if part_index == 0 else 0.0,
                        "objective_before_us": before,
                        "predicted_gain_us": gain if part_index == 0 else 0.0,
                    })
                    future_actions += part.kind in ("future", "combined")
                    copy_actions += part.kind in ("copy", "combined")
                    interval_actions += part.kind in ("interval", "combined")
        start = world.clock_us
        cost = evaluate_step(world, book_link=True)
        world.clock_us += cost.total_us
        step_count += 1
        active = list(world.requests.values())
        sum_batch += len(active)
        sum_gpu += cost.gpu_busy_us
        sum_pim += cost.pim_busy_us
        sum_boundary += cost.boundary_us
        sum_dense += cost.dense_us
        sum_cross_bytes += cost.cross_group_bytes
        sum_cross_us += cost.cross_group_us
        sum_merge += cost.merge_us
        sum_flops += cost.attention_flops
        sum_pim_scan += cost.pim_scan_bytes
        sum_gpu_scan += cost.gpu_scan_bytes
        gpu_bottlenecks += (cost.stage0_bottleneck == GPU) + (cost.stage1_bottleneck == GPU)
        pim_bottlenecks += (cost.stage0_bottleneck == PIM) + (cost.stage1_bottleneck == PIM)
        gpu_layers += cost.stage0_gpu_layers + cost.stage1_gpu_layers
        pim_layers += cost.stage0_pim_layers + cost.stage1_pim_layers
        gpu_critical += cost.gpu_critical_us
        pim_critical += cost.pim_critical_us
        if step_count == 1 or step_count % config.simulation.record_every == 0:
            gpu_usage, pim_usage = world.usage()
            iterations.append({
                "iteration": step_count, "start_us": start,
                "duration_us": cost.total_us, "stage0_us": cost.stage0_us,
                "stage1_us": cost.stage1_us, "boundary_us": cost.boundary_us,
                "stage0_bottleneck": cost.stage0_bottleneck,
                "stage1_bottleneck": cost.stage1_bottleneck,
                "stage0_gpu_layers": cost.stage0_gpu_layers,
                "stage0_pim_layers": cost.stage0_pim_layers,
                "stage1_gpu_layers": cost.stage1_gpu_layers,
                "stage1_pim_layers": cost.stage1_pim_layers,
                "gpu_busy_us": cost.gpu_busy_us, "pim_busy_us": cost.pim_busy_us,
                "cross_group_bytes": cost.cross_group_bytes,
                "cross_group_us": cost.cross_group_us,
                "batch_size": len(active), "gpu_kv_gib": sum(gpu_usage) / GiB,
                "pim_kv_and_weight_gib": sum(pim_usage) / GiB,
            })
        if config.simulation.record_pipeline_events:
            events.extend((
                {"iteration": step_count, "stage": 0, "duration_us": cost.stage0_us},
                {"iteration": step_count, "stage": 1, "duration_us": cost.stage1_us},
            ))
        finalize_preparations(world)
        for request in active:
            if request.generated == 0:
                request.first_token_us = world.clock_us
            world.append_token(request)
            output_tokens += 1
            if request.generated >= request.source.output_tokens:
                request.finished_us = world.clock_us
                completed.append(request)
                world.retire(request.source.request_id)
                completed_event = True
        if config.simulation.max_iterations is not None and step_count >= config.simulation.max_iterations:
            break

    all_requests = completed + list(world.requests.values())
    request_rows: list[dict[str, Any]] = []
    for request in all_requests:
        source = request.source
        queue_ms = (request.admitted_us - source.arrival_us) / 1e3
        elapsed_ms = ((request.finished_us or world.clock_us) - source.arrival_us) / 1e3
        tpot_ms = ((request.finished_us or world.clock_us) - request.admitted_us)
        tpot_ms /= max(request.generated, 1) * 1e3
        request_rows.append({
            "request_id": source.request_id,
            "input_tokens": source.input_tokens,
            "target_output_tokens": source.output_tokens,
            "generated_tokens": request.generated,
            "prefix_leaf_id": source.prefix_leaf_id,
            "replica": request.replica,
            "queue_ms": queue_ms,
            "e2e_from_decode_ready_ms": elapsed_ms,
            "first_token_latency_ms": ((request.first_token_us or world.clock_us)
                                       - source.arrival_us) / 1e3,
            "mean_tpot_ms": tpot_ms,
            "switches": request.switches,
            "completed": request.finished_us is not None,
        })
    finished_rows = [row for row in request_rows if row["completed"]]
    queue = [float(row["queue_ms"]) for row in finished_rows]
    e2e = [float(row["e2e_from_decode_ready_ms"]) for row in finished_rows]
    tpot = [float(row["mean_tpot_ms"]) for row in finished_rows]
    first = [float(row["first_token_latency_ms"]) for row in finished_rows]
    slo_fields = (config.simulation.slo_queue_ms, config.simulation.slo_tpot_ms,
                  config.simulation.slo_e2e_ms)
    slo_evaluated = any(value is not None for value in slo_fields)
    compliant = sum(
        (slo_fields[0] is None or row["queue_ms"] <= slo_fields[0])
        and (slo_fields[1] is None or row["mean_tpot_ms"] <= slo_fields[1])
        and (slo_fields[2] is None or row["e2e_from_decode_ready_ms"] <= slo_fields[2])
        for row in finished_rows)
    attainment = compliant / len(finished_rows) if slo_evaluated and finished_rows else None
    shared = [block for block in world.blocks.values() if block.shared]
    summary = {
        "status": "completed" if len(completed) == len(workload) else "truncated",
        "cost_backend": "layout_aware", "cost_model_revision": "layout-v4-link-calendar",
        "parameter_policy": "mechanism estimate; uncalibrated PIM dense/softmax and launch costs",
        "uncalibrated_parameters": ["PIM dense kernel", "PIM softmax kernel",
                                     "Attention KV reuse", "launch/overlap"],
        "model": config.model.name, "policy": config.simulation.policy.name,
        "policy_candidate_revision": "phase-screen-v3",
        "policy_forecast_revision": "conditional-weibull-scenarios-v1",
        "operator_mapping": "per_request_contiguous_interval",
        "fidelity": config.simulation.fidelity, "pipeline_model": "two_stage_layout",
        "workload": config.workload.type,
        "arrival_semantics": "decode_ready_after_prefill; prefill latency is not modeled",
        "attention_selector": config.simulation.attention_selector,
        "recent_window_tokens": (config.simulation.recent_blocks
                                 * config.system.pim.kv_block_tokens
                                 if config.simulation.attention_selector == "recent_blocks"
                                 else None),
        "requests": len(workload), "completed_requests": len(completed),
        "iterations": step_count, "output_tokens": output_tokens,
        "simulated_time_s": world.clock_us / 1e6,
        "throughput_tokens_s": output_tokens / max(world.clock_us / 1e6, 1e-12),
        "average_batch_size": sum_batch / max(step_count, 1),
        "peak_gpu_kv_gib": world.peak_gpu_kv_bytes / GiB,
        "peak_pim_kv_gib": world.peak_pim_kv_bytes / GiB,
        "peak_gpu_kv_with_reserve_gib": world.peak_gpu_bytes / GiB,
        "peak_pim_kv_weight_with_reserve_gib": world.peak_pim_bytes / GiB,
        "prepared_physical_gib": world.preparation_bytes / GiB,
        "preparation_transfer_s": world.preparation_us / 1e6,
        "link_busy_us": world.link_busy_us,
        "preparation_link_busy_us": (
            world.link_calendar.completed_preparation_busy_us
            + sum(max(0.0, min(world.clock_us, event.end_us) - event.start_us)
                  for event in world.link_calendar.events
                  if event.kind == "preparation")),
        "foreground_wait_us": world.foreground_wait_us,
        "gpu_kv_capacity_gib": config.system.gpu_kv_capacity_bytes(config.model) / GiB,
        "pim_kv_capacity_gib": config.system.pim.capacity_bytes / GiB,
        "layout_dense_ranksets": config.simulation.layout_dense_ranksets,
        "tail_switches": len(switches), "future_writer_actions": future_actions,
        "policy_decisions": policy_decisions,
        "decision_interval_tokens": config.simulation.policy.decision_interval,
        "planning_horizon_tokens": config.simulation.policy.planning_horizon_tokens,
        "growth_reserve_tokens": world.planning_reserve_tokens(),
        "layout_min_replan_steps": config.simulation.layout_min_replan_steps,
        "output_length_prior": {
            "family": "weibull", "mean_tokens": config.simulation.layout_survival_mean_tokens,
            "shape": config.simulation.layout_survival_shape,
            "conditioned_on_observed_generated_tokens": True,
            "scenario_count": config.simulation.layout_forecast_scenarios,
            "scenario_seed": config.simulation.policy.prediction_seed,
            "sampling": "stratified_conditional_quantiles",
        },
        "policy_search_wall_time_s": policy_wall_s,
        "candidate_screening": candidate_screening,
        "policy_search_overhead_excluded_from_simulated_time": True,
        "historical_copy_actions": copy_actions, "interval_actions": interval_actions,
        "gpu_bottleneck_phase_fraction": gpu_bottlenecks / max(2 * step_count, 1),
        "pim_bottleneck_phase_fraction": pim_bottlenecks / max(2 * step_count, 1),
        "mixed_bottleneck_phase_fraction": (
            2 * step_count - gpu_bottlenecks - pim_bottlenecks) / max(2 * step_count, 1),
        "gpu_bottleneck_layer_fraction": gpu_layers / max(2 * config.model.layers * step_count, 1),
        "pim_bottleneck_layer_fraction": pim_layers / max(2 * config.model.layers * step_count, 1),
        "tied_bottleneck_layer_fraction": (
            2 * config.model.layers * step_count - gpu_layers - pim_layers
        ) / max(2 * config.model.layers * step_count, 1),
        "gpu_critical_path_s": gpu_critical / 1e6,
        "pim_critical_path_s": pim_critical / 1e6,
        "gpu_work_s": sum_gpu / 1e6, "pim_work_s": sum_pim / 1e6,
        "boundary_s": sum_boundary / 1e6, "dense_calls_s": sum_dense / 1e6,
        "cross_group_bytes": sum_cross_bytes,
        "cross_group_us": sum_cross_us,
        "merge_calls_s": sum_merge / 1e6, "attention_flops": sum_flops,
        "pim_kv_scan_gib": sum_pim_scan / GiB,
        "gpu_kv_scan_gib": sum_gpu_scan / GiB,
        "shared_physical_blocks_remaining": len(shared),
        "shared_prefix_hits": world.prefix_hits,
        "shared_request_hit_fraction": world.prefix_hits / len(workload),
        "peak_shared_block_active_refs": world.peak_shared_refs,
        "layout_scales": {
            "pim_dense": config.simulation.layout_pim_dense_scale,
            "pim_softmax": config.simulation.layout_pim_softmax_scale,
            "pim_softmax_gelements_s_per_rankset": (
                config.simulation.layout_pim_softmax_gelements_s_per_rankset),
            "attention_bandwidth": config.simulation.layout_attention_bandwidth_scale,
            "boundary": config.simulation.layout_boundary_scale,
            "launch": config.simulation.layout_launch_scale,
            "boundary_overlap_fraction": config.simulation.layout_boundary_overlap_fraction,
            "preparation_packet_bytes": config.simulation.layout_preparation_packet_bytes,
            "weight_conversion_us_per_gib": (
                config.simulation.layout_weight_conversion_us_per_gib),
            "rankset_sync_us": config.simulation.layout_rankset_sync_us,
        },
        "latency_ms": {
            "queue_mean": fmean(queue) if queue else 0.0,
            "queue_p95": _percentile(queue, 95), "queue_p99": _percentile(queue, 99),
            "e2e_mean": fmean(e2e) if e2e else 0.0,
            "e2e_p95": _percentile(e2e, 95),
            "mean_tpot": fmean(tpot) if tpot else 0.0,
            "tpot_p95": _percentile(tpot, 95),
            "first_token_mean": fmean(first) if first else 0.0,
        },
        "slo": {
            "evaluated": slo_evaluated, "attainment": attainment,
            "goodput_rps": compliant / max(world.clock_us / 1e6, 1e-12),
            "target_met": (attainment >= config.simulation.slo_attainment_target
                           if attainment is not None else None),
        },
    }
    return SimulationResult(summary, request_rows, iterations, switches, events)
