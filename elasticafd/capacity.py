"""Segment-aware KV capacity accounting.

Performance tiles and allocation blocks are related but distinct knobs. A
segment occupies an integral number of blocks on its owning device, so repeated
PIM->GPU->PIM switches can consume more capacity even when logical token count
is unchanged.
"""

from __future__ import annotations

from math import ceil
from typing import Iterable, Mapping

from .specs import ModelSpec, SystemSpec
from .state import Location, RequestState, RequestView, make_request_view


def allocated_tokens(lengths: Iterable[int], block_tokens: int) -> int:
    return sum(ceil(length / block_tokens) * block_tokens for length in lengths)


def view_allocated_bytes(
    view: RequestView,
    location: Location,
    model: ModelSpec,
    system: SystemSpec,
) -> int:
    if location is Location.PIM:
        lengths = view.pim_segment_lengths
        block = system.pim.kv_block_tokens
    else:
        lengths = view.gpu_segment_lengths
        block = system.gpu.kv_block_tokens
    return allocated_tokens(lengths, block) * model.kv_bytes_per_token


def views_memory_usage(
    views: Iterable[RequestView], model: ModelSpec, system: SystemSpec
) -> tuple[int, int]:
    pim_bytes = 0
    gpu_bytes = 0
    for view in views:
        pim_bytes += view_allocated_bytes(view, Location.PIM, model, system)
        gpu_bytes += view_allocated_bytes(view, Location.GPU, model, system)
    return pim_bytes, gpu_bytes


def views_gpu_memory_usage_by_replica(
    views: Iterable[RequestView],
    model: ModelSpec,
    system: SystemSpec,
) -> tuple[int, ...]:
    """Return block-rounded GPU KV use for every fixed DP replica."""

    usage = [0] * model.data_parallel
    for view in views:
        if not 0 <= view.gpu_replica < model.data_parallel:
            raise ValueError(
                f"request {view.request_id}: gpu_replica={view.gpu_replica} is outside "
                f"[0, {model.data_parallel})"
            )
        usage[view.gpu_replica] += view_allocated_bytes(
            view, Location.GPU, model, system
        )
    return tuple(usage)


def views_fit_capacity(
    views: Iterable[RequestView],
    model: ModelSpec,
    system: SystemSpec,
    *,
    pim_capacity_bytes: float,
    gpu_replica_capacity_bytes: float,
) -> bool:
    """Check shared PIM capacity and each request-affine GPU replica."""

    materialized = tuple(views)
    pim_bytes, _ = views_memory_usage(materialized, model, system)
    if pim_bytes > pim_capacity_bytes:
        return False
    return all(
        used <= gpu_replica_capacity_bytes
        for used in views_gpu_memory_usage_by_replica(materialized, model, system)
    )


def requests_memory_usage(
    requests: Iterable[RequestState], model: ModelSpec, system: SystemSpec
) -> tuple[int, int]:
    return views_memory_usage(
        (make_request_view(request) for request in requests), model, system
    )


def requests_gpu_memory_usage_by_replica(
    requests: Iterable[RequestState],
    model: ModelSpec,
    system: SystemSpec,
) -> tuple[int, ...]:
    return views_gpu_memory_usage_by_replica(
        (make_request_view(request) for request in requests), model, system
    )


def projected_append_usage(
    requests: Iterable[RequestState],
    model: ModelSpec,
    system: SystemSpec,
    overrides: Mapping[str, Location] | None = None,
) -> tuple[int, int]:
    overrides = overrides or {}
    views = projected_append_views(requests, overrides)
    return views_memory_usage(views, model, system)


def projected_append_views(
    requests: Iterable[RequestState],
    overrides: Mapping[str, Location] | None = None,
) -> tuple[RequestView, ...]:
    """Materialize the placement after every active request appends one token."""

    overrides = overrides or {}
    return tuple(
        make_request_view(
            request,
            projected_tokens=1,
            projected_location=overrides.get(request.request_id, request.tail_location),
        )
        for request in requests
    )
