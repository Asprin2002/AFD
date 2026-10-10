"""Small multi-resource DAG scheduler used by rankset-level models."""

from __future__ import annotations

from bisect import bisect_right, insort
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PipelineTask:
    task_id: str
    duration_us: float
    resources: tuple[str, ...]
    predecessors: tuple[str, ...] = ()
    kind: str = "other"
    layer: int = -1
    subbatch: int = -1
    priority: int = 10
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineEvent:
    task_id: str
    kind: str
    layer: int
    subbatch: int
    start_us: float
    end_us: float
    duration_us: float
    resources: tuple[str, ...]


@dataclass(frozen=True)
class PipelineSchedule:
    makespan_us: float
    events: tuple[PipelineEvent, ...]
    resource_busy_us: Mapping[str, float]


def order_two_subbatch_stages(
    tasks: Sequence[PipelineTask],
) -> list[PipelineTask]:
    """Build paired A/F pipeline stages for exactly two sub-batches.

    Phase 0 primes B0's first QKV. Thereafter the diagonal pairs are
    ``A(B0,l) || F(B1,l-1)`` and ``F(B0,l) || A(B1,l)``. F includes the
    post-A projection/FFN of layer l and the QKV for layer l+1. A includes
    its required communication and partial-result merge. Both sides of a
    phase must finish before either side enters the next phase. The event
    scheduler still arbitrates GPU, link, and PIM resources *within* a phase;
    overlap is possible, never assumed free.

    This is a deliberately explicit two-stream interpretation of CHIME's
    diagram. The paper does not publish these exact phase-barrier edges.
    """

    qkv_kinds = {"qkv", "gpu_qkv", "pim_qkv"}
    layers = sorted({
        task.layer for task in tasks
        if task.kind in qkv_kinds and task.layer >= 0
    })
    if not layers:
        return list(tasks)
    subbatches = {
        task.subbatch for task in tasks if task.layer in layers
    }
    if subbatches != {0, 1}:
        return list(tasks)
    if layers != list(range(len(layers))):
        raise ValueError("paired sub-batches require contiguous layers")

    def stage(task: PipelineTask) -> str:
        if task.task_id.startswith(f"final:s{task.subbatch}:"):
            return "f"
        prefix = f"l{task.layer}:s{task.subbatch}:"
        if not task.task_id.startswith(prefix):
            raise ValueError(f"unrecognized pipeline task: {task.task_id}")
        suffix = task.task_id[len(prefix):]
        if suffix.startswith("qkv"):
            return "qkv"
        if suffix.startswith((
            "attention", "pim_attention", "gpu_attention", "h2d", "d2h",
            "merge", "pim_merge", "gpu_partial",
        )):
            return "a"
        if suffix.startswith((
            "proj_ff", "o_proj", "gpu_o_proj", "pim_o_proj", "ffn",
            "gpu_ffn", "pim_ffn",
        )):
            return "f"
        raise ValueError(f"unclassified pipeline task: {task.task_id}")

    layer_count = len(layers)
    phase_by_id: dict[str, int] = {}
    by_phase: dict[int, list[str]] = defaultdict(list)
    for task in tasks:
        if task.layer < 0:
            continue
        if task.layer > layer_count:
            raise ValueError("pipeline task extends past the final layer")
        if task.layer == layer_count:
            if not task.task_id.startswith(f"final:s{task.subbatch}:"):
                raise ValueError(f"unexpected post-model task: {task.task_id}")
            phase = 2 * (layer_count - 1) + 2 + task.subbatch
        else:
            role = stage(task)
            phase = 2 * task.layer + task.subbatch + {
                "qkv": 0, "a": 1, "f": 2,
            }[role]
        phase_by_id[task.task_id] = phase
        by_phase[phase].append(task.task_id)

    for task in tasks:
        task_phase = phase_by_id.get(task.task_id)
        if task_phase is None:
            continue
        for predecessor in task.predecessors:
            predecessor_phase = phase_by_id.get(predecessor)
            if predecessor_phase is not None and predecessor_phase > task_phase:
                raise ValueError(
                    f"task {task.task_id} depends on a later pipeline phase"
                )

    last_phase = 2 * layer_count + 1
    if any(not by_phase[phase] for phase in range(last_phase + 1)):
        raise ValueError("paired pipeline has an empty phase")
    paired = list(tasks)
    for phase in range(last_phase + 1):
        start_id = f"phase:{phase}:start"
        done_id = f"phase:{phase}:done"
        paired.append(PipelineTask(
            start_id, 0.0, (),
            (f"phase:{phase-1}:done",) if phase else (),
            "phase_start", priority=-1,
        ))
        paired.append(PipelineTask(
            done_id, 0.0, (), tuple(by_phase[phase]),
            "phase_done", priority=20,
        ))

    # Every original task is gated directly, including prefetch transfers and
    # branch-local O_proj work. This prevents a later phase from stealing a
    # resource before the paired A/F phase has completed on both sides.
    return [
        replace(
            task,
            predecessors=tuple(dict.fromkeys((
                *task.predecessors,
                f"phase:{phase_by_id[task.task_id]}:start",
            ))),
        ) if task.task_id in phase_by_id else task
        for task in paired
    ]


def order_two_subbatch_device_segments(
    tasks: Sequence[PipelineTask],
) -> list[PipelineTask]:
    """Interleave two GPU/PIM *visits*, not Attention/FC operator families.

    For sub-batch s and layer l, GPU visit ``2*l+s`` contains the tail of
    layer l-1, optional GPU QKV, and any GPU-local Attention partial. PIM
    visit ``2*l+s+1`` contains optional local QKV, PIM Attention/merge, and
    a contiguous PIM O_proj/FFN prefix. The final GPU tail drains in visit
    ``2*L+s``. PIM QKV is legal only with PIM-only Attention/new KV, so it
    cannot create an extra PIM->GPU->PIM crossing.

    Transfers are assigned to the visit of their destination.  They still
    occupy link/rankset resources.  This bounded two-stream schedule does not
    permit arbitrary multi-layer wavefront lead or a third device crossing.
    """
    layers = sorted({
        task.layer for task in tasks
        if task.kind in {"gpu_qkv", "pim_qkv", "qkv"} and task.layer >= 0
    })
    if not layers:
        return list(tasks)
    if layers != list(range(len(layers))):
        raise ValueError("device segments require contiguous layers")
    if {task.subbatch for task in tasks if task.layer in layers} != {0, 1}:
        return list(tasks)

    gpu_prefixes = (
        "gpu_o_proj", "o_proj_join", "merge", "pim_attention_d2h",
        "pim_merge_d2h", "o_proj_d2h", "d2h", "proj_ff",
    )
    pim_prefixes = (
        "pim_attention", "gpu_partial_h2d", "pim_merge",
        "o_proj_h2d", "pim_o_proj", "h2d",
    )

    def visit(task: PipelineTask) -> int:
        if task.layer == len(layers):
            if task.task_id.startswith(f"final:s{task.subbatch}:"):
                return 2 * len(layers) + task.subbatch
            raise ValueError(f"unexpected post-model task: {task.task_id}")
        prefix = f"l{task.layer}:s{task.subbatch}:"
        if not task.task_id.startswith(prefix):
            raise ValueError(f"unrecognized device-segment task: {task.task_id}")
        suffix = task.task_id[len(prefix):]
        if task.kind == "qkv_output_d2h":
            raise ValueError("PIM QKV return would add a third device crossing")
        if task.kind in {"qkv_input_h2d", "pim_qkv"}:
            return 2 * task.layer + task.subbatch + 1
        if suffix.startswith("ffn"):
            return 2 * task.layer + task.subbatch + (
                1 if task.kind in {
                    "ffn_h2d", "pim_ffn_up", "pim_ffn_gelu", "pim_ffn_down"
                } else 2
            )
        if suffix.startswith(gpu_prefixes):
            return 2 * (task.layer + 1) + task.subbatch
        if suffix.startswith(pim_prefixes):
            return 2 * task.layer + task.subbatch + 1
        if suffix.startswith(("qkv", "gpu_attention")):
            return 2 * task.layer + task.subbatch
        raise ValueError(f"unclassified device-segment task: {task.task_id}")

    by_visit: dict[int, list[str]] = defaultdict(list)
    visit_by_id: dict[str, int] = {}
    for task in tasks:
        if task.layer < 0:
            continue
        slot = visit(task)
        by_visit[slot].append(task.task_id)
        visit_by_id[task.task_id] = slot
    for task in tasks:
        slot = visit_by_id.get(task.task_id)
        if slot is None:
            continue
        for predecessor in task.predecessors:
            prior = visit_by_id.get(predecessor)
            if prior is not None and prior > slot:
                raise ValueError(
                    f"task {task.task_id} depends on a later device visit"
                )
    last = 2 * len(layers) + 1
    # With PIM QKV, the first GPU prefix has no modeled operator: the
    # decode-ready activation enters the first PIM visit by input DMA.
    empty_initial_gpu = all(
        task.kind == "pim_qkv" for task in tasks
        if task.kind in {"gpu_qkv", "pim_qkv", "qkv"}
    )
    if any(
        not by_visit[index]
        for index in range(last + 1)
        if index != 0 or not empty_initial_gpu
    ):
        raise ValueError("device pipeline has an empty visit")
    ordered = list(tasks)
    for index in range(last + 1):
        ordered.append(PipelineTask(
            f"visit:{index}:start", 0.0, (),
            (f"visit:{index-1}:done",) if index else (),
            "visit_start", priority=-1,
        ))
        ordered.append(PipelineTask(
            f"visit:{index}:done", 0.0, (), tuple(by_visit[index]),
            "visit_done", priority=20,
        ))
    return [
        replace(task, predecessors=tuple(dict.fromkeys((
            *task.predecessors, f"visit:{visit_by_id[task.task_id]}:start",
        )))) if task.task_id in visit_by_id else task
        for task in ordered
    ]


def _earliest_common_slot(
    calendars: Mapping[str, list[tuple[float, float]]],
    resources: Sequence[str],
    release_us: float,
    duration_us: float,
) -> float:
    """Find the first interval available simultaneously on all resources."""

    start = release_us
    while True:
        conflict_end = start
        end = start + duration_us
        for resource in resources:
            calendar = calendars[resource]
            # Calendars are sorted and non-overlapping. Jump directly to the
            # interval immediately before ``start`` instead of rescanning a
            # resource's complete history for every candidate task.
            index = max(
                0,
                bisect_right(calendar, (start, float("inf"))) - 1,
            )
            for occupied_start, occupied_end in calendar[index:]:
                if occupied_end <= start:
                    continue
                if occupied_start >= end:
                    break
                conflict_end = max(conflict_end, occupied_end)
        if conflict_end <= start:
            return start
        start = conflict_end


def schedule_pipeline(tasks: Sequence[PipelineTask]) -> PipelineSchedule:
    """List-schedule a precedence DAG on unary and multi-resource tasks.

    Resource calendars, rather than one scalar availability timestamp, allow a
    newly-ready task to fill an earlier gap even when another task has already
    reserved that resource in the future.
    """

    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("pipeline task IDs must be unique")
    known_ids = set(by_id)
    for task in tasks:
        if task.duration_us < 0:
            raise ValueError(f"task {task.task_id} has negative duration")
        missing = set(task.predecessors) - known_ids
        if missing:
            raise ValueError(
                f"task {task.task_id} has unknown predecessors: {sorted(missing)}"
            )

    resources = {resource for task in tasks for resource in task.resources}
    calendars: dict[str, list[tuple[float, float]]] = {
        resource: [] for resource in resources
    }
    completion: dict[str, float] = {}
    remaining_predecessors = {
        task.task_id: len(task.predecessors) for task in tasks
    }
    successors: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for predecessor in task.predecessors:
            successors[predecessor].append(task.task_id)
    ready = {
        task_id
        for task_id, remaining in remaining_predecessors.items()
        if remaining == 0
    }
    events: list[PipelineEvent] = []

    while ready:
        candidates: list[tuple[float, int, float, str, PipelineTask]] = []
        for task_id in ready:
            task = by_id[task_id]
            release = max(
                (completion[predecessor] for predecessor in task.predecessors),
                default=0.0,
            )
            start = _earliest_common_slot(
                calendars, task.resources, release, task.duration_us
            )
            candidates.append(
                (start, task.priority, start + task.duration_us, task.task_id, task)
            )
        start, _, end, _, selected = min(candidates)
        for resource in selected.resources:
            insort(calendars[resource], (start, end))
        completion[selected.task_id] = end
        ready.remove(selected.task_id)
        for successor in successors[selected.task_id]:
            remaining_predecessors[successor] -= 1
            if remaining_predecessors[successor] == 0:
                ready.add(successor)
        events.append(
            PipelineEvent(
                task_id=selected.task_id,
                kind=selected.kind,
                layer=selected.layer,
                subbatch=selected.subbatch,
                start_us=start,
                end_us=end,
                duration_us=selected.duration_us,
                resources=selected.resources,
            )
        )

    if len(completion) != len(tasks):
        raise ValueError("pipeline task graph contains a cycle")

    events.sort(key=lambda event: (event.start_us, event.end_us, event.task_id))
    resource_busy = {
        resource: sum(end - start for start, end in intervals)
        for resource, intervals in calendars.items()
    }
    return PipelineSchedule(
        makespan_us=max(completion.values(), default=0.0),
        events=tuple(events),
        resource_busy_us=resource_busy,
    )
