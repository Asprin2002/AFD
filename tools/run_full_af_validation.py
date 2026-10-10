#!/usr/bin/env python3
"""Two-stage full AF validation: analytic screen, then event-model replay.

The workload semantics are decode-ready: prompt KV exists on PIM at arrival.
The analytic stage is only a candidate screen. Reported replay results come
from the dependency/resource event graph and token-exact simulator.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elasticafd.config import project_config_from_dict
from elasticafd.dense_offload import (
    DenseOffloadCost,
    DenseOffloadSimulator,
    DensePlan,
    plan_menu,
)
from elasticafd.state import Location, RequestView
from elasticafd.workload import build_workload
from tools.run_dense_offload_pilot import pilot_config, validate


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n"
    )
    temporary.replace(path)


def code_hash() -> str:
    paths = [Path(__file__), *sorted((ROOT / "elasticafd").glob("*.py"))]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def config_for(batch_size: int, input_tokens: int, output_tokens: int):
    return pilot_config(
        {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "static_batch",
                "batch_size": batch_size,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "simulation": {"max_batch_size": batch_size},
        }
    )


def writer_counts(batch_size: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                0,
                batch_size,
                *(round(batch_size * fraction) for fraction in (0.25, 0.5, 0.75)),
            }
        )
    )


def request_fraction_views(
    batch_size: int,
    input_tokens: int,
    generated_tokens: int,
    gpu_writers: int,
) -> tuple[RequestView, ...]:
    if not 0 <= gpu_writers <= batch_size:
        raise ValueError("gpu_writers must be within the batch")
    views: list[RequestView] = []
    for index in range(batch_size):
        if index < gpu_writers:
            views.append(
                RequestView(
                    request_id=f"r{index}",
                    pim_segment_lengths=(input_tokens,),
                    gpu_segment_lengths=(generated_tokens,) if generated_tokens else (),
                    tail_location=Location.GPU,
                )
            )
        else:
            views.append(
                RequestView(
                    request_id=f"r{index}",
                    pim_segment_lengths=(input_tokens + generated_tokens,),
                    gpu_segment_lengths=(),
                    tail_location=Location.PIM,
                )
            )
    return tuple(views)


def sample_blocks(output_tokens: int, stride: int):
    for start in range(0, output_tokens, stride):
        end = min(output_tokens, start + stride)
        yield start, end, (start + end - 1) // 2


def _plan_dict(plan: DensePlan) -> dict[str, float]:
    return {"qkv": plan.qkv, "o_proj": plan.o_proj, "ffn": plan.ffn}


def screen_case(batch_size: int, input_tokens: int, output_tokens: int, stride: int):
    config = config_for(batch_size, input_tokens, output_tokens)
    plans = plan_menu("joint")
    cost = DenseOffloadCost(config, plans, allow_conditional_ffn=True)
    counts = writer_counts(batch_size)
    zero = DensePlan()
    blocks = list(sample_blocks(output_tokens, stride))
    score_cache: dict[tuple[int, int, DensePlan], float] = {}

    def score(generated: int, gpu_writers: int, plan: DensePlan) -> float:
        key = (generated, gpu_writers, plan)
        if key not in score_cache:
            views = request_fraction_views(
                batch_size, input_tokens, generated, gpu_writers
            )
            score_cache[key] = cost.resource_score(views, plan) * config.model.layers
        return score_cache[key]

    final_feasible_counts = tuple(
        count
        for count in counts
        if cost.fits(
            request_fraction_views(
                batch_size, input_tokens, output_tokens, count
            )
        )
    )
    if not final_feasible_counts:
        return {
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "status": "candidate_capacity_infeasible",
        }

    families = {
        "chime": ((0, zero),),
        "attention": tuple((count, zero) for count in final_feasible_counts),
        "qkv": tuple((0, plan) for plan in plan_menu("qkv")),
        "dense": tuple((0, plan) for plan in plans),
        "joint": tuple(
            (count, plan) for count in final_feasible_counts for plan in plans
        ),
    }
    dynamic_totals: dict[str, float] = {}
    dynamic_timelines: dict[str, list[dict[str, object]]] = {}
    static_totals: dict[str, tuple[float, int, DensePlan]] = {}
    for name, candidates in families.items():
        total = 0.0
        timeline: list[dict[str, object]] = []
        for start, end, sample in blocks:
            best_count, best_plan = min(
                candidates,
                key=lambda item: (score(sample, item[0], item[1]), item[0], item[1]),
            )
            block_cost = score(sample, best_count, best_plan)
            total += (end - start) * block_cost
            current = {
                "start": start,
                "end": end,
                "gpu_writers": best_count,
                **_plan_dict(best_plan),
            }
            if timeline and all(
                timeline[-1][field] == current[field]
                for field in ("gpu_writers", "qkv", "o_proj", "ffn")
            ):
                timeline[-1]["end"] = end
            else:
                timeline.append(current)
        dynamic_totals[name] = total
        dynamic_timelines[name] = timeline
        static_totals[name] = min(
            (
                (
                    sum(
                        (end - start) * score(sample, count, plan)
                        for start, end, sample in blocks
                    ),
                    count,
                    plan,
                )
                for count, plan in candidates
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )

    baseline = dynamic_totals["chime"]
    joint_static, static_count, static_plan = static_totals["joint"]
    return {
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": "completed",
        "screen_model": "independent-block analytic resource snapshot; candidate ranking only",
        "screen_semantics_warning": (
            "legacy *_oracle_* field names are optimistic independent-block "
            "scores, not executable history-preserving oracles"
        ),
        "evaluation_stride_tokens": stride,
        "candidate_writer_counts": list(final_feasible_counts),
        "candidate_dense_plans": len(plans),
        "pim_resident_dense_weight_gib": cost.pim_persistent_weight_bytes / 2**30,
        "initial_weight_copy_s": cost.initial_weight_copy_us / 1e6,
        "chime_analytic_s": baseline / 1e6,
        "attention_oracle_speedup": baseline / dynamic_totals["attention"],
        "qkv_oracle_speedup": baseline / dynamic_totals["qkv"],
        "dense_oracle_speedup": baseline / dynamic_totals["dense"],
        "joint_oracle_speedup": baseline / dynamic_totals["joint"],
        "joint_static_speedup": baseline / joint_static,
        "joint_vs_best_single": min(
            dynamic_totals["attention"],
            dynamic_totals["qkv"],
            dynamic_totals["dense"],
        )
        / dynamic_totals["joint"],
        "joint_dynamic_vs_joint_static": joint_static / dynamic_totals["joint"],
        "joint_static_gpu_writers": static_count,
        "joint_static_plan": _plan_dict(static_plan),
        "joint_oracle_timeline": dynamic_timelines["joint"],
        "attention_oracle_timeline": dynamic_timelines["attention"],
        "dense_oracle_timeline": dynamic_timelines["dense"],
        "qkv_oracle_timeline": dynamic_timelines["qkv"],
    }


def write_screen_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "batch_size",
        "input_tokens",
        "output_tokens",
        "status",
        "attention_oracle_speedup",
        "qkv_oracle_speedup",
        "dense_oracle_speedup",
        "joint_oracle_speedup",
        "joint_static_speedup",
        "joint_vs_best_single",
        "joint_dynamic_vs_joint_static",
        "pim_resident_dense_weight_gib",
        "initial_weight_copy_s",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def screen(args) -> None:
    started = perf_counter()
    rows: list[dict[str, object]] = []
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for batch_size in args.batch_sizes:
        for input_tokens in args.input_lengths:
            for output_tokens in args.output_lengths:
                row = screen_case(
                    batch_size, input_tokens, output_tokens, args.evaluation_stride
                )
                rows.append(row)
                atomic_json(
                    output / "screen.json",
                    {
                        "code_sha256": code_hash(),
                        "semantics": "decode-ready; prompt KV immutable on PIM; request-level future writers",
                        "rows": rows,
                        "wall_time_s": perf_counter() - started,
                    },
                )
            print(
                f"screen B={batch_size} I={input_tokens}: rows={len(rows)}, "
                f"wall={perf_counter()-started:.1f}s",
                flush=True,
            )
    write_screen_csv(output / "screen.csv", rows)


def selected_event_cases(rows: list[dict[str, object]], top_k: int):
    completed = [row for row in rows if row.get("status") == "completed"]
    selected: list[dict[str, object]] = []

    def add(candidates, key):
        if not candidates:
            return
        row = max(candidates, key=key)
        identity = (row["batch_size"], row["input_tokens"], row["output_tokens"])
        if all(
            (item["batch_size"], item["input_tokens"], item["output_tokens"])
            != identity
            for item in selected
        ):
            selected.append(row)

    # Prefer cases where the analytic union adds something beyond either side
    # alone; raw dense-offload speedup is included separately as an upper-bound
    # sanity point.
    for row in sorted(
        completed, key=lambda item: item["joint_vs_best_single"], reverse=True
    )[:top_k]:
        add([row], lambda item: item["joint_vs_best_single"])
    add(completed, lambda item: item["joint_oracle_speedup"])
    add(completed, lambda item: item["joint_dynamic_vs_joint_static"])
    # Always test the user's single-batch extrapolation and a large-batch edge.
    add(
        [row for row in completed if row["batch_size"] == 1],
        lambda item: item["joint_vs_best_single"],
    )
    add(
        [row for row in completed if row["batch_size"] == 32],
        lambda item: item["joint_vs_best_single"],
    )
    # Preserve the strongest prior Attention-split neighborhood even if the
    # optimistic resource bound ranks it poorly.
    add(
        [
            row
            for row in completed
            if row["batch_size"] == 16
            and row["input_tokens"] == 5200
            and row["output_tokens"] == 8192
        ],
        lambda item: item["joint_oracle_speedup"],
    )
    # Include a completed point closest to no predicted gain as a negative control.
    add(completed, lambda item: -abs(item["joint_oracle_speedup"] - 1.0))
    return selected


def sampled_event_case(row: dict[str, object], switch_execution_us: float):
    batch_size = int(row["batch_size"])
    input_tokens = int(row["input_tokens"])
    output_tokens = int(row["output_tokens"])
    stride = int(row["evaluation_stride_tokens"])
    config = config_for(batch_size, input_tokens, output_tokens)
    cost = DenseOffloadCost(
        config, plan_menu("joint"), allow_conditional_ffn=True
    )
    counts = tuple(int(value) for value in row["candidate_writer_counts"])
    all_counts = writer_counts(batch_size)
    zero = DensePlan()
    final_views = {
        count: request_fraction_views(
            batch_size, input_tokens, output_tokens, count
        )
        for count in all_counts
    }
    capacity_models = {
        "attention": DenseOffloadCost(
            config, plan_menu("attention"), allow_conditional_ffn=True
        ),
        "qkv": DenseOffloadCost(
            config, plan_menu("qkv"), allow_conditional_ffn=True
        ),
        "dense": DenseOffloadCost(
            config, plan_menu("f_all"), allow_conditional_ffn=True
        ),
    }
    attention_counts = tuple(
        count
        for count in all_counts
        if capacity_models["attention"].fits(final_views[count])
    )
    if not attention_counts:
        raise ValueError("Attention ablation has no capacity-feasible writer count")
    qkv_plans = (
        plan_menu("qkv") if capacity_models["qkv"].fits(final_views[0]) else ()
    )
    dense_plans = (
        plan_menu("f_all") if capacity_models["dense"].fits(final_views[0]) else ()
    )
    family_candidates = {
        "chime": ((0, zero),),
        # Single-family ablations do not reserve every PIM dense weight, so
        # their runtime paths retain the full writer-count menu. Joint actions
        # use only counts feasible after the all-open weight reservation.
        "attention": tuple((count, zero) for count in attention_counts),
        "qkv": tuple((0, plan) for plan in qkv_plans),
        "dense": tuple((0, plan) for plan in dense_plans),
        "joint": tuple((count, plan) for count in counts for plan in plan_menu("joint")),
    }
    static_action_totals = {
        action: 0.0 for action in family_candidates["joint"]
    }
    block_data: list[tuple[int, int, dict[tuple[int, DensePlan], object]]] = []

    for start, end, sample in sample_blocks(output_tokens, stride):
        duration = end - start
        candidates = {
            action
            for allowed in family_candidates.values()
            for action in allowed
        }
        measured: dict[tuple[int, DensePlan], object] = {}
        for action in candidates:
            count, plan = action
            views = request_fraction_views(
                batch_size, input_tokens, sample, count
            )
            measured[action] = cost.measure(views, plan)
        block_data.append((start, end, measured))
        for action in static_action_totals:
            static_action_totals[action] += duration * measured[action].cost.total_us

    def static_best(allowed):
        allowed = tuple(allowed)
        if not allowed:
            return None
        action = min(
            allowed,
            key=lambda candidate: (
                sum(
                    (end - start) * measured[candidate].cost.total_us
                    for start, end, measured in block_data
                ),
                candidate[0],
                candidate[1],
            ),
        )
        total = sum(
            (end - start) * measured[action].cost.total_us
            for start, end, measured in block_data
        )
        return action, total

    static_family = {
        name: static_best(allowed) for name, allowed in family_candidates.items()
    }
    baseline = static_family["chime"][1]
    joint_domain = set(family_candidates["joint"])
    same_domain_single = [
        static_best(
            action for action in family_candidates[name] if action in joint_domain
        )
        for name in ("attention", "qkv", "dense")
    ]
    best_single = min(result[1] for result in same_domain_single if result is not None)
    best_static_action, best_static_total = min(
        static_action_totals.items(),
        key=lambda item: (item[1], item[0][0], item[0][1]),
    )
    return {
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": "completed",
        "model": "sampled exhaustive static dependency/resource event replay",
        "evaluation_stride_tokens": stride,
        "switch_execution_us": switch_execution_us,
        "control_s": baseline / 1e6,
        "attention_path_speedup": baseline / static_family["attention"][1],
        "qkv_path_speedup": (
            baseline / static_family["qkv"][1]
            if static_family["qkv"] is not None
            else None
        ),
        "dense_path_speedup": (
            baseline / static_family["dense"][1]
            if static_family["dense"] is not None
            else None
        ),
        "event_candidate_speedup": baseline / best_static_total,
        "event_candidate_vs_best_single": best_single / best_static_total,
        "event_static_speedup": baseline / best_static_total,
        "event_static_gpu_writers": best_static_action[0],
        "event_static_plan": _plan_dict(best_static_action[1]),
        "event_candidate_switch_units": 0,
        "event_candidate_timeline": [
            {
                "start": 0,
                "end": output_tokens,
                "gpu_writers": best_static_action[0],
                **_plan_dict(best_static_action[1]),
            }
        ],
        "candidate_mean_gpu_busy_us": sum(
            (end - start) * measured[best_static_action].cost.gpu_busy_us
            for start, end, measured in block_data
        ) / output_tokens,
        "candidate_mean_pim_busy_us": sum(
            (end - start) * measured[best_static_action].cost.pim_core_busy_us
            for start, end, measured in block_data
        ) / output_tokens,
        "candidate_mean_link_busy_us": sum(
            (end - start) * measured[best_static_action].cost.link_busy_us
            for start, end, measured in block_data
        ) / output_tokens,
        "analytic_snapshot_joint_score": row["joint_oracle_speedup"],
        "analytic_snapshot_joint_vs_best_single": row["joint_vs_best_single"],
    }


def replay(args) -> None:
    screen_data = json.loads(args.screen.read_text())
    by_identity = {
        (int(row["batch_size"]), int(row["input_tokens"]), int(row["output_tokens"])): row
        for row in screen_data["rows"]
        if row.get("status") == "completed"
    }
    if args.case:
        identities = []
        for value in args.case:
            parts = tuple(int(part) for part in value.split(","))
            if len(parts) != 3:
                raise ValueError("--case must be B,I,O")
            identities.append(parts)
        cases = [by_identity[identity] for identity in identities]
    else:
        cases = selected_event_cases(screen_data["rows"], args.top_k)
    rows = []
    started = perf_counter()
    for case in cases:
        row = sampled_event_case(case, args.switch_execution_us)
        rows.append(row)
        atomic_json(
            args.output,
            {
                "code_sha256": code_hash(),
                "semantics": "sampled event replay; not token-exact dynamic execution",
                "rows": rows,
                "wall_time_s": perf_counter() - started,
            },
        )
        print(
            f"replay B={row['batch_size']} I={row['input_tokens']} "
            f"O={row['output_tokens']}: speedup={row['event_candidate_speedup']:.4f}, "
            f"joint/single={row['event_candidate_vs_best_single']:.4f}",
            flush=True,
        )


def monotonic_writer_views(
    batch_size: int,
    input_tokens: int,
    generated_tokens: int,
    switch_tokens: tuple[int, ...],
) -> tuple[RequestView, ...]:
    """Build legal views: each request may switch PIM->GPU at most once."""

    if len(switch_tokens) != batch_size:
        raise ValueError("one switch token is required per request")
    views = []
    for index, switch_token in enumerate(switch_tokens):
        if switch_token < 0:
            views.append(
                RequestView(
                    f"r{index}",
                    (input_tokens + generated_tokens,),
                    (),
                    Location.PIM,
                )
            )
        else:
            if switch_token > generated_tokens:
                raise ValueError("switch token cannot exceed generated history")
            gpu_tokens = generated_tokens - switch_token
            views.append(
                RequestView(
                    f"r{index}",
                    (input_tokens + switch_token,),
                    (gpu_tokens,) if gpu_tokens else (),
                    Location.GPU,
                )
            )
    return tuple(views)


def legal_beam_case(
    row: dict[str, object],
    *,
    beam_width: int,
    plan_shortlist: int,
    switch_execution_us: float,
    min_operator_residence_tokens: int,
):
    if beam_width <= 0 or plan_shortlist <= 0:
        raise ValueError("beam width and plan shortlist must be positive")
    batch_size = int(row["batch_size"])
    input_tokens = int(row["input_tokens"])
    output_tokens = int(row["output_tokens"])
    stride = int(row["evaluation_stride_tokens"])
    config = config_for(batch_size, input_tokens, output_tokens)
    plans = plan_menu("joint")
    cost = DenseOffloadCost(config, plans, allow_conditional_ffn=True)
    counts = tuple(int(value) for value in row["candidate_writer_counts"])
    blocks = list(sample_blocks(output_tokens, stride))
    static = sampled_event_case(row, switch_execution_us)
    static_total = static["control_s"] * 1e6 / static["event_static_speedup"]
    static_timeline = static["event_candidate_timeline"]

    def shortlist(views, current_plan, start):
        if start == 0:
            return plans
        ranked = sorted(
            plans, key=lambda plan: (cost.resource_score(views, plan), plan)
        )
        selected = set(ranked[:plan_shortlist]) | {current_plan, DensePlan()}
        for predicate in (
            lambda plan: plan.qkv > 0 and plan.o_proj == 0 and plan.ffn == 0,
            lambda plan: plan.qkv == 0 and plan.o_proj > 0 and plan.ffn == 0,
            lambda plan: plan.qkv == 0 and plan.o_proj == 0 and plan.ffn > 0,
            lambda plan: sum(value > 0 for value in (plan.qkv, plan.o_proj, plan.ffn)) > 1,
        ):
            family = [plan for plan in ranked if predicate(plan)]
            if family:
                selected.add(family[0])
        return tuple(sorted(selected))

    # state -> (total cost, action sequence). -1 means the request has never
    # switched, so no historical KV is reconstructed on another device.
    initial_switches = (-1,) * batch_size
    beam = {(initial_switches, DensePlan(), -10**18): (0.0, [])}
    measured_cache: dict[tuple[tuple[RequestView, ...], DensePlan], float] = {}
    expanded = 0
    for start, end, sample in blocks:
        next_states: dict[tuple, tuple[float, list[tuple[int, DensePlan]]]] = {}
        for (switches_at_start, current_plan, last_operator_switch), (
            accumulated,
            sequence,
        ) in beam.items():
            current_count = sum(token >= 0 for token in switches_at_start)
            for count in counts:
                if count < current_count:
                    continue
                new_switches = list(switches_at_start)
                for index in range(current_count, count):
                    new_switches[index] = start
                switch_tuple = tuple(new_switches)
                end_views = monotonic_writer_views(
                    batch_size, input_tokens, end, switch_tuple
                )
                if not cost.fits(end_views):
                    continue
                midpoint_views = monotonic_writer_views(
                    batch_size, input_tokens, sample, switch_tuple
                )
                can_change_operator = (
                    start - last_operator_switch >= min_operator_residence_tokens
                )
                plan_choices = (
                    shortlist(midpoint_views, current_plan, start)
                    if can_change_operator
                    else (current_plan,)
                )
                for plan in plan_choices:
                    expanded += 1
                    key = (midpoint_views, plan)
                    if key not in measured_cache:
                        measured_cache[key] = cost.measure(
                            midpoint_views, plan
                        ).cost.total_us
                    transition = (
                        count - current_count + int(plan != current_plan)
                    ) * switch_execution_us
                    total = (
                        accumulated
                        + (end - start) * measured_cache[key]
                        + transition
                    )
                    next_last = (
                        start if plan != current_plan else last_operator_switch
                    )
                    state = (switch_tuple, plan, next_last)
                    candidate = (total, [*sequence, (count, plan)])
                    old = next_states.get(state)
                    if old is None or candidate[0] < old[0]:
                        next_states[state] = candidate
        if not next_states:
            raise RuntimeError("legal beam has no capacity-feasible successor")
        beam = dict(
            sorted(
                next_states.items(),
                key=lambda item: (
                    item[1][0],
                    item[0][0],
                    item[0][1],
                    item[0][2],
                ),
            )[:beam_width]
        )

    _, (beam_total, sequence) = min(
        beam.items(), key=lambda item: (item[1][0], item[0][0], item[0][1])
    )
    if static_total <= beam_total:
        chosen_total = static_total
        timeline = static_timeline
        source = "static_fallback"
    else:
        chosen_total = beam_total
        timeline = []
        for (start, end, _), (count, plan) in zip(blocks, sequence):
            current = {
                "start": start,
                "end": end,
                "gpu_writers": count,
                **_plan_dict(plan),
            }
            if timeline and all(
                timeline[-1][field] == current[field]
                for field in ("gpu_writers", "qkv", "o_proj", "ffn")
            ):
                timeline[-1]["end"] = end
            else:
                timeline.append(current)
        source = "legal_monotonic_beam"
    baseline = static["control_s"] * 1e6
    return {
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "status": "completed",
        "model": "history-carrying sampled event beam",
        "history_semantics": "immutable; per-request PIM-to-GPU at most once",
        "evaluation_stride_tokens": stride,
        "beam_width": beam_width,
        "plan_shortlist": plan_shortlist,
        "expanded_candidates": expanded,
        "event_cost_cache_entries": len(measured_cache),
        "static_speedup": static["event_static_speedup"],
        "beam_speedup": baseline / chosen_total,
        "beam_vs_static": static_total / chosen_total,
        "chosen_source": source,
        "timeline": timeline,
    }


def beam(args) -> None:
    screen_data = json.loads(args.screen.read_text())
    by_identity = {
        (int(row["batch_size"]), int(row["input_tokens"]), int(row["output_tokens"])): row
        for row in screen_data["rows"]
        if row.get("status") == "completed"
    }
    rows = []
    started = perf_counter()
    for value in args.case:
        identity = tuple(int(part) for part in value.split(","))
        if len(identity) != 3:
            raise ValueError("--case must be B,I,O")
        row = legal_beam_case(
            by_identity[identity],
            beam_width=args.beam_width,
            plan_shortlist=args.plan_shortlist,
            switch_execution_us=args.switch_execution_us,
            min_operator_residence_tokens=args.min_operator_residence,
        )
        rows.append(row)
        atomic_json(
            args.output,
            {
                "code_sha256": code_hash(),
                "rows": rows,
                "wall_time_s": perf_counter() - started,
            },
        )
        print(
            f"beam B={row['batch_size']} I={row['input_tokens']} O={row['output_tokens']}: "
            f"speedup={row['beam_speedup']:.4f}, source={row['chosen_source']}",
            flush=True,
        )


def event(args) -> None:
    screen_data = json.loads(args.screen.read_text())
    by_identity = {
        (int(row["batch_size"]), int(row["input_tokens"]), int(row["output_tokens"])): row
        for row in screen_data["rows"]
        if row.get("status") == "completed"
    }
    if args.case:
        identities = []
        for value in args.case:
            parts = tuple(int(part) for part in value.split(","))
            if len(parts) != 3:
                raise ValueError("--case must be B,I,O")
            identities.append(parts)
        cases = [by_identity[identity] for identity in identities]
    else:
        cases = selected_event_cases(screen_data["rows"], args.top_k)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    code = code_hash()
    comparison: list[dict[str, object]] = []
    for case in cases:
        batch_size = int(case["batch_size"])
        input_tokens = int(case["input_tokens"])
        output_tokens = int(case["output_tokens"])
        config = config_for(batch_size, input_tokens, output_tokens)
        trace = build_workload(config.workload, ROOT)
        case_id = f"b{batch_size}-i{input_tokens}-o{output_tokens}"
        for mode in args.modes:
            target = output / f"{case_id}--{mode}.json"
            if target.exists():
                data = json.loads(target.read_text())
                if data["code_sha256"] != code:
                    raise RuntimeError(f"stale checkpoint; choose a new output: {target}")
                summary = data["summary"]
                print(f"skip {target.name}", flush=True)
            else:
                started = perf_counter()
                print(f"start {target.name}", flush=True)
                simulator = DenseOffloadSimulator(
                    config,
                    mode=mode,
                    allow_conditional_ffn=True,
                    kv_marginal_gain_threshold=args.kv_marginal_threshold,
                    lifetime_objective=(args.planning_objective == "lifetime"),
                    exhaustive_operator_menu=True,
                )
                result = simulator.run()
                validate(result, len(trace), trace)
                summary = result.summary
                atomic_json(
                    target,
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "code_sha256": code,
                        "screen_row": case,
                        "config": asdict(config),
                        "wall_time_s": perf_counter() - started,
                        "summary": summary,
                        "requests": result.requests,
                        "iterations": result.iterations,
                        "switches": result.switches,
                    },
                )
                print(
                    f"done {target.name}: simulated={summary['simulated_time_s']:.4f}s, "
                    f"wall={perf_counter()-started:.1f}s",
                    flush=True,
                )
            comparison.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "simulated_time_s": summary["simulated_time_s"],
                    "throughput_tokens_s": summary["throughput_tokens_s"],
                    "operator_switches": summary.get("operator_switches", 0),
                    "kv_switches": summary.get("switch_count", 0),
                }
            )
        atomic_json(output / "comparison.json", comparison)


def scripted(args) -> None:
    replay_data = json.loads(args.replay.read_text())
    by_identity = {
        (int(row["batch_size"]), int(row["input_tokens"]), int(row["output_tokens"])): row
        for row in replay_data["rows"]
        if row.get("status") == "completed"
    }
    identities = []
    for value in args.case:
        parts = tuple(int(part) for part in value.split(","))
        if len(parts) != 3:
            raise ValueError("--case must be B,I,O")
        identities.append(parts)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    code = code_hash()
    comparison: list[dict[str, object]] = []
    for identity in identities:
        row = by_identity[identity]
        batch_size, input_tokens, output_tokens = identity
        config = config_for(batch_size, input_tokens, output_tokens)
        trace = build_workload(config.workload, ROOT)
        case_id = f"b{batch_size}-i{input_tokens}-o{output_tokens}"
        for mode in args.modes:
            if mode not in ("control", "scripted_joint"):
                raise ValueError("scripted stage modes are control or scripted_joint")
            target = output / f"{case_id}--{mode}-{args.trajectory}.json"
            if target.exists():
                data = json.loads(target.read_text())
                if data["code_sha256"] != code:
                    raise RuntimeError(f"stale checkpoint; choose a new output: {target}")
                summary = data["summary"]
                print(f"skip {target.name}", flush=True)
            else:
                started = perf_counter()
                print(f"start {target.name}", flush=True)
                if mode == "scripted_joint" and args.trajectory == "static":
                    timeline = [
                        {
                            "start": 0,
                            "end": output_tokens,
                            "gpu_writers": row["event_static_gpu_writers"],
                            **row["event_static_plan"],
                        }
                    ]
                elif mode == "scripted_joint":
                    timeline = row.get("event_candidate_timeline", row.get("timeline"))
                    if timeline is None:
                        raise ValueError("input row has no executable timeline")
                else:
                    timeline = None
                simulator = DenseOffloadSimulator(
                    config,
                    mode=("joint" if timeline is not None else "control"),
                    allow_conditional_ffn=True,
                    scripted_timeline=timeline,
                )
                result = simulator.run()
                validate(result, len(trace), trace)
                summary = result.summary
                atomic_json(
                    target,
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "code_sha256": code,
                        "sampled_event_row": row,
                        "config": asdict(config),
                        "wall_time_s": perf_counter() - started,
                        "summary": summary,
                        "requests": result.requests,
                        "iterations": result.iterations,
                        "switches": result.switches,
                    },
                )
                print(
                    f"done {target.name}: simulated={summary['simulated_time_s']:.4f}s, "
                    f"wall={perf_counter()-started:.1f}s",
                    flush=True,
                )
            comparison.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "simulated_time_s": summary["simulated_time_s"],
                    "throughput_tokens_s": summary["throughput_tokens_s"],
                    "operator_switches": summary.get("operator_switches", 0),
                    "kv_switches": summary.get("switch_count", 0),
                }
            )
        atomic_json(output / "comparison.json", comparison)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    stages = root.add_subparsers(dest="stage", required=True)
    analytic = stages.add_parser("screen")
    analytic.add_argument("--output", type=Path, required=True)
    analytic.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32]
    )
    analytic.add_argument(
        "--input-lengths",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096, 5200, 8192],
    )
    analytic.add_argument(
        "--output-lengths",
        type=int,
        nargs="+",
        default=[256, 1024, 4096, 8192, 16384],
    )
    analytic.add_argument("--evaluation-stride", type=int, default=512)
    analytic.set_defaults(func=screen)

    sampled = stages.add_parser("replay")
    sampled.add_argument("--screen", type=Path, required=True)
    sampled.add_argument("--output", type=Path, required=True)
    sampled.add_argument("--top-k", type=int, default=3)
    sampled.add_argument(
        "--case",
        action="append",
        help="explicit B,I,O case; repeat this option for multiple cases",
    )
    sampled.add_argument("--switch-execution-us", type=float, default=100.0)
    sampled.set_defaults(func=replay)

    beam_parser = stages.add_parser("beam")
    beam_parser.add_argument("--screen", type=Path, required=True)
    beam_parser.add_argument("--output", type=Path, required=True)
    beam_parser.add_argument(
        "--case", action="append", required=True, help="B,I,O; repeatable"
    )
    beam_parser.add_argument("--beam-width", type=int, default=16)
    beam_parser.add_argument("--plan-shortlist", type=int, default=4)
    beam_parser.add_argument("--switch-execution-us", type=float, default=100.0)
    beam_parser.add_argument("--min-operator-residence", type=int, default=256)
    beam_parser.set_defaults(func=beam)

    scripted_parser = stages.add_parser("scripted")
    scripted_parser.add_argument("--replay", type=Path, required=True)
    scripted_parser.add_argument("--output", type=Path, required=True)
    scripted_parser.add_argument(
        "--case", action="append", required=True, help="B,I,O; repeatable"
    )
    scripted_parser.add_argument(
        "--modes", nargs="+", default=["control", "scripted_joint"]
    )
    scripted_parser.add_argument(
        "--trajectory", choices=("static", "dynamic"), default="static"
    )
    scripted_parser.set_defaults(func=scripted)

    event_parser = stages.add_parser("event")
    event_parser.add_argument("--screen", type=Path, required=True)
    event_parser.add_argument("--output", type=Path, required=True)
    event_parser.add_argument("--top-k", type=int, default=3)
    event_parser.add_argument(
        "--case",
        action="append",
        help="explicit B,I,O case; repeat this option for multiple cases",
    )
    event_parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "control",
            "attention",
            "qkv",
            "oproj",
            "ffn",
            "f_all",
            "joint",
        ],
    )
    event_parser.add_argument("--kv-marginal-threshold", type=float, default=0.002)
    event_parser.add_argument(
        "--planning-objective",
        choices=("lifetime", "bounded"),
        default="lifetime",
        help="lifetime uses known remaining output until the next completion",
    )
    event_parser.set_defaults(func=event)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
