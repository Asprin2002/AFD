#!/usr/bin/env python3
"""Paired, event-level Attention-only or F-only sweeps on the two-cut pipeline.

The two families are never combined. Results are lifecycle midpoint samples
unless stride=1; every sampled decode iteration schedules all model layers.
Completed shapes are checkpointed so a long run can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticafd.dense_offload import DenseOffloadCost, DensePlan
from elasticafd.specs import GiB
from elasticafd.state import Location, RequestView
from tools.attention_strategy import (all_pim, horizon_greedy, offline_windows,
                                      sampled_blocks, views as attention_views)
from tools.attention_event_search import (event_horizon_greedy,
                                          offline_event_windows)
from tools.run_contiguous_ablation import blocks, config
from tools.run_qkv_cut_f_sweep import METRICS, metrics, positive_ints


CONTROL = DensePlan()
F_PLANS = {
    "oproj_25": DensePlan(.25, 0, 0),
    "oproj_50": DensePlan(.5, 0, 0),
    "oproj_all": DensePlan(1, 0, 0),
    "qkv_only": DensePlan(0, 0, 1),
    "qkv_oproj_25": DensePlan(.25, 0, 1),
    "qkv_oproj_50": DensePlan(.5, 0, 1),
    "qkv_oproj_all": DensePlan(1, 0, 1),
    # These reuse the existing conditional PIM-FFN roofline. Residual/norm
    # locality is not calibrated, so they are explicitly segregated in output.
    "oproj_all_ffn_0625": DensePlan(1, .0625, 0),
    "oproj_all_ffn_125": DensePlan(1, .125, 0),
    "qkv_oproj_all_ffn_0625": DensePlan(1, .0625, 1),
    "qkv_oproj_all_ffn_125": DensePlan(1, .125, 1),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    plan: DensePlan = CONTROL
    gpu_requests: int = 0
    switch_after: int = 0
    conditional_ffn: bool = False


def candidates(family: str) -> tuple[Candidate, ...]:
    if family == "f":
        return tuple(Candidate(name, plan, conditional_ffn=plan.ffn > 0)
                     for name, plan in F_PLANS.items())
    raise ValueError("Attention uses horizon_greedy and offline_window planning")


def request_views(batch: int, input_tokens: int, generated: int,
                  candidate: Candidate) -> list[RequestView]:
    result = []
    for index in range(batch):
        selected = index < candidate.gpu_requests
        gpu_tokens = (
            max(0, generated - candidate.switch_after)
            if selected else 0
        )
        pim_tokens = input_tokens + generated - gpu_tokens
        gpu_tail = selected and generated >= candidate.switch_after
        result.append(RequestView(
            f"r{index}", (pim_tokens,),
            (gpu_tokens,) if gpu_tokens else (),
            Location.GPU if gpu_tail else Location.PIM,
        ))
    return result


def evaluate_shape(family: str, batch: int, input_tokens: int,
                   output_tokens: int, *, stride: int,
                   dense_compute_scale: float,
                   dense_bandwidth_scale: float) -> list[dict]:
    if family == "attention":
        return evaluate_attention_shape(batch, input_tokens, output_tokens,
                                        stride=stride)
    cfg = config(batch)
    options = candidates(family)
    baseline = DenseOffloadCost(cfg, (CONTROL,))
    models = {
        option.name: (
            baseline if family == "attention" else
            DenseOffloadCost(
                cfg, (CONTROL, option.plan),
                dense_compute_scale=dense_compute_scale,
                dense_bandwidth_scale=dense_bandwidth_scale,
                allow_conditional_ffn=True,
            )
        )
        for option in options
    }
    names = ("control", *(option.name for option in options))
    totals = {name: 0.0 for name in names}
    busy = {name: {metric: 0.0 for metric in METRICS} for name in names}
    viable = {name: True for name in names}

    for weight, generated in blocks(output_tokens, stride):
        control_views = request_views(batch, input_tokens, generated, Candidate("control"))
        if not baseline.fits(control_views):
            viable["control"] = False
            break
        measured = baseline.measure(control_views, CONTROL)
        totals["control"] += weight * measured.cost.total_us
        for metric, value in metrics(measured).items():
            busy["control"][metric] += weight * value
        for option in options:
            if not viable[option.name]:
                continue
            candidate_views = (
                request_views(batch, input_tokens, generated, option)
                if family == "attention" else control_views
            )
            model = models[option.name]
            if not model.fits(candidate_views):
                viable[option.name] = False
                continue
            measured = model.measure(candidate_views, option.plan)
            totals[option.name] += weight * measured.cost.total_us
            for metric, value in metrics(measured).items():
                busy[option.name][metric] += weight * value

    # Midpoint samples estimate time, but feasibility must also hold at the
    # largest resident KV footprint immediately before requests complete.
    if viable["control"]:
        final_control = request_views(
            batch, input_tokens, output_tokens, Candidate("control")
        )
        viable["control"] = baseline.fits(final_control)
        for option in options:
            if not viable[option.name]:
                continue
            final_views = (
                request_views(batch, input_tokens, output_tokens, option)
                if family == "attention" else final_control
            )
            viable[option.name] = models[option.name].fits(final_views)

    rows = []
    for option in (Candidate("control"), *options):
        name = option.name
        model = baseline if name == "control" else models[name]
        status = (
            "baseline_capacity_error" if not viable["control"] else
            "completed" if viable[name] else "candidate_capacity_error"
        )
        total = totals[name] if status == "completed" else None
        copy_us = model.initial_weight_copy_us if name != "control" else 0.0
        row = {
            "family": family,
            "batch_size": batch,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "candidate": name,
            "qkv_pim_fraction": option.plan.qkv,
            "o_proj_pim_fraction": option.plan.o_proj,
            "ffn_pim_fraction": option.plan.ffn,
            "conditional_ffn": option.conditional_ffn,
            "gpu_kv_requests": option.gpu_requests,
            "future_kv_switch_after_tokens": option.switch_after,
            "stride": stride,
            "status": status,
            "baseline_total_us": totals["control"] if viable["control"] else None,
            "candidate_total_us": total,
            "speedup_warm": totals["control"] / total if total else None,
            "speedup_cold": totals["control"] / (total + copy_us) if total else None,
            "pim_weight_copy_us": copy_us,
            "pim_persistent_weight_gib": model.pim_persistent_weight_bytes / GiB,
            "dense_compute_scale": dense_compute_scale,
            "dense_bandwidth_scale": dense_bandwidth_scale,
        }
        if total is not None:
            for metric in METRICS:
                row[f"baseline_{metric}"] = busy["control"][metric]
                row[f"candidate_{metric}"] = busy[name][metric]
            row["baseline_resource_bound_slack_us"] = totals["control"] - max(
                busy["control"][metric] for metric in METRICS[:3]
            )
            row["candidate_resource_bound_slack_us"] = total - max(
                busy[name][metric] for metric in METRICS[:3]
            )
        rows.append(row)
    return rows


def evaluate_attention_shape(batch: int, input_tokens: int,
                             output_tokens: int, *, stride: int) -> list[dict]:
    model = DenseOffloadCost(config(batch, scheduler="fixed"), (CONTROL,))
    control = all_pim(batch)
    baseline_feasible = model.fits(attention_views(control, input_tokens, output_tokens))
    if not baseline_feasible:
        return [{"family": "attention", "batch_size": batch,
                 "input_tokens": input_tokens, "output_tokens": output_tokens,
                 "candidate": "control", "status": "baseline_capacity_error",
                 "stride": stride}]
    greedy = event_horizon_greedy(model, batch, input_tokens, output_tokens)
    search = offline_event_windows(model, batch, input_tokens, output_tokens,
                                   final_stride=stride)
    offline = search.schedules
    best_window = offline[0] if offline else control
    schedules = [control, greedy, best_window]
    sample_points = sampled_blocks(output_tokens, stride, tuple(schedules))
    totals: dict[str, float] = {}
    busy: dict[str, dict[str, float]] = {}
    for schedule in schedules:
        total = 0.0
        accumulated = {metric: 0.0 for metric in METRICS}
        for weight, generated in sample_points:
            state = attention_views(schedule, input_tokens, generated)
            if not model.fits(state):
                total = float("inf")
                break
            measured = model.measure(state, CONTROL)
            total += weight * measured.cost.total_us
            for metric, value in metrics(measured).items():
                accumulated[metric] += weight * value
        totals[schedule.name] = total
        busy[schedule.name] = accumulated
    # The offline planner's bounded search is not an exact oracle. Include
    # the greedy schedule and all-PIM in its EVENT-VERIFIED candidate set,
    # so the reported best-of candidate cannot lose to either one.
    best = min((control, greedy, best_window), key=lambda item: totals[item.name])
    named = (("control", control), ("horizon_greedy", greedy),
             ("offline_window", best_window),
             ("offline_best_of", best))
    rows = []
    baseline_total = totals[control.name]
    for name, schedule in named:
        total = totals[schedule.name]
        row = {
            "family": "attention", "batch_size": batch,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "candidate": name, "selected_schedule": schedule.name,
            "strategy": "all_pim" if name == "control" else
                        "short_horizon_greedy" if name == "horizon_greedy" else
                        "bounded_offline_window" if name == "offline_window" else
                        "offline_window_best_of_greedy_and_control",
            "search_exact": False,
            "cut_schedule": ";".join(
                f"r{i}:" + ",".join(f"{token}:{location.value}" for token, location in cuts)
                for i, cuts in enumerate(schedule.cuts) if cuts
            ),
            "switch_count": sum(len(cuts) for cuts in schedule.cuts),
            "gpu_kv_requests": sum(bool(cuts) for cuts in schedule.cuts),
            "future_kv_switch_after_tokens": None,
            "qkv_pim_fraction": 0.0, "o_proj_pim_fraction": 0.0,
            "ffn_pim_fraction": 0.0, "conditional_ffn": False,
            "stride": stride, "status": "completed" if total < float("inf") else "candidate_capacity_error",
            "baseline_total_us": baseline_total, "candidate_total_us": total,
            "speedup_warm": baseline_total / total if total < float("inf") else None,
            "speedup_cold": baseline_total / total if total < float("inf") else None,
            "pim_weight_copy_us": 0.0,
            "pim_persistent_weight_gib": 0.0,
            "dense_compute_scale": 1.0, "dense_bandwidth_scale": 1.0,
            "offline_window_finalists": len(offline),
            "offline_coarse_event_evaluations": search.coarse_evaluations,
            "offline_final_event_evaluations": search.final_evaluations,
        }
        if total < float("inf"):
            for metric in METRICS:
                row[f"baseline_{metric}"] = busy[control.name][metric]
                row[f"candidate_{metric}"] = busy[schedule.name][metric]
            row["baseline_resource_bound_slack_us"] = baseline_total - max(
                busy[control.name][metric] for metric in METRICS[:3]
            )
            row["candidate_resource_bound_slack_us"] = total - max(
                busy[schedule.name][metric] for metric in METRICS[:3]
            )
        rows.append(row)
    return rows


def code_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "tools" / "attention_strategy.py",
        ROOT / "tools" / "attention_event_search.py",
        ROOT / "tools" / "run_contiguous_ablation.py",
        ROOT / "tools" / "run_qkv_cut_f_sweep.py",
        *sorted((ROOT / "elasticafd").glob("*.py")),
    )
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_checkpoint(path: Path, rows: list[dict], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row
        )))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps({**metadata, "rows": len(rows),
                    "completed_shapes": len({
                        (row["batch_size"], row["input_tokens"], row["output_tokens"])
                        for row in rows
                    })}, indent=2) + "\n", encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("attention", "f"), required=True)
    parser.add_argument("--batch-sizes", type=positive_ints,
                        default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--input-lengths", type=positive_ints,
                        default=(128, 2048, 5200, 8192))
    parser.add_argument("--output-lengths", type=positive_ints,
                        default=(256, 1024, 4096, 8192))
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--dense-compute-scale", type=float, default=1.0)
    parser.add_argument("--dense-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.stride <= 0 or args.dense_compute_scale <= 0
            or args.dense_bandwidth_scale <= 0):
        parser.error("stride and dense scales must be positive")
    shapes = [
        (batch, input_tokens, output_tokens)
        for batch in args.batch_sizes
        for input_tokens in args.input_lengths
        for output_tokens in args.output_lengths
    ]
    total_rows = (4 * len(shapes) if args.family == "attention" else
                  sum(1 + len(candidates(args.family))
                      for batch, _, output_tokens in shapes))
    if args.dry_run:
        print(f"family={args.family} shapes={len(shapes)} rows={total_rows} "
              f"stride={args.stride}")
        return 0
    metadata = {
        "code_sha256": code_sha256(),
        "family": args.family,
        "model": "gpt175b",
        "hardware": "chime_a100",
        "ranksets": 4,
        "compute_ranksets_per_layer": 1,
        "pipeline": "device_contiguous rankset_event",
        "subbatch_scheduler": "fixed request membership for Attention",
        "baseline": "same pipeline, all KV on PIM, GPU QKV/O_proj/FFN",
        "arrival": "static prefill-complete decode-ready",
        "sampling": "one full-layer event DAG per sampled decode age; stride=1 per token",
        "dense_cost": "conditional roofline; not CHIME-calibrated",
        "conditional_ffn": "requires uncalibrated PIM residual/norm locality; "
                           "exclude these rows from primary validated F ranking",
        "attention_policy": "64-token event-horizon group greedy; bounded full-lifecycle "
                            "count/start/end event search before decode; "
                            "all-PIM and greedy included in best-of; not exact oracle",
        "attention_policy_parameters": {
            "greedy_decision_interval_tokens": 64,
            "greedy_horizon_tokens": 256,
            "minimum_residence_tokens": 128,
            "greedy_max_group_cuts_per_decision": 1,
            "greedy_relative_gain_threshold": 0.002,
            "offline_coarse_event_samples": "approximately 6 plus cut boundaries",
            "offline_count_grid": "powers-of-two and quartiles; local refinement",
            "offline_early_capacity_candidates": True,
            "offline_event_finalists": 6,
            "offline_local_cut_refinement": True,
            "offline_includes_greedy_and_control": True,
        },
        "args": {
            "batch_sizes": list(args.batch_sizes),
            "input_lengths": list(args.input_lengths),
            "output_lengths": list(args.output_lengths),
            "stride": args.stride,
            "dense_compute_scale": args.dense_compute_scale,
            "dense_bandwidth_scale": args.dense_bandwidth_scale,
        },
    }
    rows: list[dict] = []
    if args.resume and args.output.exists():
        meta_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("code_sha256", "family", "args"):
            if previous.get(key) != metadata[key]:
                raise ValueError(f"cannot resume: {key} differs")
        with args.output.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    done = {
        (int(row["batch_size"]), int(row["input_tokens"]),
         int(row["output_tokens"])) for row in rows
    }
    for index, (batch, input_tokens, output_tokens) in enumerate(shapes, 1):
        shape = (batch, input_tokens, output_tokens)
        if shape in done:
            continue
        print(f"{args.family} {index}/{len(shapes)} B{batch} "
              f"I{input_tokens} O{output_tokens}", flush=True)
        rows.extend(evaluate_shape(
            args.family, *shape, stride=args.stride,
            dense_compute_scale=args.dense_compute_scale,
            dense_bandwidth_scale=args.dense_bandwidth_scale,
        ))
        write_checkpoint(args.output, rows, metadata)
    print(f"written={args.output.resolve()} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
