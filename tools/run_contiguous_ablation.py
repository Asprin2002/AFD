#!/usr/bin/env python3
"""Small, paired Attention-only and F-only device-contiguous ablations.

Screening uses a cheap resource lower bound. Confirmation integrates sampled
decode iterations with the same bounded, two-sub-batch device-visit event DAG
for the baseline and every candidate. This is a pilot, not a token-exact run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticafd.config import project_config_from_dict
from elasticafd.dense_offload import DenseOffloadCost, DensePlan
from elasticafd.specs import GiB
from elasticafd.state import Location, RequestView


ATTENTION = (
    "gpu_half_start", "gpu_all_start",
    "gpu_half_mid", "gpu_all_mid",
)
F_PLANS = {
    "oproj_25": DensePlan(.25, 0),
    "oproj_50": DensePlan(.5, 0),
    "oproj_all": DensePlan(1, 0),
    "oproj_all_ffn_0625": DensePlan(1, .0625),
    "oproj_all_ffn_125": DensePlan(1, .125),
}
BASE = DensePlan()
ANCHORS = ((16, 5200, 8192), (32, 128, 8192), (8, 128, 4096))


def positive_csv(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in raw.split(",") if value}))
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def config(batch: int, progression: str = "device_contiguous",
           scheduler: str = "lpt"):
    return project_config_from_dict({
        "model": "gpt175b",
        "hardware": "chime_a100",
        "workload": {
            "type": "static_batch", "batch_size": batch,
            "input_tokens": 1, "output_tokens": 1,
        },
        "simulation": {
            "max_batch_size": batch,
            "pipeline_model": "rankset_event",
            "subbatch_progression": progression,
            "subbatch_scheduler": scheduler,
            "operator_mapping": "chime",
            "policy": "chime",
        },
    })


def views(
    batch: int, input_tokens: int, generated: int,
    output_tokens: int, placement: str,
):
    result = []
    gpu_count = (
        batch if placement.startswith("gpu_all")
        else (max(1, batch // 2) if placement.startswith("gpu_half") else 0)
    )
    switch_at = output_tokens // 2 if placement.endswith("_mid") else 0
    for index in range(batch):
        selected = index < gpu_count
        gpu_tokens = max(0, generated - switch_at) if selected else 0
        pim_tokens = input_tokens + generated - gpu_tokens
        gpu_tail = selected and generated >= switch_at
        result.append(RequestView(
            f"r{index}",
            (pim_tokens,),
            (gpu_tokens,) if gpu_tokens else (),
            Location.GPU if gpu_tail else Location.PIM,
        ))
    return result


def blocks(output_tokens: int, stride: int):
    for start in range(0, output_tokens, stride):
        end = min(output_tokens, start + stride)
        yield end - start, (start + end - 1) // 2


def candidate_models(cfg):
    baseline = DenseOffloadCost(cfg, (BASE,))
    f_models = {
        name: DenseOffloadCost(
            cfg, (BASE, plan), allow_conditional_ffn=True,
        )
        for name, plan in F_PLANS.items()
    }
    return baseline, f_models


def evaluate(
    batch: int, input_tokens: int, output_tokens: int,
    *, stride: int, fidelity: str,
    progression: str = "device_contiguous",
):
    cfg = config(batch, progression)
    baseline, f_models = candidate_models(cfg)
    options = [
        ("attention", placement, BASE, baseline)
        for placement in ATTENTION
    ] + [
        ("f", name, plan, f_models[name])
        for name, plan in F_PLANS.items()
    ]
    results = []
    baseline_total = 0.0
    candidate_totals = {(family, name): 0.0 for family, name, _, _ in options}
    viable = {(family, name): True for family, name, _, _ in options}
    metric_names = (
        "gpu_busy_us", "pim_busy_us", "link_busy_us", "h2d_bytes", "d2h_bytes",
    )
    baseline_metrics = {name: 0.0 for name in metric_names}
    candidate_metrics = {
        key: {name: 0.0 for name in metric_names}
        for key in candidate_totals
    }

    def add_metrics(target, measurement, weight):
        values = {
            "gpu_busy_us": measurement.cost.gpu_busy_us,
            "pim_busy_us": measurement.cost.pim_core_busy_us,
            "link_busy_us": measurement.cost.link_busy_us,
            "h2d_bytes": measurement.h2d_bytes,
            "d2h_bytes": measurement.d2h_bytes,
        }
        for name, value in values.items():
            target[name] += weight * value

    for weight, generated in blocks(output_tokens, stride):
        baseline_views = views(
            batch, input_tokens, generated, output_tokens, "pim_all",
        )
        if not baseline.fits(baseline_views):
            return [{
                "batch_size": batch, "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "subbatch_progression": progression,
                "status": "baseline_capacity_error",
            }]
        if fidelity == "analytic":
            baseline_us = (
                baseline.resource_score(baseline_views, BASE)
                * cfg.model.layers
            )
        else:
            baseline_measurement = baseline.measure(baseline_views, BASE)
            baseline_us = baseline_measurement.cost.total_us
            add_metrics(baseline_metrics, baseline_measurement, weight)
        baseline_total += weight * baseline_us
        for family, name, plan, model in options:
            key = family, name
            if not viable[key]:
                continue
            candidate_views = views(
                batch, input_tokens, generated, output_tokens,
                name if family == "attention" else "pim_all",
            )
            if not model.fits(candidate_views):
                viable[key] = False
                continue
            if fidelity == "analytic":
                candidate_us = (
                    model.resource_score(candidate_views, plan)
                    * cfg.model.layers
                )
            else:
                candidate_measurement = model.measure(candidate_views, plan)
                candidate_us = candidate_measurement.cost.total_us
                add_metrics(candidate_metrics[key], candidate_measurement, weight)
            candidate_totals[key] += weight * candidate_us
    for family, name, plan, model in options:
        key = family, name
        total = candidate_totals[key] if viable[key] else None
        row = {
            "batch_size": batch,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "family": family,
            "candidate": name,
            "subbatch_progression": progression,
            "fidelity": fidelity,
            "stride": stride,
            "status": "completed" if viable[key] else "candidate_capacity_error",
            "baseline_total_us": baseline_total,
            "candidate_total_us": total,
            "speedup_warm": baseline_total / total if total else None,
            "pim_weight_copy_us": (
                model.initial_weight_copy_us if family == "f" else 0.0
            ),
            "pim_persistent_weight_gib": (
                model.pim_persistent_weight_bytes / GiB
            ),
            "speedup_cold": (
                baseline_total / (total + model.initial_weight_copy_us)
                if total and family == "f" else
                (baseline_total / total if total else None)
            ),
            "o_proj_pim_fraction": plan.o_proj,
            "ffn_pim_fraction": plan.ffn,
            "gpu_output_kv_requests": (
                batch if name.startswith("gpu_all")
                else max(1, batch // 2) if name.startswith("gpu_half")
                else 0
            ),
            "future_kv_switch_after_tokens": (
                output_tokens // 2 if name.endswith("_mid") else 0
            ) if family == "attention" else None,
        }
        if fidelity == "event" and viable[key]:
            for metric in metric_names:
                row[f"baseline_{metric}"] = baseline_metrics[metric]
                row[f"candidate_{metric}"] = candidate_metrics[key][metric]
            row["baseline_resource_bound_slack_us"] = (
                baseline_total - max(
                    baseline_metrics["gpu_busy_us"],
                    baseline_metrics["pim_busy_us"],
                    baseline_metrics["link_busy_us"],
                )
            )
            row["candidate_resource_bound_slack_us"] = (
                total - max(
                    candidate_metrics[key]["gpu_busy_us"],
                    candidate_metrics[key]["pim_busy_us"],
                    candidate_metrics[key]["link_busy_us"],
                )
            )
        results.append(row)
    return results


def code_sha256():
    digest = hashlib.sha256()
    for source in (Path(__file__), *sorted((ROOT / "elasticafd").glob("*.py"))):
        digest.update(source.relative_to(ROOT).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no experiment rows")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps({
            "code_sha256": code_sha256(),
            "model": "gpt175b",
            "hardware": "chime_a100",
            "subbatch_progressions": sorted({
                row.get("subbatch_progression", "device_contiguous")
                for row in rows
            }),
            "semantics": (
                "device_contiguous: GPU-prefix / PIM-middle / GPU-suffix; "
                "legacy comparison uses its original operator DAG"
            ),
            "arrival": "prefill-complete, decode-ready",
            "kv": "prompt PIM; future-only write switch at start or output midpoint; no migration",
            "f_placement": "static over the sampled output lifetime",
            "analytic": "resource lower-bound screen only",
            "event": "midpoint-sampled lifecycle, not token-exact",
            "pim_dense": "conditional roofline, not calibrated CHIME capability",
            "pim_softmax_merge": "GPU-merge time proxy at scale 1; sensitivity needed",
            "baseline": "same device-contiguous event engine, all PIM KV, GPU dense",
        }, indent=2) + "\n", encoding="utf-8",
    )


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    screen = commands.add_parser("screen")
    screen.add_argument("--batch-sizes", type=positive_csv, default=(8, 16, 32))
    screen.add_argument("--input-lengths", type=positive_csv, default=(128, 5200, 8192))
    screen.add_argument("--output-lengths", type=positive_csv, default=(4096, 8192))
    screen.add_argument("--stride", type=int, default=1024)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--dry-run", action="store_true")
    event = commands.add_parser("event")
    event.add_argument("--screen", type=Path, required=True)
    event.add_argument("--top-k", type=int, default=2)
    event.add_argument("--stride", type=int, default=512)
    event.add_argument(
        "--compare-mode", choices=("strict_ordered", "free_wavefront"),
        help="also replay the same shapes/plans with one legacy progression",
    )
    event.add_argument("--output", type=Path, required=True)
    event.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stride <= 0:
        parser.error("stride must be positive")
    if args.command == "screen":
        shapes = [
            (b, i, o) for b in args.batch_sizes
            for i in args.input_lengths for o in args.output_lengths
        ]
        if args.dry_run:
            print(f"analytic shapes={len(shapes)} candidates=9")
            return 0
        rows = []
        for index, (b, i, o) in enumerate(shapes, 1):
            print(f"analytic {index}/{len(shapes)} B{b} I{i} O{o}", flush=True)
            rows.extend(evaluate(b, i, o, stride=args.stride, fidelity="analytic"))
    else:
        metadata_path = args.screen.with_suffix(args.screen.suffix + ".metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("code_sha256") != code_sha256():
                raise RuntimeError(
                    "screen was produced by different code; run screen again"
                )
        source = read_csv(args.screen)
        shapes = set()
        available = {
            (int(row["batch_size"]), int(row["input_tokens"]),
             int(row["output_tokens"]))
            for row in source if row.get("status") == "completed"
        }
        shapes.update(shape for shape in ANCHORS if shape in available)
        for family in ("attention", "f"):
            ranked = sorted(
                (row for row in source
                 if row.get("family") == family
                 and row.get("status") == "completed"),
                key=lambda row: float(row["speedup_warm"]), reverse=True,
            )
            for row in ranked[:args.top_k]:
                shapes.add((int(row["batch_size"]),
                            int(row["input_tokens"]),
                            int(row["output_tokens"])))
        if not shapes:
            raise ValueError("screen has no completed candidates")
        if args.dry_run:
            modes = 2 if args.compare_mode else 1
            print(f"event shapes={len(shapes)} candidates_per_shape=9 modes={modes}")
            return 0
        rows = []
        for index, (b, i, o) in enumerate(sorted(shapes), 1):
            for mode in (
                ("device_contiguous", args.compare_mode)
                if args.compare_mode else ("device_contiguous",)
            ):
                print(
                    f"event {index}/{len(shapes)} B{b} I{i} O{o} mode={mode}",
                    flush=True,
                )
                rows.extend(evaluate(
                    b, i, o, stride=args.stride, fidelity="event",
                    progression=mode,
                ))
    write_csv(args.output, rows)
    print(f"written={args.output.resolve()} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
