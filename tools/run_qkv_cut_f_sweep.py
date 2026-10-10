#!/usr/bin/env python3
"""Event-only F placement sweep with a movable GPU/PIM first cut.

All historical and new KV remains on PIM. The PIM-QKV candidates therefore
need only one input-activation H2D and one final result D2H per layer; GPU
Attention and FFN-on-PIM are deliberately outside this causal ablation.
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

from elasticafd.dense_offload import DenseOffloadCost, DensePlan
from elasticafd.specs import GiB
from tools.run_contiguous_ablation import blocks, config, views


CONTROL = DensePlan()
PLANS = {
    "oproj_25": DensePlan(.25, 0, 0),
    "oproj_50": DensePlan(.5, 0, 0),
    "oproj_all": DensePlan(1, 0, 0),
    "qkv_only": DensePlan(0, 0, 1),
    "qkv_oproj_25": DensePlan(.25, 0, 1),
    "qkv_oproj_50": DensePlan(.5, 0, 1),
    "qkv_oproj_all": DensePlan(1, 0, 1),
}
METRICS = ("gpu_busy_us", "pim_busy_us", "link_busy_us", "h2d_bytes", "d2h_bytes")


def positive_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(value) for value in raw.split(",") if value}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def metrics(measurement):
    return {
        "gpu_busy_us": measurement.cost.gpu_busy_us,
        "pim_busy_us": measurement.cost.pim_core_busy_us,
        "link_busy_us": measurement.cost.link_busy_us,
        "h2d_bytes": measurement.h2d_bytes,
        "d2h_bytes": measurement.d2h_bytes,
    }


def evaluate(batch: int, input_tokens: int, output_tokens: int,
             *, stride: int, dense_compute_scale: float,
             dense_bandwidth_scale: float):
    cfg = config(batch)
    baseline = DenseOffloadCost(cfg, (CONTROL,))
    candidates = {
        name: DenseOffloadCost(
            cfg, (CONTROL, plan), dense_compute_scale=dense_compute_scale,
            dense_bandwidth_scale=dense_bandwidth_scale,
            allow_conditional_ffn=True,
        )
        for name, plan in PLANS.items()
    }
    totals = {name: 0.0 for name in ("control", *PLANS)}
    busy = {name: {metric: 0.0 for metric in METRICS} for name in totals}
    viable = {name: True for name in totals}
    for weight, generated in blocks(output_tokens, stride):
        state = views(batch, input_tokens, generated, output_tokens, "pim_all")
        if not baseline.fits(state):
            viable["control"] = False
            break
        measured = baseline.measure(state, CONTROL)
        totals["control"] += weight * measured.cost.total_us
        for metric, value in metrics(measured).items():
            busy["control"][metric] += weight * value
        for name, plan in PLANS.items():
            if not viable[name]:
                continue
            model = candidates[name]
            if not model.fits(state):
                viable[name] = False
                continue
            measured = model.measure(state, plan)
            totals[name] += weight * measured.cost.total_us
            for metric, value in metrics(measured).items():
                busy[name][metric] += weight * value

    rows = []
    for name, plan in (("control", CONTROL), *PLANS.items()):
        model = baseline if name == "control" else candidates[name]
        status = (
            "baseline_capacity_error" if not viable["control"] else
            "completed" if viable[name] else "candidate_capacity_error"
        )
        candidate_us = totals[name] if status == "completed" else None
        weight_copy_us = model.initial_weight_copy_us if name != "control" else 0.0
        row = {
            "batch_size": batch,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "candidate": name,
            "qkv_pim_fraction": plan.qkv,
            "o_proj_pim_fraction": plan.o_proj,
            "ffn_pim_fraction": plan.ffn,
            "dense_compute_scale": dense_compute_scale,
            "dense_bandwidth_scale": dense_bandwidth_scale,
            "subbatch_progression": "device_contiguous",
            "stride": stride,
            "status": status,
            "baseline_total_us": totals["control"] if viable["control"] else None,
            "candidate_total_us": candidate_us,
            "speedup_warm": totals["control"] / candidate_us if candidate_us else None,
            "speedup_cold": (
                totals["control"] / (candidate_us + weight_copy_us)
                if candidate_us else None
            ),
            "pim_weight_copy_us": weight_copy_us,
            "pim_persistent_weight_gib": model.pim_persistent_weight_bytes / GiB,
        }
        if candidate_us is not None:
            for metric in METRICS:
                row[f"baseline_{metric}"] = busy["control"][metric]
                row[f"candidate_{metric}"] = busy[name][metric]
            row["baseline_resource_bound_slack_us"] = totals["control"] - max(
                busy["control"][metric] for metric in METRICS[:3]
            )
            row["candidate_resource_bound_slack_us"] = candidate_us - max(
                busy[name][metric] for metric in METRICS[:3]
            )
        rows.append(row)
    return rows


def code_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "tools" / "run_contiguous_ablation.py",
        *sorted((ROOT / "elasticafd").glob("*.py")),
    )
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=positive_ints,
                        default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--input-lengths", type=positive_ints,
                        default=(128, 2048, 5200))
    parser.add_argument("--output-lengths", type=positive_ints,
                        default=(256, 1024, 4096, 8192))
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--dense-compute-scale", type=float, default=1.0)
    parser.add_argument("--dense-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.stride <= 0 or args.dense_compute_scale <= 0
            or args.dense_bandwidth_scale <= 0):
        parser.error("stride and both dense scales must be positive")
    shapes = [
        (b, i, o) for b in args.batch_sizes
        for i in args.input_lengths for o in args.output_lengths
    ]
    if args.dry_run:
        print(f"shapes={len(shapes)} candidates={len(PLANS)+1} "
              f"rows={len(shapes)*(len(PLANS)+1)}")
        return 0
    rows = []
    for index, (batch, input_tokens, output_tokens) in enumerate(shapes, 1):
        print(f"event {index}/{len(shapes)} B{batch} I{input_tokens} "
              f"O{output_tokens}", flush=True)
        rows.extend(evaluate(
            batch, input_tokens, output_tokens,
            stride=args.stride,
            dense_compute_scale=args.dense_compute_scale,
            dense_bandwidth_scale=args.dense_bandwidth_scale,
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row
        )))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps({
            "code_sha256": code_sha256(),
            "hardware": "chime_a100",
            "model": "gpt175b",
            "semantics": "static decode-ready, all KV and Attention on PIM; "
                         "one GPU-PIM-GPU visit cycle per layer",
            "placement": "QKV whole operator; O_proj row fraction; GPU FFN",
            "pim_dense": "conditional roofline, not CHIME-calibrated",
            "sampling": "midpoint-sampled output blocks, not token-exact",
            "candidate_weights": "plan-specific persistent PIM copies, GPU copies retained",
            "dense_compute_scale": args.dense_compute_scale,
            "dense_bandwidth_scale": args.dense_bandwidth_scale,
            "stride": args.stride,
        }, indent=2) + "\n", encoding="utf-8",
    )
    print(f"written={args.output.resolve()} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
