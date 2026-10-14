"""Paired AFD versus fixed-AF validation within the layout-aware backend.

The generated workloads are controlled mechanism tests, not measured traces.
Every pair changes only the policy. All reported times are uncalibrated model
estimates and exclude Python policy search time.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from elasticafd.config import load_config
from elasticafd.simulator import run_simulation


def _numbers(raw: str, cast: type) -> tuple:
    values = tuple(cast(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("parameter list cannot be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/layout-aware-shared.yaml")
    parser.add_argument("--input-lengths", default="4096,32768,100000")
    parser.add_argument("--shared-fractions", default="0,0.5")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--attention-bandwidth-scales", default="1")
    parser.add_argument("--output", default="layout-afd-validation.csv")
    args = parser.parse_args()

    base = load_config(args.config)
    if base.simulation.cost_backend != "layout_aware":
        raise ValueError("AFD validation requires the layout_aware backend")
    if not base.workload.requests:
        raise ValueError("validation requires inline requests for batch size")
    lengths = _numbers(args.input_lengths, int)
    fractions = _numbers(args.shared_fractions, float)
    bandwidth_scales = _numbers(args.attention_bandwidth_scales, float)
    if any(length <= 0 for length in lengths) or args.output_tokens <= 0:
        raise ValueError("token counts must be positive")
    if any(not 0 <= value < 1 for value in fractions):
        raise ValueError("shared fractions must be in [0, 1)")
    if any(value <= 0 for value in bandwidth_scales):
        raise ValueError("attention bandwidth scales must be positive")

    rows = []
    batch = len(base.workload.requests)
    for bandwidth_scale in bandwidth_scales:
        for length in lengths:
            for fraction in fractions:
                shared_tokens = int(length * fraction)
                prefix_blocks = (
                    ({"id": "shared-root", "parent_id": None,
                      "tokens": shared_tokens},) if shared_tokens else ()
                )
                requests = tuple(
                    {"request_id": f"r{index}", "arrival_us": 0,
                     "input_tokens": length, "output_tokens": args.output_tokens,
                     **({"prefix_leaf_id": "shared-root"}
                        if shared_tokens else {})}
                    for index in range(batch)
                )
                workload = replace(base.workload, type="closed_batch",
                                   requests=requests, prefix_blocks=prefix_blocks)
                paired = {}
                for policy in ("layout_fixed", "layout_greedy"):
                    simulation = replace(
                        base.simulation,
                        layout_attention_bandwidth_scale=bandwidth_scale,
                        layout_growth_reserve_tokens=args.output_tokens,
                        policy=replace(base.simulation.policy, name=policy),
                    )
                    summary = run_simulation(replace(
                        base, workload=workload, simulation=simulation)).summary
                    if summary["status"] != "completed":
                        raise RuntimeError(f"{policy} did not complete: {summary['status']}")
                    paired[policy] = summary
                fixed, afd = paired["layout_fixed"], paired["layout_greedy"]
                if fixed["output_tokens"] != afd["output_tokens"]:
                    raise RuntimeError("paired runs generated different token counts")
                rows.append({
                    "cost_model_revision": fixed["cost_model_revision"],
                    "calibration_status": "uncalibrated_mechanism_estimate",
                    "source_config": args.config,
                    "batch_size": batch,
                    "input_tokens": length,
                    "shared_prefix_tokens": shared_tokens,
                    "output_tokens_per_request": args.output_tokens,
                    "attention_bandwidth_scale": bandwidth_scale,
                    "fixed_throughput_tokens_s": fixed["throughput_tokens_s"],
                    "afd_throughput_tokens_s": afd["throughput_tokens_s"],
                    "afd_throughput_gain_pct": 100 * (
                        afd["throughput_tokens_s"]
                        / fixed["throughput_tokens_s"] - 1),
                    "fixed_mean_tpot_ms": fixed["latency_ms"]["mean_tpot"],
                    "afd_mean_tpot_ms": afd["latency_ms"]["mean_tpot"],
                    "fixed_p95_tpot_ms": fixed["latency_ms"]["tpot_p95"],
                    "afd_p95_tpot_ms": afd["latency_ms"]["tpot_p95"],
                    "fixed_p95_e2e_ms": fixed["latency_ms"]["e2e_p95"],
                    "afd_p95_e2e_ms": afd["latency_ms"]["e2e_p95"],
                    "afd_actions": afd["tail_switches"],
                    "afd_history_copies": afd["historical_copy_actions"],
                    "afd_interval_actions": afd["interval_actions"],
                    "afd_future_writer_actions": afd["future_writer_actions"],
                    "afd_foreground_wait_us": afd["foreground_wait_us"],
                    "afd_boundary_s": afd["boundary_s"],
                    "afd_pim_bottleneck_fraction": (
                        afd["pim_bottleneck_phase_fraction"]),
                    "afd_peak_gpu_with_reserve_gib": (
                        afd["peak_gpu_kv_with_reserve_gib"]),
                    "afd_peak_pim_with_reserve_gib": (
                        afd["peak_pim_kv_weight_with_reserve_gib"]),
                    "afd_policy_search_wall_time_s": (
                        afd["policy_search_wall_time_s"]),
                })
                print(
                    f"input={length} shared={shared_tokens} "
                    f"bandwidth={bandwidth_scale:g} "
                    f"gain={rows[-1]['afd_throughput_gain_pct']:.2f}% "
                    f"actions={afd['tail_switches']}", flush=True)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
