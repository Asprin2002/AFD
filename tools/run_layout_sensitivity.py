"""Run paired, uncalibrated layout-aware parameter sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from elasticafd.config import load_config
from elasticafd.simulator import run_simulation


SCALES = (
    "layout_pim_dense_scale",
    "layout_pim_softmax_scale",
    "layout_attention_bandwidth_scale",
    "layout_boundary_scale",
    "layout_launch_scale",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/layout-aware-shared.yaml")
    parser.add_argument("--factors", default="0.5,1,2")
    parser.add_argument("--overlap-fractions", default="0,0.25,0.5")
    parser.add_argument("--preparation-packet-bytes", default="1048576,4194304,16777216")
    parser.add_argument("--output", default="layout-sensitivity.csv")
    args = parser.parse_args()
    base = load_config(args.config)
    if base.simulation.cost_backend != "layout_aware":
        raise ValueError("sensitivity requires cost_backend=layout_aware")
    factors = [float(value) for value in args.factors.split(",")]
    if any(value <= 0 for value in factors):
        raise ValueError("all factors must be positive")
    overlaps = [float(value) for value in args.overlap_fractions.split(",")]
    if any(not 0 <= value <= 1 for value in overlaps):
        raise ValueError("overlap fractions must be in [0, 1]")
    packet_bytes = [int(value) for value in args.preparation_packet_bytes.split(",")]
    if any(value <= 0 for value in packet_bytes):
        raise ValueError("preparation packet sizes must be positive")
    rows = []
    settings = [(parameter, factor) for parameter in SCALES for factor in factors]
    settings += [("layout_boundary_overlap_fraction", value) for value in overlaps]
    settings += [("layout_preparation_packet_bytes", value) for value in packet_bytes]
    for parameter, factor in settings:
        simulation = replace(base.simulation, **{parameter: factor})
        for policy in ("layout_fixed", "layout_greedy"):
            configured = replace(base, simulation=replace(
                simulation, policy=replace(simulation.policy, name=policy)))
            result = run_simulation(configured)
            rows.append({
                "model_revision": result.summary["cost_model_revision"],
                "calibration_status": "uncalibrated_mechanism_estimate",
                "parameter": parameter, "factor": factor, "policy": policy,
                "status": result.summary["status"],
                "throughput_tokens_s": result.summary["throughput_tokens_s"],
                "mean_tpot_ms": result.summary["latency_ms"]["mean_tpot"],
                "peak_gpu_kv_gib": result.summary["peak_gpu_kv_gib"],
                "peak_pim_kv_gib": result.summary["peak_pim_kv_gib"],
                "actions": result.summary["tail_switches"],
                "policy_search_wall_time_s": (
                    result.summary["policy_search_wall_time_s"]),
            })
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
