"""Paired private/shared/sparse mechanism runs on one layout-aware backend."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from elasticafd.config import load_config
from elasticafd.simulator import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/layout-aware-shared.yaml")
    parser.add_argument("--output", default="layout-scenarios.csv")
    args = parser.parse_args()
    base = load_config(args.config)
    if base.simulation.cost_backend != "layout_aware":
        raise ValueError("scenario comparison requires layout_aware")
    if not base.workload.requests:
        raise ValueError("scenario comparison requires inline requests")
    private_requests = tuple(
        {key: value for key, value in row.items() if key != "prefix_leaf_id"}
        for row in base.workload.requests)
    scenarios = (
        ("private_dense", replace(base.workload, requests=private_requests,
                                   prefix_blocks=()), base.simulation),
        ("shared_dense", base.workload, base.simulation),
        ("shared_recent_blocks", base.workload,
         replace(base.simulation, attention_selector="recent_blocks",
                 recent_blocks=1)),
    )
    rows = []
    for scenario, workload, simulation in scenarios:
        for policy in ("layout_fixed", "layout_greedy"):
            config = replace(base, workload=workload,
                             simulation=replace(
                                 simulation,
                                 policy=replace(simulation.policy, name=policy)))
            summary = run_simulation(config).summary
            rows.append({
                "scenario": scenario, "policy": policy,
                "model_revision": summary["cost_model_revision"],
                "calibration_status": "uncalibrated_mechanism_estimate",
                "status": summary["status"],
                "throughput_tokens_s": summary["throughput_tokens_s"],
                "mean_tpot_ms": summary["latency_ms"]["mean_tpot"],
                "pim_scan_gib": summary["pim_kv_scan_gib"],
                "shared_prefix_hits": summary["shared_prefix_hits"],
                "actions": summary["tail_switches"],
                "policy_search_wall_time_s": summary["policy_search_wall_time_s"],
            })
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
