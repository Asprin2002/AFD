"""Run fixed-AF/greedy AFD pairs under 1 and 4 PIM dense ranksets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

from elasticafd.cli import _write_result
from elasticafd.config import WorkloadConfig, load_config
from elasticafd.simulator import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/layout-aware-long-context.yaml")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--cadence", type=int, default=128)
    parser.add_argument("--output-dir", default="results/layout-dense-policy-b16-i4096-o4096")
    args = parser.parse_args()
    if min(args.batch, args.input_tokens, args.output_tokens,
           args.horizon, args.cadence) <= 0:
        raise ValueError("batch, token counts, horizon and cadence must be positive")

    base = load_config(args.config)
    rows = tuple({"request_id": f"r{index}", "arrival_us": 0,
                  "input_tokens": args.input_tokens,
                  "output_tokens": args.output_tokens}
                 for index in range(args.batch))
    workload = WorkloadConfig("closed_batch", requests=rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_config": str(Path(args.config).resolve()),
                "batch": args.batch, "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens, "horizon": args.horizon,
                "cadence": args.cadence,
                "ranksets": [1, 4], "policies": ["layout_fixed", "layout_greedy"],
                "calibration": "uncalibrated mechanism estimate"}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summaries = []
    for ranksets in (1, 4):
        for policy in ("layout_fixed", "layout_greedy"):
            simulation = replace(
                base.simulation, layout_dense_ranksets=ranksets,
                max_batch_size=args.batch, layout_max_candidates=12,
                layout_min_replan_steps=4,
                policy=replace(base.simulation.policy, name=policy,
                               decision_interval=args.cadence,
                               planning_horizon_tokens=args.horizon))
            config = replace(base, workload=workload, simulation=simulation)
            arm = output / f"ranksets-{ranksets}" / policy
            arm.mkdir(parents=True, exist_ok=True)
            (arm / "config.json").write_text(
                json.dumps(asdict(config), indent=2, default=str) + "\n",
                encoding="utf-8")
            print(f"starting ranksets={ranksets} policy={policy}", flush=True)
            start = perf_counter()
            result = run_simulation(config)
            elapsed = perf_counter() - start
            _write_result(arm, result)
            (arm / "wall_time.json").write_text(
                json.dumps({"seconds": elapsed}, indent=2) + "\n",
                encoding="utf-8")
            summary = result.summary
            if (summary["status"] != "completed"
                    or summary["output_tokens"] != args.batch * args.output_tokens):
                raise RuntimeError(f"incomplete arm: ranksets={ranksets} policy={policy}")
            summaries.append({"ranksets": ranksets, "policy": policy,
                              "throughput_tokens_s": summary["throughput_tokens_s"],
                              "output_tokens": summary["output_tokens"],
                              "switches": summary["tail_switches"],
                              "interval_actions": summary["interval_actions"],
                              "cross_group_bytes": summary["cross_group_bytes"],
                              "wall_time_s": elapsed})
            (output / "pair_summary.json").write_text(
                json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
            print(f"finished ranksets={ranksets} policy={policy} "
                  f"wall={elapsed:.1f}s tokens={summary['output_tokens']} "
                  f"throughput={summary['throughput_tokens_s']:.3f} token/s",
                  flush=True)


if __name__ == "__main__":
    main()
