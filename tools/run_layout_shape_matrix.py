"""Paired fixed-AF/AFD checks on the historically informative decode shapes.

The old results select only input/output shapes. Every comparison here uses
the same opt-in layout-aware backend and records uncalibrated model estimates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from elasticafd.cli import _write_result
from elasticafd.config import load_config
from elasticafd.simulator import run_simulation


SHAPES = (
    ("b4-i5200-o4096", 4, 5200, 4096),
    ("b8-i8192-o1024", 8, 8192, 1024),
    ("b8-i8192-o4096", 8, 8192, 4096),
    ("b8-i8192-o8192", 8, 8192, 8192),
    ("b16-i5200-o256", 16, 5200, 256),
    ("b16-i5200-o1024", 16, 5200, 1024),
    ("b16-i5200-o4096", 16, 5200, 4096),
    ("b16-i5200-o8192", 16, 5200, 8192),
    ("b32-i2048-o1024", 32, 2048, 1024),
    ("b32-i2048-o4096", 32, 2048, 4096),
    ("b32-i2048-o8192", 32, 2048, 8192),
    ("b2-i128-o256", 2, 128, 256),
    ("b2-i128-o4096", 2, 128, 4096),
)


def policy_window(batch: int, input_tokens: int) -> tuple[int, int]:
    """Predeclared H/Delta rule using observable shape, never target output."""
    if input_tokens <= 256:
        horizon = 64
    elif input_tokens <= 4096 or batch < 8:
        horizon = 128
    else:
        horizon = 256
    cadence = min(512, max(128, 64 * batch))
    return horizon, cadence


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/layout-aware-long-context.yaml")
    parser.add_argument("--output-dir", default="results/layout-shape-matrix-v2")
    parser.add_argument("--cases", default="all",
                        help="Comma-separated case IDs, or all")
    parser.add_argument("--max-iterations", type=int,
                        help="For runtime/capacity preflight only; not a benefit result")
    args = parser.parse_args()
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")

    base = load_config(args.config)
    if base.simulation.cost_backend != "layout_aware":
        raise ValueError("shape matrix requires layout_aware")
    selected = set(args.cases.split(",")) if args.cases != "all" else None
    if selected is not None:
        unknown = selected - {shape[0] for shape in SHAPES}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "source_config": str(Path(args.config).resolve()),
        "layout_code_sha256": _file_sha256(root / "elasticafd/layout_aware.py"),
        "runner_sha256": _file_sha256(Path(__file__)),
        "historical_results_role": "shape selection only",
        "calibration_status": "uncalibrated mechanism estimate",
        "window_rule": "H=64 for prompt<=256; H=128 for prompt<=4096 or B<8; otherwise H=256; Delta=min(512,max(128,64*B))",
        "event_driven_replan": True,
        "window_uses_target_output_tokens": False,
        "max_iterations": args.max_iterations,
        "cases": [shape[0] for shape in SHAPES
                  if selected is None or shape[0] in selected],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows: list[dict] = []
    for case_id, batch, input_tokens, output_tokens in SHAPES:
        if selected is not None and case_id not in selected:
            continue
        case_dir = output / case_id
        case_dir.mkdir(exist_ok=True)
        horizon, cadence = policy_window(batch, input_tokens)
        workload = replace(
            base.workload, type="closed_batch", prefix_blocks=(),
            requests=tuple({"request_id": f"r{index:02d}", "arrival_us": 0,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens}
                           for index in range(batch)))
        pair: dict[str, dict] = {}
        for policy_name in ("layout_fixed", "layout_greedy"):
            simulation = replace(
                base.simulation, max_batch_size=batch,
                max_iterations=args.max_iterations,
                policy=replace(base.simulation.policy, name=policy_name,
                               planning_horizon_tokens=horizon,
                               decision_interval=cadence,
                               event_driven_replan=True))
            config = replace(base, workload=workload, simulation=simulation)
            policy_dir = case_dir / policy_name
            policy_dir.mkdir(exist_ok=True)
            (policy_dir / "config.json").write_text(
                json.dumps(asdict(config), indent=2, ensure_ascii=False,
                           default=str) + "\n", encoding="utf-8")
            start = perf_counter()
            try:
                result = run_simulation(config)
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)}
                (policy_dir / "failure.json").write_text(
                    json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                print(f"{case_id} {policy_name}: {failure}", flush=True)
                raise
            elapsed = perf_counter() - start
            _write_result(policy_dir, result)
            pair[policy_name] = result.summary
            (policy_dir / "wall_time.json").write_text(
                json.dumps({"wall_time_s": elapsed}, indent=2) + "\n",
                encoding="utf-8")
            print(f"{case_id} {policy_name}: {result.summary['status']} "
                  f"{elapsed:.1f}s wall, {result.summary['output_tokens']} tokens, "
                  f"{result.summary['tail_switches']} actions", flush=True)
        fixed, afd = pair["layout_fixed"], pair["layout_greedy"]
        if fixed["output_tokens"] != afd["output_tokens"]:
            raise RuntimeError(f"paired output token mismatch: {case_id}")
        if args.max_iterations is None and (
                fixed["status"] != "completed" or afd["status"] != "completed"):
            raise RuntimeError(f"full run incomplete: {case_id}")
        rows.append({
            "case_id": case_id, "batch_size": batch,
            "cost_model_revision": afd["cost_model_revision"],
            "input_tokens": input_tokens, "output_tokens_per_request": output_tokens,
            "preflight": args.max_iterations is not None,
            "fixed_status": fixed["status"], "afd_status": afd["status"],
            "fixed_tokens": fixed["output_tokens"], "afd_tokens": afd["output_tokens"],
            "fixed_throughput_tokens_s": fixed["throughput_tokens_s"],
            "afd_throughput_tokens_s": afd["throughput_tokens_s"],
            "gain_pct": 100 * (afd["throughput_tokens_s"] /
                               fixed["throughput_tokens_s"] - 1),
            "fixed_mean_tpot_ms": fixed["latency_ms"]["mean_tpot"],
            "afd_mean_tpot_ms": afd["latency_ms"]["mean_tpot"],
            "fixed_p95_tpot_ms": fixed["latency_ms"]["tpot_p95"],
            "afd_p95_tpot_ms": afd["latency_ms"]["tpot_p95"],
            "fixed_p95_e2e_ms": fixed["latency_ms"]["e2e_p95"],
            "afd_p95_e2e_ms": afd["latency_ms"]["e2e_p95"],
            "afd_actions": afd["tail_switches"],
            "afd_future_actions": afd["future_writer_actions"],
            "afd_copy_actions": afd["historical_copy_actions"],
            "afd_interval_actions": afd["interval_actions"],
            "afd_foreground_wait_us": afd["foreground_wait_us"],
            "afd_boundary_s": afd["boundary_s"],
            "afd_pim_bottleneck_fraction": afd["pim_bottleneck_phase_fraction"],
            "afd_peak_gpu_with_reserve_gib": afd["peak_gpu_kv_with_reserve_gib"],
            "afd_peak_pim_with_reserve_gib": afd["peak_pim_kv_weight_with_reserve_gib"],
            "afd_search_wall_time_s": afd["policy_search_wall_time_s"],
            "horizon": horizon,
            "decision_interval": cadence,
            "candidate_budget": base.simulation.layout_max_candidates,
            "candidate_revision": afd["policy_candidate_revision"],
            "survival_mean_tokens": simulation.layout_survival_mean_tokens,
            "survival_shape": simulation.layout_survival_shape,
        })
        _save_rows(output / "paired.csv", rows)
    print(output / "paired.csv", flush=True)


if __name__ == "__main__":
    main()
