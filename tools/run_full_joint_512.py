#!/usr/bin/env python3
"""512-request continuous decode pilot: CHIME-style baseline vs joint policy.

Both arms share the trace, hardware, admission rule and rankset-event timing.
The baseline uses the independent decode-only N=1 sub-batch formation in
ChimeDecodeBaseline; the joint policy uses its own LPT sub-batch scheduler.
This is a paper-aligned model, not the official CHIME runtime or prefill path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_burstgpt_trace import sha256_file
from tools.run_full_joint_trace import (
    MODES, FullJointSimulator, atomic_json, code_hash, config_for,
)
from elasticafd.chime_baseline import ChimeDecodeBaseline
from elasticafd.policy import build_policy
from elasticafd.simulator import Simulator
from elasticafd.state import make_request_view


def validate_trace(path: Path) -> tuple[str, int]:
    trace_hash = sha256_file(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 512:
        raise ValueError(f"pilot needs 512 requests; found {len(rows)}")
    ids = [row["request_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("request IDs must be unique")
    arrivals = [float(row["arrival_s"]) for row in rows]
    lengths = [
        (int(row["input_tokens"]), int(row["output_tokens"]))
        for row in rows
    ]
    if abs(arrivals[0]) > 1e-9 or any(
        left > right for left, right in zip(arrivals, arrivals[1:])
    ):
        raise ValueError("arrivals must start at zero and be nondecreasing")
    if any(input_tokens <= 0 or output_tokens <= 0
           for input_tokens, output_tokens in lengths):
        raise ValueError("all requests need positive input and output lengths")
    metadata_path = path.with_suffix(".metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("trace_sha256") != trace_hash:
            raise ValueError("trace metadata checksum does not match the CSV")
    return trace_hash, sum(output_tokens for _, output_tokens in lengths)


class ChimeServiceSimulator(Simulator):
    """Run the independent CHIME decode grouping in continuous batching."""

    def __init__(self, config) -> None:
        if config.simulation.subbatch_scheduler != "fixed":
            raise ValueError("CHIME baseline config must identify its own scheduler")
        super().__init__(config)
        baseline = ChimeDecodeBaseline(config.model, config.system)
        self.cost_model = baseline
        self.policy_cost_model = baseline
        self.policy = build_policy(config.simulation.policy, baseline)


def config_for_mode(config, mode: str):
    if mode == "chime_baseline":
        return replace(config, simulation=replace(
            config.simulation,
            subbatch_scheduler="fixed",
            policy=replace(config.simulation.policy, event_driven_replan=False),
        ))
    return config


def simulator_for_mode(config, mode: str):
    mode_config = config_for_mode(config, mode)
    if mode == "chime_baseline":
        return ChimeServiceSimulator(mode_config)
    return FullJointSimulator(mode_config, mode=mode)


def serialized_config(config) -> dict:
    """Match the JSON representation used for checkpoint comparison."""
    return json.loads(json.dumps(asdict(config), default=str))


def verify_chime_baseline(config) -> None:
    baseline = simulator_for_mode(config, "chime_baseline")
    if (
        not isinstance(baseline.cost_model, ChimeDecodeBaseline)
        or baseline.cost_model.operator_mapping.name != "chime"
        or baseline.cost_model.subbatch_progression != "device_contiguous"
        or baseline.policy.allows_gpu_kv
        or baseline.config.simulation.admission_scheduler != "fcfs"
    ):
        raise RuntimeError("CHIME decode baseline configuration changed")
    for count in (1, 16, 64):
        views = tuple(
            make_request_view(request.to_state())
            for request in baseline.workload[:count]
        )
        partitions = baseline.cost_model.partition_views(views)
        if (
            len(partitions) > 2
            or sum(map(len, partitions)) != count
            or any(view.gpu_tokens for part in partitions for view in part)
        ):
            raise RuntimeError(f"CHIME N=1 grouping failed at batch {count}")
    print("verified CHIME decode grouping at batch 1, 16, 64", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", type=Path,
        default=ROOT / "workloads/public/long_service_candidates/"
        "qwen-coder-capacity-512.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/full-joint-qwen-coder-512-chime-baseline",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("chime_baseline", *MODES),
        default=("chime_baseline", "joint"),
    )
    parser.add_argument(
        "--subbatch-scheduler", choices=("lpt", "alignment"), default="lpt",
        help="sub-batch formation for the policy arms; the chime_baseline arm "
             "always uses its own N=1 pairing regardless of this flag",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate the trace and configuration without simulating",
    )
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("duplicate modes are not allowed")
    trace_hash, expected_output_tokens = validate_trace(args.trace)
    config = config_for(
        args.trace, max_batch_size=64, segment_setup_us=1.0,
        subbatch_scheduler=args.subbatch_scheduler,
    )
    if (config.simulation.subbatch_progression != "device_contiguous"
            or config.simulation.pipeline_model != "rankset_event"
            or config.simulation.max_batch_size != 64):
        raise RuntimeError("pilot execution settings changed unexpectedly")
    model_hash = code_hash()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(
        f"trace={args.trace.resolve()} requests=512 "
        f"output_tokens={expected_output_tokens} batch_cap=64 "
        f"modes={','.join(args.modes)}",
        flush=True,
    )
    print(f"model_sha256={model_hash} trace_sha256={trace_hash}", flush=True)
    verify_chime_baseline(config)
    if args.dry_run:
        for mode in args.modes:
            simulator = simulator_for_mode(config, mode)
            if len(simulator.workload) != 512:
                raise RuntimeError(f"{mode}: simulator loaded wrong request count")
            print(f"validated {mode} configuration", flush=True)
        return

    for mode in args.modes:
        if code_hash() != model_hash:
            raise RuntimeError("simulator source changed during the paired run")
        target = args.output_dir / f"{mode}.json"
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            if (
                previous.get("model_code_sha256") != model_hash
                or previous.get("runner_code_sha256") != runner_hash
                or previous.get("trace_sha256") != trace_hash
                or previous.get("mode") != mode
                or previous.get("config") != serialized_config(
                    config_for_mode(config, mode)
                )
                or previous.get("summary", {}).get("status") != "completed"
            ):
                raise RuntimeError(f"stale or incomplete checkpoint: {target}")
            print(f"skip completed {mode}: {target}", flush=True)
            continue
        started = perf_counter()
        print(f"running {mode}", flush=True)
        simulator = simulator_for_mode(config, mode)
        result = simulator.run()
        if code_hash() != model_hash:
            raise RuntimeError("simulator source changed during execution")
        summary = result.summary
        summary["experiment_mode"] = mode
        summary["effective_policy"] = (
            "chime_decode_n1_predictive_pairing" if mode == "chime_baseline"
            else "fixed_all_pim_lpt" if mode == "control"
            else f"{mode}_event"
        )
        summary["comparison_scope"] = (
            "decode-only CHIME-style baseline versus joint policy"
            if mode in ("chime_baseline", "joint")
            else "joint-policy component ablation"
        )
        summary["published_chime_baseline_validated"] = False
        if mode == "chime_baseline":
            summary["subbatch_scheduler_config"] = (
                summary["subbatch_scheduler"]
            )
            summary["subbatch_scheduler"] = "chime_decode_rounds_n1"
        summary["subbatch_formation"] = (
            "decode_rounds_n1_predictive_alignment" if mode == "chime_baseline"
            else "lpt"
        )
        summary["baseline_scope"] = (
            "CHIME-style decode specialization with shared cost model; "
            "no prefill or runtime-trained RFR predictor"
        ) if mode == "chime_baseline" else None
        summary["pim_dense_action_enabled"] = mode in ("f_all", "joint")
        summary["future_gpu_kv_action_enabled"] = mode in ("attention", "joint")
        summary["control_placement_definition"] = (
            "prompt/future KV and Attention on PIM; QKV/O_proj/FFN on GPU; "
            "no placement switches"
        ) if mode in ("control", "chime_baseline") else None
        if (
            summary["status"] != "completed"
            or summary["completed_requests"] != 512
            or summary["output_tokens"] != expected_output_tokens
            or len(result.requests) != 512
            or any(
                request["generated_tokens"] != request["target_output_tokens"]
                for request in result.requests
            )
            or (mode in ("control", "chime_baseline") and (
                summary["tail_switches"] != 0
                or summary.get("operator_switches", 0) != 0
                or summary.get("initial_weight_copy_s", 0) != 0
            ))
        ):
            raise RuntimeError(f"{mode}: request or token conservation failed")
        elapsed = perf_counter() - started
        atomic_json(target, {
            "model_code_sha256": model_hash,
            "runner_code_sha256": runner_hash,
            "trace_sha256": trace_hash,
            "mode": mode,
            "config": serialized_config(simulator.config),
            "wall_time_s": elapsed,
            "summary": summary,
            "requests": result.requests,
            "iterations": result.iterations,
            "switches": result.switches,
        })
        latency = summary["latency_ms"]
        print(
            f"completed {mode}: wall={elapsed:.1f}s "
            f"mean_e2e={latency['e2e_mean']:.1f}ms "
            f"mean_tpot={latency['mean_tpot']:.2f}ms "
            f"avg_batch={summary['average_batch_size']:.2f} "
            f"kv_switches={summary['tail_switches']} "
            f"operator_switches={summary.get('operator_switches', 0)} "
            f"result={target}",
            flush=True,
        )

    pair = [args.output_dir / f"{mode}.json" for mode in ("chime_baseline", "joint")]
    if all(path.exists() for path in pair):
        baseline_result, joint = (
            json.loads(path.read_text(encoding="utf-8")) for path in pair
        )
        if all(
            item.get("model_code_sha256") == model_hash
            and item.get("runner_code_sha256") == runner_hash
            and item.get("trace_sha256") == trace_hash
            and item.get("mode") == expected_mode
            and item.get("config") == serialized_config(
                config_for_mode(config, expected_mode)
            )
            and item.get("summary", {}).get("status") == "completed"
            for item, expected_mode in zip(
                (baseline_result, joint), ("chime_baseline", "joint")
            )
        ):
            baseline = baseline_result["summary"]
            treatment = joint["summary"]
            before = baseline["latency_ms"]
            after = treatment["latency_ms"]
            gain = lambda old, new: 100 * (old - new) / old if old else None
            comparison = {
                "trace_sha256": trace_hash,
                "model_code_sha256": model_hash,
                "note": (
                    "CHIME-style N=1 predictive sub-batch grouping versus joint "
                    "policy; decode-ready continuous replay; no prefill or "
                    "runtime-trained RFR predictor; warm timing with initial "
                    "PIM weight copy separate"
                ),
                "mean_e2e_ms": {
                    "chime_baseline": before["e2e_mean"],
                    "joint": after["e2e_mean"],
                    "improvement_percent": gain(
                        before["e2e_mean"], after["e2e_mean"]
                    ),
                },
                "p95_e2e_ms": {
                    "chime_baseline": before["e2e_p95"],
                    "joint": after["e2e_p95"],
                    "improvement_percent": gain(
                        before["e2e_p95"], after["e2e_p95"]
                    ),
                },
                "mean_tpot_ms": {
                    "chime_baseline": before["mean_tpot"],
                    "joint": after["mean_tpot"],
                    "improvement_percent": gain(
                        before["mean_tpot"], after["mean_tpot"]
                    ),
                },
                "average_batch_size": {
                    "chime_baseline": baseline["average_batch_size"],
                    "joint": treatment["average_batch_size"],
                },
                "throughput_tokens_s": {
                    "chime_baseline": baseline["throughput_tokens_s"],
                    "joint": treatment["throughput_tokens_s"],
                },
                "p95_queue_ms": {
                    "chime_baseline": before["queue_p95"],
                    "joint": after["queue_p95"],
                },
                "path_utilization": {
                    mode: {
                        "gpu": item["gpu_path_utilization"],
                        "pim": item["pim_core_path_utilization"],
                        "link": item["link_path_utilization"],
                    }
                    for mode, item in (("chime_baseline", baseline),
                                       ("joint", treatment))
                },
                "joint_kv_switches": treatment["tail_switches"],
                "joint_operator_switches": treatment["operator_switches"],
                "joint_initial_weight_copy_s": treatment["initial_weight_copy_s"],
            }
            destination = args.output_dir / "paired-summary.json"
            atomic_json(destination, comparison)
            print(f"paired summary: {destination}", flush=True)


if __name__ == "__main__":
    main()
