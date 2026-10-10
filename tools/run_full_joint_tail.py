#!/usr/bin/env python3
"""Paired token-exact full-strategy replay of the BurstGPT tail stress window."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--segment-setup-us", type=float, default=1.0)
    args = parser.parse_args()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model_hash = code_hash()
    trace_hash = sha256_file(args.trace)
    config = config_for(
        args.trace,
        max_batch_size=args.max_batch_size,
        segment_setup_us=args.segment_setup_us,
    )
    for mode in args.modes:
        if code_hash() != model_hash:
            raise RuntimeError("source changed during paired experiment")
        target = args.output_dir / f"tail-p99--{mode}.json"
        if target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
            if (payload["model_code_sha256"] != model_hash
                    or payload["runner_code_sha256"] != runner_hash
                    or payload["trace_sha256"] != trace_hash):
                raise RuntimeError(f"stale checkpoint: {target}")
            print(f"skip {mode}", flush=True)
            continue
        started = perf_counter()
        print(f"running tail-p99/{mode}", flush=True)
        simulator = FullJointSimulator(config, mode=mode)
        result = simulator.run()
        if (result.summary["status"] != "completed"
                or result.summary["completed_requests"] != len(simulator.workload)):
            raise RuntimeError(f"incomplete simulation: {mode}")
        atomic_json(target, {
            "model_code_sha256": model_hash,
            "runner_code_sha256": runner_hash,
            "trace_sha256": trace_hash,
            "mode": mode,
            "config": asdict(config),
            "wall_time_s": perf_counter() - started,
            "summary": result.summary,
            "requests": result.requests,
            "iterations": result.iterations,
            "switches": result.switches,
        })
        print(f"done tail-p99/{mode}: {perf_counter()-started:.1f}s wall", flush=True)


if __name__ == "__main__":
    main()
