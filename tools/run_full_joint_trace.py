#!/usr/bin/env python3
"""Causal full-strategy replay on unconditioned BurstGPT service windows.

All four modes use the device-contiguous, rankset-event execution model.
The joint policy searches legal future-KV and resident-operator actions in one
decision, rather than combining the outcomes of separate ablations.
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

from elasticafd.config import project_config_from_dict
from elasticafd.dense_offload import (
    DenseOffloadCost,
    DenseOffloadSimulator,
    DensePlan,
    OnlineDensePolicy,
    plan_menu,
)
from elasticafd.policy import SwitchDecision
from elasticafd.state import Location, make_request_view
from tools.prepare_burstgpt_trace import (
    SOURCE_SHA256,
    SOURCE_URL,
    read_source,
    sha256_file,
)


MODES = ("control", "attention", "f_all", "joint")
QUANTILES = (("quiet", 0.1), ("typical", 0.5), ("busy", 0.9))


def legal_plans(mode: str) -> tuple[DensePlan, ...]:
    if mode in ("control", "attention"):
        return (DensePlan(),)
    # The two-cut path permits PIM FFN only after complete PIM O_proj.
    return tuple(
        plan for plan in plan_menu("joint")
        if not plan.ffn or plan.o_proj == 1.0
    )


class FullJointPolicy(OnlineDensePolicy):
    """Enumerate every legal joint action at each causal decision point."""

    def decide(self, requests, iteration, *, force=False):
        if self.mode == "control" or (
            not force and iteration % self.config.decision_interval
        ):
            return []
        cost = self.cost_model
        keep = tuple(request.tail_location for request in requests)
        known_lifetime = min(request.remaining_tokens for request in requests)
        horizon = max(1, known_lifetime if self.lifetime_objective else min(
            self.config.planning_horizon_tokens, known_lifetime
        ))
        plans = (
            cost.plans
            if iteration - self.last_operator_switch >= self.config.min_switch_interval
            else (cost.plan,)
        )
        candidates = []
        for targets in self._targets(requests, iteration):
            start = [
                make_request_view(request, projected_location=target)
                for request, target in zip(requests, targets)
            ]
            end = [
                make_request_view(
                    request, projected_tokens=horizon, projected_location=target
                )
                for request, target in zip(requests, targets)
            ]
            if not cost.fits(end):
                continue
            for plan in plans:
                # PIM QKV cannot return to a GPU Attention/KV branch without
                # adding a third crossing to the bounded two-cut schedule.
                if plan.qkv and any(
                    view.gpu_tokens or view.tail_location is Location.GPU
                    for view in (*start, *end)
                ):
                    continue
                before = cost.measure(start, plan).cost.total_us
                after = cost.measure(end, plan).cost.total_us
                switches = sum(a != b for a, b in zip(targets, keep))
                switches += plan != cost.plan
                score = (
                    (before + after) * horizon / 2
                    + switches * self.config.switch_execution_us
                )
                candidates.append((score, switches, plan, targets))

        current = next(
            (candidate for candidate in candidates
             if candidate[2] == cost.plan and candidate[3] == keep),
            None,
        )
        if current is None:
            raise RuntimeError("current plan exceeds projected capacity")
        best = min(
            candidates,
            key=lambda candidate: (
                candidate[0], candidate[1], candidate[2],
                tuple(target.value for target in candidate[3]),
            ),
        )
        if best[3] != keep and self.kv_marginal_gain_threshold:
            same_dense = next(
                candidate for candidate in candidates
                if candidate[2] == best[2] and candidate[3] == keep
            )
            if best[0] >= same_dense[0] * (1 - self.kv_marginal_gain_threshold):
                best = same_dense
                self.kv_marginal_guard_vetoes += 1
        if best[0] >= current[0] * (1 - self.config.relative_improvement_threshold):
            return []

        _, _, plan, targets = best
        old_plan = cost.plan
        changes = []
        for request, target in zip(requests, targets):
            old = request.tail_location
            if request.switch_tail(target, iteration):
                changes.append(SwitchDecision(
                    request.request_id, old, target, "full_joint_event",
                    current[0], best[0],
                ))
        if plan != old_plan:
            self.operator_switches += 1
            self.last_operator_switch = iteration
            self.pending_operator_cost_us += self.config.switch_execution_us
            cost.plan = plan
        self.decisions.append({
            "iteration": iteration,
            "horizon": horizon,
            "old_q": old_plan.qkv,
            "old_o": old_plan.o_proj,
            "old_f": old_plan.ffn,
            "new_q": plan.qkv,
            "new_o": plan.o_proj,
            "new_f": plan.ffn,
            "objective_before_us": current[0],
            "objective_after_us": best[0],
            "kv_switches": len(changes),
            "candidate_count": len(candidates),
        })
        return changes

    def summary_metadata(self):
        metadata = super().summary_metadata()
        metadata["search"] = "all legal resident plans x bounded future-KV targets"
        metadata["execution_semantics"] = "device_contiguous"
        return metadata


class FullJointSimulator(DenseOffloadSimulator):
    """Reuse the token-exact simulator with a legal two-cut operator menu."""

    def __init__(self, config, *, mode: str):
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        if config.simulation.subbatch_progression != "device_contiguous":
            raise ValueError("full strategy requires device_contiguous")
        super().__init__(config, mode="control", allow_conditional_ffn=True)
        self.cost_model = DenseOffloadCost(
            config, legal_plans(mode), allow_conditional_ffn=True,
            event_cost_cache_max_entries=256,
        )
        self.policy_cost_model = self.cost_model
        self.policy = FullJointPolicy(
            config.simulation.policy, self.cost_model, mode,
            kv_marginal_gain_threshold=0.002,
            lifetime_objective=True,
            exhaustive_operator_menu=True,
        )


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), *sorted((ROOT / "elasticafd").glob("*.py"))):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(source: Path, output_dir: Path, count: int) -> None:
    actual = sha256_file(source)
    if actual != SOURCE_SHA256:
        raise ValueError(f"BurstGPT source hash mismatch: {actual}")
    records, source_rows, failed_rows = read_source(source)
    hours: dict[int, list] = {}
    for record in records:
        hours.setdefault(int(record.timestamp_s // 3600), []).append(record)
    ranked = sorted(
        (len(values), hour) for hour, values in hours.items()
        if len(values) >= count
    )
    if len(ranked) < len(QUANTILES):
        raise ValueError("source has too few eligible hours")
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, quantile in QUANTILES:
        population, hour = ranked[int(quantile * (len(ranked) - 1))]
        values = hours[hour]
        start = (len(values) - count) // 2
        window = values[start:start + count]
        first = window[0].timestamp_s
        path = output_dir / f"{label}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "request_id", "arrival_s", "input_tokens", "output_tokens",
            ))
            writer.writeheader()
            for record in window:
                writer.writerow({
                    "request_id": f"burstgpt-{record.source_row}",
                    "arrival_s": record.timestamp_s - first,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                })
        atomic_json(path.with_suffix(".metadata.json"), {
            "source_url": SOURCE_URL,
            "source_sha256": actual,
            "source_rows": source_rows,
            "source_positive_rows": len(records),
            "source_failed_or_zero_rows": failed_rows,
            "selection": "central consecutive positive-token requests within an hour",
            "length_filter": "none beyond positive input/output tokens",
            "hour_rate_quantile": quantile,
            "source_hour_index": hour,
            "source_hour_positive_rows": population,
            "source_row_range": [window[0].source_row, window[-1].source_row],
            "request_count": count,
            "native_time_span_s": window[-1].timestamp_s - first,
            "time_rescaled": False,
            "source_model_counts": {
                model: sum(record.model == model for record in window)
                for model in sorted({record.model for record in window})
            },
            "mean_input_tokens": sum(r.input_tokens for r in window) / count,
            "mean_output_tokens": sum(r.output_tokens for r in window) / count,
        })
        print(f"prepared {path}: hour={hour} n={population} span={window[-1].timestamp_s-first:.1f}s")


def config_for(trace: Path, *, max_batch_size: int, segment_setup_us: float,
               subbatch_scheduler: str = "lpt"):
    config = project_config_from_dict({
        "model": "gpt175b",
        "hardware": "chime_a100",
        "workload": {"type": "trace_serving", "trace_file": str(trace.resolve())},
        "simulation": {
            "max_batch_size": max_batch_size,
            "fidelity": "tile",
            "pipeline_model": "rankset_event",
            "policy_pipeline_model": "rankset_event",
            "subbatch_scheduler": subbatch_scheduler,
            "subbatch_progression": "device_contiguous",
            "operator_mapping": "chime",
            "admission_scheduler": "fcfs",
            "allow_gpu_prompt_fallback": False,
            "record_every": 64,
            "policy": {
                "name": "chime",
                "decision_interval": 64,
                "planning_horizon_tokens": 64,
                "min_switch_interval": 64,
                "min_segment_tokens": 64,
                "max_switches_per_decision": max_batch_size,
                "relative_improvement_threshold": 0.002,
                "switch_execution_us": 100,
                "event_driven_replan": True,
            },
        },
    })
    return replace(config, system=replace(
        config.system,
        gpu=replace(config.system.gpu, segment_setup_us=segment_setup_us),
        pim=replace(config.system.pim, segment_setup_us=segment_setup_us),
    ))


def run(trace_dir: Path, output_dir: Path, modes: tuple[str, ...],
        max_batch_size: int, segment_setup_us: float) -> None:
    version = code_hash()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, _ in QUANTILES:
        trace = trace_dir / f"{label}.csv"
        if not trace.exists():
            raise FileNotFoundError(trace)
        trace_hash = sha256_file(trace)
        config = config_for(
            trace, max_batch_size=max_batch_size,
            segment_setup_us=segment_setup_us,
        )
        for mode in modes:
            if code_hash() != version:
                raise RuntimeError("source changed during paired experiment")
            target = output_dir / f"{label}--{mode}.json"
            if target.exists():
                payload = json.loads(target.read_text(encoding="utf-8"))
                if (payload["code_sha256"] != version
                        or payload["trace_sha256"] != trace_hash):
                    raise RuntimeError(f"stale checkpoint: {target}")
            else:
                start = perf_counter()
                print(f"running {label}/{mode}", flush=True)
                simulator = FullJointSimulator(config, mode=mode)
                result = simulator.run()
                if (result.summary["status"] != "completed"
                        or result.summary["completed_requests"] != len(simulator.workload)):
                    raise RuntimeError(f"incomplete simulation: {label}/{mode}")
                payload = {
                    "code_sha256": version,
                    "trace_sha256": trace_hash,
                    "mode": mode,
                    "trace": label,
                    "config": asdict(config),
                    "wall_time_s": perf_counter() - start,
                    "summary": result.summary,
                    "requests": result.requests,
                    "iterations": result.iterations,
                    "switches": result.switches,
                }
                atomic_json(target, payload)
                print(f"done {label}/{mode}: {payload['wall_time_s']:.1f}s wall", flush=True)
            summary = payload["summary"]
            rows.append({
                "trace": label,
                "mode": mode,
                "status": summary["status"],
                "simulated_time_s": summary["simulated_time_s"],
                "completed_requests": summary["completed_requests"],
                "output_tokens": summary["output_tokens"],
                "initial_weight_copy_s": summary["initial_weight_copy_s"],
                "operator_switches": summary["operator_switches"],
                "kv_switches": len(payload["switches"]),
                "peak_gpu_kv_gib": summary["peak_gpu_kv_gib"],
                "peak_pim_kv_gib": summary["peak_pim_kv_gib"],
                "wall_time_s": payload["wall_time_s"],
            })
    path = output_dir / "paired-summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output_dir / "metadata.json", {
        "code_sha256": version,
        "source": SOURCE_URL,
        "trace_selection": "three unconditioned native-time service windows",
        "execution": "device_contiguous rankset_event token-exact decode",
        "policy": "causal exhaustive legal joint action menu",
        "prefill": "completed before each arrival; prompt KV on PIM",
        "pim_dense": "conditional roofline, not calibrated CHIME kernel",
        "modes": modes,
        "max_batch_size": max_batch_size,
        "segment_setup_us": segment_setup_us,
    })
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--count", type=int, default=32)
    experiment = commands.add_parser("run")
    experiment.add_argument("--trace-dir", type=Path, required=True)
    experiment.add_argument("--output-dir", type=Path, required=True)
    experiment.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    experiment.add_argument("--max-batch-size", type=int, default=16)
    experiment.add_argument("--segment-setup-us", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.count < 2:
            parser.error("--count must be at least 2")
        prepare(args.source, args.output_dir, args.count)
    else:
        run(args.trace_dir, args.output_dir, tuple(args.modes),
            args.max_batch_size, args.segment_setup_us)


if __name__ == "__main__":
    main()
