"""Measure online layout decisions without changing their search settings.

Use an original online-pair manifest. Run one process at a time, and use
--reference to compare all decisions and model outputs with an earlier run.
The nested timings are inclusive; self_s excludes timed child calls.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from functools import wraps
from hashlib import sha256
import json
import inspect
from pathlib import Path
import pickle
import statistics
import sys
from time import perf_counter

if __package__:
    from .layout_search_comparison import compare_screening
else:
    from layout_search_comparison import compare_screening


class SearchTimings:
    def __init__(self, module, output: Path, capture: bool, *, detail_cost_functions=False):
        self.module = module
        self.output = output
        self.capture = capture
        self.detail_cost_functions = detail_cost_functions
        self.current = None
        self.stack = []
        self.decisions = []
        self.originals = []

    def wrap(self, owner, name):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        @wraps(original)
        def measured(*args, **kwargs):
            if self.current is None:
                return original(*args, **kwargs)
            if name in ("_reflow_preparations", "_reserve_link"):
                calendar = args[0]
                self.current["max_link_events"] = max(
                    self.current["max_link_events"], len(calendar.events))
                self.current["max_preparation_jobs"] = max(
                    self.current["max_preparation_jobs"], len(calendar.jobs))
            elif name == "_forecast_total":
                world = args[0]
                self.current["max_physical_blocks"] = max(
                    self.current["max_physical_blocks"], len(world.blocks))
            frame = [perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                duration = perf_counter() - frame[0]
                self.stack.pop()
                item = self.current["timings"][name]
                item["calls"] += 1
                item["total_s"] += duration
                item["self_s"] += duration - frame[1]
                if self.stack:
                    self.stack[-1][1] += duration

        setattr(owner, name, measured)

    def install(self):
        module = self.module
        for name in ("deepcopy", "generate_candidates", "apply_action",
                     "_forecast_total", "evaluate_step", "_stage_cost",
                     "_book_boundary", "_reflow_preparations", "_reserve_link",
                     "finalize_preparations"):
            self.wrap(module, name)
        if self.detail_cost_functions:
            for name in ("_stage_cache_key", "_compute_stage_cost", "_compute_linear_us",
                         "_compute_linear_shard_us", "_compute_attention_us"):
                if hasattr(module, name):
                    self.wrap(module, name)
        for name in ("check_capacity", "kv_usage", "append_token", "retire"):
            self.wrap(module.LayoutWorld, name)
        original = module.choose_action
        supports_reuse_statistics = "reuse_statistics" in inspect.signature(original).parameters
        self.originals.append((module, "choose_action", original))

        @wraps(original)
        def choose(world, **kwargs):
            index = len(self.decisions)
            if self.capture:
                with (self.output / f"world-{index:03}.pickle").open("wb") as handle:
                    pickle.dump(world, handle, protocol=pickle.HIGHEST_PROTOCOL)
            row = {
                "decision": index, "clock_us": world.clock_us,
                "active_requests": len(world.requests),
                "generated_tokens": {key: req.generated for key, req in world.requests.items()},
                "physical_blocks": len(world.blocks),
                "max_physical_blocks": len(world.blocks),
                "max_link_events": len(world.link_calendar.events),
                "max_preparation_jobs": len(world.link_calendar.jobs),
                "timings": defaultdict(lambda: {"calls": 0, "total_s": 0.0, "self_s": 0.0}),
            }
            self.current = row
            started = perf_counter()
            try:
                if supports_reuse_statistics:
                    reuse = {}
                    action, gain, baseline = original(world, reuse_statistics=reuse, **kwargs)
                    row["cost_reuse"] = reuse
                else:
                    action, gain, baseline = original(world, **kwargs)
            finally:
                row["wall_s"] = perf_counter() - started
                self.current = None
            row.update(action=asdict(action) if action else None,
                       gain_us=gain, baseline_us=baseline,
                       screening=kwargs.get("screening"))
            self.decisions.append(row)
            with (self.output / "decisions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"decision={index} active={row['active_requests']} "
                  f"search_s={row['wall_s']:.3f} action={action.kind if action else 'none'}",
                  flush=True)
            return action, gain, baseline

        module.choose_action = choose

    def restore(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


def percentile(values, percent):
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def compare(reference: Path, result: dict, decisions: list[dict]):
    previous = json.loads((reference / "result.json").read_text())
    prior_decisions = [json.loads(line) for line in
                       (reference / "decisions.jsonl").read_text().splitlines()]
    differences = []
    screening_exact, bounds_verified = True, 0
    if len(prior_decisions) != len(decisions):
        differences.append("decision count")
    for before, after in zip(prior_decisions, decisions):
        for key in ("clock_us", "generated_tokens", "action", "gain_us", "baseline_us"):
            # JSON normalizes tuples in screening and action fields to lists.
            if before[key] != json.loads(json.dumps(after[key])):
                differences.append(f"decision {after['decision']}: {key}")
        check = compare_screening(before['screening'], json.loads(json.dumps(after['screening'])),
                                  baseline_us=after['baseline_us'])
        differences.extend(f"decision {after['decision']}: {value}" for value in check['differences'])
        screening_exact &= check['exact_match']
        bounds_verified += check['full_score_bounds_verified']
    for key in ("requests", "switches", "iterations", "pipeline_events"):
        if previous[key] != json.loads(json.dumps(result[key])):
            differences.append(key)
    ignored = {"policy_search_wall_time_s", "candidate_screening"}
    for key in set(previous["summary"]) | set(result["summary"]):
        if key not in ignored and previous["summary"].get(key) != result["summary"].get(key):
            differences.append(f"summary.{key}")
    prior_screen = previous['summary']['candidate_screening']
    actual_screen = result['summary']['candidate_screening']
    if len(prior_screen) != len(actual_screen):
        differences.append('summary candidate screening count')
    for old, new in zip(prior_screen, actual_screen):
        if old['iteration'] != new['iteration']:
            differences.append('summary candidate iteration')
        check = compare_screening(old['rounds'], json.loads(json.dumps(new['rounds'])))
        differences.extend(check['differences'])
    return {"exact_match": not differences and screening_exact, "differences": differences,
            "decision_output_match": not differences, "full_score_bounds_verified": bounds_verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-worlds", action="store_true")
    parser.add_argument("--detail-cost-functions", action="store_true",
                        help="Time new cache-key and calculation helpers separately; adds extra overhead")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--forecast-mode", choices=('reference', 'shared', 'bounded'))
    args = parser.parse_args()
    root = args.source_root.resolve()
    sys.path.insert(0, str(root))
    import csv
    from elasticafd.config import load_config
    import elasticafd.layout_aware as layout
    from elasticafd.simulator import run_simulation
    if args.forecast_mode:
        original_choose = layout.choose_action
        if 'forecast_mode' not in inspect.signature(original_choose).parameters:
            parser.error('source does not support forecast-mode')
        @wraps(original_choose)
        def choose_with_mode(world, **kwargs):
            return original_choose(world, forecast_mode=args.forecast_mode, **kwargs)
        layout.choose_action = choose_with_mode

    manifest = json.loads(args.manifest.read_text())
    with Path(manifest["trace"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[manifest["start"]:manifest["start"] + manifest["count"]]
    arrival0 = float(selected[0]["arrival_s"])
    requests = tuple({"request_id": row["request_id"],
                      "arrival_s": (float(row["arrival_s"]) - arrival0) * manifest["arrival_scale"],
                      "input_tokens": int(row["input_tokens"]),
                      "output_tokens": int(row["output_tokens"])} for row in selected)
    base = load_config(root / "configs/layout-aware-online-validation.yaml")
    workload = replace(base.workload, requests=requests, prefix_blocks=(), trace_file=None)
    policy = replace(base.simulation.policy, name="layout_greedy", event_driven_replan=True,
                     decision_interval=manifest["decision_interval"],
                     planning_horizon_tokens=manifest["horizon"])
    config = replace(base, workload=workload, simulation=replace(
        base.simulation, policy=policy, max_batch_size=manifest["batch_size"],
        record_every=100, layout_max_candidates=manifest["candidates"],
        layout_forecast_scenarios=manifest["scenarios"],
        layout_survival_mean_tokens=manifest["prior_mean_tokens"],
        layout_growth_reserve_tokens=manifest["horizon"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "decisions.jsonl").exists():
        parser.error("output directory already contains decisions; use a fresh directory")
    sources = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
               for path in sorted((root / "elasticafd").glob("*.py"))}
    metadata = {"manifest": manifest, "source_root": str(root), "source_sha256": sources,
                "trace_sha256": sha256(Path(manifest["trace"]).read_bytes()).hexdigest(),
                "python": sys.version, "instrumented": True,
                "detail_cost_functions": args.detail_cost_functions,
                "forecast_mode": args.forecast_mode or 'source_default',
                "timing_notes": "Nested total_s overlap; self_s excludes timed child calls. "
                                "Run cases sequentially; instrumentation adds overhead."}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    timings = SearchTimings(layout, args.output_dir, args.capture_worlds,
                            detail_cost_functions=args.detail_cost_functions)
    started = perf_counter()
    timings.install()
    try:
        run = run_simulation(config)
    finally:
        timings.restore()
    elapsed = perf_counter() - started
    result = asdict(run)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["summary"]["status"] == "completed"
    assert result["summary"]["output_tokens"] == manifest["total_output_tokens"]
    values = [row["wall_s"] for row in timings.decisions]
    totals = defaultdict(lambda: {"calls": 0, "total_s": 0.0, "self_s": 0.0})
    for row in timings.decisions:
        for name, values_by_key in row["timings"].items():
            for key, value in values_by_key.items():
                totals[name][key] += value
    report = {"run_wall_s": elapsed, "search_wall_s": sum(values), "decisions": len(values),
              "mean_s": statistics.fmean(values) if values else 0.0,
              "p50_s": percentile(values, 50), "p95_s": percentile(values, 95),
              "p99_s": percentile(values, 99), "max_s": max(values, default=0),
              "timings": totals}
    reuse_totals = defaultdict(int)
    for row in timings.decisions:
        for name, value in row.get("cost_reuse", {}).items():
            if name.endswith("_peak_entries") or name in ("configuration_count", "forecast_peak_checkpoints"):
                reuse_totals[name] = max(reuse_totals[name], value)
            else:
                reuse_totals[name] += value
    if reuse_totals:
        report["cost_reuse"] = reuse_totals
    if args.reference:
        report["comparison"] = compare(args.reference, result, timings.decisions)
    (args.output_dir / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.reference and report["comparison"]["differences"]:
        raise SystemExit("reference comparison failed")


if __name__ == "__main__":
    main()
