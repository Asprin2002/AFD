#!/usr/bin/env python3
"""A7 same-state screening and paired, full-length decode-ready trace runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from elasticafd.config import project_config_from_dict
from elasticafd.dense_offload import DenseOffloadCost, DenseOffloadSimulator, DensePlan, plan_menu
from elasticafd.state import RequestState, views_for_requests
from elasticafd.suite import load_suite
from elasticafd.workload import build_workload


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Generated artifacts, not source edits. Atomic replacement is confined to
    # this runner's own named checkpoint files.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def code_hash():
    paths = [Path(__file__), *sorted((ROOT / "elasticafd").glob("*.py"))]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def pilot_config(raw, source=None, *, min_switch_interval=64, min_segment_tokens=64,
                 segment_setup_us=0.0):
    if segment_setup_us < 0:
        raise ValueError("segment setup cost must be nonnegative")
    c = project_config_from_dict(raw, source_path=source)
    c = replace(c, system=replace(
        c.system,
        gpu=replace(c.system.gpu, segment_setup_us=segment_setup_us),
        pim=replace(c.system.pim, segment_setup_us=segment_setup_us),
    ))
    policy = replace(c.simulation.policy, name="chime", decision_interval=64,
                     planning_horizon_tokens=64,
                     min_switch_interval=min_switch_interval,
                     min_segment_tokens=min_segment_tokens,
                     max_switches_per_decision=16, relative_improvement_threshold=.002,
                     switch_execution_us=100, event_driven_replan=True,
                     beam_include_future_arrivals=False)
    return replace(c, simulation=replace(c.simulation, pipeline_model="rankset_event",
                                        policy_pipeline_model="rankset_event", policy=policy,
                                        record_every=64, record_pipeline_events=False))


def validate(result, expected_count, trace):
    s = result.summary
    assert s["status"] == "completed" and s["completed_requests"] == expected_count
    assert s["output_tokens"] == sum(r.output_tokens for r in trace)
    assert s["peak_pim_kv_gib"] <= s["pim_kv_capacity_gib"] + 1e-9
    assert s["peak_gpu_kv_gib"] <= s["gpu_kv_capacity_gib"] + 1e-9
    assert s["af_actual_minus_ideal_s"] >= -1e-7
    source = {r.request_id:r for r in trace}
    for row in result.requests:
        r = source[row["request_id"]]
        assert row["input_tokens"] == r.input_tokens
        assert row["target_output_tokens"] == r.output_tokens
        assert abs(row["arrival_ms"] - r.arrival_us / 1000) < 1e-7
        first = row["segment_layout"].split("|")[0]
        assert first.startswith("pim:0-") and int(first.split("-")[-1]) >= r.input_tokens
        assert row["pim_tokens_final"] + row["gpu_tokens_final"] == r.input_tokens + r.output_tokens
    for d in s["dense_policy_decisions"]:
        if "objective_after_us" in d:
            assert d["objective_after_us"] < d["objective_before_us"]


def screen(args):
    config = pilot_config(dict(model="gpt175b", hardware="chime_a100",
                               workload=dict(type="static_batch",batch_size=4,input_tokens=2048,output_tokens=4096),
                               simulation=dict(max_batch_size=32)))
    rows = []
    started = perf_counter()
    for scale in args.scales:
        cost = DenseOffloadCost(config,plan_menu("joint"),dense_compute_scale=scale,allow_conditional_ffn=True)
        for b in (4,8,16,32):
            for context in (2048,8192,16384):
                views = views_for_requests([RequestState(f"r{i}",0,context,4096) for i in range(b)])
                if not cost.fits(views):
                    rows.append(dict(scale=scale,batch=b,context=context,status="capacity_infeasible"))
                    continue
                base = cost.measure(views,DensePlan())
                group = []
                for plan in cost.plans:
                    m = cost.measure(views,plan)
                    group.append(dict(scale=scale,batch=b,context=context,status="completed",
                        qkv_fraction=plan.qkv,o_proj_fraction=plan.o_proj,ffn_fraction=plan.ffn,
                        iteration_us=m.cost.total_us,baseline_us=base.cost.total_us,
                        speedup=base.cost.total_us/m.cost.total_us,
                        gpu_removed_us=base.cost.gpu_busy_us-m.cost.gpu_busy_us,
                        pim_added_busiest_rankset_us=m.cost.pim_core_busy_us-base.cost.pim_core_busy_us,
                        pim_added_dense_sum_us=m.pim_oproj_us+m.pim_ffn_us,
                        baseline_gpu_us=base.cost.gpu_busy_us,baseline_pim_us=base.cost.pim_core_busy_us,
                        gpu_us=m.cost.gpu_busy_us,pim_us=m.cost.pim_core_busy_us,
                        link_us=m.cost.link_busy_us,extra_h2d_bytes=m.h2d_bytes-base.h2d_bytes,
                        extra_d2h_bytes=m.d2h_bytes-base.d2h_bytes,
                        bubble_us=m.cost.total_us-max(m.cost.gpu_busy_us,m.cost.pim_core_busy_us,m.cost.link_busy_us)))
                for mode in ("qkv","oproj","ffn","joint"):
                    allowed = set(plan_menu(mode))
                    candidates = [
                        r for r in group
                        if DensePlan(
                            r["o_proj_fraction"],
                            r["ffn_fraction"],
                            r["qkv_fraction"],
                        ) in allowed
                    ]
                    best = min(candidates,key=lambda r:r["iteration_us"])
                    assert best["speedup"] >= 1 - 1e-10
                    best["best_"+mode] = True
                rows.extend(group)
        print(f"screen scale={scale}: {len(rows)} rows, wall={perf_counter()-started:.1f}s",flush=True)
    dump(args.output / "screen.json",dict(code_sha256=code_hash(),rows=rows,
        semantics="single decode iteration, original 96 layers, all historical KV on PIM; not lifecycle speedup",
        conditional_pim_dense=True,wall_time_s=perf_counter()-started))


def workloads(args):
    suite = load_suite(ROOT/"configs/suites/attention-split-v2.yaml")
    for case_id in args.cases:
        if case_id.startswith("offline-b"):
            b = int(case_id.split("-b")[1])
            raw = dict(model="gpt175b",hardware="chime_a100",
                       workload=dict(type="static_batch",batch_size=b,input_tokens=2048,output_tokens=4096),
                       simulation=dict(max_batch_size=b))
            yield case_id,pilot_config(
                raw,
                min_switch_interval=args.min_switch_interval,
                min_segment_tokens=args.min_segment_tokens,
                segment_setup_us=args.segment_setup_us,
            )
        else:
            case = next(c for c in suite.cases if c.case_id == case_id)
            yield case_id,pilot_config(
                case.config,
                suite.source_path,
                min_switch_interval=args.min_switch_interval,
                min_segment_tokens=args.min_segment_tokens,
                segment_setup_us=args.segment_setup_us,
            )


def online(args):
    code = code_hash()
    for case_id,config in workloads(args):
        trace = build_workload(config.workload, config.source_path.parent if config.source_path else ROOT)
        trace_hash = digest([asdict(r) for r in trace])
        for scale in args.scales:
            for mode in args.modes:
                target = args.output / f"{case_id}--{mode}--s{scale:g}.json"
                if target.exists():
                    data = json.loads(target.read_text())
                    if (data["code_sha256"] != code or data["trace_sha256"] != trace_hash
                            or data["summary"].get("kv_marginal_gain_threshold",0.0)
                            != args.kv_marginal_threshold):
                        raise RuntimeError(f"stale checkpoint: choose a new output directory: {target}")
                    print(f"skip {target.name}",flush=True)
                    continue
                started = perf_counter()
                print(f"start {target.name}: {len(trace)} requests",flush=True)
                class ProgressSimulator(DenseOffloadSimulator):
                    last_log = 0.0
                    def _run_iteration(self, **kwargs):
                        completed = super()._run_iteration(**kwargs)
                        now = perf_counter()
                        if now - self.last_log >= 30:
                            print(f"  {target.name}: iter={self.iteration}, active={len(self.active)}, "
                                  f"completed={len(self.completed)}/{len(self.workload)}",flush=True)
                            self.last_log = now
                        return completed
                sim = ProgressSimulator(config,mode=mode,dense_compute_scale=scale,allow_conditional_ffn=True,
                                        kv_marginal_gain_threshold=args.kv_marginal_threshold,
                                        lifetime_objective=(args.planning_objective == "lifetime"),
                                        event_cost_cache_max_entries=args.event_cache_entries)
                result = sim.run()
                validate(result,len(trace),trace)
                dump(target,dict(case_id=case_id,mode=mode,scale=scale,code_sha256=code,
                    trace_sha256=trace_hash,config=asdict(config),wall_time_s=perf_counter()-started,
                    summary=result.summary,requests=result.requests,iterations=result.iterations,
                    switches=result.switches))
                print(f"done {target.name}: {result.summary['simulated_time_s']:.3f}s simulated, "
                      f"{perf_counter()-started:.1f}s wall, O/F switches={sim.policy.operator_switches}, "
                      f"KV switches={len(result.switches)}",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("screen","online"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scales",type=float,nargs="+",default=[1.0])
    p.add_argument("--cases",nargs="+",default=["on-sl-r85","on-mix-burst-r85","burstgpt-ll-r85"])
    p.add_argument("--modes",nargs="+",default=["control","attention","qkv","oproj","ffn","f_all","joint"])
    p.add_argument("--kv-marginal-threshold",type=float,default=0.0,
                   help="additional KV-only marginal gain gate; zero reproduces r2")
    p.add_argument("--planning-objective",choices=("bounded","lifetime"),default="lifetime",
                   help="lifetime uses each arrived request's known remaining output")
    p.add_argument("--event-cache-entries",type=int,default=4096,
                   help="bounded exact event-cost cache; affects wall time, not simulated time")
    p.add_argument("--min-switch-interval",type=int,default=64,
                   help="minimum generated-token residence before another future-writer/operator switch")
    p.add_argument("--min-segment-tokens",type=int,default=64,
                   help="minimum current tail-segment length before changing its future writer")
    p.add_argument("--segment-setup-us",type=float,default=0.0,
                   help="paired per-layer setup cost for each GPU/PIM KV segment")
    args = p.parse_args()
    (screen if args.stage == "screen" else online)(args)


if __name__ == "__main__":
    main()
