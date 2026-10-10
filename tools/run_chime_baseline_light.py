#!/usr/bin/env python3
"""Small, frozen-plan CHIME-baseline check for separate A and F migrations.

This is a decode-ready homogeneous closed-batch experiment. It does not search
new policies: the candidate windows/plans below were selected before this
comparison. One midpoint represents each decode interval; --stride 1 is exact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticafd.capacity import views_fit_capacity
from elasticafd.chime_baseline import ChimeDecodeBaseline
from elasticafd.dense_offload import DenseOffloadCost, DensePlan
from elasticafd.specs import GiB
from tools.attention_strategy import Schedule, all_pim, sampled_blocks, views
from tools.run_contiguous_ablation import config


CONTROL = DensePlan()


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    batch: int
    input_tokens: int
    output_tokens: int
    # Frozen Attention window: first n requests write future KV to GPU in
    # [start, end). All historical KV remains on its original device.
    n: int = 0
    start: int = 0
    end: int = 0
    # Frozen output of the prior 64-token horizon policy, not rerun here.
    dynamic: tuple[tuple[int, int], ...] = ()
    f_candidates: tuple[tuple[str, DensePlan], ...] = ()


CASES = (
    Case("attention_long_b16", "attention", 16, 5200, 8192, 16, 0, 4466,
         dynamic=tuple((index, 0) for index in range(8)) + ((8, 2304),)),
    Case("attention_long_b32", "attention", 32, 2048, 4096, 24, 41, 3018,
         dynamic=tuple((index, 0) for index in range(16))
         + ((16, 128), (17, 2176))),
    Case("attention_long_b8", "attention", 8, 8192, 8192, 6, 487, 8192,
         dynamic=((0, 448), (1, 512), (2, 576), (3, 576),
                  (4, 832), (5, 832), (6, 3200), (7, 3200))),
    Case("attention_control_b4", "attention", 4, 5200, 4096, 2, 0, 4096),
    Case("f_short_b2", "f", 2, 128, 256, f_candidates=(
        ("qkv_oproj_all", DensePlan(1, 0, 1)),
        ("oproj_all_ffn_125", DensePlan(1, .125, 0)),
    )),
    Case("f_long_b2", "f", 2, 128, 4096, f_candidates=(
        ("qkv_oproj_50", DensePlan(.5, 0, 1)),
        ("oproj_all_ffn_125", DensePlan(1, .125, 0)),
    )),
    Case("f_control_b4", "f", 4, 128, 1024, f_candidates=(
        ("qkv_only", DensePlan(0, 0, 1)),
        ("oproj_all_ffn_125", DensePlan(1, .125, 0)),
    )),
)


def attention_schedule(case: Case) -> Schedule:
    if case.family != "attention" or not 0 < case.n <= case.batch:
        raise ValueError("invalid Attention case")
    if not 0 <= case.start < case.end <= case.output_tokens:
        raise ValueError("invalid Attention window")
    from elasticafd.state import Location
    cuts = tuple(
        ((case.start, Location.GPU),)
        + (((case.end, Location.PIM),) if case.end < case.output_tokens else ())
        if index < case.n else ()
        for index in range(case.batch)
    )
    return Schedule("offline_window", cuts)


def dynamic_schedule(case: Case) -> Schedule:
    from elasticafd.state import Location
    cuts = [[] for _ in range(case.batch)]
    for request, token in case.dynamic:
        if not 0 <= request < case.batch or not 0 <= token < case.output_tokens:
            raise ValueError("invalid frozen horizon cut")
        cuts[request].append((token, Location.GPU))
    return Schedule("frozen_horizon_64", tuple(tuple(part) for part in cuts))


def chime_model(case: Case) -> ChimeDecodeBaseline:
    # This is the baseline's own scheduler and pipeline. Candidate plan menus,
    # weight reservations, and future-KV cuts cannot modify it.
    cfg = config(case.batch, progression="device_contiguous", scheduler="fixed")
    return ChimeDecodeBaseline(cfg.model, cfg.system)


def chime_fits(model: ChimeDecodeBaseline, state) -> bool:
    return views_fit_capacity(
        state, model.model, model.system,
        pim_capacity_bytes=model.pim_kv_capacity_bytes,
        gpu_replica_capacity_bytes=model.gpu_replica_kv_capacity_bytes,
    )


def cost_metrics(cost) -> dict[str, float]:
    return {
        "gpu_busy_us": cost.gpu_busy_us,
        "pim_busy_us": cost.pim_core_busy_us,
        "link_busy_us": cost.link_busy_us,
    }


def measure_case(case: Case, stride: int,
                 dense_compute_scale: float,
                 dense_bandwidth_scale: float) -> list[dict]:
    control = all_pim(case.batch)
    if case.family == "attention":
        candidates = [("offline_window", CONTROL, attention_schedule(case))]
        if case.dynamic:
            candidates.append(("frozen_horizon_64", CONTROL, dynamic_schedule(case)))
    else:
        candidates = [(name, plan, control) for name, plan in case.f_candidates]
    baseline = chime_model(case)
    cfg = config(case.batch, progression="device_contiguous", scheduler="fixed")
    two_cut_control = DenseOffloadCost(cfg, (CONTROL,))
    candidate_models = {
        name: DenseOffloadCost(
            cfg, (CONTROL, plan), allow_conditional_ffn=True,
            dense_compute_scale=dense_compute_scale,
            dense_bandwidth_scale=dense_bandwidth_scale,
        )
        for name, plan, _ in candidates
    }
    models = {"chime_baseline": baseline, "two_cut_control": two_cut_control,
              **candidate_models}
    totals = {name: 0.0 for name in models}
    busy = {name: {metric: 0.0 for metric in
                   ("gpu_busy_us", "pim_busy_us", "link_busy_us")}
            for name in models}
    feasible = {name: True for name in models}
    # Baseline integration is *independent* of candidate schedules. Otherwise
    # adding a candidate cut would change the baseline's midpoint quadrature.
    for weight, generated in sampled_blocks(case.output_tokens, stride, (control,)):
        baseline_state = views(control, case.input_tokens, generated)
        for name in ("chime_baseline", "two_cut_control"):
            model = models[name]
            if not feasible[name]:
                continue
            fits = (chime_fits(model, baseline_state) if name == "chime_baseline"
                    else model.fits(baseline_state))
            if not fits:
                feasible[name] = False
                continue
            cost = (model.iteration_cost(baseline_state) if name == "chime_baseline"
                    else model.measure(baseline_state, CONTROL).cost)
            totals[name] += weight * cost.total_us
            for metric, value in cost_metrics(cost).items():
                busy[name][metric] += weight * value
    for name, plan, candidate_schedule in candidates:
        model = candidate_models[name]
        for weight, generated in sampled_blocks(
            case.output_tokens, stride, (candidate_schedule,)
        ):
            state = views(candidate_schedule, case.input_tokens, generated)
            if not model.fits(state):
                feasible[name] = False
                break
            cost = model.measure(state, plan).cost
            totals[name] += weight * cost.total_us
            for metric, value in cost_metrics(cost).items():
                busy[name][metric] += weight * value
    # Midpoint timing is approximate, but capacity must hold at peak KV size.
    baseline_final = views(control, case.input_tokens, case.output_tokens)
    for name in ("chime_baseline", "two_cut_control"):
        model = models[name]
        feasible[name] &= (chime_fits(model, baseline_final) if name == "chime_baseline"
                           else model.fits(baseline_final))
    for name, _, candidate_schedule in candidates:
        feasible[name] &= candidate_models[name].fits(
            views(candidate_schedule, case.input_tokens, case.output_tokens)
        )
    rows = []
    for name, plan, candidate_schedule in candidates:
        model = candidate_models[name]
        ok = all(feasible[baseline] for baseline in
                 ("chime_baseline", "two_cut_control")) and feasible[name]
        row = {
            "case": case.name, "family": case.family, "batch_size": case.batch,
            "input_tokens": case.input_tokens, "output_tokens": case.output_tokens,
            "candidate": name, "status": "completed" if ok else "capacity_error",
            "capacity_detail": ";".join(
                n for n in ("chime_baseline", "two_cut_control", name)
                if not feasible[n]
            ),
            "stride": stride,
            "sampling": "baseline_fixed_midpoints_candidate_own_cut_boundaries",
            "chime_baseline_progression": "device_contiguous",
            "chime_baseline_formation": "decode_rounds_n1_predictive_alignment",
            "candidate_progression": "device_contiguous",
            "candidate_subbatch_scheduler": "fixed",
            "qkv_pim_fraction": plan.qkv, "o_proj_pim_fraction": plan.o_proj,
            "ffn_pim_fraction": plan.ffn, "conditional_pim_dense": case.family == "f",
            "gpu_kv_request_count": sum(bool(cuts) for cuts in candidate_schedule.cuts),
            "gpu_kv_start": case.start if name == "offline_window" else None,
            "gpu_kv_end": case.end if name == "offline_window" else None,
            "kv_switches": sum(len(cuts) for cuts in candidate_schedule.cuts),
            "kv_cut_schedule": ";".join(
                f"r{index}:" + ",".join(f"{token}:{target.value}" for token, target in cuts)
                for index, cuts in enumerate(candidate_schedule.cuts) if cuts
            ),
            "pim_weight_gib": model.pim_persistent_weight_bytes / GiB,
            "pim_weight_copy_us": model.initial_weight_copy_us,
            "dense_compute_scale": dense_compute_scale,
            "dense_bandwidth_scale": dense_bandwidth_scale,
        }
        if ok:
            for key in totals:
                row[f"{key}_total_us"] = totals[key]
                for metric, value in busy[key].items():
                    row[f"{key}_{metric}"] = value
            for baseline in ("chime_baseline", "two_cut_control"):
                row[f"speedup_vs_{baseline}_warm"] = totals[baseline] / totals[name]
                row[f"speedup_vs_{baseline}_cold"] = (
                    totals[baseline] / (totals[name] + model.initial_weight_copy_us)
                )
        rows.append(row)
    return rows


def code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "tools" / "attention_strategy.py",
                 ROOT / "tools" / "run_contiguous_ablation.py",
                 *sorted((ROOT / "elasticafd").glob("*.py"))):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("attention", "f", "all"), default="all")
    parser.add_argument("--case", action="append", choices=tuple(c.name for c in CASES))
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--dense-compute-scale", type=float, default=1.0)
    parser.add_argument("--dense-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stride <= 0 or args.dense_compute_scale <= 0 or args.dense_bandwidth_scale <= 0:
        parser.error("stride and dense scales must be positive")
    selected = [case for case in CASES if
                (args.family == "all" or case.family == args.family)
                and (not args.case or case.name in args.case)]
    if not selected:
        parser.error("no cases match --family and --case")
    if args.dry_run:
        for case in selected:
            print(f"{case.name}: B{case.batch} I{case.input_tokens} "
                  f"O{case.output_tokens}; candidate rows="
                  f"{1 + bool(case.dynamic) if case.family == 'attention' else len(case.f_candidates)}")
        return 0
    rows = []
    for index, case in enumerate(selected, 1):
        print(f"event {index}/{len(selected)} {case.name}", flush=True)
        rows.extend(measure_case(case, args.stride, args.dense_compute_scale,
                                 args.dense_bandwidth_scale))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row
        )))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps({
            "code_sha256": code_sha256(), "model": "gpt175b",
            "hardware": "chime_a100", "scope": "static homogeneous decode-ready only",
            "baseline": "CHIME operator mapping: QKV/O_proj/FFN on GPU, all KV/Attention on PIM",
            "chime_baseline": "separate CHIME operator model; N=1 decode request rounds, predicted cross-device alignment, paired GPU/PIM visits without lead",
            "chime_decode_only_gap": "paper does not specify request ordering or admission of all decode-ready requests when there is no prefill; longest-context-first closed-batch specialization",
            "chime_predictor_gap": "uses the common calibrated simulator cost model, not CHIME's runtime-trained random forest predictor",
            "two_cut_control": "all-PIM Attention/KV under the candidate device-contiguous DAG; diagnostic only",
            "candidate": "frozen prior-plan replay, no search on this baseline",
            "pim_dense": "conditional roofline, not an experimentally calibrated CHIME capability",
            "sampling": "baseline uses fixed-stride midpoints independent of candidates; each candidate also splits at its own KV cuts; endpoint capacity check",
            "stride": args.stride,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"written={args.output.resolve()} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
