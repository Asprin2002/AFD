#!/usr/bin/env python3
"""M1-M3 runner: heterogeneous/homogeneous joint static matrix.

The matrix cross-product is

  scenario (recipe) x formation x strategy x cost arm

with strategies (``control`` all-PIM future KV with GPU F side;
``attn`` operator frozen, future-KV writer search; ``f_only`` KV frozen,
operator timeline search; ``joint`` the two-phase segmented search;
``ffn_pilot`` f_only on the device-contiguous FFN menu; ``qkv_pilot`` f_only on
the whole-fused QKV menu).

The formation axis reuses the validated partition kernels exactly: ``lpt``
balances Attention-scan work, ``alignment`` is the LPT-initialized local search
over the pipeline objective (the paper-aligned *idea* of CHIME's N=1 pairing,
evaluated on the shared cost predictor instead of a runtime-trained forest),
and ``fixed`` is the closed-batch membership reference.

The economics axis (M3) is ``real`` (100us per KV/operator switch plus replica
delta weight copy over the link) versus ``free`` (all zeros).  Every schedule
is evaluated under both arms so the effect of the cost model itself on the
chosen timing is visible.

M1-B cross-validation: for every scenario and formation the independent
JointAFScheduler best-static oracle (KV assignment + one O_proj fraction frozen
for the lifetime) is produced and replayed on the same event engine, and the
best ``joint`` schedule is replayed on a second freshly-constructed engine to
guard cache state.  Summaries are printed after every scenario and a final
``matrix.csv`` plus ``schedules.json`` are written to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticafd.config import project_config_from_dict
from elasticafd.dense_offload import DenseOffloadCost
from elasticafd.joint import JointAFResourceModel, JointAFScheduler
from elasticafd.segmented import (
    FREE_PRICING,
    DCSchedule,
    DensePlan,
    SwitchPricing,
    control_schedule,
    dc_plan_menu,
    eval_schedule,
    max_output,
    search_joint,
    search_kv_timeline,
    search_operator_timeline,
)
from elasticafd.specs import MODEL_PRESETS, SYSTEM_PRESETS
from elasticafd.state import Location, RequestState

try:
    from tools.prepare_static_recipes import load_recipe
    from tools.prepare_static_recipes import discover as discover_recipes
except ImportError:  # pragma: no cover - direct invocation
    from prepare_static_recipes import load_recipe
    from prepare_static_recipes import discover as discover_recipes

STRATEGIES = ("control", "attn", "f_only", "joint", "ffn_pilot", "qkv_pilot")
FORMATIONS = ("lpt", "alignment", "fixed")
ARMS = ("real", "free")
# Anchor shapes worth running when --anchors quick is used.
QUICK_ANCHORS = (
    (2, 128, 256),     # no-benefit region
    (8, 8192, 8192),   # 1.2694x
    (16, 5200, 4096),  # 1.3742x
    (16, 5200, 8192),  # headline 1.5074x
    (32, 2048, 8192),  # 1.4847x / 1.4001x
)


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in [Path(__file__), *sorted((ROOT / "elasticafd").glob("*.py")),
                 Path(__file__).with_name("prepare_static_recipes.py")]:
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def static_config(batch: int, scheduler: str) -> object:
    return project_config_from_dict({
        "model": "gpt175b",
        "hardware": "chime_a100",
        "workload": {
            "type": "closed_batch",
            "requests": [{"request_id": "dummy", "input_tokens": 1,
                          "output_tokens": 1}],
        },
        "simulation": {
            "max_batch_size": batch,
            "fidelity": "tile",
            "pipeline_model": "rankset_event",
            "policy_pipeline_model": "rankset_event",
            "subbatch_scheduler": scheduler,
            "subbatch_progression": "device_contiguous",
            "operator_mapping": "chime",
            "record_every": 64,
            "policy": {
                "name": "chime",
                "switch_execution_us": 100,
            },
        },
    })


def static_engine(config, *, cache_entries: int = 8192) -> DenseOffloadCost:
    return DenseOffloadCost(
        config,
        plans=dc_plan_menu(),
        allow_conditional_ffn=True,          # FFN/QKV conditional capabilities on
        event_cost_cache_max_entries=cache_entries,
    )


def to_request_state(spec) -> RequestState:
    return RequestState(spec.request_id, 0.0, spec.input_tokens,
                        spec.output_tokens, gpu_replica=spec.gpu_replica)


def encode_schedule(schedule: DCSchedule) -> dict:
    return {
        "name": schedule.name,
        "kv_cuts": [
            [[boundary, location.value] for boundary, location in cuts]
            for cuts in schedule.kv_cuts
        ],
        "plan_timeline": [
            [start, {"o_proj": plan.o_proj, "ffn": plan.ffn, "qkv": plan.qkv}]
            for start, plan in schedule.plan_timeline
        ],
    }


def joint_static_reference(
    specs, scheduler: str, *, engine: DenseOffloadCost | None = None,
    budget: int = 24, stride: int | None = None,
) -> tuple[DCSchedule, dict]:
    """Independent M1-B planner: JointAF best frozen (KV + O_proj) action.

    The exact JointAF token-by-token ``run_best_static`` simulation is
    exponential in batch size, so this reference runs its *candidate space*
    (the JointAF grouped assignment generator + O_proj fraction ladder) and
    scores every candidate on the shared event engine via ``eval_schedule``.
    For batch <= 6 the assignment generator is exactly exhaustive; above that a
    coordinate-only single-writer move set is used and ``reference_partial``
    records the simplification, like the JointAF online search's coordinate
    restriction.  The ``--reference-budget`` cap bounds full-lifetime evals so
    the reference stays cheap at any batch size.
    """
    model = MODEL_PRESETS["gpt175b"]
    system = SYSTEM_PRESETS["chime_a100"]
    resource = JointAFResourceModel(model, system, max_o_proj_pim_fraction=1.0)
    oracle = JointAFScheduler(
        resource,
        fractions=(0.0, 0.25, 0.5, 1.0),
        decision_interval=64,
        min_kv_segment_tokens=64,
        min_kv_switch_interval=128,
        min_operator_residence_tokens=128,
        max_kv_switches_per_decision=len(specs),
        max_exact_candidates=len(specs),
        max_cost_evaluations=budget,
    )
    requests = [to_request_state(spec) for spec in specs]
    if engine is None:
        engine = static_engine(static_config(len(specs), scheduler))
    horizon = max_output(specs)
    if stride is None:
        stride = max(1, round(horizon / 96))
    pricing = SwitchPricing()
    exact_space = len(specs) <= 6
    assignments = oracle._kv_target_assignments(
        requests, 0,
        allow_gpu_kv=True,
        coordinate_only=not exact_space,
        enforce_switch_limit=False,
        enforce_residence=False,
    )
    best_schedule: DCSchedule | None = None
    best_cost: float | None = None
    evaluations = 0
    for targets in assignments:
        for fraction in (0.0, 0.25, 0.5, 1.0):
            if evaluations >= budget:
                break
            evaluations += 1
            cuts = tuple(
                ((0, targets[spec.request_id]),) if targets.get(spec.request_id)
                is Location.GPU else ()
                for spec in specs
            )
            candidate = DCSchedule(
                "joint_best_static_cand",
                cuts,
                ((0, DensePlan(o_proj=fraction)),),
            )
            cost = eval_schedule(
                engine, candidate, specs, stride=stride, pricing=pricing,
            )
            if cost.feasible and (best_cost is None
                                  or cost.total_us < best_cost):
                best_schedule, best_cost = candidate, cost.total_us
    if best_schedule is None:
        raise RuntimeError("joint static reference found no feasible action")
    metadata = {
        "reference_partial": (evaluations >= budget) or not exact_space,
        "reference_evaluations": evaluations,
        "reference_budget": budget,
        "reference_space_exact": exact_space,
        "o_proj_fraction": best_schedule.plan_timeline[0][1].o_proj,
    }
    return best_schedule, metadata


def plan_menu_restriction(strategy: str):
    if strategy == "ffn_pilot":
        return tuple(
            p for p in dc_plan_menu()
            if p.qkv == 0.0 and p.ffn > 0.0 and p.o_proj == 1.0
        )
    if strategy == "qkv_pilot":
        return tuple(p for p in dc_plan_menu() if p.qkv == 1.0)
    return dc_plan_menu()


def build_strategy_schedule(
    strategy: str,
    engine: DenseOffloadCost,
    specs,
    *,
    pricing: SwitchPricing,
    stride: int,
    decision_interval: int,
    lookahead: int,
    min_residence: int,
    max_segments: int,
) -> DCSchedule:
    if strategy == "control":
        return control_schedule(specs)
    if strategy == "attn":
        return search_kv_timeline(
            engine, specs,
            plan_timeline=((0, DensePlan()),),
            stride=stride,
            decision_interval=decision_interval,
            lookahead=lookahead,
            min_residence=min_residence,
            pricing=pricing,
        )
    if strategy in ("f_only", "ffn_pilot", "qkv_pilot"):
        return search_operator_timeline(
            engine, specs,
            plan_menu=plan_menu_restriction(strategy),
            stride=stride,
            pricing=pricing,
            max_segments=max_segments,
        )
    if strategy == "joint":
        return search_joint(
            engine, specs,
            stride=stride,
            pricing=pricing,
            decision_interval=decision_interval,
            lookahead=lookahead,
            min_residence=min_residence,
            max_segments=max_segments,
        )
    raise ValueError(f"unknown strategy: {strategy}")


def decode_schedule(encoded: dict) -> DCSchedule:
    cuts = tuple(
        tuple((boundary, Location(location)) for boundary, location in part)
        for part in encoded["kv_cuts"]
    )
    timeline = tuple(
        (start, DensePlan(vals["o_proj"], vals["ffn"], vals["qkv"]))
        for start, vals in encoded["plan_timeline"]
    )
    return DCSchedule(encoded["name"], cuts, timeline)


def run_scenario(
    specs,
    *,
    formations: tuple[str, ...],
    strategies: tuple[str, ...],
    arms: tuple[str, ...],
    report_arms: tuple[str, ...] | None = None,
    stride: int,
    decision_interval: int,
    lookahead: int,
    min_residence: int,
    max_segments: int,
    reference_budget: int,
    progress: callable,
) -> tuple[list[dict], dict]:
    """Run one recipe across the (formation x strategy x arm) matrix."""
    rows: list[dict] = []
    # Report arms default to the search arms (each schedule is re-valued under
    # every report arm).  Decoupling lets ``--report-arms real`` re-cost a
    # free-priced search without re-running the expensive real search.
    report_arms = arms if report_arms is None else report_arms
    pricing_arms = tuple(sorted(set(arms) | set(report_arms)))
    pricings = {arm: (SwitchPricing() if arm == "real" else FREE_PRICING)
                for arm in pricing_arms}
    # Invariant guards (M1-B reference, best_constant_menu, fresh-engine replay)
    # are defined on real costs and must run even when --arms free alone.
    guard_pricing = SwitchPricing()
    # One engine per formation (fresh cache keeps every scenario independent);
    # a second engine guards fresh-engine replay of the best joint schedule.
    engines = {formation: static_engine(static_config(len(specs), formation))
               for formation in formations}

    best_joint: dict[str, dict] = {}
    violations: list[dict] = []

    for formation in formations:
        engine = engines[formation]
        reference_schedule, reference_meta, reference_cost = None, None, None
        try:
            reference_schedule, reference_meta = joint_static_reference(
                specs, formation, engine=engine, stride=stride,
                budget=reference_budget,
            )
            reference_cost = eval_schedule(
                engine, reference_schedule, specs, pricing=guard_pricing,
            )
        except Exception as error:  # capacity / schedule edge -> unavailable
            progress(f"{formation}: joint-reference failed: {error}")

        for strategy in strategies:
            progress(f"{formation}/{strategy}")
            for arm in arms:
                schedule = build_strategy_schedule(
                    strategy, engine, specs,
                    pricing=pricings[arm],
                    stride=stride,
                    decision_interval=decision_interval,
                    lookahead=lookahead,
                    min_residence=min_residence,
                    max_segments=max_segments,
                )
                for report in report_arms:
                    cost = eval_schedule(
                        engine, schedule, specs, pricing=pricings[report],
                    )
                    rows.append({
                        "formation": formation,
                        "strategy": strategy,
                        "search_arm": arm,
                        "report_arm": report,
                        "iteration_us": cost.iteration_us,
                        "switch_us": cost.switch_us,
                        "weight_copy_us": cost.weight_copy_us,
                        "total_us": cost.total_us,
                        "total_free_us": cost.total_without_overhead(),
                        "kv_switches": cost.kv_switches,
                        "operator_switches": cost.operator_switches,
                        "peak_pim_resident_gib": cost.peak_pim_resident_gib,
                        "feasible": int(cost.feasible),
                        "blocks": len(cost.curve),
                        "schedule": encode_schedule(schedule),
                    })

        # Best real-cost joint schedule for the cross-checks below.
        joint_real = [
            r for r in rows
            if r["formation"] == formation and r["strategy"] == "joint"
            and r["search_arm"] == "real" and r["report_arm"] == "real"
        ]
        if joint_real:
            best_joint[formation] = min(joint_real, key=lambda r: r["total_us"])

        if reference_schedule is not None and reference_cost is not None:
            rows.append({
                "formation": formation,
                "strategy": "joint_static_reference",
                "search_arm": "-",
                "report_arm": "real",
                "iteration_us": reference_cost.iteration_us,
                "switch_us": reference_cost.switch_us,
                "weight_copy_us": reference_cost.weight_copy_us,
                "total_us": reference_cost.total_us,
                "total_free_us": reference_cost.total_without_overhead(),
                "kv_switches": reference_cost.kv_switches,
                "operator_switches": reference_cost.operator_switches,
                "peak_pim_resident_gib": reference_cost.peak_pim_resident_gib,
                "feasible": int(reference_cost.feasible),
                "blocks": len(reference_cost.curve),
                "schedule": encode_schedule(reference_schedule),
                "reference_meta": json.dumps(reference_meta, sort_keys=True),
            })
            mine = best_joint.get(formation)
            if mine is not None and reference_cost.feasible and \
                    mine["total_us"] > reference_cost.total_us + 1e-6:
                violations.append({
                    "formation": formation,
                    "kind": "segmented loses to best_static",
                    "segmented_total_us": mine["total_us"],
                    "reference_total_us": reference_cost.total_us,
                })

        # Primary always-on bound: best *constant* plan over the full legal DC
        # menu (a strict superset of the JointAF o_proj-only subspace).  This
        # stays cheap at any batch size and is the honest "segmentation must not
        # lose to a smart frozen plan" check.
        zero_cuts = tuple(tuple() for _ in specs)
        best_constant_cost = None
        best_constant_plan = None
        for plan in dc_plan_menu():
            trial = DCSchedule("const", zero_cuts, ((0, plan),))
            cost = eval_schedule(engine, trial, specs, pricing=guard_pricing)
            if cost.feasible and (
                best_constant_cost is None
                or cost.total_us < best_constant_cost
            ):
                best_constant_cost = cost.total_us
                best_constant_plan = (trial, cost)
        constant_schedule, constant_cost = best_constant_plan or (
            control_schedule(specs, name="best_constant_menu"),
            eval_schedule(engine, control_schedule(specs, name="best_constant_menu"),
                          specs, pricing=guard_pricing),
        )
        rows.append({
            "formation": formation,
            "strategy": "best_constant_menu",
            "search_arm": "-",
            "report_arm": "real",
            "iteration_us": constant_cost.iteration_us,
            "switch_us": constant_cost.switch_us,
            "weight_copy_us": constant_cost.weight_copy_us,
            "total_us": constant_cost.total_us,
            "total_free_us": constant_cost.total_without_overhead(),
            "kv_switches": constant_cost.kv_switches,
            "operator_switches": constant_cost.operator_switches,
            "peak_pim_resident_gib": constant_cost.peak_pim_resident_gib,
            "feasible": int(constant_cost.feasible),
            "blocks": len(constant_cost.curve),
            "schedule": encode_schedule(constant_schedule),
        })
        mine = best_joint.get(formation)
        if mine is not None and constant_cost.feasible and \
                mine["total_us"] > constant_cost.total_us + 1e-6:
            violations.append({
                "formation": formation,
                "kind": "segmented loses to best constant legal plan",
                "segmented_total_us": mine["total_us"],
                "constant_total_us": constant_cost.total_us,
                "constant_timeline": encode_schedule(constant_schedule),
            })

        # Fresh-engine replay guard on the best real joint schedule: a second
        # engine with the SAME configuration as this formation (a different
        # scheduler would legitimately re-partition and break equality).
        best = best_joint.get(formation)
        if best is not None and best["total_us"] < float("inf"):
            schedule = decode_schedule(best["schedule"])
            replay_engine = static_engine(static_config(len(specs), formation))
            replay = eval_schedule(
                replay_engine, schedule, specs, pricing=guard_pricing,
            )
            if replay.total_us != best["total_us"]:
                violations.append({
                    "formation": formation,
                    "kind": "fresh-engine replay mismatch",
                    "first_total_us": best["total_us"],
                    "replay_total_us": replay.total_us,
                })

    return rows, {"best_joint": best_joint, "violations": violations}


def summarize(rows: list[dict], name: str, control: dict | None = None) -> list[str]:
    lines = []

    def best(strategy: str, arm: str) -> dict | None:
        key = [r for r in rows
               if r["strategy"] == strategy and r["search_arm"] == arm
               and r["report_arm"] == "real"]
        key = [r for r in key if math_isfinite(r["total_us"])]
        return min(key, key=lambda r: r["total_us"]) if key else None

    if control is not None and math_isfinite(control["total_us"]):
        for strategy in ("attn", "f_only", "joint", "ffn_pilot", "qkv_pilot",
                         "joint_static_reference"):
            found = best(strategy, "real")
            if strategy == "joint_static_reference":
                found = best(strategy, "-")
            if found is None:
                continue
            ratio = control["total_us"] / found["total_us"]
            lines.append(f"  {strategy:>22s}: {found['total_us']:10.0f}us "
                         f"{ratio:6.3f}x  kv={found['kv_switches']:3d} "
                         f"op={found['operator_switches']:2d} "
                         f"peak={found['peak_pim_resident_gib']:6.1f}GiB")
        for strategy in ("attn", "joint"):
            found = best(strategy, "free")
            if found is None:
                continue
            ratio = control["total_us"] / found["total_us"]
            lines.append(f"  {('free/' + strategy):>22s}: {found['total_us']:10.0f}us "
                         f"{ratio:6.3f}x")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes-dir", type=Path, default=ROOT / "workloads/static")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/heterogeneous-joint-matrix")
    parser.add_argument("--only", default=None,
                        help="substring filter on recipe filenames")
    parser.add_argument("--only-batch", type=int, default=None)
    parser.add_argument("--anchors", choices=("all", "quick", "none"),
                        default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of recipes processed")
    parser.add_argument("--formations", nargs="+", default=("lpt", "alignment"),
                        choices=FORMATIONS)
    parser.add_argument("--strategies", nargs="+", default=None,
                        choices=STRATEGIES)
    parser.add_argument("--arms", nargs="+", default=("real", "free"),
                        choices=ARMS)
    parser.add_argument("--report-arms", nargs="+", default=None,
                        choices=ARMS,
                        help="arms under which each searched schedule is valued"
                             " (default: same as --arms, i.e. the full"
                             " search x report cartesian product)")
    parser.add_argument("--stride", type=int, default=0,
                        help="block stride for lifetime sampling (0=auto)")
    parser.add_argument("--decision-interval", type=int, default=64)
    parser.add_argument("--lookahead", type=int, default=256)
    parser.add_argument("--min-residence", type=int, default=128)
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--reference-budget", type=int, default=24,
                        help="max full-lifetime evals for the JointAF reference"
                             " (small batches stay exact; larger record partial)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the recipe/strategy plan and exit")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    formations = tuple(args.formations)
    strategies = tuple(args.strategies) if args.strategies else STRATEGIES
    arms = tuple(args.arms)
    report_arms = (tuple(args.report_arms) if args.report_arms else arms)
    versions = code_hash()

    def scenario_name(path: Path) -> str:
        parts = path.relative_to(args.recipes_dir).parts
        if len(parts) > 1 and parts[-2] != "homogeneous":
            return f"{parts[-2]}_{path.stem}"
        return path.stem

    paths = discover_recipes(args.recipes_dir)
    selected: list[Path] = []
    for path in paths:
        name = scenario_name(path)
        parts = path.stem.split("_")
        if args.only is not None and args.only not in name:
            continue
        if path.parent.name == "homogeneous" and args.anchors == "none":
            continue
        if path.parent.name == "homogeneous" and args.anchors == "quick":
            shape = (int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:]))
            if shape not in QUICK_ANCHORS:
                continue
        if args.only_batch is not None:
            if int(parts[1][1:]) != args.only_batch:
                continue
        selected.append(path)
    selected.sort(key=lambda p: p.stem)
    if args.limit:
        selected = selected[: args.limit]

    matrix_rows: list[dict] = []
    all_violations: list[dict] = []
    for path in selected:
        name = scenario_name(path)
        specs = load_recipe(path)
        horizon = max_output(specs)
        stride = args.stride or max(1, round(horizon / 96))
        print(f"scenario {name}: batch={len(specs)} horizon={horizon} "
              f"stride={stride} formations={','.join(formations)} "
              f"strategies={','.join(strategies)} arms={','.join(arms)}",
              flush=True)
        if args.dry_run:
            print(f"  (dry) search-cells = "
                  f"{len(formations) * len(strategies) * len(arms)}",
                  flush=True)
            continue
        t0 = perf_counter()
        rows, extra = run_scenario(
            specs,
            formations=formations,
            strategies=strategies,
            arms=arms,
            report_arms=report_arms,
            stride=stride,
            decision_interval=args.decision_interval,
            lookahead=args.lookahead,
            min_residence=args.min_residence,
            max_segments=args.max_segments,
            reference_budget=args.reference_budget,
            progress=lambda msg: print(f"  {name}/{msg}", flush=True),
        )
        for row in rows:
            row["scenario"] = name
            row["batch"] = len(specs)
            row["horizon"] = horizon
        matrix_rows.extend(rows)
        all_violations.extend(extra["violations"])

        control = next(
            (r for r in rows
             if r["strategy"] == "control" and r["report_arm"] == "real"
             and r["search_arm"] == "real"), None
        )
        for line in summarize(rows, name, control):
            print(line, flush=True)
        best = extra["best_joint"].get("lpt", extra["best_joint"].get("alignment"))
        if best is not None:
            ref = next(
                (r for r in rows if r["strategy"] == "joint_static_reference"
                 and r["report_arm"] == "real"), None
            )
            if ref is not None and math_isfinite(ref["total_us"]):
                delta = best["total_us"] - ref["total_us"]
                print(f"  [M1-B] segmented[joint]={best['total_us']:.0f}us vs "
                      f"best_static={ref['total_us']:.0f}us "
                      f"(delta={delta:+.0f}us)", flush=True)
        elapsed = perf_counter() - t0
        print(f"  finished in {elapsed:.1f}s", flush=True)

    if args.dry_run:
        return
    out_path = output / "matrix.csv"
    fieldnames = ("scenario", "batch", "horizon", "formation", "strategy",
                  "search_arm", "report_arm", "iteration_us", "switch_us",
                  "weight_copy_us", "total_us", "total_free_us",
                  "kv_switches", "operator_switches",
                  "peak_pim_resident_gib", "feasible", "blocks",
                  "reference_meta")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow(row)
    json_path = output / "schedules.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "runner_code_sha256": versions,
            "schema_version": "heterogeneous-joint-matrix-v1",
            "formations": list(formations),
            "strategies": list(strategies),
            "arms": list(arms),
            "report_arms": list(report_arms),
            "decision_interval": args.decision_interval,
            "lookahead": args.lookahead,
            "min_residence": args.min_residence,
            "max_segments": args.max_segments,
            "reference_budget": args.reference_budget,
            "violations": all_violations,
            "scenarios": sorted({r["scenario"] for r in matrix_rows}),
        }, handle, indent=2, sort_keys=True)
    print(f"wrote {out_path} and {json_path}\n"
          f"violations={len(all_violations)}")
    for violation in all_violations:
        print(f"  VIOLATION {violation}")


def math_isfinite(value: float) -> bool:
    import math
    return math.isfinite(value)


if __name__ == "__main__":
    main()