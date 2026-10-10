"""Command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from .config import ProjectConfig, load_config
from .cost import ChimeCostModel
from .mapping import OPERATOR_MAPPINGS
from .experiments import (
    aggregate_rankset_sensitivity_rows,
    beam_terminal_ablation_rows,
    dynamic_operator_offload_rows,
    future_only_benefit_rows,
    future_only_composition_rows,
    hardware_ablation_rows,
    homogeneous_lifecycle_rows,
    joint_af_event_replay_rows,
    joint_af_heterogeneous_oracle_rows,
    joint_af_small_oracle_rows,
    lifetime_policy_comparison_rows,
    open_arrival_slo_rows,
    operator_offload_rows,
    policy_comparison_rows,
    scalability_rows,
    static_operator_mapping_rows,
)
from .simulator import SimulationCapacityError, SimulationResult, run_simulation
from .specs import GiB, MODEL_PRESETS, SYSTEM_PRESETS
from .state import RequestState, views_for_requests
from .suite import (
    comparison_rows,
    environment_manifest,
    load_suite,
    parse_shard,
    read_index,
    run_cases,
    select_cases,
    validate_cases,
    write_index,
    write_manifest,
)
from .trace_profiles import PAPER_TRACE_PROFILES, get_trace_profile, sample_profile_lengths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_result(output_dir: Path, result: SimulationResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_csv(output_dir / "requests.csv", result.requests)
    _write_csv(output_dir / "iterations.csv", result.iterations)
    _write_csv(output_dir / "switches.csv", result.switches)
    _write_csv(output_dir / "pipeline_events.csv", result.pipeline_events)


def _print_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency_ms"]
    print(
        f"status={summary['status']} model={summary['model']} "
        f"policy={summary['policy']} mapping={summary['operator_mapping']} "
        f"fidelity={summary['fidelity']} pipeline={summary['pipeline_model']}"
    )
    print(
        f"requests={summary['completed_requests']}/{summary['requests']} "
        f"tokens={summary['output_tokens']} iterations={summary['iterations']} "
        f"time={summary['simulated_time_s']:.6f}s"
    )
    print(
        f"throughput={summary['throughput_tokens_s']:.3f} token/s "
        f"avg_batch={summary['average_batch_size']:.2f} "
        f"mean_request_TPOT={latency['mean_tpot']:.3f}ms"
    )
    print(
        f"queue: mean={latency['queue_mean']:.3f}ms "
        f"p95={latency['queue_p95']:.3f}ms "
        f"p99={latency['queue_p99']:.3f}ms"
    )
    if summary["slo"]["evaluated"]:
        print(
            f"SLO attainment={summary['slo']['attainment']:.1%} "
            f"goodput={summary['slo']['goodput_rps']:.6f} request/s "
            f"target_met={summary['slo']['target_met']}"
        )
    print(
        f"peak_KV: PIM={summary['peak_pim_kv_gib']:.3f}/"
        f"{summary['pim_kv_capacity_gib']:.1f}GiB, "
        f"GPU={summary['peak_gpu_kv_gib']:.3f}/"
        f"{summary['gpu_kv_capacity_gib']:.1f}GiB, "
        f"switches={summary['tail_switches']}"
    )
    print(
        f"phase bottlenecks: GPU={summary['gpu_bottleneck_phase_fraction']:.1%}, "
        f"PIM={summary['pim_bottleneck_phase_fraction']:.1%}"
    )


def _apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    simulation = config.simulation
    policy = simulation.policy
    if getattr(args, "policy", None):
        policy = replace(policy, name=args.policy)
    values: dict[str, Any] = {"policy": policy}
    if getattr(args, "fidelity", None):
        values["fidelity"] = args.fidelity
    if getattr(args, "record_every", None) is not None:
        values["record_every"] = args.record_every
    if getattr(args, "pipeline_model", None):
        values["pipeline_model"] = args.pipeline_model
    if getattr(args, "subbatch_scheduler", None):
        values["subbatch_scheduler"] = args.subbatch_scheduler
    if getattr(args, "operator_mapping", None):
        values["operator_mapping"] = args.operator_mapping
    if getattr(args, "record_pipeline_events", False):
        values["record_pipeline_events"] = True
    if getattr(args, "admission_scheduler", None):
        values["admission_scheduler"] = args.admission_scheduler
    return replace(config, simulation=replace(simulation, **values))


def command_simulate(args: argparse.Namespace) -> int:
    config = _apply_overrides(load_config(args.config), args)
    result = run_simulation(config)
    _print_summary(result.summary)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        _write_result(output_dir, result)
        print(f"results={output_dir}")
    if args.json:
        print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))
    return 0


def _numbers(raw: str | None, cast: type[int]) -> list[int] | None:
    if raw is None:
        return None
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def _gpu_budgets(raw: str) -> list[float | None]:
    budgets: list[float | None] = []
    for value in raw.split(","):
        normalized = value.strip().lower()
        if not normalized:
            continue
        budgets.append(None if normalized in ("max", "physical") else float(normalized))
    if not budgets:
        raise ValueError("at least one GPU KV budget is required")
    return budgets


def command_sweep(args: argparse.Namespace) -> int:
    base = _apply_overrides(load_config(args.config), args)
    if base.workload.type != "static_batch":
        raise ValueError("sweep currently requires a static_batch config")
    batches = _numbers(args.batch_sizes, int) or [int(base.workload.batch_size or 1)]
    inputs = _numbers(args.input_lengths, int) or [int(base.workload.input_tokens or 1)]
    outputs = _numbers(args.output_lengths, int) or [int(base.workload.output_tokens or 1)]
    rows: list[dict[str, Any]] = []
    mappings = (
        [value.strip() for value in args.operator_mappings.split(",") if value.strip()]
        if args.operator_mappings
        else [base.simulation.operator_mapping]
    )
    comparison_uses_event_model = len(set(mappings)) > 1
    if comparison_uses_event_model and base.simulation.pipeline_model != "rankset_event":
        print(
            "mapping comparison: using rankset_event for CHIME so every row "
            "shares the same event engine",
            file=sys.stderr,
        )
    for mapping, batch, input_tokens, output_tokens in product(
        mappings, batches, inputs, outputs
    ):
        workload = replace(
            base.workload,
            batch_size=batch,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        simulation = replace(
            base.simulation,
            max_batch_size=max(batch, base.simulation.max_batch_size),
            operator_mapping=mapping,
            pipeline_model=(
                "rankset_event"
                if comparison_uses_event_model
                else base.simulation.pipeline_model
            ),
        )
        result = run_simulation(
            replace(base, workload=workload, simulation=simulation)
        )
        rows.append(
            {
                "batch_size": batch,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "policy": result.summary["policy"],
                "operator_mapping": result.summary["operator_mapping"],
                "operator_mapping_label": result.summary[
                    "operator_mapping_label"
                ],
                "pipeline_model": result.summary["pipeline_model"],
                "fidelity": result.summary["fidelity"],
                "throughput_tokens_s": result.summary["throughput_tokens_s"],
                "simulated_time_s": result.summary["simulated_time_s"],
                "mean_tpot_ms": result.summary["latency_ms"]["mean_tpot"],
                "peak_pim_kv_gib": result.summary["peak_pim_kv_gib"],
                "peak_gpu_kv_gib": result.summary["peak_gpu_kv_gib"],
                "pim_persistent_weight_gib": result.summary[
                    "pim_persistent_weight_gib"
                ],
                "pim_kv_capacity_gib": result.summary["pim_kv_capacity_gib"],
                "gpu_kv_capacity_gib": result.summary["gpu_kv_capacity_gib"],
                "tail_switches": result.summary["tail_switches"],
                "pim_bottleneck_fraction": result.summary[
                    "pim_bottleneck_phase_fraction"
                ],
            }
        )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output, rows)
        print(f"results={output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


def command_benefit_map(args: argparse.Namespace) -> int:
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unknown = set(models) - set(MODEL_PRESETS)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    rows = future_only_benefit_rows(
        model_names=models,
        batch_sizes=_numbers(args.batch_sizes, int) or [4, 12],
        input_lengths=_numbers(args.input_lengths, int) or [128, 8192],
        output_lengths=_numbers(args.output_lengths, int) or [128, 4096],
        gpu_kv_budgets_gib=_gpu_budgets(args.gpu_kv_budgets),
        decision_interval=args.decision_interval,
        planning_horizon_tokens=args.planning_horizon,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_homogeneous_lifecycle(args: argparse.Namespace) -> int:
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unknown = set(models) - set(MODEL_PRESETS)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    rows = homogeneous_lifecycle_rows(
        model_names=models,
        batch_sizes=_numbers(args.batch_sizes, int) or [2, 4, 8, 12, 16, 32],
        input_lengths=_numbers(args.input_lengths, int) or [128, 2048, 5200, 8192],
        output_lengths=_numbers(args.output_lengths, int) or [256, 1024, 4096],
        gpu_kv_budgets_gib=_gpu_budgets(args.gpu_kv_budgets),
        mode_grid_points=args.mode_grid_points,
        quadrature_samples=args.quadrature_samples,
        exact_finalists=args.exact_finalists,
        pipeline_model=args.pipeline_model,
        policy_pipeline_model=args.policy_pipeline_model,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_composition_map(args: argparse.Namespace) -> int:
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unknown = set(models) - set(MODEL_PRESETS)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    rows = future_only_composition_rows(
        model_names=models,
        batch_sizes=_numbers(args.batch_sizes, int) or [12],
        total_contexts=_numbers(args.total_contexts, int) or [8192],
        output_lengths=_numbers(args.output_lengths, int)
        or [256, 1024, 2048, 4096, 6144],
        gpu_kv_budgets_gib=_gpu_budgets(args.gpu_kv_budgets),
        decision_interval=args.decision_interval,
        planning_horizon_tokens=args.planning_horizon,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_lifetime_comparison(args: argparse.Namespace) -> int:
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unknown = set(models) - set(MODEL_PRESETS)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(sorted(unknown))}")
    rows = lifetime_policy_comparison_rows(
        model_names=models,
        batch_sizes=_numbers(args.batch_sizes, int) or [2, 4],
        input_lengths=_numbers(args.input_lengths, int) or [4096, 5200],
        output_lengths=_numbers(args.output_lengths, int) or [64],
        gpu_kv_budgets_gib=_gpu_budgets(args.gpu_kv_budgets),
        decision_interval=args.decision_interval,
        planning_horizon_tokens=args.planning_horizon,
        lifetime_evaluation_stride_tokens=args.lifetime_stride,
        beam_depth=args.beam_depth,
        beam_width=args.beam_width,
        beam_request_candidates=args.beam_request_candidates,
        beam_evaluation_stride_tokens=args.beam_evaluation_stride,
        beam_terminal_samples=args.beam_terminal_samples,
        beam_terminal_mode=args.beam_terminal_mode,
        beam_relative_improvement_threshold=args.beam_threshold,
        include_global_exact=not args.skip_global_exact,
        exact_max_candidates=args.exact_max_candidates,
        exact_max_cost_evaluations=args.exact_max_cost_evaluations,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_calibrate(_: argparse.Namespace) -> int:
    model = MODEL_PRESETS["gpt175b"]
    system = SYSTEM_PRESETS["chime_a100"]
    cost_model = ChimeCostModel(model, system, "tile")
    requests = [
        RequestState(f"cal-{index}", 0.0, input_tokens=5200, output_tokens=1)
        for index in range(12)
    ]
    cost = cost_model.iteration_cost(views_for_requests(requests))
    half = cost.subbatches[0]
    lower_bound = cost_model.physical_pim_lower_bound_us(
        cost_model.partition_views(views_for_requests(requests))[0]
    )
    checks = {
        "calibration_scope": "Fig.17 in-sample; not independent validation",
        "gpu_dense_hbm_efficiency": system.gpu.dense_hbm_efficiency,
        "effective_link_gbps": system.link.host_to_pim_gbps,
        "removed_fig14_correction_factors": True,
        "gpu_qkv_us": half.dense.qkv_us,
        "gpu_proj_ff_us": half.dense.proj_ff_us,
        "pim_attention_core_us": half.pim.core_us,
        "pim_attention_total_us": half.pim.total_us,
        "qkv_h2d_us": half.pim.h2d_us,
        "output_d2h_us": half.pim.d2h_us,
        "single_layer_two_subbatch_us": cost.per_layer_us,
        "pim_physical_lower_bound_us": lower_bound,
        "gpt175b_kv_mib_per_token": model.kv_bytes_per_token / 2**20,
        "gpu_free_kv_gib": system.gpu_kv_capacity_bytes(model) / GiB,
    }
    # Fig.17 uses a representative OpenR1 batch whose exact request lengths are
    # not published. A 6-request half-batch at 5.2K context reconstructs its
    # reported component ranges and guards against unit/topology mistakes.
    assert 120 <= checks["gpu_qkv_us"] <= 150
    assert 370 <= checks["gpu_proj_ff_us"] <= 440
    assert 430 <= checks["pim_attention_core_us"] <= 530
    assert 7.2 <= checks["qkv_h2d_us"] <= 7.4
    assert 2.4 <= checks["output_d2h_us"] <= 2.5
    assert half.pim.bank_us + 1e-9 >= lower_bound
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("calibration_checks=passed")
    return 0


def _emit_rows(rows: list[dict[str, Any]], output: str | None) -> None:
    if output:
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(path, rows)
        print(f"results={path}")
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def command_trace_profile(args: argparse.Namespace) -> int:
    profile = get_trace_profile(args.profile)
    lengths = sample_profile_lengths(args.profile, args.count, args.seed)
    arrival_rng = random.Random(args.seed + 1)
    now_us = 0.0
    rows: list[dict[str, Any]] = []
    for index, (input_tokens, output_tokens) in enumerate(lengths):
        if index and args.arrival_rate_rps > 0:
            now_us += arrival_rng.expovariate(args.arrival_rate_rps) * 1e6
        rows.append(
            {
                "request_id": f"{args.profile}-{index}",
                "arrival_ms": now_us / 1e3,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
    inputs = [value[0] for value in lengths]
    outputs = [value[1] for value in lengths]
    print(
        f"profile={profile.name} source={profile.source} count={args.count} seed={args.seed}",
        file=sys.stderr if not args.output else sys.stdout,
    )
    print(
        "sample moments: "
        f"input={fmean(inputs):.2f}+/-{pstdev(inputs):.2f}, "
        f"output={fmean(outputs):.2f}+/-{pstdev(outputs):.2f}",
        file=sys.stderr if not args.output else sys.stdout,
    )
    _emit_rows(rows, args.output)
    return 0


def command_hardware_ablation(args: argparse.Namespace) -> int:
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    lengths = _numbers(args.token_lengths, int) or [512, 1024, 2048, 4096, 8192]
    rows = hardware_ablation_rows(models, lengths)
    for model in models:
        model_label = MODEL_PRESETS[model].name
        model_rows = [row for row in rows if row["model"] == model_label]
        relayout_gain = max(
            1.0 - float(row["normalized_latency"])
            for row in model_rows
            if row["variant"] == "+Relayout"
        )
        longest = max(lengths)
        pipeline = next(
            row
            for row in model_rows
            if row["variant"] == "+Pipeline" and row["token_length"] == longest
        )
        print(
            f"{model_label}: max relayout reduction={relayout_gain:.1%}, "
            f"pipeline reduction at {longest}={1-float(pipeline['normalized_latency']):.1%}"
        )
    _emit_rows(rows, args.output)
    return 0


def command_scalability(args: argparse.Namespace) -> int:
    factors = _numbers(args.factors, int) or [1, 2, 4, 8]
    rows = scalability_rows(
        model_name=args.model,
        profile=args.profile,
        request_count=args.requests,
        seed=args.seed,
        factors=factors,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_rankset_sensitivity(args: argparse.Namespace) -> int:
    rows = aggregate_rankset_sensitivity_rows(
        model_name=args.model,
        batch_size=args.batch_size,
        context_tokens=args.context_tokens,
    )
    _emit_rows(rows, args.output)
    return 0


def command_policy_comparison(args: argparse.Namespace) -> int:
    rows = policy_comparison_rows(
        model_name=args.model,
        profile=args.profile,
        request_count=args.requests,
        seed=args.seed,
        arrival_rate_rps=args.arrival_rate_rps,
        decision_interval=args.decision_interval,
        planning_horizon_tokens=args.planning_horizon,
        lifetime_evaluation_stride_tokens=args.lifetime_stride,
        beam_depth=args.beam_depth,
        beam_width=args.beam_width,
        beam_request_candidates=args.beam_request_candidates,
        beam_evaluation_stride_tokens=args.beam_evaluation_stride,
        beam_terminal_samples=args.beam_terminal_samples,
        beam_terminal_mode=args.beam_terminal_mode,
        beam_relative_improvement_threshold=args.beam_threshold,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_beam_terminal_ablation(args: argparse.Namespace) -> int:
    modes = tuple(
        value.strip() for value in args.terminal_modes.split(",") if value.strip()
    )
    rows = beam_terminal_ablation_rows(
        model_name=args.model,
        profile=args.profile,
        request_count=args.requests,
        seed=args.seed,
        arrival_rate_rps=args.arrival_rate_rps,
        terminal_modes=modes,
        decision_interval=args.decision_interval,
        planning_horizon_tokens=args.planning_horizon,
        beam_depth=args.beam_depth,
        beam_width=args.beam_width,
        beam_request_candidates=args.beam_request_candidates,
        beam_evaluation_stride_tokens=args.beam_evaluation_stride,
        beam_terminal_samples=args.beam_terminal_samples,
        beam_relative_improvement_threshold=args.beam_threshold,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_open_slo_sweep(args: argparse.Namespace) -> int:
    budgets = _gpu_budgets(args.gpu_kv_budget)
    if len(budgets) != 1:
        raise ValueError("open-slo-sweep accepts exactly one GPU KV budget")
    rows = open_arrival_slo_rows(
        model_name=args.model,
        request_count=args.requests,
        seeds=tuple(_numbers(args.seeds, int) or [0]),
        arrival_rates_rps=tuple(_numbers(args.arrival_rates, float) or [0.02]),
        max_batch_size=args.max_batch_size,
        input_mean=args.input_mean,
        input_std=args.input_std,
        input_min=args.input_min,
        input_max=args.input_max,
        output_mean=args.output_mean,
        output_std=args.output_std,
        output_min=args.output_min,
        output_max=args.output_max,
        gpu_kv_budget_gib=budgets[0],
        slo_queue_ms=args.slo_queue_ms,
        slo_tpot_ms=args.slo_tpot_ms,
        slo_e2e_ms=args.slo_e2e_ms,
        slo_attainment_target=args.slo_attainment_target,
        decision_interval=args.decision_interval,
        planning_horizon_tokens=args.planning_horizon,
        beam_depth=args.beam_depth,
        beam_width=args.beam_width,
        beam_request_candidates=args.beam_request_candidates,
        beam_evaluation_stride_tokens=args.beam_evaluation_stride,
        beam_terminal_samples=args.beam_terminal_samples,
        beam_relative_improvement_threshold=args.beam_threshold,
        include_event_driven_variant=args.event_driven_ablation,
        progress=lambda label: print(f"running={label}", file=sys.stderr),
    )
    _emit_rows(rows, args.output)
    return 0


def command_operator_offload(args: argparse.Namespace) -> int:
    contexts = _numbers(args.context_lengths, int) or [512, 1024, 2048, 4096, 8192]
    fractions = [
        float(value.strip())
        for value in args.fractions.split(",")
        if value.strip()
    ]
    rows = operator_offload_rows(
        model_name=args.model,
        subbatch_size=args.subbatch_size,
        context_lengths=contexts,
        fractions=fractions,
    )
    summary_stream = sys.stdout if args.output else sys.stderr
    for context in contexts:
        best = max(
            (row for row in rows if row["context_tokens"] == context),
            key=lambda row: float(row["speedup"]),
        )
        print(
            f"context={context}: best_fraction={best['o_proj_fraction_on_pim']}, "
            f"optimistic_speedup={float(best['speedup']):.3f}x, "
            f"persistent_weight={float(best['persistent_pim_weight_gib']):.2f}GiB",
            file=summary_stream,
        )
    _emit_rows(rows, args.output)
    return 0


def command_dynamic_operator_offload(args: argparse.Namespace) -> int:
    rows = dynamic_operator_offload_rows(
        model_name=args.model,
        system_name=args.system,
        subbatch_sizes=_numbers(args.subbatch_sizes, int) or (1, 4, 8, 16),
        input_lengths=_numbers(args.input_lengths, int) or (128, 2048, 8192),
        output_lengths=_numbers(args.output_lengths, int) or (4096,),
        fractions=(
            float(value.strip())
            for value in args.fractions.split(",")
            if value.strip()
        ),
        decision_interval=args.decision_interval,
        min_residence_tokens=args.min_residence,
        switch_penalty_us=args.switch_penalty_us,
    )
    summary_stream = sys.stdout if args.output else sys.stderr
    for row in rows:
        print(
            f"model={row['model']}:SB{row['subbatch_size']}:I{row['input_tokens']}:"
            f"O{row['output_tokens']}: static={float(row['best_static_speedup']):.4f}x "
            f"dynamic={float(row['constrained_dynamic_speedup']):.4f}x "
            f"dynamic/static={float(row['dynamic_vs_best_static']):.4f}x "
            f"switches={row['constrained_dynamic_switches']} "
            f"schedule={row['constrained_dynamic_schedule']}",
            file=summary_stream,
        )
    _emit_rows(rows, args.output)
    return 0


def command_static_operator_mapping(args: argparse.Namespace) -> int:
    rows = static_operator_mapping_rows(
        model_names=tuple(
            value.strip() for value in args.models.split(",") if value.strip()
        ),
        mappings=tuple(
            value.strip() for value in args.operator_mappings.split(",") if value.strip()
        ),
        batch_sizes=_numbers(args.batch_sizes, int) or (1, 4, 8, 16, 32),
        context_lengths=_numbers(args.context_lengths, int)
        or (128, 2048, 8192, 32768),
        output_tokens=args.output_tokens,
    )
    _emit_rows(rows, args.output)
    return 0


def command_joint_af_small_oracle(args: argparse.Namespace) -> int:
    models = tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    batch_sizes = tuple(_numbers(args.batch_sizes, int) or (2, 4))
    input_lengths = tuple(_numbers(args.input_lengths, int) or (2048, 8192))
    output_lengths = tuple(_numbers(args.output_lengths, int) or (32, 64))
    fractions = tuple(
        float(value.strip())
        for value in args.fractions.split(",")
        if value.strip()
    )
    common = {
        "fractions": fractions,
        "decision_interval": args.decision_interval,
        "min_residence_tokens": args.min_residence,
        "max_kv_switches_per_decision": args.max_kv_switches,
        "max_cost_evaluations": args.max_cost_evaluations,
        "greedy_terminal_samples": args.greedy_terminal_samples,
        "progress": lambda label: print(f"running={label}", file=sys.stderr),
    }
    rows: list[dict[str, Any]] = []
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for model, batch, input_tokens, output_tokens in product(
            models, batch_sizes, input_lengths, output_lengths
        ):
            fraction_key = "_".join(f"{value:g}" for value in fractions)
            max_switches = (
                args.max_kv_switches
                if args.max_kv_switches is not None
                else batch
            )
            filename = (
                f"jointv7-{model.replace('.', '_')}-b{batch}-"
                f"i{input_tokens}-o{output_tokens}-d{args.decision_interval}-"
                f"r{args.min_residence}-k{max_switches}-f{fraction_key}-"
                f"g{args.greedy_terminal_samples}.csv"
            )
            checkpoint = checkpoint_dir / filename
            if checkpoint.exists():
                with checkpoint.open(encoding="utf-8", newline="") as handle:
                    cached = list(csv.DictReader(handle))
                if len(cached) != 1:
                    raise ValueError(
                        f"checkpoint {checkpoint} must contain exactly one row"
                    )
                rows.extend(cached)
                print(f"cached={checkpoint}", file=sys.stderr)
                continue
            point_rows = joint_af_small_oracle_rows(
                model_names=(model,),
                batch_sizes=(batch,),
                input_lengths=(input_tokens,),
                output_lengths=(output_tokens,),
                **common,
            )
            _write_csv(checkpoint, point_rows)
            rows.extend(point_rows)
            print(f"checkpoint={checkpoint}", file=sys.stderr)
    else:
        rows = joint_af_small_oracle_rows(
            model_names=models,
            batch_sizes=batch_sizes,
            input_lengths=input_lengths,
            output_lengths=output_lengths,
            **common,
        )
    _emit_rows(rows, args.output)
    return 0


def command_joint_af_heterogeneous(args: argparse.Namespace) -> int:
    models = tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    profiles = tuple(
        value.strip() for value in args.profiles.split(",") if value.strip()
    )
    input_lengths = tuple(_numbers(args.input_lengths, int) or (12000,))
    fractions = tuple(
        float(value.strip())
        for value in args.fractions.split(",")
        if value.strip()
    )
    common = {
        "fractions": fractions,
        "decision_interval": args.decision_interval,
        "min_residence_tokens": args.min_residence,
        "max_kv_switches_per_decision": args.max_kv_switches,
        "max_cost_evaluations": args.max_cost_evaluations,
        "greedy_terminal_samples": args.greedy_terminal_samples,
        "progress": lambda label: print(f"running={label}", file=sys.stderr),
    }
    rows: list[dict[str, Any]] = []
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for model, profile, input_tokens in product(
            models, profiles, input_lengths
        ):
            fraction_key = "_".join(f"{value:g}" for value in fractions)
            filename = (
                f"jointhetv4-{model.replace('.', '_')}-{profile}-"
                f"i{input_tokens}-d{args.decision_interval}-"
                f"r{args.min_residence}-k{args.max_kv_switches}-"
                f"f{fraction_key}-g{args.greedy_terminal_samples}.csv"
            )
            checkpoint = checkpoint_dir / filename
            if checkpoint.exists():
                with checkpoint.open(encoding="utf-8", newline="") as handle:
                    cached = list(csv.DictReader(handle))
                if len(cached) != 1:
                    raise ValueError(
                        f"checkpoint {checkpoint} must contain exactly one row"
                    )
                rows.extend(cached)
                print(f"cached={checkpoint}", file=sys.stderr)
                continue
            point_rows = joint_af_heterogeneous_oracle_rows(
                model_names=(model,),
                profile_names=(profile,),
                input_lengths=(input_tokens,),
                **common,
            )
            _write_csv(checkpoint, point_rows)
            rows.extend(point_rows)
            print(f"checkpoint={checkpoint}", file=sys.stderr)
    else:
        rows = joint_af_heterogeneous_oracle_rows(
            model_names=models,
            profile_names=profiles,
            input_lengths=input_lengths,
            **common,
        )
    _emit_rows(rows, args.output)
    return 0


def command_joint_af_event_replay(args: argparse.Namespace) -> int:
    source_path = Path(args.source).resolve()
    with source_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    selected_indices = tuple(_numbers(args.row_indices, int))
    if not selected_indices:
        selected_indices = tuple(range(len(source_rows)))
    for index in selected_indices:
        if not 0 <= index < len(source_rows):
            raise ValueError(
                f"row index {index} is outside source with {len(source_rows)} rows"
            )
    strides = tuple(_numbers(args.evaluation_strides, int) or (64,))
    common = {
        "model_name": args.model,
        "evaluation_strides": strides,
        "include_split_aware_oproj": args.split_aware_oproj,
        "include_owner_coupled_merge": args.owner_coupled_merge,
        "owner_coupled_pim_merge_cost_scale": args.pim_merge_cost_scale,
        "progress": lambda label: print(f"running={label}", file=sys.stderr),
    }
    rows: list[dict[str, Any]] = []
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stride_key = "_".join(map(str, strides))
        for index in selected_indices:
            if args.owner_coupled_merge:
                scale_key = f"{args.pim_merge_cost_scale:g}".replace(".", "p")
                version = f"jointeventv3owner-s{scale_key}"
            elif args.split_aware_oproj:
                version = "jointeventv2split"
            else:
                version = "jointeventv1"
            filename = (
                f"{version}-{source_path.stem}-row{index}-"
                f"stride{stride_key}.csv"
            )
            checkpoint = checkpoint_dir / filename
            if checkpoint.exists():
                with checkpoint.open(encoding="utf-8", newline="") as handle:
                    cached = list(csv.DictReader(handle))
                rows.extend(cached)
                print(f"cached={checkpoint}", file=sys.stderr)
                continue
            point_rows = joint_af_event_replay_rows(
                (source_rows[index],), **common
            )
            for row in point_rows:
                row["source_index"] = index
                row["source_file"] = str(source_path)
            _write_csv(checkpoint, point_rows)
            rows.extend(point_rows)
            print(f"checkpoint={checkpoint}", file=sys.stderr)
    else:
        for index in selected_indices:
            point_rows = joint_af_event_replay_rows(
                (source_rows[index],), **common
            )
            for row in point_rows:
                row["source_index"] = index
                row["source_file"] = str(source_path)
            rows.extend(point_rows)
    _emit_rows(rows, args.output)
    return 0


def _suite_scenarios(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _selected_suite_cases(args: argparse.Namespace):
    suite = load_suite(args.suite)
    shard = parse_shard(getattr(args, "shard", None))
    cases = select_cases(
        suite,
        stage=args.stage,
        tier=args.tier,
        scenarios=_suite_scenarios(getattr(args, "scenarios", None)),
        shard=shard,
        max_cases=getattr(args, "max_cases", None),
    )
    if not cases:
        raise ValueError("suite selection contains no cases")
    return suite, shard, cases


def command_suite_list(args: argparse.Namespace) -> int:
    suite, shard, cases = _selected_suite_cases(args)
    scenario_counts = Counter(case.scenario for case in cases)
    tier_counts = Counter(case.tier for case in cases)
    print(
        f"suite={suite.name} version={suite.version} stage={args.stage} "
        f"tier<={args.tier} cases={len(cases)}"
    )
    if shard:
        print(f"shard={shard[0]}/{shard[1]}")
    print(
        "scenarios="
        + ",".join(f"{name}:{count}" for name, count in sorted(scenario_counts.items()))
    )
    print(
        "case_tiers="
        + ",".join(f"{name}:{count}" for name, count in sorted(tier_counts.items()))
    )
    if args.verbose:
        for case in cases:
            print(
                f"{case.case_id}\ttier={case.tier}\tscenario={case.scenario}\t"
                f"{case.description}"
            )
    return 0


def command_suite_run(args: argparse.Namespace) -> int:
    suite, shard, cases = _selected_suite_cases(args)
    if args.dry_run:
        print(
            f"dry_run suite={suite.name} stage={args.stage} tier<={args.tier} "
            f"cases={len(cases)}"
        )
        for case in cases:
            print(f"{case.case_id}\t{case.scenario}\t{case.tier}")
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = environment_manifest(
        suite,
        cases,
        stage=args.stage,
        tier=args.tier,
        shard=shard,
        workers=args.workers,
    )
    manifest["resume"] = not args.rerun
    manifest["status"] = "running"
    write_manifest(output_dir / "run.json", manifest)
    print(
        f"suite={suite.name} stage={args.stage} tier<={args.tier} "
        f"cases={len(cases)} workers={args.workers} output={output_dir}"
    )
    rows = run_cases(
        cases,
        output_dir,
        workers=args.workers,
        resume=not args.rerun,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    write_index(output_dir / "suite-index.csv", rows)
    status_counts = Counter(str(row["status"]) for row in rows)
    manifest["status"] = "failed" if status_counts.get("failed", 0) else "finished"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status_counts"] = dict(sorted(status_counts.items()))
    write_manifest(output_dir / "run.json", manifest)
    print(
        "status_counts="
        + ",".join(f"{name}:{count}" for name, count in sorted(status_counts.items()))
    )
    print(f"index={output_dir / 'suite-index.csv'}")
    return 1 if status_counts.get("failed", 0) else 0


def command_suite_validate(args: argparse.Namespace) -> int:
    suite, _, cases = _selected_suite_cases(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, issues = validate_cases(cases, output_dir)
    status_counts = Counter(str(row["status"]) for row in rows)
    print(
        f"suite={suite.name} stage={args.stage} tier<={args.tier} "
        f"expected={len(cases)} valid={len(rows)} issues={len(issues)}"
    )
    if status_counts:
        print(
            "status_counts="
            + ",".join(
                f"{name}:{count}" for name, count in sorted(status_counts.items())
            )
        )
    for issue in issues:
        print(f"issue={issue}")
    if args.rebuild_index:
        write_index(output_dir / "suite-index.csv", rows)
        print(f"rebuilt_index={output_dir / 'suite-index.csv'}")
    return 3 if issues else 0


def command_suite_compare(args: argparse.Namespace) -> int:
    index_path = Path(args.index).expanduser().resolve()
    rows = comparison_rows(read_index(index_path), args.baseline_variant)
    if not rows:
        raise ValueError("suite index contains no non-baseline variants")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, rows)
    counts = Counter(str(row["classification"]) for row in rows)
    print(f"comparisons={len(rows)}")
    print(
        "classification_counts="
        + ",".join(f"{name}:{count}" for name, count in sorted(counts.items()))
    )
    print(f"results={output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elasticafd",
        description="unified CHIME and operator-mapped GPU + DIMM-PIM simulator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="run one workload")
    simulate.add_argument("config")
    simulate.add_argument("--policy", choices=("chime", "elastic"))
    simulate.add_argument("--fidelity", choices=("roofline", "tile"))
    simulate.add_argument("--record-every", type=int)
    simulate.add_argument(
        "--pipeline-model", choices=("analytic", "rankset_event")
    )
    simulate.add_argument(
        "--subbatch-scheduler", choices=("lpt", "alignment")
    )
    simulate.add_argument(
        "--operator-mapping", choices=tuple(sorted(OPERATOR_MAPPINGS))
    )
    simulate.add_argument("--record-pipeline-events", action="store_true")
    simulate.add_argument(
        "--admission-scheduler", choices=("fcfs", "alignment")
    )
    simulate.add_argument("--output-dir")
    simulate.add_argument("--json", action="store_true")
    simulate.set_defaults(func=command_simulate)

    sweep = subparsers.add_parser("sweep", help="sweep a static-batch config")
    sweep.add_argument("config")
    sweep.add_argument("--batch-sizes")
    sweep.add_argument("--input-lengths")
    sweep.add_argument("--output-lengths")
    sweep.add_argument(
        "--operator-mappings",
        help="comma-separated mappings, e.g. gpu_only,chime,dops_af",
    )
    sweep.add_argument("--policy", choices=("chime", "elastic"))
    sweep.add_argument("--fidelity", choices=("roofline", "tile"))
    sweep.add_argument("--record-every", type=int)
    sweep.add_argument(
        "--pipeline-model", choices=("analytic", "rankset_event")
    )
    sweep.add_argument(
        "--subbatch-scheduler", choices=("lpt", "alignment")
    )
    sweep.add_argument("--output")
    sweep.set_defaults(func=command_sweep)

    benefit = subparsers.add_parser(
        "benefit-map",
        help="pair CHIME and future-only Elastic across a static workload grid",
    )
    benefit.add_argument("--models", default="gpt175b,qwen72b")
    benefit.add_argument("--batch-sizes", default="4,12")
    benefit.add_argument("--input-lengths", default="128,8192")
    benefit.add_argument("--output-lengths", default="128,4096")
    benefit.add_argument("--gpu-kv-budgets", default="0,32,max")
    benefit.add_argument("--decision-interval", type=int, default=64)
    benefit.add_argument("--planning-horizon", type=int, default=256)
    benefit.add_argument("--output")
    benefit.set_defaults(func=command_benefit_map)

    homogeneous = subparsers.add_parser(
        "homogeneous-lifecycle",
        help=(
            "compare CHIME, rolling H64, and a preplanned homogeneous "
            "lifecycle mode"
        ),
    )
    homogeneous.add_argument("--models", default="gpt175b,opt66b,qwen72b")
    homogeneous.add_argument("--batch-sizes", default="2,4,8,12,16,32")
    homogeneous.add_argument("--input-lengths", default="128,2048,5200,8192")
    homogeneous.add_argument("--output-lengths", default="256,1024,4096")
    homogeneous.add_argument("--gpu-kv-budgets", default="32,max")
    homogeneous.add_argument("--mode-grid-points", type=int, default=5)
    homogeneous.add_argument("--quadrature-samples", type=int, default=24)
    homogeneous.add_argument("--exact-finalists", type=int, default=8)
    homogeneous.add_argument(
        "--pipeline-model",
        choices=("analytic", "rankset_event"),
        default="analytic",
        help="execution cost model used for reported results",
    )
    homogeneous.add_argument(
        "--policy-pipeline-model",
        choices=("analytic", "rankset_event"),
        help=(
            "optional cost model used only for placement decisions; use "
            "analytic with rankset_event execution for bounded event replay"
        ),
    )
    homogeneous.add_argument("--output")
    homogeneous.set_defaults(func=command_homogeneous_lifecycle)

    composition = subparsers.add_parser(
        "composition-map",
        help="hold final context fixed while varying prompt/output composition",
    )
    composition.add_argument("--models", default="gpt175b")
    composition.add_argument("--batch-sizes", default="12")
    composition.add_argument("--total-contexts", default="8192")
    composition.add_argument(
        "--output-lengths", default="256,1024,2048,4096,6144"
    )
    composition.add_argument("--gpu-kv-budgets", default="32,max")
    composition.add_argument("--decision-interval", type=int, default=64)
    composition.add_argument("--planning-horizon", type=int, default=256)
    composition.add_argument("--output")
    composition.set_defaults(func=command_composition_map)

    lifetime = subparsers.add_parser(
        "lifetime-comparison",
        help=(
            "compare horizon, lifetime, rolling-beam, and small closed-batch "
            "exact placement"
        ),
    )
    lifetime.add_argument("--models", default="gpt175b")
    lifetime.add_argument("--batch-sizes", default="2,4")
    lifetime.add_argument("--input-lengths", default="4096,5200")
    lifetime.add_argument("--output-lengths", default="64")
    lifetime.add_argument("--gpu-kv-budgets", default="32,max")
    lifetime.add_argument("--decision-interval", type=int, default=32)
    lifetime.add_argument("--planning-horizon", type=int, default=256)
    lifetime.add_argument("--lifetime-stride", type=int, default=16)
    lifetime.add_argument("--beam-depth", type=int, default=4)
    lifetime.add_argument("--beam-width", type=int, default=4)
    lifetime.add_argument("--beam-request-candidates", type=int, default=6)
    lifetime.add_argument("--beam-evaluation-stride", type=int, default=16)
    lifetime.add_argument("--beam-terminal-samples", type=int, default=8)
    lifetime.add_argument(
        "--beam-terminal-mode",
        choices=("horizon", "pim_lifetime", "tail_lifetime"),
        default="horizon",
    )
    lifetime.add_argument("--beam-threshold", type=float, default=0.001)
    lifetime.add_argument("--exact-max-candidates", type=int, default=8)
    lifetime.add_argument(
        "--exact-max-cost-evaluations", type=int, default=200000
    )
    lifetime.add_argument("--skip-global-exact", action="store_true")
    lifetime.add_argument("--output")
    lifetime.set_defaults(func=command_lifetime_comparison)

    calibrate = subparsers.add_parser(
        "calibrate", help="run CHIME Fig.17-scale component sanity checks"
    )
    calibrate.set_defaults(func=command_calibrate)

    trace_profile = subparsers.add_parser(
        "trace-profile", help="materialize a CHIME Table III length profile"
    )
    trace_profile.add_argument(
        "--profile", choices=tuple(PAPER_TRACE_PROFILES), default="openr1"
    )
    trace_profile.add_argument("--count", type=int, default=1000)
    trace_profile.add_argument("--seed", type=int, default=0)
    trace_profile.add_argument("--arrival-rate-rps", type=float, default=0.0)
    trace_profile.add_argument("--output")
    trace_profile.set_defaults(func=command_trace_profile)

    ablation = subparsers.add_parser(
        "hardware-ablation", help="run a CHIME Fig.14-style PIM ablation"
    )
    ablation.add_argument("--models", default="gpt175b,qwen72b")
    ablation.add_argument("--token-lengths", default="512,1024,2048,4096,8192")
    ablation.add_argument("--output")
    ablation.set_defaults(func=command_hardware_ablation)

    scalability = subparsers.add_parser(
        "scalability", help="run a CHIME Fig.13-style capacity/bandwidth study"
    )
    scalability.add_argument("--model", choices=tuple(MODEL_PRESETS), default="gpt175b")
    scalability.add_argument(
        "--profile", choices=tuple(PAPER_TRACE_PROFILES), default="openr1"
    )
    scalability.add_argument("--requests", type=int, default=32)
    scalability.add_argument("--seed", type=int, default=0)
    scalability.add_argument("--factors", default="1,2,4,8")
    scalability.add_argument("--output")
    scalability.set_defaults(func=command_scalability)

    sensitivity = subparsers.add_parser(
        "rankset-sensitivity",
        help="compare per-layer and aggregate-rankset bandwidth assumptions",
    )
    sensitivity.add_argument("--model", choices=tuple(MODEL_PRESETS), default="gpt175b")
    sensitivity.add_argument("--batch-size", type=int, default=12)
    sensitivity.add_argument("--context-tokens", type=int, default=5200)
    sensitivity.add_argument("--output")
    sensitivity.set_defaults(func=command_rankset_sensitivity)

    policies = subparsers.add_parser(
        "policy-comparison",
        help="compare fixed/profile and greedy/rolling oracle placement",
    )
    policies.add_argument("--model", choices=tuple(MODEL_PRESETS), default="gpt175b")
    policies.add_argument(
        "--profile", choices=tuple(PAPER_TRACE_PROFILES), default="openr1"
    )
    policies.add_argument("--requests", type=int, default=16)
    policies.add_argument("--seed", type=int, default=0)
    policies.add_argument("--arrival-rate-rps", type=float, default=0.0)
    policies.add_argument("--decision-interval", type=int, default=64)
    policies.add_argument("--planning-horizon", type=int, default=256)
    policies.add_argument("--lifetime-stride", type=int, default=64)
    policies.add_argument("--beam-depth", type=int, default=4)
    policies.add_argument("--beam-width", type=int, default=4)
    policies.add_argument("--beam-request-candidates", type=int, default=6)
    policies.add_argument("--beam-evaluation-stride", type=int, default=32)
    policies.add_argument("--beam-terminal-samples", type=int, default=8)
    policies.add_argument(
        "--beam-terminal-mode",
        choices=("horizon", "pim_lifetime", "tail_lifetime"),
        default="horizon",
    )
    policies.add_argument("--beam-threshold", type=float, default=0.001)
    policies.add_argument("--output")
    policies.set_defaults(func=command_policy_comparison)

    terminal = subparsers.add_parser(
        "beam-terminal-ablation",
        help="compare rolling-beam terminal cost approximations",
    )
    terminal.add_argument(
        "--model", choices=tuple(MODEL_PRESETS), default="gpt175b"
    )
    terminal.add_argument(
        "--profile", choices=tuple(PAPER_TRACE_PROFILES), default="openr1"
    )
    terminal.add_argument("--requests", type=int, default=8)
    terminal.add_argument("--seed", type=int, default=0)
    terminal.add_argument("--arrival-rate-rps", type=float, default=0.0)
    terminal.add_argument(
        "--terminal-modes", default="horizon,pim_lifetime,tail_lifetime"
    )
    terminal.add_argument("--decision-interval", type=int, default=64)
    terminal.add_argument("--planning-horizon", type=int, default=256)
    terminal.add_argument("--beam-depth", type=int, default=4)
    terminal.add_argument("--beam-width", type=int, default=4)
    terminal.add_argument("--beam-request-candidates", type=int, default=2)
    terminal.add_argument("--beam-evaluation-stride", type=int, default=64)
    terminal.add_argument("--beam-terminal-samples", type=int, default=8)
    terminal.add_argument("--beam-threshold", type=float, default=0.001)
    terminal.add_argument("--output")
    terminal.set_defaults(func=command_beam_terminal_ablation)

    open_slo = subparsers.add_parser(
        "open-slo-sweep",
        help="compare active-only and future-aware placement under open arrivals",
    )
    open_slo.add_argument(
        "--model", choices=tuple(MODEL_PRESETS), default="gpt175b"
    )
    open_slo.add_argument("--requests", type=int, default=24)
    open_slo.add_argument("--seeds", default="0")
    open_slo.add_argument("--arrival-rates", default="0.02,0.04,0.06")
    open_slo.add_argument("--max-batch-size", type=int, default=8)
    open_slo.add_argument("--input-mean", type=float, default=4096.0)
    open_slo.add_argument("--input-std", type=float, default=2048.0)
    open_slo.add_argument("--input-min", type=int, default=512)
    open_slo.add_argument("--input-max", type=int, default=8192)
    open_slo.add_argument("--output-mean", type=float, default=2048.0)
    open_slo.add_argument("--output-std", type=float, default=1024.0)
    open_slo.add_argument("--output-min", type=int, default=256)
    open_slo.add_argument("--output-max", type=int, default=4096)
    open_slo.add_argument("--gpu-kv-budget", default="max")
    open_slo.add_argument("--slo-queue-ms", type=float, default=2000.0)
    open_slo.add_argument("--slo-tpot-ms", type=float, default=130.0)
    open_slo.add_argument("--slo-e2e-ms", type=float)
    open_slo.add_argument("--slo-attainment-target", type=float, default=0.99)
    open_slo.add_argument("--decision-interval", type=int, default=64)
    open_slo.add_argument("--planning-horizon", type=int, default=256)
    open_slo.add_argument("--beam-depth", type=int, default=4)
    open_slo.add_argument("--beam-width", type=int, default=3)
    open_slo.add_argument("--beam-request-candidates", type=int, default=2)
    open_slo.add_argument("--beam-evaluation-stride", type=int, default=8)
    open_slo.add_argument("--beam-terminal-samples", type=int, default=8)
    open_slo.add_argument("--beam-threshold", type=float, default=0.001)
    open_slo.add_argument(
        "--event-driven-ablation",
        action="store_true",
        help="add a future-aware variant that replans on admission/completion",
    )
    open_slo.add_argument("--output")
    open_slo.set_defaults(func=command_open_slo_sweep)

    offload = subparsers.add_parser(
        "operator-offload",
        help="evaluate an optimistic row-wise O_proj-to-PIM upper bound",
    )
    offload.add_argument("--model", choices=tuple(MODEL_PRESETS), default="gpt175b")
    offload.add_argument("--subbatch-size", type=int, default=6)
    offload.add_argument("--context-lengths", default="512,1024,2048,4096,8192")
    offload.add_argument("--fractions", default="0,0.125,0.25,0.5,0.75,1")
    offload.add_argument("--output")
    offload.set_defaults(func=command_operator_offload)

    dynamic_offload = subparsers.add_parser(
        "dynamic-af-offload",
        help="compare static and constrained dynamic O-projection PIM offload",
    )
    dynamic_offload.add_argument(
        "--model", choices=tuple(MODEL_PRESETS), default="gpt175b"
    )
    dynamic_offload.add_argument(
        "--system", choices=tuple(SYSTEM_PRESETS), default="chime_a100"
    )
    dynamic_offload.add_argument("--subbatch-sizes", default="1,4,8,16")
    dynamic_offload.add_argument("--input-lengths", default="128,2048,8192")
    dynamic_offload.add_argument("--output-lengths", default="4096")
    dynamic_offload.add_argument(
        "--fractions", default="0,0.125,0.25,0.5,0.75,1"
    )
    dynamic_offload.add_argument("--decision-interval", type=int, default=64)
    dynamic_offload.add_argument("--min-residence", type=int, default=512)
    dynamic_offload.add_argument("--switch-penalty-us", type=float, default=0.0)
    dynamic_offload.add_argument("--output")
    dynamic_offload.set_defaults(func=command_dynamic_operator_offload)

    mapping_map = subparsers.add_parser(
        "operator-mapping-map",
        help="compare paired coarse static A/F operator bundles without YAML",
    )
    mapping_map.add_argument("--models", default="gpt175b")
    mapping_map.add_argument(
        "--operator-mappings",
        default=(
            "chime,attn_pim_control,gpu_only,dops_af,dops_ffn,dops_all_pim"
        ),
    )
    mapping_map.add_argument("--batch-sizes", default="1,4,8,16,32")
    mapping_map.add_argument("--context-lengths", default="128,2048,8192,32768")
    mapping_map.add_argument("--output-tokens", type=int, default=1)
    mapping_map.add_argument("--output")
    mapping_map.set_defaults(func=command_static_operator_mapping)

    joint_small = subparsers.add_parser(
        "joint-af-small",
        help="run a small exact joint future-KV and O-projection oracle",
    )
    joint_small.add_argument("--models", default="gpt175b")
    joint_small.add_argument("--batch-sizes", default="2,4")
    joint_small.add_argument("--input-lengths", default="2048,8192")
    joint_small.add_argument("--output-lengths", default="32,64")
    joint_small.add_argument("--fractions", default="0,0.25,0.5,1")
    joint_small.add_argument("--decision-interval", type=int, default=16)
    joint_small.add_argument("--min-residence", type=int, default=16)
    joint_small.add_argument(
        "--max-kv-switches",
        type=int,
        help="maximum request-tail changes at one exact decision",
    )
    joint_small.add_argument(
        "--max-cost-evaluations", type=int, default=200000
    )
    joint_small.add_argument(
        "--greedy-terminal-samples",
        type=int,
        default=8,
        help="sample count for exact-length lifetime scoring in the heuristic",
    )
    joint_small.add_argument(
        "--checkpoint-dir",
        help="write and reuse one CSV checkpoint per parameter point",
    )
    joint_small.add_argument("--output")
    joint_small.set_defaults(func=command_joint_af_small_oracle)

    joint_heterogeneous = subparsers.add_parser(
        "joint-af-heterogeneous",
        help="run an exact completion-event AF scheduling comparison",
    )
    joint_heterogeneous.add_argument("--models", default="gpt175b")
    joint_heterogeneous.add_argument(
        "--profiles",
        default="balanced,staircase,two_wave,three_then_long",
    )
    joint_heterogeneous.add_argument("--input-lengths", default="12000")
    joint_heterogeneous.add_argument("--fractions", default="0,0.5,0.75,1")
    joint_heterogeneous.add_argument(
        "--decision-interval", type=int, default=64
    )
    joint_heterogeneous.add_argument("--min-residence", type=int, default=64)
    joint_heterogeneous.add_argument("--max-kv-switches", type=int, default=1)
    joint_heterogeneous.add_argument(
        "--max-cost-evaluations", type=int, default=3000000
    )
    joint_heterogeneous.add_argument(
        "--greedy-terminal-samples", type=int, default=8
    )
    joint_heterogeneous.add_argument("--checkpoint-dir")
    joint_heterogeneous.add_argument("--output")
    joint_heterogeneous.set_defaults(func=command_joint_af_heterogeneous)

    joint_event = subparsers.add_parser(
        "joint-af-event-replay",
        help="replay Stage 3B1 candidates on the dependency-aware event DAG",
    )
    joint_event.add_argument("--source", required=True)
    joint_event.add_argument("--model", default="gpt175b")
    joint_event.add_argument(
        "--row-indices",
        help="comma-separated zero-based source CSV rows; default is all",
    )
    joint_event.add_argument("--evaluation-strides", default="64")
    joint_event.add_argument(
        "--split-aware-oproj",
        action="store_true",
        help=(
            "also evaluate request-level routing that keeps O_proj on GPU "
            "for split/GPU-attention requests"
        ),
    )
    joint_event.add_argument(
        "--owner-coupled-merge",
        action="store_true",
        help=(
            "also evaluate PIM softmax merge whenever O_proj has a PIM "
            "fraction, avoiding merge-to-PIM activation round trips"
        ),
    )
    joint_event.add_argument(
        "--pim-merge-cost-scale",
        type=float,
        default=0.0,
        help=(
            "PIM merge duration as a sensitivity multiple of the GPU merge "
            "estimate; zero is an optimistic lower bound"
        ),
    )
    joint_event.add_argument("--checkpoint-dir")
    joint_event.add_argument("--output")
    joint_event.set_defaults(func=command_joint_af_event_replay)

    suite_list = subparsers.add_parser(
        "suite-list",
        help="inspect a versioned workload-suite selection without running it",
    )
    suite_list.add_argument("--suite", default="workloads-v1")
    suite_list.add_argument("--stage", default="e0")
    suite_list.add_argument(
        "--tier", choices=("smoke", "core", "full"), default="core"
    )
    suite_list.add_argument(
        "--scenarios", help="comma-separated scenario filter"
    )
    suite_list.add_argument(
        "--shard", help="zero-based INDEX/COUNT selection, e.g. 0/4"
    )
    suite_list.add_argument("--max-cases", type=int)
    suite_list.add_argument("--verbose", action="store_true")
    suite_list.set_defaults(func=command_suite_list)

    suite_run = subparsers.add_parser(
        "suite-run",
        help="run a workload suite with case checkpoints and optional workers",
    )
    suite_run.add_argument("--suite", default="workloads-v1")
    suite_run.add_argument("--stage", default="e0")
    suite_run.add_argument(
        "--tier", choices=("smoke", "core", "full"), default="core"
    )
    suite_run.add_argument(
        "--scenarios", help="comma-separated scenario filter"
    )
    suite_run.add_argument(
        "--shard", help="zero-based INDEX/COUNT selection, e.g. 0/4"
    )
    suite_run.add_argument("--max-cases", type=int)
    suite_run.add_argument("--workers", type=int, default=1)
    suite_run.add_argument("--output-dir", required=True)
    suite_run.add_argument(
        "--rerun",
        action="store_true",
        help="rerun selected cases instead of reusing matching completed checkpoints",
    )
    suite_run.add_argument("--dry-run", action="store_true")
    suite_run.set_defaults(func=command_suite_run)

    suite_validate = subparsers.add_parser(
        "suite-validate",
        help="check that every selected suite case has a matching checkpoint",
    )
    suite_validate.add_argument("--suite", default="workloads-v1")
    suite_validate.add_argument("--stage", default="e0")
    suite_validate.add_argument(
        "--tier", choices=("smoke", "core", "full"), default="core"
    )
    suite_validate.add_argument(
        "--scenarios", help="comma-separated scenario filter"
    )
    suite_validate.add_argument(
        "--shard", help="zero-based INDEX/COUNT selection, e.g. 0/4"
    )
    suite_validate.add_argument("--max-cases", type=int)
    suite_validate.add_argument("--output-dir", required=True)
    suite_validate.add_argument(
        "--rebuild-index",
        action="store_true",
        help="reconstruct suite-index.csv from valid per-case checkpoints",
    )
    suite_validate.set_defaults(func=command_suite_validate)

    suite_compare = subparsers.add_parser(
        "suite-compare",
        help="pair suite variants with a baseline and calculate scenario-aware speedups",
    )
    suite_compare.add_argument("--index", required=True)
    suite_compare.add_argument("--baseline-variant", default="chime")
    suite_compare.add_argument("--output", required=True)
    suite_compare.set_defaults(func=command_suite_compare)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SimulationCapacityError as error:
        print(f"capacity_error: {error}", file=sys.stderr)
        return 2
