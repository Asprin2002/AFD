"""Reproducible CHIME-oriented experiment helpers.

These experiments intentionally expose the assumptions that are easy to hide
inside an end-to-end number: per-layer rankset bandwidth, re-layout, internal
cross-chip synchronization, and offline KV-capacity admission.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import product
from math import ceil, inf, isclose
from typing import Callable, Iterable, Mapping

from .config import project_config_from_dict
from .cost import ChimeCostModel, PimLayerCost
from .joint import (
    JointAFEventModel,
    JointAFResourceModel,
    JointAFScheduler,
    JointAction,
    JointRunResult,
    assert_exact_not_slower,
    format_joint_schedule,
    parse_joint_schedule,
)
from .simulator import SimulationCapacityError, run_simulation
from .specs import GiB, MODEL_PRESETS, SYSTEM_PRESETS, TB, SystemSpec
from .state import Location, RequestState, views_for_requests


def _benefit_classification(
    *,
    baseline_status: str,
    elastic_status: str,
    speedup: float | None,
    gpu_budget_gib: float,
    peak_gpu_kv_gib: float,
    switches: int,
    baseline_pim_bottleneck_fraction: float,
    output_tokens: int,
    decision_interval: int,
    neutral_threshold: float = 0.001,
) -> str:
    """Return a diagnostic label, not a proof of root cause."""

    if baseline_status != "completed":
        return "baseline_capacity_error"
    if elastic_status != "completed":
        return "elastic_capacity_error"
    assert speedup is not None
    if speedup > 1.0 + neutral_threshold:
        if gpu_budget_gib > 0 and peak_gpu_kv_gib >= 0.95 * gpu_budget_gib:
            return "benefit_gpu_capacity_bound"
        return "benefit_load_balancing"
    if speedup < 1.0 - neutral_threshold:
        return "regression_placement_overhead_or_myopia"
    if gpu_budget_gib <= 0:
        return "neutral_zero_gpu_kv_budget"
    if switches == 0 and baseline_pim_bottleneck_fraction < 0.5:
        return "neutral_gpu_path_dominant"
    if switches == 0 and output_tokens <= decision_interval:
        return "neutral_insufficient_future_tokens"
    if switches == 0:
        return "neutral_no_profitable_switch"
    return "neutral_after_switch"


def future_only_benefit_rows(
    model_names: Iterable[str] = ("gpt175b", "qwen72b"),
    batch_sizes: Iterable[int] = (4, 12),
    input_lengths: Iterable[int] = (128, 8192),
    output_lengths: Iterable[int] = (128, 4096),
    gpu_kv_budgets_gib: Iterable[float | None] = (0.0, 32.0, None),
    decision_interval: int = 64,
    planning_horizon_tokens: int = 256,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Pair CHIME and current future-only Elastic on identical static batches.

    ``None`` selects the physical GPU KV maximum after weights and runtime
    reserve. Each row contains both policies so workload equality is explicit.
    The classification is a first-pass diagnostic for experiment triage.
    """

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        for batch_size in batch_sizes:
            for input_tokens in input_lengths:
                for output_tokens in output_lengths:
                    for requested_budget in gpu_kv_budgets_gib:
                        label = (
                            "max" if requested_budget is None else f"{requested_budget:g}"
                        )
                        if progress:
                            progress(
                                f"{model_name}:B{batch_size}:I{input_tokens}:"
                                f"O{output_tokens}:GPU-KV={label}GiB"
                            )
                        hardware: str | dict[str, object]
                        if requested_budget is None:
                            hardware = "chime_a100"
                        else:
                            hardware = {
                                "preset": "chime_a100",
                                "gpu": {
                                    "kv_capacity_limit_gib": requested_budget,
                                },
                            }
                        common = {
                            "model": model_name,
                            "hardware": hardware,
                            "workload": {
                                "type": "static_batch",
                                "batch_size": batch_size,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                            },
                            "simulation": {
                                "max_batch_size": batch_size,
                                "fidelity": "tile",
                                "pipeline_model": "analytic",
                                "subbatch_scheduler": "lpt",
                                "operator_mapping": "chime",
                                "record_every": max(1, output_tokens),
                                "allow_gpu_prompt_fallback": False,
                            },
                        }

                        summaries: dict[str, dict[str, object]] = {}
                        errors: dict[str, str] = {}
                        policies: tuple[tuple[str, object], ...] = (
                            ("chime", "chime"),
                            (
                                "elastic",
                                {
                                    "name": "elastic",
                                    "prediction": "oracle",
                                    "decision_interval": decision_interval,
                                    "planning_horizon_tokens": planning_horizon_tokens,
                                    "min_segment_tokens": max(1, decision_interval),
                                    "min_switch_interval": max(
                                        1, decision_interval * 2
                                    ),
                                    "relative_improvement_threshold": 0.002,
                                    "max_switches_per_decision": min(
                                        8, batch_size
                                    ),
                                },
                            ),
                        )
                        for name, policy in policies:
                            simulation = dict(common["simulation"])
                            simulation["policy"] = policy
                            config = project_config_from_dict(
                                {**common, "simulation": simulation}
                            )
                            try:
                                summaries[name] = run_simulation(config).summary
                            except SimulationCapacityError as error:
                                errors[name] = str(error)

                        baseline = summaries.get("chime")
                        elastic = summaries.get("elastic")
                        baseline_status = (
                            str(baseline["status"]) if baseline else "capacity_error"
                        )
                        elastic_status = (
                            str(elastic["status"]) if elastic else "capacity_error"
                        )
                        speedup = (
                            float(elastic["throughput_tokens_s"])
                            / float(baseline["throughput_tokens_s"])
                            if baseline and elastic
                            else None
                        )
                        effective_budget = (
                            float(elastic["gpu_kv_capacity_gib"])
                            if elastic
                            else (
                                float(baseline["gpu_kv_capacity_gib"])
                                if baseline
                                else float(requested_budget or 0.0)
                            )
                        )
                        peak_gpu = (
                            float(elastic["peak_gpu_kv_gib"])
                            if elastic
                            else 0.0
                        )
                        switches = int(elastic["tail_switches"]) if elastic else 0
                        baseline_pim_fraction = (
                            float(baseline["pim_bottleneck_phase_fraction"])
                            if baseline
                            else 0.0
                        )
                        rows.append(
                            {
                                "model": MODEL_PRESETS[model_name].name,
                                "batch_size": batch_size,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "gpu_kv_budget_requested_gib": label,
                                "gpu_kv_budget_effective_gib": effective_budget,
                                "baseline_status": baseline_status,
                                "elastic_status": elastic_status,
                                "baseline_throughput_tokens_s": (
                                    baseline["throughput_tokens_s"]
                                    if baseline
                                    else None
                                ),
                                "elastic_throughput_tokens_s": (
                                    elastic["throughput_tokens_s"]
                                    if elastic
                                    else None
                                ),
                                "speedup_vs_chime": speedup,
                                "makespan_reduction_s": (
                                    float(baseline["simulated_time_s"])
                                    - float(elastic["simulated_time_s"])
                                    if baseline and elastic
                                    else None
                                ),
                                "baseline_mean_tpot_ms": (
                                    baseline["latency_ms"]["mean_tpot"]
                                    if baseline
                                    else None
                                ),
                                "elastic_mean_tpot_ms": (
                                    elastic["latency_ms"]["mean_tpot"]
                                    if elastic
                                    else None
                                ),
                                "baseline_pim_bottleneck_fraction": (
                                    baseline_pim_fraction
                                ),
                                "baseline_gpu_path_utilization": (
                                    baseline["gpu_path_utilization"]
                                    if baseline
                                    else None
                                ),
                                "baseline_pim_core_path_utilization": (
                                    baseline["pim_core_path_utilization"]
                                    if baseline
                                    else None
                                ),
                                "elastic_pim_bottleneck_fraction": (
                                    elastic["pim_bottleneck_phase_fraction"]
                                    if elastic
                                    else None
                                ),
                                "elastic_gpu_path_utilization": (
                                    elastic["gpu_path_utilization"]
                                    if elastic
                                    else None
                                ),
                                "elastic_pim_core_path_utilization": (
                                    elastic["pim_core_path_utilization"]
                                    if elastic
                                    else None
                                ),
                                "elastic_link_path_utilization": (
                                    elastic["link_path_utilization"]
                                    if elastic
                                    else None
                                ),
                                "elastic_peak_gpu_kv_gib": peak_gpu,
                                "elastic_peak_pim_kv_gib": (
                                    elastic["peak_pim_kv_gib"]
                                    if elastic
                                    else None
                                ),
                                "elastic_tail_switches": switches,
                                "classification": _benefit_classification(
                                    baseline_status=baseline_status,
                                    elastic_status=elastic_status,
                                    speedup=speedup,
                                    gpu_budget_gib=effective_budget,
                                    peak_gpu_kv_gib=peak_gpu,
                                    switches=switches,
                                    baseline_pim_bottleneck_fraction=(
                                        baseline_pim_fraction
                                    ),
                                    output_tokens=output_tokens,
                                    decision_interval=decision_interval,
                                ),
                                "baseline_error": errors.get("chime", ""),
                                "elastic_error": errors.get("elastic", ""),
                            }
                        )
    return rows


def homogeneous_lifecycle_rows(
    model_names: Iterable[str] = ("gpt175b", "opt66b", "qwen72b"),
    batch_sizes: Iterable[int] = (2, 4, 8, 12, 16, 32),
    input_lengths: Iterable[int] = (128, 2048, 5200, 8192),
    output_lengths: Iterable[int] = (256, 1024, 4096),
    gpu_kv_budgets_gib: Iterable[float | None] = (32.0, None),
    mode_grid_points: int = 5,
    quadrature_samples: int = 24,
    exact_finalists: int = 8,
    pipeline_model: str = "analytic",
    policy_pipeline_model: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Compare CHIME, rolling H64, and a preplanned homogeneous lifecycle mode.

    The offline policy solves a three-variable problem before decode starts:
    ``(request_count, start_token, end_token)``.  Those exchangeable requests
    create new KV on GPU only over ``[start_token, end_token)``.  Capacity is
    checked exactly from interval length and block rounding; the makespan
    integral uses bounded quadrature plus exact replay of a few finalists.
    """

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        for batch_size in batch_sizes:
            for input_tokens in input_lengths:
                for output_tokens in output_lengths:
                    for requested_budget in gpu_kv_budgets_gib:
                        budget_label = (
                            "max"
                            if requested_budget is None
                            else f"{requested_budget:g}"
                        )
                        label = (
                            f"{model_name}:B{batch_size}:I{input_tokens}:"
                            f"O{output_tokens}:GPU-KV={budget_label}GiB"
                        )
                        if progress:
                            progress(label)
                        hardware: str | dict[str, object]
                        if requested_budget is None:
                            hardware = "chime_a100"
                        else:
                            hardware = {
                                "preset": "chime_a100",
                                "gpu": {
                                    "kv_capacity_limit_gib": requested_budget
                                },
                            }
                        common = {
                            "model": model_name,
                            "hardware": hardware,
                            "workload": {
                                "type": "static_batch",
                                "batch_size": batch_size,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                            },
                            "simulation": {
                                "max_batch_size": batch_size,
                                "fidelity": "tile",
                                "pipeline_model": pipeline_model,
                                "policy_pipeline_model": (
                                    policy_pipeline_model
                                ),
                                "subbatch_scheduler": "lpt",
                                "operator_mapping": "chime",
                                "record_every": max(1, output_tokens),
                                "allow_gpu_prompt_fallback": False,
                            },
                        }
                        policies: tuple[tuple[str, str | dict[str, object]], ...] = (
                            ("chime", "chime"),
                            (
                                "horizon64",
                                {
                                    "name": "elastic",
                                    "optimizer": "horizon_greedy",
                                    "prediction": "oracle",
                                    "decision_interval": 64,
                                    "planning_horizon_tokens": 256,
                                    "min_segment_tokens": 64,
                                    "min_switch_interval": 128,
                                    "relative_improvement_threshold": 0.002,
                                    "max_switches_per_decision": min(
                                        8, batch_size
                                    ),
                                },
                            ),
                            (
                                "offline_mode",
                                {
                                    "name": "elastic",
                                    "optimizer": "homogeneous_modes",
                                    "prediction": "oracle",
                                    # The plan owns explicit event tokens; this
                                    # field is only simulator metadata.
                                    "decision_interval": 1,
                                    "planning_horizon_tokens": 256,
                                    "min_segment_tokens": min(64, output_tokens),
                                    "min_switch_interval": min(64, output_tokens),
                                    "relative_improvement_threshold": 0.0,
                                    "max_switches_per_decision": batch_size,
                                    "homogeneous_mode_grid_points": mode_grid_points,
                                    "homogeneous_mode_quadrature_samples": (
                                        quadrature_samples
                                    ),
                                    "homogeneous_mode_exact_finalists": (
                                        exact_finalists
                                    ),
                                },
                            ),
                        )
                        results: dict[str, object] = {}
                        errors: dict[str, str] = {}
                        for policy_name, policy in policies:
                            simulation = dict(common["simulation"])
                            simulation["policy"] = policy
                            try:
                                results[policy_name] = run_simulation(
                                    project_config_from_dict(
                                        {**common, "simulation": simulation}
                                    )
                                )
                            except (SimulationCapacityError, RuntimeError) as error:
                                errors[policy_name] = str(error)

                        def summary(name: str) -> dict[str, object] | None:
                            result = results.get(name)
                            return result.summary if result is not None else None

                        baseline = summary("chime")
                        rolling = summary("horizon64")
                        offline = summary("offline_mode")
                        baseline_time = (
                            float(baseline["simulated_time_s"])
                            if baseline
                            else None
                        )
                        rolling_time = (
                            float(rolling["simulated_time_s"])
                            if rolling
                            else None
                        )
                        offline_time = (
                            float(offline["simulated_time_s"])
                            if offline
                            else None
                        )
                        offline_peak = (
                            float(offline["peak_gpu_kv_gib"])
                            if offline
                            else 0.0
                        )
                        effective_budget = (
                            float(offline["gpu_kv_capacity_gib"])
                            if offline
                            else float(requested_budget or 0.0)
                        )
                        if not baseline:
                            classification = "baseline_capacity_error"
                        elif not offline:
                            classification = "offline_mode_error"
                        elif int(offline["homogeneous_mode_count"]) == 0:
                            classification = "no_offload_profitable"
                        elif offline_peak >= 0.95 * effective_budget:
                            classification = "benefit_capacity_bound"
                        elif offline_time is not None and baseline_time is not None and offline_time < baseline_time * 0.999:
                            classification = "benefit_load_balancing"
                        else:
                            classification = "neutral_mode"

                        def value(
                            item: dict[str, object] | None,
                            key: str,
                        ) -> object | None:
                            return item.get(key) if item else None

                        def latency(
                            item: dict[str, object] | None, key: str
                        ) -> object | None:
                            return item["latency_ms"][key] if item else None

                        offline_result = results.get("offline_mode")
                        capacity_switches = (
                            sum(
                                switch["reason"] == "capacity"
                                for switch in offline_result.switches
                            )
                            if offline_result is not None
                            else None
                        )
                        rolling_result = results.get("horizon64")
                        rolling_capacity_switches = (
                            sum(
                                switch["reason"] == "capacity"
                                for switch in rolling_result.switches
                            )
                            if rolling_result is not None
                            else None
                        )
                        rows.append(
                            {
                                "model": MODEL_PRESETS[model_name].name,
                                "pipeline_model": pipeline_model,
                                "policy_pipeline_model": (
                                    policy_pipeline_model or pipeline_model
                                ),
                                "batch_size": batch_size,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "gpu_kv_budget_requested_gib": budget_label,
                                "gpu_kv_budget_effective_gib": effective_budget,
                                "baseline_status": (
                                    baseline["status"]
                                    if baseline
                                    else "capacity_error"
                                ),
                                "horizon64_status": (
                                    rolling["status"]
                                    if rolling
                                    else "error"
                                ),
                                "offline_mode_status": (
                                    offline["status"]
                                    if offline
                                    else "error"
                                ),
                                "baseline_makespan_s": baseline_time,
                                "horizon64_makespan_s": rolling_time,
                                "offline_mode_makespan_s": offline_time,
                                "horizon64_speedup_vs_chime": (
                                    baseline_time / rolling_time
                                    if baseline_time is not None
                                    and rolling_time is not None
                                    else None
                                ),
                                "offline_mode_speedup_vs_chime": (
                                    baseline_time / offline_time
                                    if baseline_time is not None
                                    and offline_time is not None
                                    else None
                                ),
                                "offline_mode_speedup_vs_horizon64": (
                                    rolling_time / offline_time
                                    if rolling_time is not None
                                    and offline_time is not None
                                    else None
                                ),
                                "baseline_mean_tpot_ms": latency(
                                    baseline, "mean_tpot"
                                ),
                                "horizon64_mean_tpot_ms": latency(
                                    rolling, "mean_tpot"
                                ),
                                "offline_mode_mean_tpot_ms": latency(
                                    offline, "mean_tpot"
                                ),
                                "baseline_pim_bottleneck_fraction": value(
                                    baseline, "pim_bottleneck_phase_fraction"
                                ),
                                "horizon64_pim_bottleneck_fraction": value(
                                    rolling, "pim_bottleneck_phase_fraction"
                                ),
                                "offline_mode_pim_bottleneck_fraction": value(
                                    offline, "pim_bottleneck_phase_fraction"
                                ),
                                "offline_mode_gpu_path_utilization": value(
                                    offline, "gpu_path_utilization"
                                ),
                                "offline_mode_pim_path_utilization": value(
                                    offline, "pim_core_path_utilization"
                                ),
                                "offline_mode_link_path_utilization": value(
                                    offline, "link_path_utilization"
                                ),
                                "horizon64_peak_gpu_kv_gib": value(
                                    rolling, "peak_gpu_kv_gib"
                                ),
                                "offline_mode_peak_gpu_kv_gib": value(
                                    offline, "peak_gpu_kv_gib"
                                ),
                                "offline_mode_peak_pim_kv_gib": value(
                                    offline, "peak_pim_kv_gib"
                                ),
                                "horizon64_tail_switches": value(
                                    rolling, "tail_switches"
                                ),
                                "offline_mode_tail_switches": value(
                                    offline, "tail_switches"
                                ),
                                "horizon64_capacity_switches": (
                                    rolling_capacity_switches
                                ),
                                "offline_mode_capacity_switches": (
                                    capacity_switches
                                ),
                                "offline_mode_request_count": value(
                                    offline, "homogeneous_mode_count"
                                ),
                                "offline_mode_start_token": value(
                                    offline, "homogeneous_mode_start_token"
                                ),
                                "offline_mode_end_token": value(
                                    offline, "homogeneous_mode_end_token"
                                ),
                                "offline_mode_duration_tokens": value(
                                    offline, "homogeneous_mode_duration_tokens"
                                ),
                                "offline_mode_search_wall_time_s": value(
                                    offline, "policy_search_wall_time_s"
                                ),
                                "offline_mode_cost_evaluations": value(
                                    offline,
                                    "homogeneous_mode_cost_model_evaluations",
                                ),
                                "classification": classification,
                                "historical_kv_migration": False,
                                "baseline_error": errors.get("chime", ""),
                                "horizon64_error": errors.get("horizon64", ""),
                                "offline_mode_error": errors.get(
                                    "offline_mode", ""
                                ),
                            }
                        )
    return rows


def future_only_composition_rows(
    model_names: Iterable[str] = ("gpt175b",),
    batch_sizes: Iterable[int] = (12,),
    total_contexts: Iterable[int] = (8192,),
    output_lengths: Iterable[int] = (256, 1024, 2048, 4096, 6144),
    gpu_kv_budgets_gib: Iterable[float | None] = (32.0, None),
    decision_interval: int = 64,
    planning_horizon_tokens: int = 256,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Hold final context fixed while changing prompt/output composition."""

    rows: list[dict[str, object]] = []
    materialized_models = tuple(model_names)
    materialized_batches = tuple(batch_sizes)
    materialized_budgets = tuple(gpu_kv_budgets_gib)
    for total_context in total_contexts:
        if total_context <= 1:
            raise ValueError("total context must be greater than one token")
        for output_tokens in output_lengths:
            if not 0 < output_tokens < total_context:
                raise ValueError(
                    f"output length {output_tokens} must be between 1 and "
                    f"total_context-1 ({total_context - 1})"
                )
            input_tokens = total_context - output_tokens
            composed = future_only_benefit_rows(
                model_names=materialized_models,
                batch_sizes=materialized_batches,
                input_lengths=(input_tokens,),
                output_lengths=(output_tokens,),
                gpu_kv_budgets_gib=materialized_budgets,
                decision_interval=decision_interval,
                planning_horizon_tokens=planning_horizon_tokens,
                progress=progress,
            )
            for row in composed:
                row["final_context_tokens"] = total_context
                row["output_fraction"] = output_tokens / total_context
            rows.extend(composed)
    return rows


def lifetime_policy_comparison_rows(
    model_names: Iterable[str] = ("gpt175b",),
    batch_sizes: Iterable[int] = (2, 4),
    input_lengths: Iterable[int] = (4096, 5200),
    output_lengths: Iterable[int] = (64,),
    gpu_kv_budgets_gib: Iterable[float | None] = (32.0, None),
    decision_interval: int = 32,
    planning_horizon_tokens: int = 256,
    lifetime_evaluation_stride_tokens: int = 16,
    beam_depth: int = 4,
    beam_width: int = 4,
    beam_request_candidates: int = 6,
    beam_evaluation_stride_tokens: int = 16,
    beam_terminal_samples: int = 8,
    beam_terminal_mode: str = "horizon",
    beam_relative_improvement_threshold: float = 0.001,
    include_global_exact: bool = True,
    exact_max_candidates: int = 8,
    exact_max_cost_evaluations: int = 200000,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Compare greedy, rolling-beam, and exact closed-workload placement.

    ``global_exact`` enumerates all eligible switch combinations at every
    decision boundary and is therefore limited to small closed static batches.
    Its search CPU time is deliberately outside simulated makespan. The
    Rolling beam is a bounded receding-horizon approximation. The
    lifetime-greedy objective values the selected tail under a remaining-life
    continuation assumption while committing capacity for only one interval.
    """

    variants: list[tuple[str, str | dict[str, object], str]] = [
        ("CHIME", "chime", "all KV on PIM"),
        (
            "Horizon-greedy",
            {
                "name": "elastic",
                "optimizer": "horizon_greedy",
                "prediction": "oracle",
            },
            "one projected iteration at a capped horizon",
        ),
        (
            "Lifetime-greedy",
            {
                "name": "elastic",
                "optimizer": "lifetime_greedy",
                "prediction": "oracle",
            },
            "remaining-life continuation; one-block capacity commitment",
        ),
        (
            "Rolling-beam",
            {
                "name": "elastic",
                "optimizer": "rolling_beam",
                "prediction": "oracle",
            },
            "multistage beam search with configurable terminal heuristic",
        ),
    ]
    if include_global_exact:
        variants.append(
            (
                "Global-exact",
                {
                    "name": "elastic",
                    "optimizer": "global_exact",
                    "prediction": "oracle",
                },
                "exact remaining makespan for a closed static batch",
            )
        )

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        for batch_size in batch_sizes:
            for input_tokens in input_lengths:
                for output_tokens in output_lengths:
                    for requested_budget in gpu_kv_budgets_gib:
                        budget_label = (
                            "max" if requested_budget is None else f"{requested_budget:g}"
                        )
                        hardware: str | dict[str, object]
                        if requested_budget is None:
                            hardware = "chime_a100"
                        else:
                            hardware = {
                                "preset": "chime_a100",
                                "gpu": {
                                    "kv_capacity_limit_gib": requested_budget,
                                },
                            }
                        common = {
                            "model": model_name,
                            "hardware": hardware,
                            "workload": {
                                "type": "static_batch",
                                "batch_size": batch_size,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                            },
                            "simulation": {
                                "max_batch_size": batch_size,
                                "fidelity": "tile",
                                "pipeline_model": "analytic",
                                "subbatch_scheduler": "lpt",
                                "operator_mapping": "chime",
                                "record_every": max(1, output_tokens),
                                "allow_gpu_prompt_fallback": False,
                            },
                        }

                        results: list[
                            tuple[str, str, dict[str, object], str]
                        ] = []
                        for label, raw_policy, objective_scope in variants:
                            if progress:
                                progress(
                                    f"{model_name}:B{batch_size}:I{input_tokens}:"
                                    f"O{output_tokens}:GPU-KV={budget_label}GiB:{label}"
                                )
                            policy: str | dict[str, object]
                            if isinstance(raw_policy, str):
                                policy = raw_policy
                            else:
                                policy = {
                                    **raw_policy,
                                    "decision_interval": decision_interval,
                                    "planning_horizon_tokens": planning_horizon_tokens,
                                    "lifetime_evaluation_stride_tokens": (
                                        lifetime_evaluation_stride_tokens
                                    ),
                                    "beam_depth": beam_depth,
                                    "beam_width": beam_width,
                                    "beam_request_candidates": (
                                        beam_request_candidates
                                    ),
                                    "beam_evaluation_stride_tokens": (
                                        beam_evaluation_stride_tokens
                                    ),
                                    "beam_terminal_samples": beam_terminal_samples,
                                    "beam_terminal_mode": beam_terminal_mode,
                                    "min_segment_tokens": max(1, decision_interval),
                                    "min_switch_interval": max(1, decision_interval),
                                    # Zero isolates optimizer quality. Stage 1's
                                    # production-like 0.2% hysteresis remains in
                                    # benefit-map and is evaluated separately.
                                    "relative_improvement_threshold": (
                                        beam_relative_improvement_threshold
                                        if label == "Rolling-beam"
                                        else 0.0
                                    ),
                                    "max_switches_per_decision": min(8, batch_size),
                                    "exact_max_candidates": exact_max_candidates,
                                    "exact_max_cost_evaluations": (
                                        exact_max_cost_evaluations
                                    ),
                                }
                            simulation = dict(common["simulation"])
                            simulation["policy"] = policy
                            config = project_config_from_dict(
                                {**common, "simulation": simulation}
                            )
                            try:
                                summary = run_simulation(config).summary
                                results.append(
                                    (label, objective_scope, summary, "")
                                )
                            except SimulationCapacityError as error:
                                results.append(
                                    (label, objective_scope, {}, str(error))
                                )

                        baseline = next(
                            summary
                            for label, _, summary, _ in results
                            if label == "CHIME"
                        )
                        exact = next(
                            (
                                summary
                                for label, _, summary, _ in results
                                if label == "Global-exact" and summary
                            ),
                            None,
                        )
                        baseline_throughput = float(
                            baseline["throughput_tokens_s"]
                        )
                        exact_time = (
                            float(exact["simulated_time_s"]) if exact else None
                        )
                        for label, objective_scope, summary, error in results:
                            completed = bool(summary)
                            simulated_time = (
                                float(summary["simulated_time_s"])
                                if completed
                                else None
                            )
                            throughput = (
                                float(summary["throughput_tokens_s"])
                                if completed
                                else None
                            )
                            rows.append(
                                {
                                    "model": MODEL_PRESETS[model_name].name,
                                    "batch_size": batch_size,
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "gpu_kv_budget_requested_gib": budget_label,
                                    "policy": label,
                                    "optimizer": (
                                        summary.get("policy_optimizer", "chime")
                                        if completed
                                        else "error"
                                    ),
                                    "objective_scope": objective_scope,
                                    "status": (
                                        summary.get("status", "completed")
                                        if completed
                                        else "capacity_error"
                                    ),
                                    "throughput_tokens_s": throughput,
                                    "speedup_vs_chime": (
                                        throughput / baseline_throughput
                                        if throughput is not None
                                        else None
                                    ),
                                    "simulated_time_s": simulated_time,
                                    "makespan_gap_vs_global_exact": (
                                        simulated_time / exact_time - 1.0
                                        if simulated_time is not None
                                        and exact_time is not None
                                        else None
                                    ),
                                    "mean_tpot_ms": (
                                        summary["latency_ms"]["mean_tpot"]
                                        if completed
                                        else None
                                    ),
                                    "pim_bottleneck_fraction": (
                                        summary["pim_bottleneck_phase_fraction"]
                                        if completed
                                        else None
                                    ),
                                    "gpu_path_utilization": (
                                        summary["gpu_path_utilization"]
                                        if completed
                                        else None
                                    ),
                                    "pim_path_utilization": (
                                        summary["pim_core_path_utilization"]
                                        if completed
                                        else None
                                    ),
                                    "peak_gpu_kv_gib": (
                                        summary["peak_gpu_kv_gib"]
                                        if completed
                                        else None
                                    ),
                                    "peak_pim_kv_gib": (
                                        summary["peak_pim_kv_gib"]
                                        if completed
                                        else None
                                    ),
                                    "tail_switches": (
                                        summary["tail_switches"]
                                        if completed
                                        else None
                                    ),
                                    "relative_improvement_threshold": (
                                        beam_relative_improvement_threshold
                                        if label == "Rolling-beam"
                                        else 0.0
                                    ),
                                    "beam_depth": (
                                        beam_depth if label == "Rolling-beam" else None
                                    ),
                                    "beam_width": (
                                        beam_width if label == "Rolling-beam" else None
                                    ),
                                    "beam_terminal_mode": (
                                        beam_terminal_mode
                                        if label == "Rolling-beam"
                                        else None
                                    ),
                                    "error": error,
                                }
                            )
    return rows


def hardware_ablation_rows(
    model_names: Iterable[str] = ("gpt175b", "qwen72b"),
    token_lengths: Iterable[int] = (512, 1024, 2048, 4096, 8192),
) -> list[dict[str, object]]:
    """Return a Fig.14-style, single-layer PIM hardware ablation.

    ``Base`` is CHIME's naive bank-level implementation: CPU re-layout,
    serialized bank/cross-chip/softmax work, and an untuned head mapping.
    ``+Relayout`` adds the rank-PU hardware re-layout. ``+Pipeline`` additionally
    enables the bubble-free pipeline and MHA/GQA-specific head mapping.

    R-DP is kept as a separate roofline reference. Its 1.6 TB/s module bandwidth
    is divided over four layer-interleaved ranksets, matching the same topology
    convention used for CHIME rather than fitting the red labels in Fig.14.
    """

    base_system = SYSTEM_PRESETS["chime_a100"]
    rows: list[dict[str, object]] = []
    variants = (
        ("Base", False, False, False),
        ("+Relayout", False, False, True),
        ("+Pipeline", True, True, True),
    )

    for model_name in model_names:
        model = MODEL_PRESETS[model_name]
        for token_length in token_lengths:
            views = views_for_requests(
                [RequestState("ablation", 0.0, token_length, 1)]
            )
            costs: dict[str, PimLayerCost] = {}
            for variant, pipeline, head_mapping, hardware_relayout in variants:
                pim = replace(
                    base_system.pim,
                    bubble_free_pipeline=pipeline,
                    optimized_head_mapping=head_mapping,
                    hardware_relayout=hardware_relayout,
                )
                cost = ChimeCostModel(
                    model, replace(base_system, pim=pim), "tile"
                ).pim_layer_cost(views)
                costs[variant] = cost

            base_latency = costs["Base"].total_us
            for variant, *_ in variants:
                cost = costs[variant]
                rows.append(
                    {
                        "model": model.name,
                        "attention": (
                            "MHA" if model.gqa_group_size == 1 else f"GQA-{model.gqa_group_size}"
                        ),
                        "token_length": token_length,
                        "variant": variant,
                        "latency_us": cost.total_us,
                        "normalized_latency": cost.total_us / base_latency,
                        "bank_us": cost.bank_us,
                        "cross_chip_us": cost.cross_chip_us,
                        "softmax_us": cost.softmax_us,
                        "relayout_us": cost.relayout_us,
                        "setup_us": cost.setup_us,
                        "exposed_link_us": cost.exposed_link_us,
                        "assumption": "CHIME layer-interleaved: one rankset per layer",
                    }
                )

            r_dp_pim = replace(
                base_system.pim,
                aggregate_bandwidth_tbps=1.6,
                # R-DP uses the same four-rankset layer interleaving. Internal
                # bank-PU synchronization/re-layout terms do not apply.
                bubble_free_pipeline=True,
                optimized_head_mapping=True,
                hardware_relayout=True,
            )
            r_dp_system = replace(base_system, pim=r_dp_pim)
            chime_link = ChimeCostModel(
                model, r_dp_system, "tile"
            ).pim_layer_cost(views)
            logical_bytes = token_length * model.kv_bytes_per_token_per_layer
            r_dp_core_us = (
                logical_bytes / r_dp_pim.layer_bandwidth_bytes_s * 1e6
            )
            r_dp_total_us = (
                r_dp_core_us + chime_link.setup_us + chime_link.exposed_link_us
            )
            rows.append(
                {
                    "model": model.name,
                    "attention": (
                        "MHA" if model.gqa_group_size == 1 else f"GQA-{model.gqa_group_size}"
                    ),
                    "token_length": token_length,
                    "variant": "R-DP roofline",
                    "latency_us": r_dp_total_us,
                    "normalized_latency": r_dp_total_us / base_latency,
                    "bank_us": r_dp_core_us,
                    "cross_chip_us": 0.0,
                    "softmax_us": 0.0,
                    "relayout_us": 0.0,
                    "setup_us": chime_link.setup_us,
                    "exposed_link_us": chime_link.exposed_link_us,
                    "assumption": "1.6 TB/s module / 4 layer-interleaved ranksets",
                }
            )
    return rows


def scalability_rows(
    model_name: str = "gpt175b",
    profile: str = "openr1",
    request_count: int = 32,
    seed: int = 0,
    factors: Iterable[int] = (1, 2, 4, 8),
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Run a Fig.13-style bandwidth/capacity scaling study.

    The experiment starts from the paper's ``1 rankset, 512 GB`` memory and
    uses target-length reservation to isolate the capacity envelope. For this
    steady-state throughput experiment, independent requests/layers are allowed
    to occupy all added ranksets; this represents inter-rankset pipeline
    concurrency. It is deliberately distinct from the single-layer latency
    convention used by :func:`hardware_ablation_rows`.

    This is an *offline trace* experiment: target lengths are used only by
    admission, not by the execution cost model or placement policy.
    """

    if request_count <= 0:
        raise ValueError("request_count must be positive")
    factor_values = tuple(dict.fromkeys(int(value) for value in factors))
    if not factor_values or any(value <= 0 for value in factor_values):
        raise ValueError("scaling factors must be positive")

    rows: list[dict[str, object]] = []
    baselines: dict[str, float] = {}
    configurations: list[tuple[str, int]] = [("Base", 1)]
    configurations.extend(
        (mode, factor)
        for mode in ("Bw-only", "Cap-only", "Bw-Cap")
        for factor in factor_values
        if factor != 1
    )

    for mode, factor in configurations:
        bandwidth_factor = factor if mode in ("Bw-only", "Bw-Cap") else 1
        capacity_factor = factor if mode in ("Cap-only", "Bw-Cap") else 1
        ranksets = bandwidth_factor
        if progress:
            progress(f"{model_name} {mode} {factor}x")
        config = project_config_from_dict(
            {
                "model": model_name,
                "hardware": {
                    "preset": "chime_a100",
                    "pim": {
                        "capacity_gib": 512.0 * capacity_factor,
                        "aggregate_bandwidth_tbps": 3.25 * bandwidth_factor,
                        "ranksets": ranksets,
                        "ranks_per_channel": ranksets,
                        # Effective steady-state concurrency across independent
                        # requests/layers, not one layer's isolated latency.
                        "compute_ranksets_per_layer": ranksets,
                    },
                },
                "workload": {
                    "type": "trace_serving",
                    "synthetic": {
                        "profile": profile,
                        "count": request_count,
                        "seed": seed,
                        "arrival_rate_rps": 0,
                    },
                },
                "simulation": {
                    "max_batch_size": request_count,
                    "fidelity": "tile",
                    "record_every": 1000000,
                    "admission_reservation": "pim_target",
                    "policy": "chime",
                },
            }
        )
        result = run_simulation(config)
        throughput = float(result.summary["throughput_tokens_s"])
        if mode == "Base":
            baselines[model_name] = throughput
        rows.append(
            {
                "model": result.summary["model"],
                "profile": profile,
                "requests": request_count,
                "seed": seed,
                "mode": mode,
                "factor": factor,
                "capacity_gib": 512.0 * capacity_factor,
                "aggregate_bandwidth_tbps": 3.25 * bandwidth_factor,
                "effective_attention_bandwidth_tbps": 3.25 * bandwidth_factor,
                "ranksets": ranksets,
                "throughput_tokens_s": throughput,
                "normalized_throughput": throughput / baselines[model_name],
                "simulated_time_s": result.summary["simulated_time_s"],
                "average_batch_size": result.summary["average_batch_size"],
                "pim_bottleneck_fraction": result.summary[
                    "pim_bottleneck_phase_fraction"
                ],
                "peak_pim_kv_gib": result.summary["peak_pim_kv_gib"],
                "peak_pim_reserved_gib": result.summary["peak_pim_reserved_gib"],
                "admission": "offline target-length reservation",
                "execution_scope": "steady-state inter-rankset concurrency",
            }
        )
    return rows


def aggregate_rankset_sensitivity_rows(
    model_name: str = "gpt175b",
    batch_size: int = 12,
    context_tokens: int = 5200,
) -> list[dict[str, object]]:
    """Compare CHIME layer mapping with a hypothetical rankset-striped layer."""

    model = MODEL_PRESETS[model_name]
    system = SYSTEM_PRESETS["chime_a100"]
    views = views_for_requests(
        [RequestState(f"sensitivity-{i}", 0.0, context_tokens, 1) for i in range(batch_size)]
    )
    rows: list[dict[str, object]] = []
    for active_ranksets, label in (
        (1, "CHIME layer-interleaved"),
        (system.pim.ranksets, "hypothetical layer-striped"),
    ):
        pim = replace(system.pim, compute_ranksets_per_layer=active_ranksets)
        cost = ChimeCostModel(model, replace(system, pim=pim), "tile").iteration_cost(views)
        rows.append(
            {
                "mapping": label,
                "compute_ranksets_per_layer": active_ranksets,
                "aggregate_bandwidth_tbps": pim.aggregate_bandwidth_tbps,
                "per_layer_bandwidth_tbps": pim.layer_bandwidth_bytes_s / TB,
                "iteration_ms": cost.total_us / 1e3,
                "per_layer_us": cost.per_layer_us,
                "pim_core_us_per_subbatch_layer": cost.subbatches[0].pim.core_us,
                "gpu_bottleneck_phases": cost.phase_gpu_bottlenecks,
                "pim_bottleneck_phases": cost.phase_pim_bottlenecks,
            }
        )
    return rows


def policy_comparison_rows(
    model_name: str = "gpt175b",
    profile: str = "openr1",
    request_count: int = 16,
    seed: int = 0,
    arrival_rate_rps: float = 0.0,
    decision_interval: int = 64,
    planning_horizon_tokens: int = 256,
    lifetime_evaluation_stride_tokens: int = 64,
    beam_depth: int = 4,
    beam_width: int = 4,
    beam_request_candidates: int = 6,
    beam_evaluation_stride_tokens: int = 32,
    beam_terminal_samples: int = 8,
    beam_terminal_mode: str = "horizon",
    beam_relative_improvement_threshold: float = 0.001,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Compare no prediction, trace prediction, and two oracle objectives.

    All policies receive exactly the same sampled requests and arrival process.
    The profile predictor conditions only on generated-token progress; the
    oracle length is explicitly unavailable to an online system. The lifetime
    variant is a scalable heuristic and must not be confused with Global-exact.
    """

    variants = (
        ("CHIME", "chime", "fixed", "none"),
        ("Elastic-fixed", "elastic", "fixed", "horizon_greedy"),
        ("Elastic-profile", "elastic", "profile", "horizon_greedy"),
        ("Elastic-oracle", "elastic", "oracle", "horizon_greedy"),
        (
            "Elastic-lifetime-oracle",
            "elastic",
            "oracle",
            "lifetime_greedy",
        ),
        (
            "Elastic-rolling-beam",
            "elastic",
            "oracle",
            "rolling_beam",
        ),
    )
    rows: list[dict[str, object]] = []
    baseline_throughput = 0.0
    for label, policy_name, prediction, optimizer in variants:
        if progress:
            progress(label)
        policy: dict[str, object] | str
        if policy_name == "chime":
            policy = "chime"
        else:
            policy = {
                "name": "elastic",
                "decision_interval": decision_interval,
                "planning_horizon_tokens": planning_horizon_tokens,
                "optimizer": optimizer,
                "prediction": prediction,
                "prediction_profile": profile,
                "prediction_samples": 10000,
                "prediction_seed": seed,
                "lifetime_evaluation_stride_tokens": (
                    lifetime_evaluation_stride_tokens
                ),
                "beam_depth": beam_depth,
                "beam_width": beam_width,
                "beam_request_candidates": beam_request_candidates,
                "beam_evaluation_stride_tokens": (
                    beam_evaluation_stride_tokens
                ),
                "beam_terminal_samples": beam_terminal_samples,
                "beam_terminal_mode": beam_terminal_mode,
                "min_segment_tokens": max(1, decision_interval),
                "min_switch_interval": max(1, decision_interval * 2),
                "relative_improvement_threshold": (
                    beam_relative_improvement_threshold
                    if optimizer == "rolling_beam"
                    else 0.002
                ),
                "max_switches_per_decision": 8,
            }
        config = project_config_from_dict(
            {
                "model": model_name,
                "hardware": "chime_a100",
                "workload": {
                    "type": "trace_serving",
                    "synthetic": {
                        "profile": profile,
                        "count": request_count,
                        "seed": seed,
                        "arrival_rate_rps": arrival_rate_rps,
                    },
                },
                "simulation": {
                    "max_batch_size": request_count,
                    "fidelity": "tile",
                    "record_every": 1000000,
                    "policy": policy,
                },
            }
        )
        result = run_simulation(config)
        throughput = float(result.summary["throughput_tokens_s"])
        if label == "CHIME":
            baseline_throughput = throughput
        rows.append(
            {
                "model": result.summary["model"],
                "profile": profile,
                "requests": request_count,
                "arrival_rate_rps": arrival_rate_rps,
                "policy": label,
                "prediction": "none" if label == "CHIME" else prediction,
                "optimizer": optimizer,
                "beam_terminal_mode": (
                    beam_terminal_mode if optimizer == "rolling_beam" else ""
                ),
                "throughput_tokens_s": throughput,
                "speedup_vs_chime": throughput / baseline_throughput,
                "simulated_time_s": result.summary["simulated_time_s"],
                "average_batch_size": result.summary["average_batch_size"],
                "mean_tpot_ms": result.summary["latency_ms"]["mean_tpot"],
                "p95_request_mean_tpot_ms": result.summary["latency_ms"][
                    "p95_request_mean_tpot"
                ],
                "p95_e2e_ms": result.summary["latency_ms"]["e2e_p95"],
                "pim_bottleneck_fraction": result.summary[
                    "pim_bottleneck_phase_fraction"
                ],
                "tail_switches": result.summary["tail_switches"],
                "peak_pim_kv_gib": result.summary["peak_pim_kv_gib"],
                "peak_gpu_kv_gib": result.summary["peak_gpu_kv_gib"],
            }
        )
    return rows


def beam_terminal_ablation_rows(
    model_name: str = "gpt175b",
    profile: str = "openr1",
    request_count: int = 8,
    seed: int = 0,
    arrival_rate_rps: float = 0.0,
    terminal_modes: Iterable[str] = (
        "horizon",
        "pim_lifetime",
        "tail_lifetime",
    ),
    decision_interval: int = 64,
    planning_horizon_tokens: int = 256,
    beam_depth: int = 4,
    beam_width: int = 4,
    beam_request_candidates: int = 2,
    beam_evaluation_stride_tokens: int = 64,
    beam_terminal_samples: int = 8,
    beam_relative_improvement_threshold: float = 0.001,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Ablate the rolling beam's treatment of undecided future KV."""

    modes = tuple(dict.fromkeys(terminal_modes))
    unknown = set(modes) - {"horizon", "pim_lifetime", "tail_lifetime"}
    if unknown:
        raise ValueError(f"unknown beam terminal modes: {', '.join(sorted(unknown))}")
    variants: tuple[tuple[str, str | dict[str, object]], ...] = (
        ("CHIME", "chime"),
        *tuple(
            (
                mode,
                {
                    "name": "elastic",
                    "optimizer": "rolling_beam",
                    "prediction": "oracle",
                    "decision_interval": decision_interval,
                    "planning_horizon_tokens": planning_horizon_tokens,
                    "beam_depth": beam_depth,
                    "beam_width": beam_width,
                    "beam_request_candidates": beam_request_candidates,
                    "beam_evaluation_stride_tokens": (
                        beam_evaluation_stride_tokens
                    ),
                    "beam_terminal_samples": beam_terminal_samples,
                    "beam_terminal_mode": mode,
                    "min_segment_tokens": max(1, decision_interval),
                    "min_switch_interval": max(1, decision_interval * 2),
                    "relative_improvement_threshold": (
                        beam_relative_improvement_threshold
                    ),
                    "max_switches_per_decision": min(8, request_count),
                },
            )
            for mode in modes
        ),
    )

    rows: list[dict[str, object]] = []
    baseline_throughput = 0.0
    for label, policy in variants:
        if progress:
            progress(label)
        config = project_config_from_dict(
            {
                "model": model_name,
                "hardware": "chime_a100",
                "workload": {
                    "type": "trace_serving",
                    "synthetic": {
                        "profile": profile,
                        "count": request_count,
                        "seed": seed,
                        "arrival_rate_rps": arrival_rate_rps,
                    },
                },
                "simulation": {
                    "max_batch_size": request_count,
                    "fidelity": "tile",
                    "record_every": 1000000,
                    "policy": policy,
                },
            }
        )
        summary = run_simulation(config).summary
        throughput = float(summary["throughput_tokens_s"])
        if label == "CHIME":
            baseline_throughput = throughput
        rows.append(
            {
                "model": summary["model"],
                "profile": profile,
                "requests": request_count,
                "seed": seed,
                "arrival_rate_rps": arrival_rate_rps,
                "terminal_mode": "none" if label == "CHIME" else label,
                "throughput_tokens_s": throughput,
                "speedup_vs_chime": throughput / baseline_throughput,
                "simulated_time_s": summary["simulated_time_s"],
                "mean_tpot_ms": summary["latency_ms"]["mean_tpot"],
                "p95_request_mean_tpot_ms": summary["latency_ms"][
                    "p95_request_mean_tpot"
                ],
                "p95_e2e_ms": summary["latency_ms"]["e2e_p95"],
                "tail_switches": summary["tail_switches"],
                "peak_gpu_kv_gib": summary["peak_gpu_kv_gib"],
                "peak_pim_kv_gib": summary["peak_pim_kv_gib"],
            }
        )
    return rows


def open_arrival_slo_rows(
    model_name: str = "gpt175b",
    request_count: int = 24,
    seeds: Iterable[int] = (0,),
    arrival_rates_rps: Iterable[float] = (0.02, 0.04, 0.06),
    max_batch_size: int = 8,
    input_mean: float = 4096.0,
    input_std: float = 2048.0,
    input_min: int = 512,
    input_max: int = 8192,
    output_mean: float = 2048.0,
    output_std: float = 1024.0,
    output_min: int = 256,
    output_max: int = 4096,
    gpu_kv_budget_gib: float | None = None,
    slo_queue_ms: float = 2000.0,
    slo_tpot_ms: float = 130.0,
    slo_e2e_ms: float | None = None,
    slo_attainment_target: float = 0.99,
    decision_interval: int = 64,
    planning_horizon_tokens: int = 256,
    beam_depth: int = 4,
    beam_width: int = 3,
    beam_request_candidates: int = 2,
    beam_evaluation_stride_tokens: int = 8,
    beam_terminal_samples: int = 8,
    beam_relative_improvement_threshold: float = 0.001,
    include_event_driven_variant: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Compare active-only and future-arrival-aware beam on open traces.

    Arrival times and request lengths are identical across all policies for a
    given seed/rate pair. The future-aware variant is an offline oracle: it
    receives the pending trace, while the active-only variant receives exactly
    the same per-request length oracle but cannot see requests before arrival.
    Goodput counts requests satisfying every configured SLO and divides by the
    complete finite-trace makespan, including drain time.
    """

    seed_values = tuple(seeds)
    rate_values = tuple(arrival_rates_rps)
    if any(rate <= 0 for rate in rate_values):
        raise ValueError("open-arrival rates must be positive")
    variants: list[tuple[str, str | dict[str, object]]] = [
        ("CHIME", "chime"),
        (
            "Rolling-active-only",
            {
                "name": "elastic",
                "optimizer": "rolling_beam",
                "prediction": "oracle",
                "beam_include_future_arrivals": False,
            },
        ),
        (
            "Rolling-future-arrivals",
            {
                "name": "elastic",
                "optimizer": "rolling_beam",
                "prediction": "oracle",
                "beam_include_future_arrivals": True,
            },
        ),
    ]
    if include_event_driven_variant:
        variants.append(
            (
                "Rolling-active-only-event-driven",
                {
                    "name": "elastic",
                    "optimizer": "rolling_beam",
                    "prediction": "oracle",
                    "beam_include_future_arrivals": False,
                    "event_driven_replan": True,
                },
            )
        )
        variants.append(
            (
                "Rolling-future-arrivals-event-driven",
                {
                    "name": "elastic",
                    "optimizer": "rolling_beam",
                    "prediction": "oracle",
                    "beam_include_future_arrivals": True,
                    "event_driven_replan": True,
                },
            )
        )
    hardware: str | dict[str, object]
    if gpu_kv_budget_gib is None:
        hardware = "chime_a100"
    else:
        hardware = {
            "preset": "chime_a100",
            "gpu": {"kv_capacity_limit_gib": gpu_kv_budget_gib},
        }

    rows: list[dict[str, object]] = []
    for seed in seed_values:
        for arrival_rate in rate_values:
            baseline_throughput = 0.0
            baseline_goodput = 0.0
            for label, raw_policy in variants:
                if progress:
                    progress(
                        f"seed={seed}:arrival={arrival_rate:g}rps:{label}"
                    )
                if isinstance(raw_policy, str):
                    policy: str | dict[str, object] = raw_policy
                else:
                    policy = {
                        **raw_policy,
                        "decision_interval": decision_interval,
                        "planning_horizon_tokens": planning_horizon_tokens,
                        "beam_depth": beam_depth,
                        "beam_width": beam_width,
                        "beam_request_candidates": beam_request_candidates,
                        "beam_evaluation_stride_tokens": (
                            beam_evaluation_stride_tokens
                        ),
                        "beam_terminal_samples": beam_terminal_samples,
                        "beam_terminal_mode": "tail_lifetime",
                        "min_segment_tokens": max(1, decision_interval),
                        "min_switch_interval": max(1, decision_interval * 2),
                        "relative_improvement_threshold": (
                            beam_relative_improvement_threshold
                        ),
                        "max_switches_per_decision": min(
                            8, max_batch_size
                        ),
                    }
                config = project_config_from_dict(
                    {
                        "model": model_name,
                        "hardware": hardware,
                        "workload": {
                            "type": "trace_serving",
                            "synthetic": {
                                "count": request_count,
                                "seed": seed,
                                "arrival_rate_rps": arrival_rate,
                                "input_tokens": {
                                    "distribution": "normal",
                                    "mean": input_mean,
                                    "std": input_std,
                                    "min": input_min,
                                    "max": input_max,
                                },
                                "output_tokens": {
                                    "distribution": "normal",
                                    "mean": output_mean,
                                    "std": output_std,
                                    "min": output_min,
                                    "max": output_max,
                                },
                            },
                        },
                        "simulation": {
                            "max_batch_size": max_batch_size,
                            "fidelity": "tile",
                            "record_every": 1000000,
                            "admission_scheduler": "fcfs",
                            "admission_reservation": "none",
                            "allow_gpu_prompt_fallback": False,
                            "slo_queue_ms": slo_queue_ms,
                            "slo_tpot_ms": slo_tpot_ms,
                            "slo_e2e_ms": slo_e2e_ms,
                            "slo_attainment_target": slo_attainment_target,
                            "policy": policy,
                        },
                    }
                )
                result = run_simulation(config)
                summary = result.summary
                throughput = float(summary["throughput_tokens_s"])
                goodput = float(summary["slo"]["goodput_rps"])
                if label == "CHIME":
                    baseline_throughput = throughput
                    baseline_goodput = goodput
                rows.append(
                    {
                        "model": summary["model"],
                        "workload": "synthetic-long-heterogeneous",
                        "requests": request_count,
                        "seed": seed,
                        "configured_arrival_rate_rps": arrival_rate,
                        "realized_offered_rate_rps": summary[
                            "trace_offered_rate_rps"
                        ],
                        "max_batch_size": max_batch_size,
                        "gpu_kv_budget_gib": (
                            "max"
                            if gpu_kv_budget_gib is None
                            else gpu_kv_budget_gib
                        ),
                        "policy": label,
                        "future_arrivals_visible": (
                            label.startswith("Rolling-future-arrivals")
                        ),
                        "event_driven_replan": summary[
                            "policy_event_driven_replan"
                        ],
                        "event_policy_decisions": summary[
                            "event_policy_decisions"
                        ],
                        "throughput_tokens_s": throughput,
                        "speedup_vs_chime": (
                            throughput / baseline_throughput
                        ),
                        "request_completion_rate_rps": summary[
                            "request_completion_rate_rps"
                        ],
                        "slo_goodput_rps": goodput,
                        "goodput_vs_chime": (
                            goodput / baseline_goodput
                            if baseline_goodput > 0
                            else None
                        ),
                        "slo_attainment": summary["slo"]["attainment"],
                        "slo_target_met": summary["slo"]["target_met"],
                        "slo_queue_ms": slo_queue_ms,
                        "slo_tpot_ms": slo_tpot_ms,
                        "slo_e2e_ms": slo_e2e_ms,
                        "slo_attainment_target": slo_attainment_target,
                        "queue_mean_ms": summary["latency_ms"]["queue_mean"],
                        "queue_p95_ms": summary["latency_ms"]["queue_p95"],
                        "queue_p99_ms": summary["latency_ms"]["queue_p99"],
                        "mean_tpot_ms": summary["latency_ms"]["mean_tpot"],
                        "p95_request_mean_tpot_ms": summary["latency_ms"][
                            "p95_request_mean_tpot"
                        ],
                        "p95_e2e_ms": summary["latency_ms"]["e2e_p95"],
                        "simulated_time_s": summary["simulated_time_s"],
                        "arrival_span_s": summary["trace_arrival_span_s"],
                        "drain_time_s": summary[
                            "drain_time_after_last_arrival_s"
                        ],
                        "average_batch_size": summary["average_batch_size"],
                        "pim_bottleneck_fraction": summary[
                            "pim_bottleneck_phase_fraction"
                        ],
                        "gpu_path_utilization": summary[
                            "gpu_path_utilization"
                        ],
                        "pim_path_utilization": summary[
                            "pim_core_path_utilization"
                        ],
                        "policy_search_wall_time_s": summary[
                            "policy_search_wall_time_s"
                        ],
                        "tail_switches": summary["tail_switches"],
                        "peak_gpu_kv_gib": summary["peak_gpu_kv_gib"],
                        "peak_pim_kv_gib": summary["peak_pim_kv_gib"],
                    }
                )
    return rows


def operator_offload_rows(
    model_name: str = "gpt175b",
    subbatch_size: int = 6,
    context_lengths: Iterable[int] = (512, 1024, 2048, 4096, 8192),
    fractions: Iterable[float] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
    system: SystemSpec | None = None,
) -> list[dict[str, object]]:
    """Optimistic upper bound for row-wise O_proj placement on DIMM-PIM.

    A fraction of O_proj output rows (and their weights) is placed permanently
    in PIM. The bound assumes ideal bank-PU MAC utilization and no DRAM command
    conflicts. Partial placement needs both the raw attention vector for GPU
    rows and the PIM-produced output slice, so its incremental link traffic is
    charged. This helper is a feasibility test, not a claim that CHIME exposes
    the required dense kernel today.
    """

    if subbatch_size <= 0:
        raise ValueError("subbatch_size must be positive")
    fraction_values = tuple(dict.fromkeys(float(value) for value in fractions))
    if not fraction_values or any(not 0.0 <= value <= 1.0 for value in fraction_values):
        raise ValueError("O_proj fractions must lie in [0, 1]")

    model = MODEL_PRESETS[model_name]
    system = system or SYSTEM_PRESETS["chime_a100"]
    gpu = system.gpu
    pim = system.pim
    local_batch = ceil(subbatch_size / model.data_parallel)
    gpu_group_bandwidth = gpu.bandwidth_bytes_s / model.data_parallel
    gpu_group_flops = gpu.flops_s / model.data_parallel

    def gpu_matrix_us(elements: float, batch: int, launch_us: float) -> float:
        if elements <= 0:
            return 0.0
        memory_us = (
            elements
            * model.dtype_bytes
            / (gpu_group_bandwidth * gpu.dense_hbm_efficiency)
            * 1e6
        )
        compute_us = (
            2
            * elements
            * batch
            / (gpu_group_flops * gpu.compute_efficiency)
            * 1e6
        )
        return max(memory_us, compute_us) + launch_us

    qkv_us = ChimeCostModel(model, system, "tile").dense_layer_cost(
        subbatch_size
    ).qkv_us
    proj_ff_elements = model.proj_ff_weight_elements_per_layer
    o_launch_us = gpu.proj_ff_launch_us * (
        model.output_projection_weight_elements_per_layer / proj_ff_elements
    )
    ff_launch_us = gpu.proj_ff_launch_us - o_launch_us
    ff_us = gpu_matrix_us(
        model.ff_weight_elements_per_layer,
        local_batch,
        launch_us=ff_launch_us,
    )
    rows: list[dict[str, object]] = []
    for context in context_lengths:
        views = views_for_requests(
            [
                RequestState(f"offload-{index}", 0.0, int(context), 1)
                for index in range(subbatch_size)
            ]
        )
        pim_attention = ChimeCostModel(model, system, "tile").pim_layer_cost(
            views
        ).total_us
        baseline_o_proj = gpu_matrix_us(
            model.output_projection_weight_elements_per_layer,
            local_batch,
            launch_us=o_launch_us,
        )
        baseline_gpu_path = qkv_us + baseline_o_proj + ff_us
        baseline_phase = max(pim_attention, baseline_gpu_path)

        for fraction in fraction_values:
            gpu_o_proj = gpu_matrix_us(
                model.output_projection_weight_elements_per_layer
                * (1.0 - fraction),
                local_batch,
                launch_us=o_launch_us,
            )
            pim_o_proj_elements = (
                model.output_projection_weight_elements_per_layer * fraction
            )
            pim_o_proj_weight_bytes = pim_o_proj_elements * model.dtype_bytes
            pim_o_proj_memory_us = (
                pim_o_proj_weight_bytes / pim.layer_bandwidth_bytes_s * 1e6
                if fraction
                else 0.0
            )
            pim_dense_flops_s = (
                pim.layer_bandwidth_bytes_s
                * pim.dense_mac_flops_per_byte
                * pim.dense_compute_efficiency
            )
            pim_o_proj_compute_us = (
                2
                * pim_o_proj_elements
                * subbatch_size
                / pim_dense_flops_s
                * 1e6
                if fraction
                else 0.0
            )
            pim_o_proj_us = max(pim_o_proj_memory_us, pim_o_proj_compute_us)

            # At f=1 the normal attention-output transfer is replaced by a
            # same-sized O_proj output. A partial split additionally returns
            # only its output rows while GPU still needs the raw input vector.
            extra_link_bytes = (
                subbatch_size
                * model.output_bytes_per_token_per_layer
                * fraction
                if 0.0 < fraction < 1.0
                else 0.0
            )
            extra_link_us = (
                extra_link_bytes / system.link.pim_to_host_bytes_s * 1e6
            )
            gpu_path = qkv_us + gpu_o_proj + ff_us
            pim_path = pim_attention + pim_o_proj_us + extra_link_us
            phase_us = max(gpu_path, pim_path)
            persistent_weight_gib = (
                pim_o_proj_weight_bytes * model.layers / GiB
            )
            rows.append(
                {
                    "model": model.name,
                    "subbatch_size": subbatch_size,
                    "context_tokens": int(context),
                    "o_proj_fraction_on_pim": fraction,
                    "baseline_phase_us": baseline_phase,
                    "offload_phase_us": phase_us,
                    "speedup": baseline_phase / phase_us,
                    "gpu_path_us": gpu_path,
                    "pim_attention_us": pim_attention,
                    "pim_o_proj_us": pim_o_proj_us,
                    "extra_link_us": extra_link_us,
                    "persistent_pim_weight_gib": persistent_weight_gib,
                    "optimistic_bound": True,
                }
            )
    return rows


def dynamic_operator_offload_rows(
    model_name: str = "gpt175b",
    system_name: str = "chime_a100",
    subbatch_sizes: Iterable[int] = (1, 4, 8, 16),
    input_lengths: Iterable[int] = (128, 2048, 8192),
    output_lengths: Iterable[int] = (4096,),
    fractions: Iterable[float] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
    decision_interval: int = 64,
    min_residence_tokens: int = 512,
    switch_penalty_us: float = 0.0,
) -> list[dict[str, object]]:
    """Decode-lifetime upper bound for floating O-projection placement.

    This deliberately remains a narrow feasibility experiment rather than a
    general dynamic operator scheduler.  A prefix of O-projection output rows
    is assumed to have a persistent weight replica on PIM.  The chosen row
    fraction may change only at ``decision_interval`` boundaries and must obey
    ``min_residence_tokens`` after a change.  A small dynamic program uses the
    known output length to find the minimum-cost schedule.

    The comparison includes CHIME, the best one-fraction static placement, an
    unconstrained per-token upper bound, and the constrained dynamic schedule.
    Historical KV placement is unchanged by this experiment.
    """

    if decision_interval <= 0:
        raise ValueError("decision_interval must be positive")
    if min_residence_tokens < 0:
        raise ValueError("min_residence_tokens cannot be negative")
    if switch_penalty_us < 0:
        raise ValueError("switch_penalty_us cannot be negative")
    fraction_values = tuple(
        sorted(dict.fromkeys(float(value) for value in fractions))
    )
    if not fraction_values or any(
        not 0.0 <= value <= 1.0 for value in fraction_values
    ):
        raise ValueError("O_proj fractions must lie in [0, 1]")
    if 0.0 not in fraction_values:
        raise ValueError("dynamic AF comparison requires fraction 0 for CHIME")

    model = MODEL_PRESETS[model_name]
    system = SYSTEM_PRESETS[system_name]
    zero_index = fraction_values.index(0.0)
    rows: list[dict[str, object]] = []

    for subbatch_size in subbatch_sizes:
        if subbatch_size <= 0:
            raise ValueError("subbatch sizes must be positive")
        for input_tokens in input_lengths:
            if input_tokens <= 0:
                raise ValueError("input lengths must be positive")
            for output_tokens in output_lengths:
                if output_tokens <= 0:
                    raise ValueError("output lengths must be positive")

                contexts = tuple(
                    range(int(input_tokens), int(input_tokens) + int(output_tokens))
                )
                point_rows = operator_offload_rows(
                    model_name=model_name,
                    subbatch_size=int(subbatch_size),
                    context_lengths=contexts,
                    fractions=fraction_values,
                    system=system,
                )
                phase_by_context_fraction = {
                    (
                        int(row["context_tokens"]),
                        float(row["o_proj_fraction_on_pim"]),
                    ): float(row["offload_phase_us"]) * model.layers
                    for row in point_rows
                }
                persistent_weight_by_fraction = {
                    float(row["o_proj_fraction_on_pim"]): float(
                        row["persistent_pim_weight_gib"]
                    )
                    for row in point_rows[: len(fraction_values)]
                }

                token_costs = tuple(
                    tuple(
                        phase_by_context_fraction[(context, fraction)]
                        for fraction in fraction_values
                    )
                    for context in contexts
                )
                baseline_us = sum(costs[zero_index] for costs in token_costs)
                static_totals = tuple(
                    sum(costs[index] for costs in token_costs)
                    for index in range(len(fraction_values))
                )
                best_static_index = min(
                    range(len(fraction_values)),
                    key=lambda index: (static_totals[index], index),
                )

                ideal_indices = tuple(
                    min(
                        range(len(fraction_values)),
                        key=lambda index: (costs[index], index),
                    )
                    for costs in token_costs
                )
                ideal_us = sum(
                    token_costs[token][index]
                    for token, index in enumerate(ideal_indices)
                )
                ideal_switches = sum(
                    left != right
                    for left, right in zip(
                        (zero_index, *ideal_indices[:-1]), ideal_indices
                    )
                )

                blocks = tuple(
                    (start, min(start + decision_interval, output_tokens))
                    for start in range(0, output_tokens, decision_interval)
                )
                block_costs = tuple(
                    tuple(
                        sum(token_costs[token][fraction_index] for token in range(start, end))
                        for fraction_index in range(len(fraction_values))
                    )
                    for start, end in blocks
                )

                # Forward dynamic programming avoids Python's recursion limit
                # when experiments use fine decision intervals and long output
                # lengths.  Once a residence lock expires, all past expiry
                # values are equivalent and are canonicalized to ``unlocked``.
                unlocked = -10**18
                states: dict[
                    tuple[int, int], tuple[float, int, tuple[int, ...]]
                ] = {(zero_index, unlocked): (0.0, 0, ())}
                for block_index, (start, end) in enumerate(blocks):
                    next_states: dict[
                        tuple[int, int], tuple[float, int, tuple[int, ...]]
                    ] = {}
                    next_start = (
                        blocks[block_index + 1][0]
                        if block_index + 1 < len(blocks)
                        else end
                    )
                    for (current_fraction_index, lock_until_token), value in states.items():
                        total_us, switches, path = value
                        allowed = (
                            range(len(fraction_values))
                            if start >= lock_until_token
                            else (current_fraction_index,)
                        )
                        for target_index in allowed:
                            changed = target_index != current_fraction_index
                            next_lock = (
                                start + min_residence_tokens
                                if changed
                                else lock_until_token
                            )
                            if next_lock <= next_start:
                                next_lock = unlocked
                            candidate = (
                                total_us
                                + block_costs[block_index][target_index]
                                + (switch_penalty_us if changed else 0.0),
                                switches + int(changed),
                                (*path, target_index),
                            )
                            state_key = (target_index, next_lock)
                            incumbent = next_states.get(state_key)
                            if incumbent is None or candidate < incumbent:
                                next_states[state_key] = candidate
                    states = next_states

                dynamic_us, dynamic_switches, dynamic_path = min(states.values())
                schedule: list[str] = []
                previous_index: int | None = None
                for block_index, fraction_index in enumerate(dynamic_path):
                    if fraction_index == previous_index:
                        continue
                    schedule.append(
                        f"{blocks[block_index][0]}:{fraction_values[fraction_index]:g}"
                    )
                    previous_index = fraction_index
                max_fraction = max(
                    (fraction_values[index] for index in dynamic_path), default=0.0
                )
                persistent_weight_gib = persistent_weight_by_fraction[max_fraction]
                initial_weight_copy_us = (
                    persistent_weight_gib
                    * GiB
                    / system.link.host_to_pim_bytes_s
                    * 1e6
                )
                final_pim_kv_gib = (
                    int(subbatch_size)
                    * (int(input_tokens) + int(output_tokens))
                    * model.kv_bytes_per_token
                    / GiB
                )
                # CHIME forms two decode sub-batches whenever the active batch
                # contains more than one request.  The phase model above takes
                # one sub-batch size, so report both the local footprint and an
                # explicit equal-sized two-sub-batch capacity estimate.
                implied_full_batch_size = (
                    int(subbatch_size)
                    if int(subbatch_size) == 1
                    else 2 * int(subbatch_size)
                )
                final_full_batch_pim_kv_gib = (
                    implied_full_batch_size
                    * (int(input_tokens) + int(output_tokens))
                    * model.kv_bytes_per_token
                    / GiB
                )
                capacity_feasible = (
                    final_full_batch_pim_kv_gib + persistent_weight_gib
                    <= system.pim.capacity_gib
                )
                constrained_speedup = baseline_us / dynamic_us
                dynamic_vs_static = (
                    static_totals[best_static_index] / dynamic_us
                )
                cold_start_speedup = baseline_us / (
                    dynamic_us + initial_weight_copy_us
                )
                if not capacity_feasible:
                    classification = "infeasible_pim_capacity"
                elif constrained_speedup <= 1.001:
                    classification = "neutral_no_profitable_o_proj_offload"
                elif dynamic_vs_static <= 1.001:
                    classification = "benefit_static_offload_only"
                else:
                    classification = "benefit_dynamic_fraction_crossover"
                if not capacity_feasible:
                    cold_start_classification = "infeasible_pim_capacity"
                elif cold_start_speedup <= 1.001:
                    cold_start_classification = (
                        "cold_start_overhead_dominates_or_neutral"
                    )
                else:
                    cold_start_classification = "cold_start_net_benefit"

                rows.append(
                    {
                        "model": model.name,
                        "system": system_name,
                        "subbatch_size": int(subbatch_size),
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "decision_interval": decision_interval,
                        "min_residence_tokens": min_residence_tokens,
                        "switch_penalty_us": switch_penalty_us,
                        "baseline_decode_us": baseline_us,
                        "best_static_fraction": fraction_values[best_static_index],
                        "best_static_speedup": baseline_us
                        / static_totals[best_static_index],
                        "ideal_dynamic_speedup": baseline_us / ideal_us,
                        "ideal_dynamic_switches": ideal_switches,
                        "constrained_dynamic_speedup": constrained_speedup,
                        "dynamic_vs_best_static": dynamic_vs_static,
                        "constrained_dynamic_switches": dynamic_switches,
                        "constrained_dynamic_schedule": "|".join(schedule),
                        "max_pim_o_proj_fraction": max_fraction,
                        "persistent_pim_weight_gib": persistent_weight_by_fraction[
                            max_fraction
                        ],
                        "initial_weight_copy_s": initial_weight_copy_us / 1e6,
                        "dynamic_speedup_with_initial_weight_copy": (
                            cold_start_speedup
                        ),
                        "final_subbatch_pim_kv_gib": final_pim_kv_gib,
                        "subbatch_plus_weight_capacity_feasible": (
                            final_pim_kv_gib + persistent_weight_gib
                            <= system.pim.capacity_gib
                        ),
                        "implied_full_batch_size": implied_full_batch_size,
                        "final_implied_full_batch_pim_kv_gib": (
                            final_full_batch_pim_kv_gib
                        ),
                        "pim_capacity_feasible": capacity_feasible,
                        "capacity_scope": "one_or_two_equal_chime_subbatches",
                        "classification": classification,
                        "cold_start_classification": (
                            cold_start_classification
                        ),
                        "optimistic_bound": True,
                        "weight_policy": "persistent_replica_up_to_max_fraction",
                    }
                )
    return rows


def static_operator_mapping_rows(
    model_names: Iterable[str] = ("gpt175b",),
    mappings: Iterable[str] = (
        "chime",
        "attn_pim_control",
        "gpu_only",
        "dops_af",
        "dops_ffn",
        "dops_all_pim",
    ),
    batch_sizes: Iterable[int] = (1, 4, 8, 16, 32),
    context_lengths: Iterable[int] = (128, 2048, 8192, 32768),
    output_tokens: int = 1,
) -> list[dict[str, object]]:
    """Compare coarse static A/F operator bundles on paired workloads.

    All mappings use the operator-event scheduler so a CHIME row and every
    alternative share one timing engine.  These DOPS-inspired mappings are
    feasibility abstractions, not reproductions of DOPS hardware.
    """

    mapping_values = tuple(dict.fromkeys(str(value) for value in mappings))
    if "chime" not in mapping_values or "attn_pim_control" not in mapping_values:
        raise ValueError(
            "static operator comparison requires CHIME and attn_pim_control baselines"
        )
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        for batch_size in batch_sizes:
            for context_tokens in context_lengths:
                paired: list[dict[str, object]] = []
                for mapping in mapping_values:
                    raw = {
                        "model": model_name,
                        "hardware": "chime_a100",
                        "workload": {
                            "type": "static_batch",
                            "batch_size": int(batch_size),
                            "input_tokens": int(context_tokens),
                            "output_tokens": int(output_tokens),
                        },
                        "simulation": {
                            "max_batch_size": int(batch_size),
                            "fidelity": "tile",
                            "pipeline_model": "rankset_event",
                            "subbatch_scheduler": "lpt",
                            "operator_mapping": mapping,
                            "record_every": max(1, int(output_tokens)),
                            "policy": "chime",
                        },
                    }
                    try:
                        result = run_simulation(project_config_from_dict(raw))
                        summary = result.summary
                        paired.append(
                            {
                                "model": summary["model"],
                                "batch_size": int(batch_size),
                                "context_tokens": int(context_tokens),
                                "output_tokens": int(output_tokens),
                                "operator_mapping": mapping,
                                "operator_mapping_label": summary[
                                    "operator_mapping_label"
                                ],
                                "status": summary["status"],
                                "throughput_tokens_s": summary[
                                    "throughput_tokens_s"
                                ],
                                "simulated_time_s": summary["simulated_time_s"],
                                "mean_tpot_ms": summary["latency_ms"]["mean_tpot"],
                                "pim_persistent_weight_gib": summary[
                                    "pim_persistent_weight_gib"
                                ],
                                "pim_kv_capacity_gib": summary[
                                    "pim_kv_capacity_gib"
                                ],
                                "gpu_offloaded_weight_gib": summary[
                                    "gpu_offloaded_weight_gib"
                                ],
                                "gpu_kv_capacity_gib": summary[
                                    "gpu_kv_capacity_gib"
                                ],
                                "pim_bottleneck_fraction": summary[
                                    "pim_bottleneck_phase_fraction"
                                ],
                                "error": "",
                            }
                        )
                    except (SimulationCapacityError, ValueError) as error:
                        paired.append(
                            {
                                "model": MODEL_PRESETS[model_name].name,
                                "batch_size": int(batch_size),
                                "context_tokens": int(context_tokens),
                                "output_tokens": int(output_tokens),
                                "operator_mapping": mapping,
                                "operator_mapping_label": mapping,
                                "status": "capacity_error",
                                "throughput_tokens_s": None,
                                "simulated_time_s": None,
                                "mean_tpot_ms": None,
                                "pim_persistent_weight_gib": None,
                                "pim_kv_capacity_gib": None,
                                "gpu_offloaded_weight_gib": None,
                                "gpu_kv_capacity_gib": None,
                                "pim_bottleneck_fraction": None,
                                "error": str(error),
                            }
                        )

                baseline = next(
                    row for row in paired if row["operator_mapping"] == "chime"
                )
                placement_control = next(
                    row
                    for row in paired
                    if row["operator_mapping"] == "attn_pim_control"
                )
                baseline_throughput = baseline["throughput_tokens_s"]
                control_throughput = placement_control["throughput_tokens_s"]
                for row in paired:
                    throughput = row["throughput_tokens_s"]
                    row["speedup_vs_chime"] = (
                        float(throughput) / float(baseline_throughput)
                        if throughput is not None and baseline_throughput is not None
                        else None
                    )
                    row["speedup_vs_attn_pim_control"] = (
                        float(throughput) / float(control_throughput)
                        if throughput is not None and control_throughput is not None
                        else None
                    )
                    rows.append(row)
    return rows


def joint_af_small_oracle_rows(
    model_names: Iterable[str] = ("gpt175b",),
    batch_sizes: Iterable[int] = (2, 4),
    input_lengths: Iterable[int] = (2048, 8192),
    output_lengths: Iterable[int] = (32, 64),
    fractions: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    decision_interval: int = 16,
    min_residence_tokens: int = 16,
    max_kv_switches_per_decision: int | None = None,
    max_cost_evaluations: int = 200_000,
    greedy_terminal_samples: int = 8,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Small closed-batch exact oracle for joint KV and O_proj decisions.

    Every comparison uses the same optimistic GPU/PIM/link resource bound.
    ``joint_exact`` searches both future-only request-tail placement and a
    global row fraction for O_proj.  The two single-mechanism exact searches,
    best lifetime-static action, and sampled lifetime-aware coordinate policy
    are paired on the identical homogeneous workload.
    """

    fraction_values = tuple(
        sorted(dict.fromkeys(float(value) for value in fractions))
    )
    if not fraction_values or fraction_values[0] != 0.0:
        raise ValueError("joint AF experiment fractions must include 0")
    rows: list[dict[str, object]] = []

    for model_name in model_names:
        model = MODEL_PRESETS[model_name]
        system = SYSTEM_PRESETS["chime_a100"]
        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError("batch sizes must be positive")
            for input_tokens in input_lengths:
                for output_tokens in output_lengths:
                    label = (
                        f"{model_name}:B{batch_size}:I{input_tokens}:"
                        f"O{output_tokens}"
                    )
                    if progress:
                        progress(label)
                    requests = [
                        RequestState(
                            f"r{index}",
                            0.0,
                            int(input_tokens),
                            int(output_tokens),
                            gpu_replica=index % model.data_parallel,
                        )
                        for index in range(int(batch_size))
                    ]

                    fixed_cost = JointAFResourceModel(
                        model,
                        system,
                        max_o_proj_pim_fraction=0.0,
                    )
                    flexible_cost = JointAFResourceModel(
                        model,
                        system,
                        max_o_proj_pim_fraction=max(fraction_values),
                    )

                    def scheduler(
                        cost_model: JointAFResourceModel,
                        available_fractions: tuple[float, ...],
                    ) -> JointAFScheduler:
                        return JointAFScheduler(
                            cost_model,
                            fractions=available_fractions,
                            decision_interval=decision_interval,
                            min_kv_segment_tokens=min_residence_tokens,
                            min_kv_switch_interval=min_residence_tokens,
                            min_operator_residence_tokens=(
                                min_residence_tokens
                            ),
                            max_kv_switches_per_decision=min(
                                int(batch_size),
                                (
                                    int(max_kv_switches_per_decision)
                                    if max_kv_switches_per_decision is not None
                                    else int(batch_size)
                                ),
                            ),
                            max_exact_candidates=int(batch_size),
                            max_cost_evaluations=max_cost_evaluations,
                        )

                    baseline = scheduler(fixed_cost, (0.0,)).run_best_static(
                        requests,
                        allow_gpu_kv=False,
                        allow_operator=False,
                    )
                    kv_only = scheduler(fixed_cost, (0.0,)).run_exact(
                        requests,
                        allow_gpu_kv=True,
                        allow_operator=False,
                    )
                    flexible = scheduler(flexible_cost, fraction_values)
                    o_proj_only = flexible.run_exact(
                        requests,
                        allow_gpu_kv=False,
                        allow_operator=True,
                    )
                    best_static = flexible.run_best_static(
                        requests,
                        allow_gpu_kv=True,
                        allow_operator=True,
                    )
                    joint_exact = flexible.run_exact(
                        requests,
                        allow_gpu_kv=True,
                        allow_operator=True,
                    )
                    joint_greedy = flexible.run_greedy(
                        requests,
                        allow_gpu_kv=True,
                        allow_operator=True,
                        terminal_samples=greedy_terminal_samples,
                    )

                    assert_exact_not_slower(joint_exact, o_proj_only)
                    assert_exact_not_slower(joint_exact, best_static)
                    assert_exact_not_slower(joint_exact, joint_greedy)
                    # KV-only has no O_proj replica reservation. When the joint
                    # run fits, its action space contains the same timing path.
                    assert_exact_not_slower(joint_exact, kv_only)

                    best_single_us = min(
                        kv_only.total_us, o_proj_only.total_us
                    )
                    speedup = baseline.total_us / joint_exact.total_us
                    complementarity = best_single_us / joint_exact.total_us
                    dynamic_vs_static = (
                        best_static.total_us / joint_exact.total_us
                    )
                    if complementarity > 1.001:
                        classification = "joint_complementarity"
                    elif dynamic_vs_static > 1.001:
                        classification = "dynamic_timing_only"
                    elif speedup > 1.001:
                        classification = "single_or_static_action_suffices"
                    else:
                        classification = "neutral_joint_upper_bound"

                    initial_copy_us = flexible_cost.initial_weight_copy_us
                    rows.append(
                        {
                            "model": model.name,
                            "hardware": "chime_a100",
                            "batch_size": int(batch_size),
                            "input_tokens": int(input_tokens),
                            "output_tokens": int(output_tokens),
                            "decision_interval": decision_interval,
                            "min_residence_tokens": min_residence_tokens,
                            "max_kv_switches_per_decision": (
                                int(max_kv_switches_per_decision)
                                if max_kv_switches_per_decision is not None
                                else int(batch_size)
                            ),
                            "o_proj_fractions": "|".join(
                                f"{value:g}" for value in fraction_values
                            ),
                            "chime_resource_bound_us": baseline.total_us,
                            "kv_only_exact_us": kv_only.total_us,
                            "o_proj_only_exact_us": o_proj_only.total_us,
                            "joint_best_static_us": best_static.total_us,
                            "joint_exact_us": joint_exact.total_us,
                            "joint_lifetime_greedy_us": (
                                joint_greedy.total_us
                            ),
                            "kv_only_speedup_vs_chime": (
                                baseline.total_us / kv_only.total_us
                            ),
                            "o_proj_only_speedup_vs_chime": (
                                baseline.total_us / o_proj_only.total_us
                            ),
                            "joint_exact_speedup_vs_chime": speedup,
                            "joint_vs_best_single": complementarity,
                            "joint_dynamic_vs_best_static": (
                                dynamic_vs_static
                            ),
                            "greedy_slowdown_vs_exact": (
                                joint_greedy.total_us / joint_exact.total_us
                            ),
                            "joint_speedup_with_initial_weight_copy": (
                                baseline.total_us
                                / (joint_exact.total_us + initial_copy_us)
                            ),
                            "persistent_pim_o_proj_weight_gib": (
                                flexible_cost.pim_o_proj_weight_gib
                            ),
                            "initial_weight_copy_s": initial_copy_us / 1e6,
                            "pim_kv_capacity_after_replica_gib": (
                                flexible_cost.pim_kv_capacity_bytes / GiB
                            ),
                            "gpu_replica_kv_capacity_gib": (
                                flexible_cost.gpu_replica_kv_capacity_bytes
                                / GiB
                            ),
                            "kv_only_switches": kv_only.kv_switches,
                            "o_proj_only_switches": (
                                o_proj_only.operator_switches
                            ),
                            "joint_kv_switches": joint_exact.kv_switches,
                            "joint_operator_switches": (
                                joint_exact.operator_switches
                            ),
                            "kv_only_schedule": format_joint_schedule(
                                kv_only.schedule
                            ),
                            "o_proj_only_schedule": format_joint_schedule(
                                o_proj_only.schedule
                            ),
                            "joint_best_static_schedule": (
                                format_joint_schedule(best_static.schedule)
                            ),
                            "joint_exact_schedule": format_joint_schedule(
                                joint_exact.schedule
                            ),
                            "joint_greedy_schedule": format_joint_schedule(
                                joint_greedy.schedule
                            ),
                            "joint_exact_cost_evaluations": (
                                joint_exact.cost_evaluations
                            ),
                            "joint_greedy_cost_evaluations": (
                                joint_greedy.cost_evaluations
                            ),
                            "greedy_terminal_samples": (
                                greedy_terminal_samples
                            ),
                            "classification": classification,
                            "historical_kv_migration": False,
                            "cost_model": (
                                "optimistic_joint_gpu_pim_link_resource_bound"
                            ),
                        }
                    )
    return rows


JOINT_HETEROGENEOUS_PROFILES: dict[str, tuple[int, ...]] = {
    # All profiles contain four requests and exactly 640 generated tokens.
    # They differ only in when requests leave the active batch.
    "balanced": (160, 160, 160, 160),
    "staircase": (64, 128, 192, 256),
    "two_wave": (64, 64, 256, 256),
    "three_then_long": (128, 128, 128, 256),
}


def joint_af_heterogeneous_oracle_rows(
    model_names: Iterable[str] = ("gpt175b",),
    profile_names: Iterable[str] = tuple(JOINT_HETEROGENEOUS_PROFILES),
    input_lengths: Iterable[int] = (12000,),
    fractions: Iterable[float] = (0.0, 0.5, 0.75, 1.0),
    decision_interval: int = 64,
    min_residence_tokens: int = 64,
    max_kv_switches_per_decision: int = 1,
    max_cost_evaluations: int = 3_000_000,
    greedy_terminal_samples: int = 8,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Exact closed-batch comparison with completion-event heterogeneity.

    The built-in profiles hold both initial batch size and total generated
    tokens constant. Only the request completion pattern changes, isolating
    whether exact output-length knowledge makes dynamic AF rebalancing useful.
    """

    fraction_values = tuple(
        sorted(dict.fromkeys(float(value) for value in fractions))
    )
    if not fraction_values or fraction_values[0] != 0.0:
        raise ValueError("joint AF experiment fractions must include 0")
    rows: list[dict[str, object]] = []

    for model_name in model_names:
        model = MODEL_PRESETS[model_name]
        system = SYSTEM_PRESETS["chime_a100"]
        for profile_name in profile_names:
            try:
                output_lengths = JOINT_HETEROGENEOUS_PROFILES[profile_name]
            except KeyError as error:
                choices = ", ".join(JOINT_HETEROGENEOUS_PROFILES)
                raise ValueError(
                    f"unknown heterogeneous profile {profile_name!r}; "
                    f"choose from {choices}"
                ) from error
            batch_size = len(output_lengths)
            for input_tokens in input_lengths:
                label = (
                    f"{model_name}:{profile_name}:B{batch_size}:"
                    f"I{input_tokens}:O{'|'.join(map(str, output_lengths))}"
                )
                if progress:
                    progress(label)
                requests = [
                    RequestState(
                        f"r{index}",
                        0.0,
                        int(input_tokens),
                        int(output_tokens),
                        gpu_replica=index % model.data_parallel,
                    )
                    for index, output_tokens in enumerate(output_lengths)
                ]
                fixed_cost = JointAFResourceModel(
                    model, system, max_o_proj_pim_fraction=0.0
                )
                flexible_cost = JointAFResourceModel(
                    model,
                    system,
                    max_o_proj_pim_fraction=max(fraction_values),
                )

                def scheduler(
                    cost_model: JointAFResourceModel,
                    available_fractions: tuple[float, ...],
                ) -> JointAFScheduler:
                    return JointAFScheduler(
                        cost_model,
                        fractions=available_fractions,
                        decision_interval=decision_interval,
                        min_kv_segment_tokens=min_residence_tokens,
                        min_kv_switch_interval=min_residence_tokens,
                        min_operator_residence_tokens=min_residence_tokens,
                        max_kv_switches_per_decision=min(
                            batch_size, max_kv_switches_per_decision
                        ),
                        max_exact_candidates=batch_size,
                        max_cost_evaluations=max_cost_evaluations,
                    )

                baseline = scheduler(fixed_cost, (0.0,)).run_best_static(
                    requests,
                    allow_gpu_kv=False,
                    allow_operator=False,
                )
                kv_only = scheduler(fixed_cost, (0.0,)).run_exact(
                    requests,
                    allow_gpu_kv=True,
                    allow_operator=False,
                )
                flexible = scheduler(flexible_cost, fraction_values)
                o_proj_only = flexible.run_exact(
                    requests,
                    allow_gpu_kv=False,
                    allow_operator=True,
                )
                best_static = flexible.run_best_static(requests)
                joint_exact = flexible.run_exact(requests)
                joint_greedy = flexible.run_greedy(
                    requests, terminal_samples=greedy_terminal_samples
                )

                assert_exact_not_slower(joint_exact, kv_only)
                assert_exact_not_slower(joint_exact, o_proj_only)
                assert_exact_not_slower(joint_exact, best_static)
                assert_exact_not_slower(joint_exact, joint_greedy)

                best_single_us = min(kv_only.total_us, o_proj_only.total_us)
                speedup = baseline.total_us / joint_exact.total_us
                complementarity = best_single_us / joint_exact.total_us
                dynamic_vs_static = best_static.total_us / joint_exact.total_us
                if complementarity > 1.001:
                    classification = "joint_complementarity"
                elif dynamic_vs_static > 1.001:
                    classification = "dynamic_timing_only"
                elif speedup > 1.001:
                    classification = "single_or_static_action_suffices"
                else:
                    classification = "neutral_joint_upper_bound"

                initial_copy_us = flexible_cost.initial_weight_copy_us
                rows.append(
                    {
                        "model": model.name,
                        "hardware": "chime_a100",
                        "workload_profile": profile_name,
                        "batch_size": batch_size,
                        "input_tokens": int(input_tokens),
                        "output_tokens_by_request": "|".join(
                            map(str, output_lengths)
                        ),
                        "total_output_tokens": sum(output_lengths),
                        "completion_waves": len(set(output_lengths)),
                        "decision_interval": decision_interval,
                        "min_residence_tokens": min_residence_tokens,
                        "max_kv_switches_per_decision": (
                            max_kv_switches_per_decision
                        ),
                        "o_proj_fractions": "|".join(
                            f"{value:g}" for value in fraction_values
                        ),
                        "chime_resource_bound_us": baseline.total_us,
                        "kv_only_exact_us": kv_only.total_us,
                        "o_proj_only_exact_us": o_proj_only.total_us,
                        "joint_best_static_us": best_static.total_us,
                        "joint_exact_us": joint_exact.total_us,
                        "joint_lifetime_greedy_us": joint_greedy.total_us,
                        "kv_only_speedup_vs_chime": (
                            baseline.total_us / kv_only.total_us
                        ),
                        "o_proj_only_speedup_vs_chime": (
                            baseline.total_us / o_proj_only.total_us
                        ),
                        "joint_exact_speedup_vs_chime": speedup,
                        "joint_vs_best_single": complementarity,
                        "joint_dynamic_vs_best_static": dynamic_vs_static,
                        "greedy_slowdown_vs_exact": (
                            joint_greedy.total_us / joint_exact.total_us
                        ),
                        "joint_speedup_with_initial_weight_copy": (
                            baseline.total_us
                            / (joint_exact.total_us + initial_copy_us)
                        ),
                        "persistent_pim_o_proj_weight_gib": (
                            flexible_cost.pim_o_proj_weight_gib
                        ),
                        "initial_weight_copy_s": initial_copy_us / 1e6,
                        "pim_kv_capacity_after_replica_gib": (
                            flexible_cost.pim_kv_capacity_bytes / GiB
                        ),
                        "gpu_replica_kv_capacity_gib": (
                            flexible_cost.gpu_replica_kv_capacity_bytes / GiB
                        ),
                        "kv_only_switches": kv_only.kv_switches,
                        "o_proj_only_switches": o_proj_only.operator_switches,
                        "joint_kv_switches": joint_exact.kv_switches,
                        "joint_operator_switches": (
                            joint_exact.operator_switches
                        ),
                        "kv_only_schedule": format_joint_schedule(
                            kv_only.schedule
                        ),
                        "o_proj_only_schedule": format_joint_schedule(
                            o_proj_only.schedule
                        ),
                        "joint_best_static_schedule": format_joint_schedule(
                            best_static.schedule
                        ),
                        "joint_exact_schedule": format_joint_schedule(
                            joint_exact.schedule
                        ),
                        "joint_greedy_schedule": format_joint_schedule(
                            joint_greedy.schedule
                        ),
                        "joint_exact_cost_evaluations": (
                            joint_exact.cost_evaluations
                        ),
                        "joint_greedy_cost_evaluations": (
                            joint_greedy.cost_evaluations
                        ),
                        "greedy_terminal_samples": greedy_terminal_samples,
                        "classification": classification,
                        "historical_kv_migration": False,
                        "cost_model": (
                            "optimistic_joint_gpu_pim_link_resource_bound"
                        ),
                    }
                )
    return rows


def joint_af_event_replay_rows(
    source_rows: Iterable[Mapping[str, object]],
    *,
    model_name: str = "gpt175b",
    evaluation_strides: Iterable[int] = (64,),
    include_split_aware_oproj: bool = False,
    include_owner_coupled_merge: bool = False,
    owner_coupled_pim_merge_cost_scale: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Replay Stage 3B1 candidates on the dependency-aware event model.

    The expensive event DAG is not placed inside the exponential token-level
    exact search. Instead, this function exhaustively optimizes every
    creation-time future-KV assignment and every legal block-level O_proj
    fraction sequence, and also replays the Stage 3B1 runtime-dynamic joint
    schedule. This is a bounded candidate oracle, not a proof of the global
    runtime-KV optimum.
    """

    model = MODEL_PRESETS[model_name]
    system = SYSTEM_PRESETS["chime_a100"]
    rows: list[dict[str, object]] = []

    for source_index, source in enumerate(source_rows):
        batch_size = int(source["batch_size"])
        input_tokens = int(source["input_tokens"])
        if source.get("output_tokens_by_request"):
            output_lengths = tuple(
                int(value)
                for value in str(source["output_tokens_by_request"]).split("|")
            )
        else:
            output_lengths = (int(source["output_tokens"]),) * batch_size
        if len(output_lengths) != batch_size:
            raise ValueError("source output-length count does not match batch")
        request_ids = tuple(f"r{index}" for index in range(batch_size))
        requests = [
            RequestState(
                request_id,
                0.0,
                input_tokens,
                output_tokens,
                gpu_replica=index % model.data_parallel,
            )
            for index, (request_id, output_tokens) in enumerate(
                zip(request_ids, output_lengths)
            )
        ]
        fractions = tuple(
            float(value)
            for value in str(source["o_proj_fractions"]).split("|")
        )
        decision_interval = int(source["decision_interval"])
        min_residence = int(source["min_residence_tokens"])
        max_kv_switches = int(source["max_kv_switches_per_decision"])
        if min_residence > decision_interval:
            raise ValueError(
                "event candidate fraction enumeration currently requires "
                "min_residence_tokens <= decision_interval"
            )

        resource_cost = JointAFResourceModel(
            model,
            system,
            max_o_proj_pim_fraction=max(fractions),
        )
        event_cost = JointAFEventModel(
            model,
            system,
            max_o_proj_pim_fraction=max(fractions),
        )
        split_aware_cost = (
            JointAFEventModel(
                model,
                system,
                max_o_proj_pim_fraction=max(fractions),
                o_proj_routing="split_aware_requests",
            )
            if include_split_aware_oproj
            else None
        )
        owner_coupled_cost = (
            JointAFEventModel(
                model,
                system,
                max_o_proj_pim_fraction=max(fractions),
                split_merge_owner="pim_when_pim_oproj",
                pim_merge_cost_scale=owner_coupled_pim_merge_cost_scale,
            )
            if include_owner_coupled_merge
            else None
        )

        def scheduler(cost_model: JointAFResourceModel) -> JointAFScheduler:
            return JointAFScheduler(
                cost_model,
                fractions=fractions,
                decision_interval=decision_interval,
                min_kv_segment_tokens=min_residence,
                min_kv_switch_interval=min_residence,
                min_operator_residence_tokens=min_residence,
                max_kv_switches_per_decision=max_kv_switches,
                max_exact_candidates=batch_size,
                max_cost_evaluations=3_000_000,
            )

        resource_scheduler = scheduler(resource_cost)
        event_scheduler = scheduler(event_cost)
        split_aware_scheduler = (
            scheduler(split_aware_cost)
            if split_aware_cost is not None
            else None
        )
        owner_coupled_scheduler = (
            scheduler(owner_coupled_cost)
            if owner_coupled_cost is not None
            else None
        )
        assignments = resource_scheduler._kv_target_assignments(
            requests,
            0,
            allow_gpu_kv=True,
            coordinate_only=False,
            enforce_switch_limit=False,
            enforce_residence=False,
        )
        all_pim_assignment = {
            request_id: Location.PIM for request_id in request_ids
        }
        decision_starts = tuple(
            range(0, max(output_lengths), decision_interval)
        )
        if len(decision_starts) > 6:
            raise ValueError(
                "event candidate enumeration supports at most six decision "
                "blocks; increase decision_interval or shorten output"
            )

        def source_schedule(
            field: str,
            *,
            allow_gpu_kv: bool,
            allow_operator: bool,
        ) -> tuple[tuple[int, JointAction], ...]:
            encoded = str(source.get(field, "") or "").strip()
            if encoded:
                return parse_joint_schedule(encoded, request_ids)
            # Early v4 checkpoints predate the extra ablation schedule columns.
            if field == "o_proj_only_schedule":
                result = resource_scheduler.run_exact(
                    requests,
                    allow_gpu_kv=allow_gpu_kv,
                    allow_operator=allow_operator,
                )
            elif field == "joint_best_static_schedule":
                result = resource_scheduler.run_best_static(
                    requests,
                    allow_gpu_kv=allow_gpu_kv,
                    allow_operator=allow_operator,
                )
            else:
                raise ValueError(f"source row is missing required {field}")
            return result.schedule

        resource_o_proj_schedule = source_schedule(
            "o_proj_only_schedule",
            allow_gpu_kv=False,
            allow_operator=True,
        )
        resource_joint_schedule = source_schedule(
            "joint_exact_schedule",
            allow_gpu_kv=True,
            allow_operator=True,
        )
        resource_greedy_encoded = str(
            source.get("joint_greedy_schedule", "") or ""
        ).strip()
        resource_greedy_schedule = (
            parse_joint_schedule(resource_greedy_encoded, request_ids)
            if resource_greedy_encoded
            else resource_joint_schedule
        )
        chime_schedule = (
            (0, JointAction.from_mapping(all_pim_assignment, 0.0)),
        )

        for stride in evaluation_strides:
            stride = int(stride)
            if stride <= 0:
                raise ValueError("event replay strides must be positive")
            label = (
                f"row{source_index}:B{batch_size}:I{input_tokens}:"
                f"O{'|'.join(map(str, output_lengths))}:stride{stride}"
            )
            if progress:
                progress(label)

            replay_calls = 0
            split_aware_replay_calls = 0
            owner_coupled_replay_calls = 0

            def replay(
                schedule: tuple[tuple[int, JointAction], ...]
            ) -> JointRunResult:
                nonlocal replay_calls
                replay_calls += 1
                return event_scheduler.replay_sampled(
                    requests,
                    schedule,
                    evaluation_stride_tokens=stride,
                )

            def replay_split_aware(
                schedule: tuple[tuple[int, JointAction], ...]
            ) -> JointRunResult:
                nonlocal split_aware_replay_calls
                if split_aware_scheduler is None:
                    raise AssertionError("split-aware replay was not requested")
                split_aware_replay_calls += 1
                return split_aware_scheduler.replay_sampled(
                    requests,
                    schedule,
                    evaluation_stride_tokens=stride,
                )

            def replay_owner_coupled(
                schedule: tuple[tuple[int, JointAction], ...]
            ) -> JointRunResult:
                nonlocal owner_coupled_replay_calls
                if owner_coupled_scheduler is None:
                    raise AssertionError(
                        "owner-coupled replay was not requested"
                    )
                owner_coupled_replay_calls += 1
                return owner_coupled_scheduler.replay_sampled(
                    requests,
                    schedule,
                    evaluation_stride_tokens=stride,
                )

            chime = replay(chime_schedule)
            resource_o_proj = replay(resource_o_proj_schedule)
            resource_joint = replay(resource_joint_schedule)
            resource_greedy = replay(resource_greedy_schedule)
            split_resource_joint = (
                replay_split_aware(resource_joint_schedule)
                if include_split_aware_oproj
                else None
            )
            split_resource_greedy = (
                replay_split_aware(resource_greedy_schedule)
                if include_split_aware_oproj
                else None
            )
            owner_resource_joint = (
                replay_owner_coupled(resource_joint_schedule)
                if include_owner_coupled_merge
                else None
            )
            owner_resource_greedy = (
                replay_owner_coupled(resource_greedy_schedule)
                if include_owner_coupled_merge
                else None
            )

            best_static_us = inf
            best_static_schedule: tuple[tuple[int, JointAction], ...] = ()
            best_dynamic_us = inf
            best_dynamic_schedule: tuple[tuple[int, JointAction], ...] = ()
            best_o_proj_us = inf
            best_o_proj_schedule: tuple[tuple[int, JointAction], ...] = ()
            split_best_static_us = inf
            split_best_static_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()
            split_best_dynamic_us = inf
            split_best_dynamic_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()
            split_best_o_proj_us = inf
            split_best_o_proj_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()
            owner_best_static_us = inf
            owner_best_static_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()
            owner_best_dynamic_us = inf
            owner_best_dynamic_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()
            owner_best_o_proj_us = inf
            owner_best_o_proj_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()

            fraction_sequences = tuple(
                product(fractions, repeat=len(decision_starts))
            )
            for assignment in assignments:
                for fraction in fractions:
                    candidate = (
                        (0, JointAction.from_mapping(assignment, fraction)),
                    )
                    cost = replay(candidate).total_us
                    if cost < best_static_us - 1e-9:
                        best_static_us = cost
                        best_static_schedule = candidate
                    if include_split_aware_oproj:
                        split_cost = replay_split_aware(candidate).total_us
                        if split_cost < split_best_static_us - 1e-9:
                            split_best_static_us = split_cost
                            split_best_static_schedule = candidate
                    if include_owner_coupled_merge:
                        owner_cost = replay_owner_coupled(candidate).total_us
                        if owner_cost < owner_best_static_us - 1e-9:
                            owner_best_static_us = owner_cost
                            owner_best_static_schedule = candidate

                for sequence in fraction_sequences:
                    candidate = tuple(
                        (
                            iteration,
                            JointAction.from_mapping(assignment, fraction),
                        )
                        for iteration, fraction in zip(
                            decision_starts, sequence
                        )
                    )
                    cost = replay(candidate).total_us
                    if cost < best_dynamic_us - 1e-9:
                        best_dynamic_us = cost
                        best_dynamic_schedule = candidate
                    if assignment == all_pim_assignment and cost < best_o_proj_us - 1e-9:
                        best_o_proj_us = cost
                        best_o_proj_schedule = candidate
                    if include_split_aware_oproj:
                        split_cost = replay_split_aware(candidate).total_us
                        if split_cost < split_best_dynamic_us - 1e-9:
                            split_best_dynamic_us = split_cost
                            split_best_dynamic_schedule = candidate
                        if (
                            assignment == all_pim_assignment
                            and split_cost < split_best_o_proj_us - 1e-9
                        ):
                            split_best_o_proj_us = split_cost
                            split_best_o_proj_schedule = candidate
                    if include_owner_coupled_merge:
                        owner_cost = replay_owner_coupled(candidate).total_us
                        if owner_cost < owner_best_dynamic_us - 1e-9:
                            owner_best_dynamic_us = owner_cost
                            owner_best_dynamic_schedule = candidate
                        if (
                            assignment == all_pim_assignment
                            and owner_cost < owner_best_o_proj_us - 1e-9
                        ):
                            owner_best_o_proj_us = owner_cost
                            owner_best_o_proj_schedule = candidate

            candidates = {
                "event_creation_kv_dynamic_oproj": (
                    best_dynamic_us,
                    best_dynamic_schedule,
                ),
                "resource_joint_replay": (
                    resource_joint.total_us,
                    resource_joint_schedule,
                ),
                "resource_greedy_replay": (
                    resource_greedy.total_us,
                    resource_greedy_schedule,
                ),
            }
            best_candidate_name, (
                best_candidate_us,
                best_candidate_schedule,
            ) = min(candidates.items(), key=lambda item: item[1][0])

            split_best_candidate_name = ""
            split_best_candidate_us = inf
            split_best_candidate_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()
            if include_split_aware_oproj:
                if split_resource_joint is None or split_resource_greedy is None:
                    raise AssertionError("split-aware resource replay is missing")
                split_candidates = {
                    "split_aware_creation_kv_dynamic_oproj": (
                        split_best_dynamic_us,
                        split_best_dynamic_schedule,
                    ),
                    "split_aware_resource_joint_replay": (
                        split_resource_joint.total_us,
                        resource_joint_schedule,
                    ),
                    "split_aware_resource_greedy_replay": (
                        split_resource_greedy.total_us,
                        resource_greedy_schedule,
                    ),
                }
                split_best_candidate_name, (
                    split_best_candidate_us,
                    split_best_candidate_schedule,
                ) = min(
                    split_candidates.items(), key=lambda item: item[1][0]
                )
                if not isclose(
                    split_best_o_proj_us,
                    best_o_proj_us,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ):
                    raise AssertionError(
                        "split-aware and uniform all-PIM O_proj controls diverged"
                    )

            owner_best_candidate_name = ""
            owner_best_candidate_us = inf
            owner_best_candidate_schedule: tuple[
                tuple[int, JointAction], ...
            ] = ()
            if include_owner_coupled_merge:
                if owner_resource_joint is None or owner_resource_greedy is None:
                    raise AssertionError(
                        "owner-coupled resource replay is missing"
                    )
                owner_candidates = {
                    "owner_coupled_creation_kv_dynamic_oproj": (
                        owner_best_dynamic_us,
                        owner_best_dynamic_schedule,
                    ),
                    "owner_coupled_resource_joint_replay": (
                        owner_resource_joint.total_us,
                        resource_joint_schedule,
                    ),
                    "owner_coupled_resource_greedy_replay": (
                        owner_resource_greedy.total_us,
                        resource_greedy_schedule,
                    ),
                }
                owner_best_candidate_name, (
                    owner_best_candidate_us,
                    owner_best_candidate_schedule,
                ) = min(
                    owner_candidates.items(), key=lambda item: item[1][0]
                )
                if not isclose(
                    owner_best_o_proj_us,
                    best_o_proj_us,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ):
                    raise AssertionError(
                        "owner-coupled and GPU-merge all-PIM O_proj controls "
                        "diverged"
                    )

            row: dict[str, object] = {
                    "model": model.name,
                    "hardware": "chime_a100",
                    "source_index": source_index,
                    "workload_profile": source.get(
                        "workload_profile", "homogeneous"
                    ),
                    "batch_size": batch_size,
                    "input_tokens": input_tokens,
                    "output_tokens_by_request": "|".join(
                        map(str, output_lengths)
                    ),
                    "evaluation_stride_tokens": stride,
                    "decision_interval": decision_interval,
                    "min_residence_tokens": min_residence,
                    "o_proj_fractions": "|".join(
                        f"{fraction:g}" for fraction in fractions
                    ),
                    "event_chime_us": chime.total_us,
                    "event_best_static_us": best_static_us,
                    "event_best_oproj_dynamic_us": best_o_proj_us,
                    "event_best_creation_kv_dynamic_oproj_us": (
                        best_dynamic_us
                    ),
                    "event_resource_joint_replay_us": (
                        resource_joint.total_us
                    ),
                    "event_resource_greedy_replay_us": (
                        resource_greedy.total_us
                    ),
                    "event_best_candidate_us": best_candidate_us,
                    "event_best_static_speedup_vs_chime": (
                        chime.total_us / best_static_us
                    ),
                    "event_oproj_dynamic_speedup_vs_chime": (
                        chime.total_us / best_o_proj_us
                    ),
                    "event_best_candidate_speedup_vs_chime": (
                        chime.total_us / best_candidate_us
                    ),
                    "event_joint_vs_oproj_dynamic": (
                        best_o_proj_us / best_candidate_us
                    ),
                    "event_dynamic_vs_best_static": (
                        best_static_us / best_candidate_us
                    ),
                    "resource_joint_replay_speedup_vs_chime": (
                        chime.total_us / resource_joint.total_us
                    ),
                    "resource_o_proj_replay_speedup_vs_chime": (
                        chime.total_us / resource_o_proj.total_us
                    ),
                    "event_best_candidate_name": best_candidate_name,
                    "event_best_static_schedule": format_joint_schedule(
                        best_static_schedule
                    ),
                    "event_best_oproj_dynamic_schedule": (
                        format_joint_schedule(best_o_proj_schedule)
                    ),
                    "event_best_creation_joint_schedule": (
                        format_joint_schedule(best_dynamic_schedule)
                    ),
                    "resource_joint_schedule": format_joint_schedule(
                        resource_joint_schedule
                    ),
                    "event_best_candidate_schedule": format_joint_schedule(
                        best_candidate_schedule
                    ),
                    "candidate_replay_calls": replay_calls,
                    "unique_event_iteration_costs": (
                        event_cost.cached_iteration_costs
                    ),
                    "candidate_oracle_scope": (
                        "all_creation_kv_assignments_x_all_legal_block_oproj_"
                        "sequences_plus_resource_runtime_kv_schedules"
                    ),
                    "historical_kv_migration": False,
                    "cost_model": "dependency_aware_joint_operator_event",
                }
            if include_split_aware_oproj:
                row.update(
                    {
                        "split_aware_best_static_us": split_best_static_us,
                        "split_aware_best_oproj_dynamic_us": (
                            split_best_o_proj_us
                        ),
                        "split_aware_best_creation_joint_us": (
                            split_best_dynamic_us
                        ),
                        "split_aware_best_candidate_us": (
                            split_best_candidate_us
                        ),
                        "split_aware_best_candidate_speedup_vs_chime": (
                            chime.total_us / split_best_candidate_us
                        ),
                        "split_aware_joint_vs_oproj_dynamic": (
                            split_best_o_proj_us / split_best_candidate_us
                        ),
                        "split_aware_dynamic_vs_best_static": (
                            split_best_static_us / split_best_candidate_us
                        ),
                        "split_aware_vs_uniform_best_candidate": (
                            best_candidate_us / split_best_candidate_us
                        ),
                        "split_aware_best_candidate_name": (
                            split_best_candidate_name
                        ),
                        "split_aware_best_static_schedule": (
                            format_joint_schedule(split_best_static_schedule)
                        ),
                        "split_aware_best_oproj_dynamic_schedule": (
                            format_joint_schedule(split_best_o_proj_schedule)
                        ),
                        "split_aware_best_creation_joint_schedule": (
                            format_joint_schedule(split_best_dynamic_schedule)
                        ),
                        "split_aware_best_candidate_schedule": (
                            format_joint_schedule(
                                split_best_candidate_schedule
                            )
                        ),
                        "split_aware_candidate_replay_calls": (
                            split_aware_replay_calls
                        ),
                        "split_aware_unique_event_iteration_costs": (
                            split_aware_cost.cached_iteration_costs
                            if split_aware_cost is not None
                            else 0
                        ),
                        "split_aware_dataflow": (
                            "pim_only_requests_use_selected_pim_oproj_rows;"
                            "split_or_gpu_origin_requests_keep_full_oproj_on_gpu"
                        ),
                    }
                )
            if include_owner_coupled_merge:
                row.update(
                    {
                        "owner_coupled_pim_merge_cost_scale": (
                            owner_coupled_pim_merge_cost_scale
                        ),
                        "owner_coupled_best_static_us": owner_best_static_us,
                        "owner_coupled_best_oproj_dynamic_us": (
                            owner_best_o_proj_us
                        ),
                        "owner_coupled_best_creation_joint_us": (
                            owner_best_dynamic_us
                        ),
                        "owner_coupled_best_candidate_us": (
                            owner_best_candidate_us
                        ),
                        "owner_coupled_best_candidate_speedup_vs_chime": (
                            chime.total_us / owner_best_candidate_us
                        ),
                        "owner_coupled_joint_vs_oproj_dynamic": (
                            owner_best_o_proj_us / owner_best_candidate_us
                        ),
                        "owner_coupled_dynamic_vs_best_static": (
                            owner_best_static_us / owner_best_candidate_us
                        ),
                        "owner_coupled_vs_gpu_merge_best_candidate": (
                            best_candidate_us / owner_best_candidate_us
                        ),
                        "owner_coupled_best_candidate_name": (
                            owner_best_candidate_name
                        ),
                        "owner_coupled_best_static_schedule": (
                            format_joint_schedule(owner_best_static_schedule)
                        ),
                        "owner_coupled_best_oproj_dynamic_schedule": (
                            format_joint_schedule(owner_best_o_proj_schedule)
                        ),
                        "owner_coupled_best_creation_joint_schedule": (
                            format_joint_schedule(owner_best_dynamic_schedule)
                        ),
                        "owner_coupled_best_candidate_schedule": (
                            format_joint_schedule(
                                owner_best_candidate_schedule
                            )
                        ),
                        "owner_coupled_resource_joint_replay_us": (
                            owner_resource_joint.total_us
                            if owner_resource_joint is not None
                            else inf
                        ),
                        "owner_coupled_resource_greedy_replay_us": (
                            owner_resource_greedy.total_us
                            if owner_resource_greedy is not None
                            else inf
                        ),
                        "owner_coupled_candidate_replay_calls": (
                            owner_coupled_replay_calls
                        ),
                        "owner_coupled_unique_event_iteration_costs": (
                            owner_coupled_cost.cached_iteration_costs
                            if owner_coupled_cost is not None
                            else 0
                        ),
                        "owner_coupled_dataflow": (
                            "gpu_merge_when_oproj_f0;"
                            "pim_merge_when_any_pim_oproj_fraction"
                        ),
                    }
                )
            rows.append(row)
    return rows
