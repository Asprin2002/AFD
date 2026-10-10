"""Versioned workload-suite loading and resumable execution.

The suite runner deliberately keeps experiment definition separate from policy
implementation.  A case is a fully materialized :class:`ProjectConfig`, so a
later stage can compare policies on exactly the same workload geometry.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import project_config_from_dict
from .simulator import SimulationCapacityError, SimulationResult, run_simulation


SUITE_FORMAT_VERSION = 1
TIER_ORDER = {"smoke": 0, "core": 1, "full": 2}
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class SuiteCase:
    suite_name: str
    suite_version: int
    source_path: Path
    stage: str
    case_id: str
    tier: str
    scenario: str
    description: str
    tags: tuple[str, ...]
    config: Mapping[str, Any]
    config_sha256: str
    workload_id: str = ""
    variant: str = ""


@dataclass(frozen=True)
class SuiteDefinition:
    name: str
    version: int
    description: str
    source_path: Path
    stages: Mapping[str, str]
    cases: tuple[SuiteCase, ...]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_suite_path(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested]
    if not requested.suffix:
        filename = f"{requested.name}.yaml"
        candidates.extend(
            (
                Path.cwd() / "configs" / "suites" / filename,
                Path(__file__).resolve().parent.parent
                / "configs"
                / "suites"
                / filename,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"suite {value!r} was not found; tried: {tried}")


def load_suite(value: str | Path) -> SuiteDefinition:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "loading suite YAML requires PyYAML; install with "
            "`python3 -m pip install -e .`"
        ) from error

    source_path = resolve_suite_path(value)
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("suite root must be a mapping")

    values = dict(raw)
    version = int(values.pop("version", 0))
    if version != SUITE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported suite version {version}; expected {SUITE_FORMAT_VERSION}"
        )
    name = str(values.pop("name"))
    description = str(values.pop("description", ""))
    templates_raw = values.pop("templates", {})
    stages_raw = values.pop("stages", {})
    if values:
        raise TypeError(f"unknown suite fields: {', '.join(sorted(values))}")
    if not isinstance(templates_raw, Mapping) or not isinstance(stages_raw, Mapping):
        raise TypeError("suite templates and stages must be mappings")

    templates = {str(key): value for key, value in templates_raw.items()}
    cases: list[SuiteCase] = []
    stage_descriptions: dict[str, str] = {}
    seen_ids: set[str] = set()
    for stage_name_raw, stage_raw in stages_raw.items():
        stage_name = str(stage_name_raw).lower()
        if not isinstance(stage_raw, Mapping):
            raise TypeError(f"suite stage {stage_name} must be a mapping")
        stage_values = dict(stage_raw)
        stage_descriptions[stage_name] = str(stage_values.pop("description", ""))
        case_values = stage_values.pop("cases", ())
        derived_values = stage_values.pop("derived_cases", ())
        if stage_values:
            raise TypeError(
                f"unknown fields in stage {stage_name}: "
                + ", ".join(sorted(stage_values))
            )
        if not isinstance(case_values, Sequence) or isinstance(case_values, str):
            raise TypeError(f"suite stage {stage_name}.cases must be a sequence")
        if not isinstance(derived_values, Sequence) or isinstance(
            derived_values, str
        ):
            raise TypeError(
                f"suite stage {stage_name}.derived_cases must be a sequence"
            )
        for case_raw in case_values:
            if not isinstance(case_raw, Mapping):
                raise TypeError(f"case in stage {stage_name} must be a mapping")
            case = dict(case_raw)
            case_id = str(case.pop("id"))
            if not _CASE_ID.fullmatch(case_id):
                raise ValueError(
                    f"case id {case_id!r} must match {_CASE_ID.pattern}"
                )
            if case_id in seen_ids:
                raise ValueError(f"duplicate suite case id {case_id!r}")
            seen_ids.add(case_id)
            tier = str(case.pop("tier", "full")).lower()
            if tier not in TIER_ORDER:
                raise ValueError(
                    f"case {case_id} has unknown tier {tier!r}; "
                    f"choose from {', '.join(TIER_ORDER)}"
                )
            scenario = str(case.pop("scenario"))
            workload_id = str(case.pop("workload_id", case_id))
            variant = str(case.pop("variant", "chime"))
            case_description = str(case.pop("description", ""))
            tags_raw = case.pop("tags", ())
            if not isinstance(tags_raw, Sequence) or isinstance(tags_raw, str):
                raise TypeError(f"case {case_id}.tags must be a sequence")
            template_name = str(case.pop("template"))
            if template_name not in templates:
                raise KeyError(
                    f"case {case_id} references unknown template {template_name!r}"
                )
            overrides = case.pop("overrides", {})
            if case:
                raise TypeError(
                    f"unknown fields in case {case_id}: {', '.join(sorted(case))}"
                )
            if not isinstance(templates[template_name], Mapping) or not isinstance(
                overrides, Mapping
            ):
                raise TypeError(f"case {case_id} template and overrides must be mappings")
            config = _deep_merge(templates[template_name], overrides)
            # Validate every case at load time.  A malformed full-tier case should
            # never surface only after a multi-hour run has begun.
            project_config_from_dict(config, source_path=source_path)
            digest = _canonical_hash(
                {
                    "suite": name,
                    "suite_version": version,
                    "stage": stage_name,
                    "case_id": case_id,
                    "config": config,
                }
            )
            cases.append(
                SuiteCase(
                    suite_name=name,
                    suite_version=version,
                    source_path=source_path,
                    stage=stage_name,
                    case_id=case_id,
                    tier=tier,
                    scenario=scenario,
                    description=case_description,
                    tags=tuple(str(tag) for tag in tags_raw),
                    config=config,
                    config_sha256=digest,
                    workload_id=workload_id,
                    variant=variant,
                )
            )
        for derived_raw in derived_values:
            if not isinstance(derived_raw, Mapping):
                raise TypeError(
                    f"derived case group in stage {stage_name} must be a mapping"
                )
            derived = dict(derived_raw)
            source_stage = str(derived.pop("from_stage")).lower()
            id_prefix = str(derived.pop("id_prefix"))
            derived_variant = str(derived.pop("variant"))
            scenario_values = derived.pop("scenarios", ())
            tier_values = derived.pop("tiers", ())
            workload_id_values = derived.pop("workload_ids", ())
            output_tier_raw = derived.pop("tier", None)
            workload_id_suffix = str(derived.pop("workload_id_suffix", ""))
            derived_overrides = derived.pop("overrides", {})
            extra_tags = derived.pop("tags", ())
            description_prefix = str(
                derived.pop("description_prefix", derived_variant)
            )
            if derived:
                raise TypeError(
                    f"unknown derived-case fields in stage {stage_name}: "
                    + ", ".join(sorted(derived))
                )
            for field, raw_values in (
                ("scenarios", scenario_values),
                ("tiers", tier_values),
                ("workload_ids", workload_id_values),
                ("tags", extra_tags),
            ):
                if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
                    raise TypeError(
                        f"derived case group {stage_name}.{field} must be a sequence"
                    )
            if not isinstance(derived_overrides, Mapping):
                raise TypeError(
                    f"derived case group {stage_name}.overrides must be a mapping"
                )
            wanted_scenarios = {str(value) for value in scenario_values}
            wanted_tiers = {str(value) for value in tier_values}
            wanted_workload_ids = {
                str(value) for value in workload_id_values
            }
            output_tier = (
                str(output_tier_raw).lower()
                if output_tier_raw is not None
                else None
            )
            if output_tier is not None and output_tier not in TIER_ORDER:
                raise ValueError(
                    f"derived case group in stage {stage_name} has unknown "
                    f"tier {output_tier!r}"
                )
            sources = [
                case
                for case in cases
                if case.stage == source_stage
                and (not wanted_scenarios or case.scenario in wanted_scenarios)
                and (not wanted_tiers or case.tier in wanted_tiers)
                and (
                    not wanted_workload_ids
                    or (case.workload_id or case.case_id) in wanted_workload_ids
                )
            ]
            if not sources:
                raise ValueError(
                    f"derived case group in stage {stage_name} selected no cases "
                    f"from stage {source_stage}"
                )
            for source in sources:
                derived_id = f"{id_prefix}{source.case_id}"
                if not _CASE_ID.fullmatch(derived_id):
                    raise ValueError(
                        f"derived case id {derived_id!r} must match {_CASE_ID.pattern}"
                    )
                if derived_id in seen_ids:
                    raise ValueError(f"duplicate suite case id {derived_id!r}")
                seen_ids.add(derived_id)
                config = _deep_merge(source.config, derived_overrides)
                project_config_from_dict(config, source_path=source_path)
                workload_id = (
                    f"{source.workload_id or source.case_id}{workload_id_suffix}"
                )
                digest = _canonical_hash(
                    {
                        "suite": name,
                        "suite_version": version,
                        "stage": stage_name,
                        "case_id": derived_id,
                        "workload_id": workload_id,
                        "variant": derived_variant,
                        "config": config,
                    }
                )
                cases.append(
                    SuiteCase(
                        suite_name=name,
                        suite_version=version,
                        source_path=source_path,
                        stage=stage_name,
                        case_id=derived_id,
                        tier=output_tier or source.tier,
                        scenario=source.scenario,
                        description=f"{description_prefix}: {source.description}",
                        tags=tuple(
                            dict.fromkeys(
                                (*source.tags, *(str(tag) for tag in extra_tags))
                            )
                        ),
                        config=config,
                        config_sha256=digest,
                        workload_id=workload_id,
                        variant=derived_variant,
                    )
                )
    if not cases:
        raise ValueError("suite contains no cases")
    return SuiteDefinition(
        name=name,
        version=version,
        description=description,
        source_path=source_path,
        stages=stage_descriptions,
        cases=tuple(cases),
    )


def parse_shard(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    try:
        index_raw, count_raw = raw.split("/", 1)
        index, count = int(index_raw), int(count_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("shard must use zero-based INDEX/COUNT syntax, e.g. 0/4") from error
    if count <= 0 or not 0 <= index < count:
        raise ValueError("shard requires COUNT > 0 and 0 <= INDEX < COUNT")
    return index, count


def select_cases(
    suite: SuiteDefinition,
    *,
    stage: str,
    tier: str,
    scenarios: Iterable[str] = (),
    shard: tuple[int, int] | None = None,
    max_cases: int | None = None,
) -> list[SuiteCase]:
    normalized_stage = stage.lower()
    normalized_tier = tier.lower()
    if normalized_stage not in suite.stages:
        raise KeyError(
            f"unknown stage {stage!r}; choices: {', '.join(sorted(suite.stages))}"
        )
    if normalized_tier not in TIER_ORDER:
        raise KeyError(
            f"unknown tier {tier!r}; choices: {', '.join(TIER_ORDER)}"
        )
    wanted_scenarios = {value.strip() for value in scenarios if value.strip()}
    selected = sorted(
        (
            case
            for case in suite.cases
            if case.stage == normalized_stage
            and TIER_ORDER[case.tier] <= TIER_ORDER[normalized_tier]
            and (not wanted_scenarios or case.scenario in wanted_scenarios)
        ),
        key=lambda case: case.case_id,
    )
    if shard is not None:
        index, count = shard
        selected = [case for position, case in enumerate(selected) if position % count == index]
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive")
        selected = selected[:max_cases]
    return selected


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def write_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_csv(path, sorted(rows, key=lambda row: str(row["case_id"])))


def write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(path, value)


def read_index(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def comparison_rows(
    rows: Sequence[Mapping[str, Any]], baseline_variant: str = "chime"
) -> list[dict[str, Any]]:
    """Pair every variant with the baseline for the same workload ID."""

    def number(row: Mapping[str, Any], field: str) -> float | None:
        value = row.get(field)
        return None if value in (None, "") else float(value)

    def ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        workload_id = str(row.get("workload_id", ""))
        if not workload_id:
            raise ValueError("suite index row is missing workload_id")
        grouped.setdefault(workload_id, []).append(row)

    comparisons: list[dict[str, Any]] = []
    for workload_id, group in sorted(grouped.items()):
        baselines = [row for row in group if row.get("variant") == baseline_variant]
        if len(baselines) != 1:
            raise ValueError(
                f"workload {workload_id} requires exactly one {baseline_variant!r} "
                f"baseline row, found {len(baselines)}"
            )
        baseline = baselines[0]
        for variant in sorted(group, key=lambda row: str(row.get("variant", ""))):
            variant_name = str(variant.get("variant", ""))
            if variant_name == baseline_variant:
                continue
            baseline_time = number(baseline, "simulated_time_s")
            variant_time = number(variant, "simulated_time_s")
            makespan_speedup = ratio(baseline_time, variant_time)
            throughput_speedup = ratio(
                number(variant, "throughput_tokens_s"),
                number(baseline, "throughput_tokens_s"),
            )
            mean_tpot_speedup = ratio(
                number(baseline, "mean_tpot_ms"),
                number(variant, "mean_tpot_ms"),
            )
            p95_e2e_speedup = ratio(
                number(baseline, "p95_e2e_from_decode_ready_ms"),
                number(variant, "p95_e2e_from_decode_ready_ms"),
            )
            goodput_speedup = ratio(
                number(variant, "goodput_rps"), number(baseline, "goodput_rps")
            )
            online = (
                str(variant.get("workload_type", "")) == "trace_serving"
                or str(variant.get("scenario", "")).startswith("online")
            )
            if online and goodput_speedup is not None:
                primary_metric = "goodput_rps"
                primary_speedup = goodput_speedup
            else:
                primary_metric = "makespan"
                primary_speedup = makespan_speedup
            baseline_status = str(baseline.get("status", "missing"))
            variant_status = str(variant.get("status", "missing"))
            if baseline_status != "completed":
                classification = "baseline_not_completed"
            elif variant_status != "completed":
                classification = "variant_not_completed"
            elif primary_speedup is None:
                classification = "metric_unavailable"
            elif primary_speedup >= 1.01:
                classification = "improved_ge_1pct"
            elif primary_speedup <= 0.99:
                classification = "regressed_ge_1pct"
            else:
                classification = "neutral_within_1pct"
            comparisons.append(
                {
                    "workload_id": workload_id,
                    "scenario": variant.get("scenario"),
                    "case_tier": variant.get("case_tier"),
                    "model_preset": variant.get("model_preset"),
                    "system_preset": variant.get("system_preset"),
                    "workload_type": variant.get("workload_type"),
                    "input_batch_size": variant.get("input_batch_size"),
                    "input_tokens": variant.get("input_tokens"),
                    "output_tokens_per_request": variant.get(
                        "output_tokens_per_request"
                    ),
                    "length_profile": variant.get("length_profile"),
                    "arrival_rate_rps": (
                        variant.get("trace_offered_rate_rps")
                        if str(variant.get("workload_type", ""))
                        == "trace_serving"
                        else variant.get("arrival_rate_rps")
                    ),
                    "baseline_variant": baseline_variant,
                    "variant": variant_name,
                    "baseline_case_id": baseline.get("case_id"),
                    "variant_case_id": variant.get("case_id"),
                    "baseline_status": baseline_status,
                    "variant_status": variant_status,
                    "baseline_pipeline_model": baseline.get("pipeline_model"),
                    "variant_pipeline_model": variant.get("pipeline_model"),
                    "primary_metric": primary_metric,
                    "primary_speedup": primary_speedup,
                    "classification": classification,
                    "makespan_speedup": makespan_speedup,
                    "throughput_speedup": throughput_speedup,
                    "mean_tpot_speedup": mean_tpot_speedup,
                    "p95_e2e_speedup": p95_e2e_speedup,
                    "goodput_speedup": goodput_speedup,
                    "baseline_slo_attainment": number(baseline, "slo_attainment"),
                    "variant_slo_attainment": number(variant, "slo_attainment"),
                    "slo_attainment_delta": (
                        number(variant, "slo_attainment")
                        - number(baseline, "slo_attainment")
                        if number(variant, "slo_attainment") is not None
                        and number(baseline, "slo_attainment") is not None
                        else None
                    ),
                    "baseline_pim_bottleneck_fraction": number(
                        baseline, "pim_bottleneck_fraction"
                    ),
                    "variant_pim_bottleneck_fraction": number(
                        variant, "pim_bottleneck_fraction"
                    ),
                    "baseline_gpu_path_utilization": number(
                        baseline, "gpu_path_utilization"
                    ),
                    "variant_gpu_path_utilization": number(
                        variant, "gpu_path_utilization"
                    ),
                    "baseline_peak_gpu_kv_gib": number(
                        baseline, "peak_gpu_kv_gib"
                    ),
                    "variant_peak_gpu_kv_gib": number(
                        variant, "peak_gpu_kv_gib"
                    ),
                    "gpu_kv_capacity_gib": number(variant, "gpu_kv_capacity_gib"),
                    "tail_switches": number(variant, "tail_switches"),
                    "switch_execution_time_s": number(
                        variant, "switch_execution_time_s"
                    ),
                    "switch_execution_us": number(
                        variant, "switch_execution_us"
                    ),
                    "gpu_segment_setup_us": number(
                        variant, "gpu_segment_setup_us"
                    ),
                    "pim_segment_setup_us": number(
                        variant, "pim_segment_setup_us"
                    ),
                    "policy_search_wall_time_s": number(
                        variant, "policy_search_wall_time_s"
                    ),
                }
            )
    return comparisons


def _write_result(output_dir: Path, result: SimulationResult) -> None:
    _atomic_json(output_dir / "summary.json", result.summary)
    _atomic_csv(output_dir / "requests.csv", result.requests)
    _atomic_csv(output_dir / "iterations.csv", result.iterations)
    _atomic_csv(output_dir / "switches.csv", result.switches)
    _atomic_csv(output_dir / "pipeline_events.csv", result.pipeline_events)


def _input_columns(case: SuiteCase) -> dict[str, Any]:
    config = case.config
    workload = dict(config["workload"])
    model = config["model"]
    hardware = config.get("hardware", "chime_a100")
    simulation = dict(config.get("simulation", {}))
    model_name = model if isinstance(model, str) else model.get("preset", "custom")
    system_name = (
        hardware
        if isinstance(hardware, str)
        else hardware.get("preset", "custom")
    )
    row: dict[str, Any] = {
        "model_preset": model_name,
        "system_preset": system_name,
        "workload_type": workload.get("type"),
        "max_batch_size": simulation.get("max_batch_size"),
        "input_batch_size": workload.get("batch_size"),
        "input_tokens": workload.get("input_tokens"),
        "output_tokens_per_request": workload.get("output_tokens"),
    }
    trace_file = workload.get("trace_file")
    requests = workload.get("requests") or ()
    synthetic = workload.get("synthetic") or {}
    if trace_file:
        # A template may still contain a synthetic fallback after a recursive
        # YAML merge. Execution correctly prefers trace_file; input metadata
        # must do the same rather than reporting the unused synthetic rate.
        row["trace_file"] = trace_file
    elif requests:
        inputs = [int(request["input_tokens"]) for request in requests]
        outputs = [int(request["output_tokens"]) for request in requests]
        row.update(
            {
                "input_request_count": len(requests),
                "input_tokens_min": min(inputs),
                "input_tokens_max": max(inputs),
                "output_tokens_min": min(outputs),
                "output_tokens_max": max(outputs),
                "input_total_output_tokens": sum(outputs),
            }
        )
    elif synthetic:
        row.update(
            {
                "input_request_count": synthetic.get("count"),
                "length_profile": synthetic.get("profile"),
                "arrival_rate_rps": synthetic.get("arrival_rate_rps"),
                "seed": synthetic.get("seed"),
            }
        )
    else:
        batch_size = workload.get("batch_size")
        output_tokens = workload.get("output_tokens")
        if batch_size is not None and output_tokens is not None:
            row["input_request_count"] = batch_size
            row["input_total_output_tokens"] = int(batch_size) * int(output_tokens)
    return row


def _base_index_row(case: SuiteCase) -> dict[str, Any]:
    return {
        "suite": case.suite_name,
        "suite_version": case.suite_version,
        "stage": case.stage,
        "case_id": case.case_id,
        "workload_id": case.workload_id or case.case_id,
        "variant": case.variant or "unspecified",
        "case_tier": case.tier,
        "scenario": case.scenario,
        "tags": "|".join(case.tags),
        "config_sha256": case.config_sha256,
        **_input_columns(case),
    }


def _completed_index_row(
    case: SuiteCase, result: SimulationResult, wall_time_s: float
) -> dict[str, Any]:
    summary = result.summary
    latency = summary["latency_ms"]
    slo = summary["slo"]
    return {
        **_base_index_row(case),
        "status": "completed",
        "cache_hit": False,
        "wall_time_s": wall_time_s,
        "completed_requests": summary["completed_requests"],
        "simulated_output_tokens": summary["output_tokens"],
        "simulated_time_s": summary["simulated_time_s"],
        "throughput_tokens_s": summary["throughput_tokens_s"],
        "request_completion_rate_rps": summary["request_completion_rate_rps"],
        "mean_tpot_ms": latency["mean_tpot"],
        "p95_request_tpot_ms": latency["p95_request_mean_tpot"],
        "mean_e2e_from_decode_ready_ms": latency["e2e_mean"],
        "p95_e2e_from_decode_ready_ms": latency["e2e_p95"],
        "mean_queue_ms": latency["queue_mean"],
        "p95_queue_ms": latency["queue_p95"],
        "trace_offered_rate_rps": summary["trace_offered_rate_rps"],
        "slo_attainment": slo["attainment"],
        "goodput_rps": slo["goodput_rps"],
        "average_batch_size": summary["average_batch_size"],
        "gpu_path_utilization": summary["gpu_path_utilization"],
        "pim_path_utilization": summary["pim_core_path_utilization"],
        "link_path_utilization": summary["link_path_utilization"],
        "af_ta_work_s": summary["af_ta_work_s"],
        "af_tf_work_s": summary["af_tf_work_s"],
        "af_ideal_stage_s": summary["af_ideal_stage_s"],
        "af_iteration_cost_s": summary["af_iteration_cost_s"],
        "af_actual_minus_ideal_s": summary["af_actual_minus_ideal_s"],
        "af_imbalance_s": summary["af_imbalance_s"],
        "af_balance_ratio": summary["af_balance_ratio"],
        "h2d_transfer_gib": summary["h2d_transfer_gib"],
        "d2h_transfer_gib": summary["d2h_transfer_gib"],
        "pim_kv_scan_gib": summary["pim_kv_scan_gib"],
        "gpu_kv_scan_gib": summary["gpu_kv_scan_gib"],
        "split_overhead_scale": summary["split_overhead_scale"],
        "gpu_bottleneck_fraction": summary["gpu_bottleneck_phase_fraction"],
        "pim_bottleneck_fraction": summary["pim_bottleneck_phase_fraction"],
        "peak_pim_kv_gib": summary["peak_pim_kv_gib"],
        "peak_gpu_kv_gib": summary["peak_gpu_kv_gib"],
        "tail_switches": summary["tail_switches"],
        "switch_execution_time_s": summary["switch_execution_time_s"],
        "switch_execution_us": summary["switch_execution_us"],
        "policy_switch_penalty_us": summary["policy_switch_penalty_us"],
        "gpu_segment_setup_us": summary["gpu_segment_setup_us"],
        "pim_segment_setup_us": summary["pim_segment_setup_us"],
        "event_policy_decisions": summary["event_policy_decisions"],
        "policy_search_wall_time_s": summary["policy_search_wall_time_s"],
        "pim_kv_capacity_gib": summary["pim_kv_capacity_gib"],
        "gpu_kv_capacity_gib": summary["gpu_kv_capacity_gib"],
        "pipeline_model": summary["pipeline_model"],
        "fidelity": summary["fidelity"],
        "operator_mapping": summary["operator_mapping"],
        "policy": summary["policy"],
        "error": "",
    }


def _checkpoint_row(case: SuiteCase, case_dir: Path) -> dict[str, Any] | None:
    metadata_path = case_dir / "case.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    found_hash = metadata.get("config_sha256")
    if found_hash != case.config_sha256:
        raise ValueError(
            f"checkpoint collision for {case.case_id}: expected hash "
            f"{case.config_sha256}, found {found_hash}; choose another output directory"
        )
    status = metadata.get("status")
    if status not in ("completed", "capacity_error"):
        return None
    if status == "completed" and not (case_dir / "summary.json").is_file():
        return None
    row = dict(metadata["index_row"])
    row["cache_hit"] = True
    return row


def _execute_case(case: SuiteCase, output_root: Path) -> dict[str, Any]:
    case_dir = output_root / case.case_id
    started = perf_counter()
    try:
        config = project_config_from_dict(case.config, source_path=case.source_path)
        result = run_simulation(config)
        wall_time_s = perf_counter() - started
        _write_result(case_dir, result)
        row = _completed_index_row(case, result, wall_time_s)
        status = "completed"
    except SimulationCapacityError as error:
        wall_time_s = perf_counter() - started
        row = {
            **_base_index_row(case),
            "status": "capacity_error",
            "cache_hit": False,
            "wall_time_s": wall_time_s,
            "error": str(error),
        }
        status = "capacity_error"
    except Exception as error:  # Keep an independent sweep running after one bad case.
        wall_time_s = perf_counter() - started
        row = {
            **_base_index_row(case),
            "status": "failed",
            "cache_hit": False,
            "wall_time_s": wall_time_s,
            "error": f"{type(error).__name__}: {error}",
        }
        status = "failed"
    metadata = {
        "suite": case.suite_name,
        "suite_version": case.suite_version,
        "stage": case.stage,
        "case_id": case.case_id,
        "workload_id": case.workload_id or case.case_id,
        "variant": case.variant or "unspecified",
        "scenario": case.scenario,
        "tier": case.tier,
        "description": case.description,
        "tags": list(case.tags),
        "config_sha256": case.config_sha256,
        "config": case.config,
        "status": status,
        "wall_time_s": wall_time_s,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_row": row,
    }
    _atomic_json(case_dir / "case.json", metadata)
    return row


def run_cases(
    cases: Sequence[SuiteCase],
    output_root: Path,
    *,
    workers: int = 1,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pending: list[SuiteCase] = []
    for case in cases:
        cached = _checkpoint_row(case, output_root / case.case_id) if resume else None
        if cached is not None:
            rows.append(cached)
            if progress:
                progress(f"cached {case.case_id}")
        else:
            pending.append(case)

    def run_sequential(sequence: Sequence[SuiteCase]) -> None:
        for position, case in enumerate(sequence, 1):
            if progress:
                progress(f"running {position}/{len(sequence)} {case.case_id}")
            row = _execute_case(case, output_root)
            rows.append(row)
            if progress:
                progress(f"{row['status']} {case.case_id}")

    if workers == 1:
        run_sequential(pending)
    else:
        try:
            executor_context = ProcessPoolExecutor(max_workers=workers)
        except (OSError, PermissionError) as error:
            if progress:
                progress(
                    f"parallel workers unavailable ({type(error).__name__}: {error}); "
                    "falling back to one worker"
                )
            run_sequential(pending)
        else:
            with executor_context as executor:
                futures = {
                    executor.submit(_execute_case, case, output_root): case
                    for case in pending
                }
                completed = 0
                for future in as_completed(futures):
                    case = futures[future]
                    completed += 1
                    try:
                        row = future.result()
                    except Exception as error:
                        # This branch covers process-pool failures outside the worker's
                        # normal exception boundary.
                        row = {
                            **_base_index_row(case),
                            "status": "failed",
                            "cache_hit": False,
                            "wall_time_s": "",
                            "error": f"worker failure: {type(error).__name__}: {error}",
                        }
                    rows.append(row)
                    if progress:
                        progress(
                            f"{row['status']} {completed}/{len(pending)} {case.case_id}"
                        )
    return sorted(rows, key=lambda row: str(row["case_id"]))


def validate_cases(
    cases: Sequence[SuiteCase], output_root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for case in cases:
        case_dir = output_root / case.case_id
        try:
            row = _checkpoint_row(case, case_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            issues.append(f"{case.case_id}: invalid checkpoint: {error}")
            continue
        if row is None:
            metadata_path = case_dir / "case.json"
            if metadata_path.is_file():
                try:
                    with metadata_path.open("r", encoding="utf-8") as handle:
                        status = json.load(handle).get("status", "unknown")
                except (OSError, json.JSONDecodeError):
                    status = "unreadable"
                issues.append(f"{case.case_id}: incomplete status={status}")
            else:
                issues.append(f"{case.case_id}: missing")
            continue
        rows.append(row)
    return rows, issues


def environment_manifest(
    suite: SuiteDefinition,
    cases: Sequence[SuiteCase],
    *,
    stage: str,
    tier: str,
    shard: tuple[int, int] | None,
    workers: int,
) -> dict[str, Any]:
    return {
        "suite": suite.name,
        "suite_version": suite.version,
        "suite_source": str(suite.source_path),
        "suite_source_sha256": hashlib.sha256(
            suite.source_path.read_bytes()
        ).hexdigest(),
        "stage": stage,
        "tier": tier,
        "shard": f"{shard[0]}/{shard[1]}" if shard else None,
        "workers": workers,
        "selected_case_count": len(cases),
        "selected_case_ids": [case.case_id for case in cases],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
