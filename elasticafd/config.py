"""YAML configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .cost import Fidelity
from .specs import ModelSpec, SystemSpec, model_from_config, system_from_config


@dataclass(frozen=True)
class WorkloadConfig:
    type: str
    batch_size: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    trace_file: str | None = None
    max_requests: int | None = None
    requests: tuple[Mapping[str, Any], ...] = ()
    synthetic: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "chime"
    optimizer: str = "horizon_greedy"
    decision_interval: int = 64
    planning_horizon_tokens: int = 256
    lifetime_evaluation_stride_tokens: int = 32
    beam_depth: int = 4
    beam_width: int = 4
    beam_request_candidates: int = 6
    beam_evaluation_stride_tokens: int = 16
    beam_terminal_samples: int = 8
    beam_terminal_mode: str = "horizon"
    beam_include_future_arrivals: bool = False
    exact_max_candidates: int = 8
    exact_max_cost_evaluations: int = 200000
    homogeneous_mode_grid_points: int = 5
    homogeneous_mode_quadrature_samples: int = 24
    homogeneous_mode_exact_finalists: int = 8
    prediction: str = "fixed"
    prediction_profile: str = "openr1"
    prediction_samples: int = 10000
    prediction_seed: int = 0
    min_segment_tokens: int = 128
    min_switch_interval: int = 128
    relative_improvement_threshold: float = 0.002
    max_switches_per_decision: int = 8
    switch_penalty_us: float = 0.0
    # Runtime control/metadata delay charged once for every tail-location
    # change.  This is distinct from ``switch_penalty_us``, which influences
    # the policy objective but was historically not added to simulated time.
    switch_execution_us: float = 0.0
    event_driven_replan: bool = False
    # Online anti-thrashing guard: a normal performance switch may move work
    # away only from the currently slower AF side. Capacity-relief switches
    # remain unconditional.
    require_source_bottleneck: bool = False
    # Deterministic A1 mechanism sweep.  This selects the fraction of active
    # requests whose future KV is written to GPU at decode start; prompt KV
    # remains on PIM.  It is used only by optimizer=forced_fraction.
    forced_gpu_fraction: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    max_batch_size: int = 64
    fidelity: Fidelity = "tile"
    pipeline_model: str = "analytic"
    # Optional lower-cost model used only to choose policy actions. Execution
    # and all reported performance still use ``pipeline_model``.
    policy_pipeline_model: str | None = None
    subbatch_scheduler: str = "lpt"
    # strict_ordered is the older operator-family phase model;
    # device_contiguous is the GPU/PIM/GPU two-cut model used by the new
    # focused ablations; free_wavefront is the unconstrained event DAG.
    subbatch_progression: str = "strict_ordered"
    operator_mapping: str = "chime"
    record_every: int = 100
    record_pipeline_events: bool = False
    max_iterations: int | None = None
    admission_reservation: str = "none"
    admission_scheduler: str = "fcfs"
    admission_lookahead: int = 32
    admission_starvation_ms: float = 1000.0
    allow_gpu_prompt_fallback: bool = False
    slo_queue_ms: float | None = None
    slo_tpot_ms: float | None = None
    slo_e2e_ms: float | None = None
    slo_attainment_target: float = 0.99
    # Scale only the split-specific partial statistics and GPU merge cost.
    # Zero is an explicitly labelled ideal upper-bound ablation, not hardware.
    split_overhead_scale: float = 1.0
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelSpec
    system: SystemSpec
    workload: WorkloadConfig
    simulation: SimulationConfig
    source_path: Path | None = None


def _policy_from_raw(raw: str | Mapping[str, Any] | None) -> PolicyConfig:
    if raw is None:
        return PolicyConfig()
    if isinstance(raw, str):
        return PolicyConfig(name=raw)
    return PolicyConfig(**dict(raw))


def _workload_from_raw(raw: Mapping[str, Any]) -> WorkloadConfig:
    values = dict(raw)
    if "type" not in values:
        raise ValueError("workload.type is required")
    requests = tuple(values.pop("requests", ()))
    synthetic = dict(values.pop("synthetic", {}))
    config = WorkloadConfig(requests=requests, synthetic=synthetic, **values)
    if config.type == "static_batch":
        missing = [
            name
            for name in ("batch_size", "input_tokens", "output_tokens")
            if getattr(config, name) is None
        ]
        if missing:
            raise ValueError(
                "static_batch workload is missing: " + ", ".join(missing)
            )
    elif config.type == "closed_batch":
        if not config.requests and not config.trace_file:
            raise ValueError("closed_batch requires inline requests or trace_file")
    elif config.type != "trace_serving":
        raise ValueError(
            "workload.type must be static_batch, closed_batch, or trace_serving"
        )
    return config


def project_config_from_dict(
    raw: Mapping[str, Any], source_path: Path | None = None
) -> ProjectConfig:
    values = dict(raw)
    model = model_from_config(values.pop("model"))
    system = system_from_config(values.pop("hardware", "chime_a100"))
    workload = _workload_from_raw(values.pop("workload"))
    simulation_raw = dict(values.pop("simulation", {}))
    policy = _policy_from_raw(simulation_raw.pop("policy", None))
    simulation = SimulationConfig(policy=policy, **simulation_raw)
    if values:
        raise TypeError(f"unknown top-level fields: {', '.join(sorted(values))}")
    if simulation.fidelity not in ("roofline", "tile"):
        raise ValueError("simulation.fidelity must be roofline or tile")
    if simulation.pipeline_model not in ("analytic", "rankset_event"):
        raise ValueError(
            "simulation.pipeline_model must be analytic or rankset_event"
        )
    if simulation.policy_pipeline_model not in (
        None,
        "analytic",
        "rankset_event",
    ):
        raise ValueError(
            "simulation.policy_pipeline_model must be analytic, rankset_event, or null"
        )
    if simulation.subbatch_scheduler not in ("lpt", "alignment", "fixed"):
        raise ValueError("simulation.subbatch_scheduler must be lpt, alignment, or fixed")
    if simulation.subbatch_progression not in (
        "strict_ordered", "free_wavefront", "device_contiguous"
    ):
        raise ValueError(
            "simulation.subbatch_progression must be strict_ordered, "
            "free_wavefront, or device_contiguous"
        )
    from .mapping import get_operator_mapping

    mapping = get_operator_mapping(simulation.operator_mapping)
    if simulation.subbatch_scheduler == "alignment" and mapping.name != "chime":
        raise ValueError(
            "alignment subbatch scheduling is currently defined for operator_mapping=chime"
        )
    if mapping.name != "chime" and policy.name.lower().replace("-", "_") in (
        "elastic",
        "elasticafd",
        "elastic_afd",
    ):
        raise ValueError(
            "elastic KV-segment placement currently requires operator_mapping=chime"
        )
    if simulation.max_batch_size <= 0:
        raise ValueError("simulation.max_batch_size must be positive")
    if simulation.record_every <= 0:
        raise ValueError("simulation.record_every must be positive")
    if policy.decision_interval <= 0 or policy.planning_horizon_tokens <= 0:
        raise ValueError("policy decision interval and planning horizon must be positive")
    if policy.lifetime_evaluation_stride_tokens <= 0:
        raise ValueError("policy lifetime evaluation stride must be positive")
    if (
        policy.beam_depth <= 0
        or policy.beam_width <= 0
        or policy.beam_request_candidates <= 0
        or policy.beam_evaluation_stride_tokens <= 0
        or policy.beam_terminal_samples <= 0
    ):
        raise ValueError("policy beam-search parameters must be positive")
    if policy.beam_terminal_mode not in (
        "horizon",
        "pim_lifetime",
        "tail_lifetime",
    ):
        raise ValueError(
            "policy.beam_terminal_mode must be horizon, pim_lifetime, or "
            "tail_lifetime"
        )
    if (
        policy.exact_max_candidates <= 0
        or policy.exact_max_cost_evaluations <= 0
    ):
        raise ValueError("policy exact search limits must be positive")
    if (
        policy.homogeneous_mode_grid_points < 3
        or policy.homogeneous_mode_quadrature_samples <= 0
        or policy.homogeneous_mode_exact_finalists <= 0
    ):
        raise ValueError("homogeneous mode-search parameters must be positive")
    if policy.min_segment_tokens < 0 or policy.min_switch_interval < 0:
        raise ValueError("policy segment and switch intervals cannot be negative")
    if policy.max_switches_per_decision < 0:
        raise ValueError("policy.max_switches_per_decision cannot be negative")
    if policy.switch_penalty_us < 0 or policy.switch_execution_us < 0:
        raise ValueError("policy switch penalty and execution cost cannot be negative")
    if not 0.0 <= policy.forced_gpu_fraction <= 1.0:
        raise ValueError("policy.forced_gpu_fraction must lie in [0, 1]")
    if simulation.split_overhead_scale < 0.0:
        raise ValueError("simulation.split_overhead_scale cannot be negative")
    if policy.prediction not in ("fixed", "profile", "oracle"):
        raise ValueError("policy.prediction must be fixed, profile, or oracle")
    if policy.optimizer not in (
        "horizon_greedy",
        "lifetime_greedy",
        "rolling_beam",
        "global_exact",
        "homogeneous_modes",
        "balance_curve",
        "forced_fraction",
        "online_static_fraction",
    ):
        raise ValueError(
            "policy.optimizer must be horizon_greedy, lifetime_greedy, "
            "rolling_beam, global_exact, homogeneous_modes, balance_curve, "
            "forced_fraction, or online_static_fraction"
        )
    if policy.optimizer in (
        "lifetime_greedy",
        "rolling_beam",
        "global_exact",
        "homogeneous_modes",
        "balance_curve",
    ) and (
        policy.prediction != "oracle"
    ):
        raise ValueError(
            f"policy.optimizer={policy.optimizer} requires prediction=oracle"
        )
    if policy.optimizer in ("global_exact", "homogeneous_modes") and workload.type not in (
        "static_batch",
        "closed_batch",
    ):
        raise ValueError(
            f"{policy.optimizer} is defined only for static_batch or closed_batch workloads"
        )
    if policy.optimizer == "forced_fraction" and workload.type not in (
        "static_batch",
        "closed_batch",
    ):
        raise ValueError(
            "forced_fraction is a decode-start closed-workload mechanism probe"
        )
    if (
        policy.optimizer == "online_static_fraction"
        and workload.type != "trace_serving"
    ):
        raise ValueError(
            "online_static_fraction is defined only for trace_serving workloads"
        )
    if policy.optimizer == "balance_curve" and workload.type != "static_batch":
        raise ValueError(
            "balance_curve is currently defined only for static_batch workloads"
        )
    if policy.beam_include_future_arrivals:
        if policy.optimizer != "rolling_beam" or policy.prediction != "oracle":
            raise ValueError(
                "beam_include_future_arrivals requires oracle rolling_beam"
            )
        if workload.type != "trace_serving":
            raise ValueError(
                "beam_include_future_arrivals requires trace_serving workload"
            )
        if (
            simulation.admission_scheduler != "fcfs"
            or simulation.admission_reservation != "none"
            or simulation.allow_gpu_prompt_fallback
        ):
            raise ValueError(
                "future-arrival beam currently requires FCFS admission, no "
                "reservation, and no GPU prompt fallback"
            )
    if policy.prediction_samples <= 0:
        raise ValueError("policy.prediction_samples must be positive")
    if simulation.admission_reservation not in ("none", "pim_target"):
        raise ValueError("simulation.admission_reservation must be none or pim_target")
    if simulation.admission_scheduler not in ("fcfs", "alignment"):
        raise ValueError("simulation.admission_scheduler must be fcfs or alignment")
    if simulation.admission_lookahead <= 0:
        raise ValueError("simulation.admission_lookahead must be positive")
    if simulation.admission_starvation_ms < 0:
        raise ValueError("simulation.admission_starvation_ms cannot be negative")
    for name in ("slo_queue_ms", "slo_tpot_ms", "slo_e2e_ms"):
        value = getattr(simulation, name)
        if value is not None and value <= 0:
            raise ValueError(f"simulation.{name} must be positive when set")
    if not 0 < simulation.slo_attainment_target <= 1:
        raise ValueError("simulation.slo_attainment_target must be in (0, 1]")
    if simulation.admission_reservation == "pim_target" and policy.name.lower() not in (
        "chime",
        "baseline",
    ):
        raise ValueError("pim_target reservation is only defined for the CHIME baseline")
    if (
        simulation.admission_reservation == "pim_target"
        and mapping.attention.value == "gpu"
    ):
        raise ValueError("pim_target reservation requires PIM-resident attention KV")
    return ProjectConfig(
        model=model,
        system=system,
        workload=workload,
        simulation=simulation,
        source_path=source_path,
    )


def load_config(path: str | Path) -> ProjectConfig:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "loading YAML configs requires PyYAML; install the project with "
            "`python -m pip install -e .`"
        ) from error
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("configuration root must be a mapping")
    return project_config_from_dict(raw, source_path=config_path)
