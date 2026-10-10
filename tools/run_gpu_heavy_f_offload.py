#!/usr/bin/env python3
"""Legacy factorial F-placement experiment for GPU-heavy decode states.

This entry point retains the earlier independent-operator placements,
including FFN-only PIM paths. For the GPU/PIM/GPU two-cut pipeline use
``tools/run_contiguous_ablation.py`` instead.

Attention and every KV segment stay on PIM in this experiment.  QKV, O_proj
and FFN are independently placed on GPU or PIM, so the measured difference is
caused by F-side placement rather than by future-KV/Attention splitting.

PIM dense kernels are conditional roofline estimates.  The default reproduces
the repository's one-FP16-MAC-per-weight-read bound; independent bandwidth and
compute scales are exposed for sensitivity analysis without changing PIM
Attention performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import ceil
from pathlib import Path
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elasticafd.config import project_config_from_dict
from elasticafd.cost import ChimeCostModel, IterationCost, MatrixLayerCost
from elasticafd.mapping import OPERATOR_MAPPINGS
from elasticafd.specs import GiB
from elasticafd.state import RequestState, views_for_requests


CHIME = "chime"
CONTROL = "attn_pim_control"
F_MAPPINGS = (
    CONTROL,
    "f_qkv_pim",
    "f_oproj_pim",
    "f_ffn_pim",
    "f_qkv_oproj_pim",
    "f_qkv_ffn_pim",
    "f_oproj_ffn_pim",
    "f_all_pim",
)
ALL_F_OPERATORS = ("qkv", "o_proj", "ffn")


class FactorialFCostModel(ChimeCostModel):
    """Operator-event model with an explicit persistent PIM weight set.

    The normal static mapping model removes PIM-resident weights from GPU HBM.
    Dynamic placement cannot do that because every operator may later return to
    GPU.  This experiment therefore keeps the GPU copy and separately reserves
    every opened PIM copy.  Runtime switching never teleports weights.
    """

    def __init__(
        self,
        config,
        mapping: str,
        *,
        resident_operators: Sequence[str],
        pim_dense_bandwidth_scale: float = 1.0,
        pim_dense_compute_scale: float = 1.0,
    ) -> None:
        if pim_dense_bandwidth_scale <= 0 or pim_dense_compute_scale <= 0:
            raise ValueError("PIM dense roofline scales must be positive")
        self.resident_operators = tuple(dict.fromkeys(resident_operators))
        unknown = set(self.resident_operators) - set(ALL_F_OPERATORS)
        if unknown:
            raise ValueError(f"unknown resident F operators: {sorted(unknown)}")
        self.pim_dense_bandwidth_scale = pim_dense_bandwidth_scale
        self.pim_dense_compute_scale = pim_dense_compute_scale
        super().__init__(
            config.model,
            config.system,
            "tile",
            pipeline_model="rankset_event",
            subbatch_scheduler="lpt",
            operator_mapping=mapping,
        )

    @property
    def pim_persistent_weight_bytes(self) -> int:
        per_copy = sum(
            self._operator_weight_elements(operator) * self.model.dtype_bytes
            for operator in self.resident_operators
        ) * self.model.layers
        return per_copy * self.model.data_parallel

    @property
    def gpu_offloaded_weight_bytes(self) -> int:
        # All-open placement requires a GPU copy even while an operator happens
        # to execute on PIM in the current interval.
        return 0

    def pim_matrix_layer_cost(self, operator: str, batch_size: int) -> MatrixLayerCost:
        base = super().pim_matrix_layer_cost(operator, batch_size)
        memory_us = base.memory_us / self.pim_dense_bandwidth_scale
        compute_us = base.compute_us / self.pim_dense_compute_scale
        return MatrixLayerCost(
            total_us=max(memory_us, compute_us),
            memory_us=memory_us,
            compute_us=compute_us,
            weight_bytes=base.weight_bytes,
        )

    @property
    def initial_weight_copy_us(self) -> float:
        if not self.pim_persistent_weight_bytes:
            return 0.0
        return (
            self.pim_persistent_weight_bytes
            / self.system.link.host_to_pim_bytes_s
            * 1e6
            + self.model.layers * self.system.link.transfer_latency_us
        )


def base_config(batch_size: int):
    return project_config_from_dict(
        {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "static_batch",
                "batch_size": batch_size,
                "input_tokens": 1,
                "output_tokens": 1,
            },
            "simulation": {
                "max_batch_size": batch_size,
                "fidelity": "tile",
                "pipeline_model": "rankset_event",
                "subbatch_scheduler": "lpt",
                "operator_mapping": CONTROL,
                "policy": "chime",
            },
        }
    )


def make_views(batch_size: int, context_tokens: int):
    return views_for_requests(
        [
            RequestState(f"r{index}", 0.0, context_tokens, 1)
            for index in range(batch_size)
        ]
    )


def event_sum(cost: IterationCost, kinds: Iterable[str]) -> float:
    selected = set(kinds)
    return sum(event.duration_us for event in cost.events if event.kind in selected)


def runtime_metrics(cost: IterationCost) -> dict[str, float]:
    ta_us = max(cost.pim_core_busy_us, cost.link_busy_us)
    tf_us = cost.gpu_busy_us
    return {
        "iteration_us": cost.total_us,
        "gpu_busy_us": tf_us,
        "pim_busy_us": cost.pim_core_busy_us,
        "link_busy_us": cost.link_busy_us,
        "ta_resource_us": ta_us,
        "tf_resource_us": tf_us,
        "gpu_to_a_pressure_ratio": tf_us / max(ta_us, 1e-12),
        "event_bubble_us": cost.total_us
        - max(tf_us, cost.pim_core_busy_us, cost.link_busy_us),
        "gpu_qkv_us": event_sum(cost, ("gpu_qkv",)),
        "gpu_oproj_us": event_sum(cost, ("gpu_o_proj",)),
        "gpu_ffn_us": event_sum(cost, ("gpu_ffn",)),
        "pim_qkv_us": event_sum(cost, ("pim_qkv",)),
        "pim_oproj_us": event_sum(cost, ("pim_o_proj",)),
        "pim_ffn_us": event_sum(cost, ("pim_ffn",)),
        "pim_attention_us": event_sum(cost, ("pim_attention",)),
        "h2d_us": event_sum(cost, ("h2d",)),
        "d2h_us": event_sum(cost, ("d2h",)),
        "pim_internal_us": event_sum(cost, ("pim_internal",)),
    }


def cost_model(
    config,
    mapping: str,
    *,
    all_open: bool,
    bandwidth_scale: float,
    compute_scale: float,
) -> FactorialFCostModel:
    return FactorialFCostModel(
        config,
        mapping,
        resident_operators=ALL_F_OPERATORS if all_open else (),
        pim_dense_bandwidth_scale=bandwidth_scale,
        pim_dense_compute_scale=compute_scale,
    )


def max_context_that_fits(model: FactorialFCostModel, batch_size: int) -> int:
    return int(model.pim_kv_capacity_bytes // (batch_size * model.model.kv_bytes_per_token))


def baseline_at(model: FactorialFCostModel, batch_size: int, context_tokens: int):
    return model.iteration_cost(make_views(batch_size, context_tokens))


def context_for_ratio(
    model: FactorialFCostModel,
    batch_size: int,
    target: float,
    max_context: int,
) -> int:
    """Find the context whose baseline TF/TA ratio is nearest ``target``."""

    cache: dict[int, float] = {}

    def ratio(context: int) -> float:
        if context not in cache:
            cache[context] = runtime_metrics(
                baseline_at(model, batch_size, context)
            )["gpu_to_a_pressure_ratio"]
        return cache[context]

    low, high = 1, max_context
    while low <= high:
        middle = (low + high) // 2
        if ratio(middle) > target:
            low = middle + 1
        else:
            high = middle - 1
    candidates = {max(1, min(max_context, value)) for value in (low, high)}
    return min(candidates, key=lambda context: (abs(ratio(context) - target), context))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write an empty experiment table")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def code_hash() -> str:
    paths = [Path(__file__), ROOT / "elasticafd" / "cost.py", ROOT / "elasticafd" / "mapping.py"]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def write_manifest(path: Path, args, rows: Sequence[dict[str, object]]) -> None:
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "func"
    }
    manifest = {
        "code_sha256": code_hash(),
        "arguments": arguments,
        "rows": len(rows),
        "semantics": {
            "arrival": "prefill-complete, decode-ready",
            "kv": "all prompt and generated KV stays on PIM; no historical migration",
            "attention": "PIM-only control so F-placement causality is isolated",
            "open_operators": list(ALL_F_OPERATORS),
            "placement_granularity": "whole logical operator, fixed for one sampled interval",
            "pim_dense": "conditional roofline; independent bandwidth/compute sensitivity",
            "weights": "all opened weights persist on PIM and remain duplicated on GPU",
            "execution": "dependency/resource operator-event DAG",
        },
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest_path)


def screen(args) -> None:
    rows: list[dict[str, object]] = []
    explicit_contexts = tuple(args.context_lengths or ())
    for bandwidth_scale in args.pim_dense_bandwidth_scales:
        for compute_scale in args.pim_dense_compute_scales:
            for batch_size in args.batch_sizes:
                print(
                    "screen "
                    f"B={batch_size} dense_bw={bandwidth_scale:g} "
                    f"dense_compute={compute_scale:g}",
                    flush=True,
                )
                config = base_config(batch_size)
                control = cost_model(
                    config,
                    CONTROL,
                    all_open=False,
                    bandwidth_scale=bandwidth_scale,
                    compute_scale=compute_scale,
                )
                chime = cost_model(
                    config,
                    CHIME,
                    all_open=False,
                    bandwidth_scale=bandwidth_scale,
                    compute_scale=compute_scale,
                )
                resident_control = cost_model(
                    config,
                    CONTROL,
                    all_open=True,
                    bandwidth_scale=bandwidth_scale,
                    compute_scale=compute_scale,
                )
                capacity_limit = min(
                    args.max_context,
                    max_context_that_fits(resident_control, batch_size),
                )
                if capacity_limit < 1:
                    continue
                contexts = set(explicit_contexts)
                for target in args.target_gpu_ratios:
                    contexts.add(
                        context_for_ratio(chime, batch_size, target, capacity_limit)
                    )
                contexts = {value for value in contexts if 1 <= value <= capacity_limit}
                for context_tokens in sorted(contexts):
                    views = make_views(batch_size, context_tokens)
                    baseline_cost = control.iteration_cost(views)
                    baseline = runtime_metrics(baseline_cost)
                    chime_metrics = runtime_metrics(chime.iteration_cost(views))
                    for mapping in F_MAPPINGS:
                        model = cost_model(
                            config,
                            mapping,
                            all_open=True,
                            bandwidth_scale=bandwidth_scale,
                            compute_scale=compute_scale,
                        )
                        measured_cost = model.iteration_cost(views)
                        measured = runtime_metrics(measured_cost)
                        saving_us = baseline["iteration_us"] - measured["iteration_us"]
                        copy_us = model.initial_weight_copy_us
                        row: dict[str, object] = {
                            "model": config.model.name,
                            "batch_size": batch_size,
                            "context_tokens": context_tokens,
                            "mapping": mapping,
                            "mapping_label": OPERATOR_MAPPINGS[mapping].label,
                            "qkv_target": OPERATOR_MAPPINGS[mapping].qkv.value,
                            "attention_target": "pim",
                            "o_proj_target": OPERATOR_MAPPINGS[mapping].o_proj.value,
                            "ffn_target": OPERATOR_MAPPINGS[mapping].ffn.value,
                            "pim_dense_bandwidth_scale": bandwidth_scale,
                            "pim_dense_compute_scale": compute_scale,
                            "baseline_iteration_us": baseline["iteration_us"],
                            "runtime_speedup_vs_control": (
                                baseline["iteration_us"] / measured["iteration_us"]
                            ),
                            "runtime_speedup_vs_chime": (
                                chime_metrics["iteration_us"] / measured["iteration_us"]
                            ),
                            "control_speedup_vs_chime": (
                                chime_metrics["iteration_us"] / baseline["iteration_us"]
                            ),
                            "control_chime_relative_gap": abs(
                                baseline["iteration_us"] - chime_metrics["iteration_us"]
                            )
                            / chime_metrics["iteration_us"],
                            "control_chime_within_2pct": abs(
                                baseline["iteration_us"] - chime_metrics["iteration_us"]
                            )
                            <= 0.02 * chime_metrics["iteration_us"],
                            "runtime_saving_us_per_token": saving_us,
                            "chime_gpu_to_a_pressure_ratio": chime_metrics[
                                "gpu_to_a_pressure_ratio"
                            ],
                            "chime_gpu_heavy": chime_metrics[
                                "gpu_to_a_pressure_ratio"
                            ]
                            >= args.gpu_heavy_threshold,
                            "all_open_pim_weight_gib": (
                                model.pim_persistent_weight_bytes / GiB
                            ),
                            "all_open_pim_kv_capacity_gib": (
                                model.pim_kv_capacity_bytes / GiB
                            ),
                            "initial_weight_copy_s": copy_us / 1e6,
                            "break_even_decode_tokens": (
                                ceil(copy_us / saving_us) if saving_us > 0 else None
                            ),
                            "classification": (
                                "not_gpu_heavy"
                                if chime_metrics["gpu_to_a_pressure_ratio"]
                                < args.gpu_heavy_threshold
                                else (
                                    "f_offload_profitable_warm"
                                    if saving_us > 0
                                    else "f_offload_not_profitable"
                                )
                            ),
                        }
                        row.update({f"chime_{key}": value for key, value in chime_metrics.items()})
                        row.update({f"baseline_{key}": value for key, value in baseline.items()})
                        row.update(measured)
                        rows.append(row)
    write_csv(args.output, rows)
    write_manifest(args.output, args, rows)


def read_screen_anchors(path: Path, top_k: int, threshold: float):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates = [
        row
        for row in rows
        if row["mapping"] != CONTROL
        and row["chime_gpu_heavy"].lower() == "true"
        and float(row["runtime_speedup_vs_control"]) > threshold
    ]
    candidates.sort(key=lambda row: float(row["runtime_speedup_vs_control"]), reverse=True)
    anchors = []
    seen = set()
    for row in candidates:
        key = (
            int(row["batch_size"]),
            int(row["context_tokens"]),
            float(row["pim_dense_bandwidth_scale"]),
            float(row["pim_dense_compute_scale"]),
        )
        if key in seen:
            continue
        seen.add(key)
        anchors.append(key)
        if len(anchors) >= top_k:
            break
    if not anchors:
        raise ValueError("screen contains no profitable GPU-heavy anchor")
    return anchors


def sample_blocks(output_tokens: int, stride: int):
    if output_tokens <= 0 or stride <= 0:
        raise ValueError("output tokens and evaluation stride must be positive")
    for start in range(0, output_tokens, stride):
        end = min(output_tokens, start + stride)
        yield start, end, (start + end - 1) // 2


def compress_timeline(sequence: Sequence[tuple[int, int, str]]) -> str:
    compact: list[list[object]] = []
    for start, end, mapping in sequence:
        if compact and compact[-1][2] == mapping and compact[-1][1] == start:
            compact[-1][1] = end
        else:
            compact.append([start, end, mapping])
    return ";".join(f"{start}-{end}:{mapping}" for start, end, mapping in compact)


def aggregate_strategy(
    name: str,
    chosen: Sequence[tuple[int, int, str, dict[str, float]]],
    *,
    switch_execution_us: float,
    copy_us: float,
    baseline_total_us: float,
    chime_total_us: float,
) -> dict[str, object]:
    runtime_us = sum((end - start) * metrics["iteration_us"] for start, end, _, metrics in chosen)
    switches = sum(
        previous[2] != current[2]
        for previous, current in zip(chosen, chosen[1:])
    )
    if chosen and chosen[0][2] != CONTROL:
        switches += 1
    runtime_us += switches * switch_execution_us
    total_tokens = sum(end - start for start, end, _, _ in chosen)
    weighted = {}
    for field in ("gpu_busy_us", "pim_busy_us", "link_busy_us", "event_bubble_us"):
        weighted[field] = sum(
            (end - start) * metrics[field] for start, end, _, metrics in chosen
        ) / max(total_tokens, 1)
    return {
        "strategy": name,
        "status": "completed",
        "runtime_us": runtime_us,
        "cold_runtime_us": runtime_us + copy_us,
        "speedup_vs_control_warm": baseline_total_us / runtime_us,
        "speedup_vs_control_cold": baseline_total_us / (runtime_us + copy_us),
        "speedup_vs_chime_warm": chime_total_us / runtime_us,
        "speedup_vs_chime_cold": chime_total_us / (runtime_us + copy_us),
        "operator_switches": switches,
        "mapping_timeline": compress_timeline(
            [(start, end, mapping) for start, end, mapping, _ in chosen]
        ),
        **{f"mean_{key}": value for key, value in weighted.items()},
    }


def lifecycle(args) -> None:
    anchors = read_screen_anchors(
        args.screen, args.top_k, args.minimum_screen_speedup
    )
    rows: list[dict[str, object]] = []
    for batch_size, input_tokens, bandwidth_scale, compute_scale in anchors:
        print(
            "lifecycle "
            f"B={batch_size} I={input_tokens} dense_bw={bandwidth_scale:g} "
            f"dense_compute={compute_scale:g}",
            flush=True,
        )
        config = base_config(batch_size)
        baseline_model = cost_model(
            config,
            CONTROL,
            all_open=False,
            bandwidth_scale=bandwidth_scale,
            compute_scale=compute_scale,
        )
        chime_model = cost_model(
            config,
            CHIME,
            all_open=False,
            bandwidth_scale=bandwidth_scale,
            compute_scale=compute_scale,
        )
        models = {
            mapping: cost_model(
                config,
                mapping,
                all_open=True,
                bandwidth_scale=bandwidth_scale,
                compute_scale=compute_scale,
            )
            for mapping in F_MAPPINGS
        }
        copy_us = models[CONTROL].initial_weight_copy_us
        for output_tokens in args.output_lengths:
            common = {
                "model": config.model.name,
                "batch_size": batch_size,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "evaluation_stride_tokens": args.evaluation_stride,
                "pim_dense_bandwidth_scale": bandwidth_scale,
                "pim_dense_compute_scale": compute_scale,
                "all_open_pim_weight_gib": (
                    models[CONTROL].pim_persistent_weight_bytes / GiB
                ),
                "initial_weight_copy_s": copy_us / 1e6,
                "switch_execution_us": args.switch_execution_us,
                "relative_improvement_threshold": args.relative_improvement_threshold,
                "min_residence_tokens": args.min_residence_tokens,
            }
            final_tokens = input_tokens + output_tokens
            baseline_required = (
                batch_size * final_tokens * config.model.kv_bytes_per_token
            )
            if baseline_required > baseline_model.pim_kv_capacity_bytes:
                rows.append({**common, "strategy": "control", "status": "baseline_capacity_error"})
                continue
            candidate_required = baseline_required
            if candidate_required > models[CONTROL].pim_kv_capacity_bytes:
                rows.append({**common, "strategy": "all_open", "status": "candidate_capacity_error"})
                continue

            blocks = list(sample_blocks(output_tokens, args.evaluation_stride))
            baseline_metrics = []
            chime_metrics = []
            table: dict[str, list[dict[str, float]]] = {mapping: [] for mapping in F_MAPPINGS}
            for start, end, sample in blocks:
                context = input_tokens + sample
                views = make_views(batch_size, context)
                baseline_metrics.append(runtime_metrics(baseline_model.iteration_cost(views)))
                chime_metrics.append(runtime_metrics(chime_model.iteration_cost(views)))
                for mapping, model in models.items():
                    table[mapping].append(runtime_metrics(model.iteration_cost(views)))

            baseline_total_us = sum(
                (end - start) * metrics["iteration_us"]
                for (start, end, _), metrics in zip(blocks, baseline_metrics)
            )
            chime_total_us = sum(
                (end - start) * metrics["iteration_us"]
                for (start, end, _), metrics in zip(blocks, chime_metrics)
            )
            baseline_choice = [
                (start, end, CONTROL, metrics)
                for (start, end, _), metrics in zip(blocks, baseline_metrics)
            ]
            baseline_row = aggregate_strategy(
                "control",
                baseline_choice,
                switch_execution_us=0.0,
                copy_us=0.0,
                baseline_total_us=baseline_total_us,
                chime_total_us=chime_total_us,
            )
            rows.append({**common, **baseline_row})

            for mapping in F_MAPPINGS[1:]:
                chosen = [
                    (start, end, mapping, metrics)
                    for (start, end, _), metrics in zip(blocks, table[mapping])
                ]
                row = aggregate_strategy(
                    f"static:{mapping}",
                    chosen,
                    switch_execution_us=args.switch_execution_us,
                    copy_us=copy_us,
                    baseline_total_us=baseline_total_us,
                    chime_total_us=chime_total_us,
                )
                rows.append({**common, **row})

            oracle_choice = []
            for index, (start, end, _) in enumerate(blocks):
                mapping = min(F_MAPPINGS, key=lambda item: table[item][index]["iteration_us"])
                oracle_choice.append((start, end, mapping, table[mapping][index]))
            oracle_row = aggregate_strategy(
                "dynamic_oracle_no_switch_cost",
                oracle_choice,
                switch_execution_us=0.0,
                copy_us=copy_us,
                baseline_total_us=baseline_total_us,
                chime_total_us=chime_total_us,
            )
            rows.append({**common, **oracle_row})

            current = CONTROL
            last_switch_token = -10**18
            hysteresis_choice = []
            for index, (start, end, _) in enumerate(blocks):
                best = min(F_MAPPINGS, key=lambda item: table[item][index]["iteration_us"])
                current_us = table[current][index]["iteration_us"]
                best_us = table[best][index]["iteration_us"]
                eligible = start - last_switch_token >= args.min_residence_tokens
                if (
                    best != current
                    and eligible
                    and best_us < current_us * (1 - args.relative_improvement_threshold)
                ):
                    current = best
                    last_switch_token = start
                hysteresis_choice.append((start, end, current, table[current][index]))
            hysteresis_row = aggregate_strategy(
                "dynamic_hysteresis",
                hysteresis_choice,
                switch_execution_us=args.switch_execution_us,
                copy_us=copy_us,
                baseline_total_us=baseline_total_us,
                chime_total_us=chime_total_us,
            )
            rows.append({**common, **hysteresis_row})

    write_csv(args.output, rows)
    write_manifest(args.output, args, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    screen_parser = subparsers.add_parser(
        "screen", help="predict GPU-heavy contexts and enumerate all F placements"
    )
    screen_parser.add_argument("--output", type=Path, required=True)
    screen_parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    screen_parser.add_argument(
        "--target-gpu-ratios", type=float, nargs="+", default=[1.25, 1.5, 2.0, 4.0]
    )
    screen_parser.add_argument("--context-lengths", type=int, nargs="+")
    screen_parser.add_argument("--max-context", type=int, default=32768)
    screen_parser.add_argument("--gpu-heavy-threshold", type=float, default=1.25)
    screen_parser.add_argument(
        "--pim-dense-bandwidth-scales", type=float, nargs="+", default=[1.0]
    )
    screen_parser.add_argument(
        "--pim-dense-compute-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    screen_parser.set_defaults(func=screen)

    lifecycle_parser = subparsers.add_parser(
        "lifecycle", help="evaluate static, oracle and hysteretic F placement"
    )
    lifecycle_parser.add_argument("--screen", type=Path, required=True)
    lifecycle_parser.add_argument("--output", type=Path, required=True)
    lifecycle_parser.add_argument("--top-k", type=int, default=12)
    lifecycle_parser.add_argument(
        "--output-lengths", type=int, nargs="+", default=[256, 1024, 4096]
    )
    lifecycle_parser.add_argument("--evaluation-stride", type=int, default=64)
    lifecycle_parser.add_argument("--minimum-screen-speedup", type=float, default=1.001)
    lifecycle_parser.add_argument("--relative-improvement-threshold", type=float, default=0.005)
    lifecycle_parser.add_argument("--min-residence-tokens", type=int, default=256)
    lifecycle_parser.add_argument("--switch-execution-us", type=float, default=100.0)
    lifecycle_parser.set_defaults(func=lifecycle)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
