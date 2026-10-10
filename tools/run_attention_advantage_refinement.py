#!/usr/bin/env python3
"""Search and validate the B/I/O region where future-KV splitting helps.

The broad and refined phases use the existing analytic homogeneous lifecycle
planner.  The event phase re-plans with the same analytic policy model but
executes the selected schedule on the dependency-aware rankset event model.
Each B/I/O point is checkpointed independently so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from itertools import product
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from elasticafd.experiments import homogeneous_lifecycle_rows


DEFAULT_BATCHES = (8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
DEFAULT_INPUTS = (512, 1024, 2048, 3072, 4096, 5200, 8192, 12288, 16384)
DEFAULT_OUTPUTS = (1024, 2048, 4096, 6144, 8192)
SMOKE_BATCHES = (16, 32)
SMOKE_INPUTS = (2048, 5200)
SMOKE_OUTPUTS = (2048, 4096)

# The first anchor is a known no-benefit case.  The second captures the
# delayed-switch short-prompt positive case that the focused Cartesian grid
# intentionally omits.  The two strongest old points are already in the grid.
ANCHORS = (
    (4, 5200, 4096),
    (32, 128, 4096),
)


@dataclass(frozen=True, order=True)
class Point:
    batch_size: int
    input_tokens: int
    output_tokens: int

    @property
    def label(self) -> str:
        return (
            f"B{self.batch_size}-I{self.input_tokens}-O{self.output_tokens}"
        )


def _positive_ints(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in raw.split(",") if value}))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _gpu_budget(raw: str) -> float | None:
    if raw.lower() == "max":
        return None
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("GPU KV budget must be positive or 'max'")
    return value


def coarse_points(
    batches: Iterable[int],
    inputs: Iterable[int],
    outputs: Iterable[int],
    *,
    include_anchors: bool = True,
) -> tuple[Point, ...]:
    points = {
        Point(batch, input_tokens, output_tokens)
        for batch, input_tokens, output_tokens in product(
            batches, inputs, outputs
        )
    }
    if include_anchors:
        points.update(Point(*values) for values in ANCHORS)
    return tuple(sorted(points))


def refinement_points(
    seeds: Iterable[Point],
    existing: Iterable[Point],
    *,
    batch_step: int,
    input_step: int,
    output_step: int,
) -> tuple[Point, ...]:
    existing_set = set(existing)
    points: set[Point] = set()
    for seed in seeds:
        for batch_delta, input_delta, output_delta in product(
            (-batch_step, 0, batch_step),
            (-input_step, 0, input_step),
            (-output_step, 0, output_step),
        ):
            point = Point(
                seed.batch_size + batch_delta,
                seed.input_tokens + input_delta,
                seed.output_tokens + output_delta,
            )
            if (
                point.batch_size > 0
                and point.input_tokens > 0
                and point.output_tokens > 0
                and point not in existing_set
            ):
                points.add(point)
    return tuple(sorted(points))


def _code_sha256() -> str:
    paths = [Path(__file__), *sorted((REPOSITORY_ROOT / "elasticafd").glob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPOSITORY_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _budget_label(budget: float | None) -> str:
    return "max" if budget is None else f"{budget:g}GiB"


def _checkpoint_path(
    output_dir: Path,
    pipeline_model: str,
    point: Point,
    budget: float | None,
) -> Path:
    return (
        output_dir
        / "checkpoints"
        / pipeline_model
        / _budget_label(budget)
        / f"{point.label}.json"
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_settings(
    *,
    pipeline_model: str,
    budget: float | None,
    mode_grid_points: int,
    quadrature_samples: int,
    exact_finalists: int,
) -> dict[str, object]:
    return {
        "model": "gpt175b",
        "hardware": "chime_a100",
        "gpu_kv_budget": _budget_label(budget),
        "pipeline_model": pipeline_model,
        "policy_pipeline_model": (
            "analytic" if pipeline_model == "rankset_event" else "analytic"
        ),
        "mode_grid_points": mode_grid_points,
        "quadrature_samples": quadrature_samples,
        "exact_finalists": exact_finalists,
        "historical_kv_migration": False,
        "prefill_semantics": "complete_decode_ready",
    }


def _run_one(
    point: Point,
    *,
    output_dir: Path,
    source_stage: str,
    pipeline_model: str,
    budget: float | None,
    mode_grid_points: int,
    quadrature_samples: int,
    exact_finalists: int,
    code_sha256: str,
) -> dict[str, object]:
    checkpoint = _checkpoint_path(output_dir, pipeline_model, point, budget)
    settings = _run_settings(
        pipeline_model=pipeline_model,
        budget=budget,
        mode_grid_points=mode_grid_points,
        quadrature_samples=quadrature_samples,
        exact_finalists=exact_finalists,
    )
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("code_sha256") != code_sha256:
            raise RuntimeError(
                f"stale checkpoint {checkpoint}; use a new output directory"
            )
        if payload.get("settings") != settings:
            raise RuntimeError(
                f"checkpoint settings differ for {checkpoint}; "
                "use a new output directory"
            )
        print(f"skip={pipeline_model}:{point.label}", flush=True)
        return payload["row"]

    print(f"start={pipeline_model}:{point.label}", flush=True)
    started = perf_counter()
    rows = homogeneous_lifecycle_rows(
        model_names=("gpt175b",),
        batch_sizes=(point.batch_size,),
        input_lengths=(point.input_tokens,),
        output_lengths=(point.output_tokens,),
        gpu_kv_budgets_gib=(budget,),
        mode_grid_points=mode_grid_points,
        quadrature_samples=quadrature_samples,
        exact_finalists=exact_finalists,
        pipeline_model=pipeline_model,
        policy_pipeline_model=(
            "analytic" if pipeline_model == "rankset_event" else None
        ),
    )
    if len(rows) != 1:
        raise AssertionError(f"expected one result row, received {len(rows)}")
    row = dict(rows[0])
    wall_time_s = perf_counter() - started
    row["source_stage"] = source_stage
    row["experiment_wall_time_s"] = wall_time_s
    payload = {
        "code_sha256": code_sha256,
        "settings": settings,
        "point": {
            "batch_size": point.batch_size,
            "input_tokens": point.input_tokens,
            "output_tokens": point.output_tokens,
        },
        "row": row,
    }
    _atomic_json(checkpoint, payload)
    speedup = row.get("offline_mode_speedup_vs_chime")
    print(
        f"done={pipeline_model}:{point.label} speedup={speedup} "
        f"wall={wall_time_s:.2f}s",
        flush=True,
    )
    return row


def _run_many(
    points: Iterable[Point],
    *,
    output_dir: Path,
    source_stage: str,
    pipeline_model: str,
    budget: float | None,
    mode_grid_points: int,
    quadrature_samples: int,
    exact_finalists: int,
) -> None:
    code_sha256 = _code_sha256()
    points_tuple = tuple(points)
    print(
        f"points={len(points_tuple)} pipeline={pipeline_model} "
        f"budget={_budget_label(budget)}",
        flush=True,
    )
    for index, point in enumerate(points_tuple, 1):
        print(f"progress={index}/{len(points_tuple)}", flush=True)
        _run_one(
            point,
            output_dir=output_dir,
            source_stage=source_stage,
            pipeline_model=pipeline_model,
            budget=budget,
            mode_grid_points=mode_grid_points,
            quadrature_samples=quadrature_samples,
            exact_finalists=exact_finalists,
            code_sha256=code_sha256,
        )


def _load_records(
    output_dir: Path,
    pipeline_model: str,
    budget: float | None,
) -> list[dict[str, object]]:
    directory = (
        output_dir / "checkpoints" / pipeline_model / _budget_label(budget)
    )
    if not directory.exists():
        return []
    records: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(payload["row"])
    return records


def _point_from_row(row: dict[str, object]) -> Point:
    return Point(
        int(row["batch_size"]),
        int(row["input_tokens"]),
        int(row["output_tokens"]),
    )


def _speedup(row: dict[str, object]) -> float:
    value = row.get("offline_mode_speedup_vs_chime")
    if value is None:
        return float("-inf")
    return float(value)


def _completed(row: dict[str, object]) -> bool:
    return (
        row.get("baseline_status") == "completed"
        and row.get("offline_mode_status") == "completed"
    )


def _ranked(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        (row for row in rows if _completed(row)),
        key=lambda row: (
            -_speedup(row),
            int(row["batch_size"]),
            int(row["input_tokens"]),
            int(row["output_tokens"]),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_dir: Path, budget: float | None) -> None:
    summary_dir = output_dir / "summary" / _budget_label(budget)
    analytic = _load_records(output_dir, "analytic", budget)
    event = _load_records(output_dir, "rankset_event", budget)
    analytic_ranked = _ranked(analytic)
    event_ranked = _ranked(event)
    _write_csv(summary_dir / "analytic_all.csv", analytic)
    _write_csv(summary_dir / "analytic_ranked.csv", analytic_ranked)
    _write_csv(summary_dir / "event_all.csv", event)
    _write_csv(summary_dir / "event_ranked.csv", event_ranked)

    for dimension in ("batch_size", "input_tokens", "output_tokens"):
        best: dict[int, dict[str, object]] = {}
        for row in analytic_ranked:
            value = int(row[dimension])
            if value not in best:
                best[value] = row
        _write_csv(
            summary_dir / f"best_by_{dimension}.csv",
            [best[value] for value in sorted(best)],
        )

    payload = {
        "gpu_kv_budget": _budget_label(budget),
        "analytic_points": len(analytic),
        "analytic_completed": len(analytic_ranked),
        "event_points": len(event),
        "event_completed": len(event_ranked),
        "best_analytic": analytic_ranked[:10],
        "best_event": event_ranked[:10],
        "speedup_definition": "CHIME makespan / candidate makespan",
        "primary_claim_rule": (
            "rankset_event is confirmatory; analytic results are screening only"
        ),
    }
    _atomic_json(summary_dir / "summary.json", payload)
    print(f"summary={summary_dir.resolve()}")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "attention-advantage-refinement",
    )
    parser.add_argument("--gpu-kv-budget", type=_gpu_budget, default=None)
    parser.add_argument("--mode-grid-points", type=int, default=5)
    parser.add_argument("--quadrature-samples", type=int, default=24)
    parser.add_argument("--exact-finalists", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned point count without running the simulator",
    )


def _validate_run_options(args: argparse.Namespace) -> None:
    if args.mode_grid_points < 3:
        raise ValueError("mode-grid-points must be at least 3")
    if args.quadrature_samples <= 0 or args.exact_finalists <= 0:
        raise ValueError("quadrature-samples and exact-finalists must be positive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    coarse = subparsers.add_parser("coarse", help="run the broad analytic grid")
    _add_run_options(coarse)
    coarse.add_argument("--preset", choices=("smoke", "focused"), default="focused")
    coarse.add_argument("--batch-sizes", type=_positive_ints)
    coarse.add_argument("--input-lengths", type=_positive_ints)
    coarse.add_argument("--output-lengths", type=_positive_ints)
    coarse.add_argument("--no-anchors", action="store_true")

    refine = subparsers.add_parser(
        "refine", help="refine a 3x3x3 neighborhood around top analytic points"
    )
    _add_run_options(refine)
    refine.add_argument("--top-k", type=int, default=8)
    refine.add_argument("--minimum-speedup", type=float, default=1.01)
    refine.add_argument("--batch-step", type=int, default=2)
    refine.add_argument("--input-step", type=int, default=512)
    refine.add_argument("--output-step", type=int, default=512)

    event = subparsers.add_parser(
        "event", help="validate top analytic points on the rankset event model"
    )
    _add_run_options(event)
    event.add_argument("--top-k", type=int, default=5)
    event.add_argument("--minimum-speedup", type=float, default=1.01)

    summary_parser = subparsers.add_parser(
        "summarize", help="build ranked and per-dimension CSV summaries"
    )
    summary_parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "attention-advantage-refinement",
    )
    summary_parser.add_argument("--gpu-kv-budget", type=_gpu_budget, default=None)

    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.output_dir, args.gpu_kv_budget)
        return 0

    _validate_run_options(args)
    if args.command == "coarse":
        if args.preset == "smoke":
            default_batches = SMOKE_BATCHES
            default_inputs = SMOKE_INPUTS
            default_outputs = SMOKE_OUTPUTS
        else:
            default_batches = DEFAULT_BATCHES
            default_inputs = DEFAULT_INPUTS
            default_outputs = DEFAULT_OUTPUTS
        points = coarse_points(
            args.batch_sizes or default_batches,
            args.input_lengths or default_inputs,
            args.output_lengths or default_outputs,
            include_anchors=not args.no_anchors,
        )
        if args.dry_run:
            print(f"coarse_points={len(points)}")
            print(f"first={points[0].label} last={points[-1].label}")
            return 0
        _run_many(
            points,
            output_dir=args.output_dir,
            source_stage="coarse",
            pipeline_model="analytic",
            budget=args.gpu_kv_budget,
            mode_grid_points=args.mode_grid_points,
            quadrature_samples=args.quadrature_samples,
            exact_finalists=args.exact_finalists,
        )
    elif args.command == "refine":
        analytic = _ranked(
            _load_records(args.output_dir, "analytic", args.gpu_kv_budget)
        )
        seeds = [
            _point_from_row(row)
            for row in analytic
            if _speedup(row) >= args.minimum_speedup
        ][: args.top_k]
        if not seeds:
            raise RuntimeError("no qualifying analytic checkpoints to refine")
        existing = [_point_from_row(row) for row in analytic]
        points = refinement_points(
            seeds,
            existing,
            batch_step=args.batch_step,
            input_step=args.input_step,
            output_step=args.output_step,
        )
        if args.dry_run:
            print(f"refinement_seeds={len(seeds)} refinement_points={len(points)}")
            print("seeds=" + ",".join(point.label for point in seeds))
            return 0
        _run_many(
            points,
            output_dir=args.output_dir,
            source_stage="refine",
            pipeline_model="analytic",
            budget=args.gpu_kv_budget,
            mode_grid_points=args.mode_grid_points,
            quadrature_samples=args.quadrature_samples,
            exact_finalists=args.exact_finalists,
        )
    elif args.command == "event":
        analytic = _ranked(
            _load_records(args.output_dir, "analytic", args.gpu_kv_budget)
        )
        points = tuple(
            _point_from_row(row)
            for row in analytic
            if _speedup(row) >= args.minimum_speedup
        )[: args.top_k]
        if not points:
            raise RuntimeError("no qualifying analytic checkpoints to validate")
        if args.dry_run:
            print(f"event_points={len(points)}")
            print("points=" + ",".join(point.label for point in points))
            return 0
        _run_many(
            points,
            output_dir=args.output_dir,
            source_stage="event_top",
            pipeline_model="rankset_event",
            budget=args.gpu_kv_budget,
            mode_grid_points=args.mode_grid_points,
            quadrature_samples=args.quadrature_samples,
            exact_finalists=args.exact_finalists,
        )
    else:
        raise AssertionError(args.command)

    summarize(args.output_dir, args.gpu_kv_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
