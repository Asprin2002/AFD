#!/usr/bin/env python3
"""M0: deterministic heterogeneous static-batch recipe generator.

Three heterogeneity profiles are generated over a seeded RNG, for closed
static batches (all requests arrive at clock 0, decode-ready, shared decode
clock):

  - ``long_in``: fixed output length, input lengths spanning an order of
    magnitude (1024 .. 12288).  Stress the Attention-side scan disparity.
  - ``long_out``: fixed input, output lengths spanning 128 .. 2048.  Stress
    stretched tails and the late-hot migration region.
  - ``mixed``: two K clusters (short-context/short-output and
    long-context/long-output).  Stress the abrupt workload switch that the
    segmented joint policy is designed for.

A deterministic preflight capacity check (all-PIM control at every request's
finish line) keeps the generated recipes runnable; the runner repeats the
authoritative per-block capacity check.  Homogeneous anchor shapes from the
trusted static grid are emitted alongside as paired references.

Output layout under ``workloads/static``:

  manifest.json
  homogeneous/anchor_b{batch}_i{input}_o{output}.csv
  heterogeneous/{profile}/batch{batch}_seed{seed}.csv

Every CSV is accompanied by a ``.metadata.json`` with the exact sampling
parameters, source hashes of this generator, the model-code hash, and the
determinism seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticafd.segmented import (
    RequestSpec,
    block_sample_boundaries,
    control_schedule,
    max_output,
    state_views,
)
from elasticafd.specs import model_from_config, system_from_config

GIB = float(1024 ** 3)

# Trusted-grid homogeneous anchors: benefit region, no-benefit region, tail.
HOMOGENEOUS_ANCHORS = (
    (2, 128, 256),     # no-benefit short region
    (2, 128, 4096),
    (4, 2048, 1024),
    (4, 5200, 4096),
    (4, 2048, 8192),
    (8, 8192, 8192),   # known 1.2694x shape
    (8, 2048, 1024),
    (8, 5200, 4096),
    (16, 128, 256),
    (16, 2048, 256),
    (16, 2048, 1024),
    (16, 5200, 4096),  # known 1.3742x shape
    (16, 5200, 8192),  # known best-of statement 1.5074x
    (32, 2048, 1024),
    (32, 2048, 4096),
    (32, 2048, 8192),  # known 1.4001x / 1.4847x shapes
    (32, 5200, 4096),
)
BATCHES = (8, 16, 32)
SEEDS = (20260922, 20260923)
PROFILES = ("long_in", "long_out", "mixed")
DECISION_STRIDE = 64  # outputs are rounded to this multiple


def round_output(tokens: int) -> int:
    return max(DECISION_STRIDE, (tokens + DECISION_STRIDE - 1) // DECISION_STRIDE
               * DECISION_STRIDE)


def _sample(rng: random.Random, choices, weights):
    return rng.choices(choices, weights=weights, k=1)[0]


def build_heterogeneous(profile: str, batch: int, seed: int) -> list[RequestSpec]:
    """Sampled, long-context-first request list for one recipe."""
    rng = random.Random(f"{profile}:{batch}:{seed}")
    specs: list[RequestSpec] = []
    for index in range(batch):
        if profile == "long_in":
            input_tokens = _sample(rng, (1024, 2048, 4096, 8192, 12288),
                                   weights=(25, 30, 25, 15, 5))
            output_tokens = round_output(rng.randint(512, 2048))
        elif profile == "long_out":
            input_tokens = rng.randint(2048, 4096)
            output_tokens = _sample(rng, (128, 256, 512, 1024, 2048),
                                    weights=(20, 25, 25, 20, 10))
            output_tokens = round_output(output_tokens)
        elif profile == "mixed":
            if rng.random() < 0.5:
                input_tokens = rng.randint(512, 2048)
                output_tokens = round_output(rng.randint(64, 512))
            else:
                input_tokens = rng.randint(4096, 16384)
                output_tokens = round_output(rng.randint(1024, 4096))
        else:
            raise ValueError(f"unknown profile: {profile}")
        specs.append(RequestSpec(f"r{index}", input_tokens, output_tokens))
    specs.sort(key=lambda spec: (-spec.input_tokens, -spec.output_tokens,
                                 spec.request_id))
    return specs


def build_anchor(batch: int, input_tokens: int, output_tokens: int) -> list[RequestSpec]:
    return [RequestSpec(f"r{i}", input_tokens, output_tokens)
            for i in range(batch)]


def capacity_feasible(specs: list[RequestSpec]) -> bool:
    """Cheap all-PIM preflight: aggregate PIM KV at each request finish line.

    Control keeps every token on PIM, so this is the same ``views_fit_capacity``
    the runtime uses; the authoritative per-block check is repeated by the
    runner.  A 1 GiB workspace allowance stands in for activation buffers.
    """
    model = model_from_config("gpt175b")
    system = system_from_config("chime_a100")
    schedule = control_schedule(tuple(specs))
    boundaries = block_sample_boundaries(schedule, tuple(specs), 1)
    workspace_gib = 1.0
    capacity_bytes = system.pim.capacity_bytes - int(workspace_gib * GIB)
    from elasticafd.capacity import views_memory_usage

    check_points = {spec.output_tokens for spec in specs} | {max_output(tuple(specs))}
    for generated in sorted(check_points):
        views = state_views(schedule, tuple(specs), generated)
        pim_bytes, _ = views_memory_usage(views, model, system)
        if pim_bytes > capacity_bytes:
            return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash() -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update(Path("elasticafd/segmented.py").read_bytes())
    digest.update(Path("elasticafd/dense_offload.py").read_bytes())
    digest.update(Path("elasticafd/specs.py").read_bytes())
    digest.update(Path("elasticafd/state.py").read_bytes())
    return digest.hexdigest()


def write_csv(path: Path, specs: list[RequestSpec], extra: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "request_id", "arrival_s", "input_tokens", "output_tokens",
            "gpu_replica",
        ))
        writer.writeheader()
        for spec in specs:
            writer.writerow({
                "request_id": spec.request_id,
                "arrival_s": 0,
                "input_tokens": spec.input_tokens,
                "output_tokens": spec.output_tokens,
                "gpu_replica": spec.gpu_replica,
            })
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "trace_sha256": sha256_file(path),
        "generator_sha256": code_hash(),
        "profile": extra["profile"],
        "seed": extra["seed"],
        "batch_size": len(specs),
        "capacity_preflight": extra.get("capacity_preflight", True),
        "request_count": len(specs),
        "total_output_tokens": sum(s.output_tokens for s in specs),
        "total_input_tokens": sum(s.input_tokens for s in specs),
        "mean_input_tokens": (
            sum(s.input_tokens for s in specs) / len(specs) if specs else None
        ),
        "mean_output_tokens": (
            sum(s.output_tokens for s in specs) / len(specs) if specs else None
        ),
        "note": extra.get("note", ""),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate(output_dir: Path, *, dry: bool = False) -> dict:
    output_dir = output_dir.resolve()
    (output_dir / "homogeneous").mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generator_sha256": code_hash(),
        "schema_version": "static-recipe-v1",
        "decision_stride": DECISION_STRIDE,
        "homogeneous_anchors": list(HOMOGENEOUS_ANCHORS),
        "batches": list(BATCHES),
        "seeds": list(SEEDS),
        "profiles": list(PROFILES),
        "files": {},
    }
    # Homogeneous anchors.
    for batch, input_tokens, output_tokens in HOMOGENEOUS_ANCHORS:
        specs = build_anchor(batch, input_tokens, output_tokens)
        name = f"anchor_b{batch}_i{input_tokens}_o{output_tokens}"
        path = output_dir / "homogeneous" / f"{name}.csv"
        feasible = capacity_feasible(specs)
        if not dry:
            write_csv(path, specs, {
                "profile": "anchor",
                "seed": None,
                "capacity_preflight": feasible,
                "note": "trusted-grid homogeneous anchor",
            })
        manifest["files"][name] = {
            "kind": "anchor",
            "batch": batch,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "capacity_preflight": feasible,
            "path": str(path.relative_to(output_dir)),
        }
    # Heterogeneous profiles.
    for profile in PROFILES:
        (output_dir / "heterogeneous" / profile).mkdir(parents=True,
                                                       exist_ok=True)
        for batch in BATCHES:
            for seed in SEEDS:
                specs = build_heterogeneous(profile, batch, seed)
                name = f"{profile}_b{batch}_s{seed}"
                path = (output_dir / "heterogeneous" / profile / f"b{batch}_s{seed}.csv")
                feasible = capacity_feasible(specs)
                if not dry:
                    write_csv(path, specs, {
                        "profile": profile,
                        "seed": seed,
                        "capacity_preflight": feasible,
                        "note": f"deterministic {profile} recipe, seed {seed}",
                    })
                manifest["files"][name] = {
                    "kind": "heterogeneous",
                    "profile": profile,
                    "batch": batch,
                    "seed": seed,
                    "capacity_preflight": feasible,
                    "path": str(path.relative_to(output_dir)),
                }
    if not dry:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def load_recipe(path: Path) -> tuple[RequestSpec, ...]:
    """Load a generated static recipe back into request specs."""
    path = path.resolve()
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    specs = tuple(
        RequestSpec(
            row["request_id"],
            int(row["input_tokens"]),
            int(row["output_tokens"]),
            int(row.get("gpu_replica", 0)),
        )
        for row in rows
    )
    return specs


def discover(output_dir: Path) -> list[Path]:
    """Deterministic discovery of generated recipe CSV files."""
    output_dir = output_dir.resolve()
    paths = sorted((output_dir / "homogeneous").glob("anchor_*.csv"))
    for profile in PROFILES:
        paths.extend(sorted((output_dir / "heterogeneous" / profile).glob("*.csv")))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("workloads/static"),
        help="output directory under the project root",
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="validate profiles and capacity preflight without writing files",
    )
    args = parser.parse_args()
    manifest = generate(args.output, dry=args.dry)
    infeasible = [name for name, entry in manifest["files"].items()
                  if not entry["capacity_preflight"]]
    count = len(manifest["files"])
    print(f"{'dry-validation' if args.dry else 'generated'} {count} recipes in "
          f"{args.output.resolve()}")
    if infeasible:
        print("WARNING: preflight-infeasible recipes (skipped by runner):")
        for name in infeasible:
            print(f"  {name}")
    else:
        print("all recipe preflights pass the all-PIM control capacity check")
    if not args.dry:
        print(f"generator_sha256={manifest['generator_sha256']}")


if __name__ == "__main__":
    main()