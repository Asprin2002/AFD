#!/usr/bin/env python3
"""Rescale an existing online trace's arrival axis to a target offered load.

For a single-replica small device, replaying a large-deployment trace at its
native arrival rate overloads the device (the Qwen-512 source offers roughly
4,500 token/s against ~110 token/s of measured capacity).  This tool keeps the
full request set — lengths, order, and relative inter-arrival structure — and
affine-scales only arrival_s so that the offered output-token rate matches a
target below capacity.  This is the same timestamp transform used by
``prepare_burstgpt_trace.py`` (``time_scale``), generalised to an existing CSV.

All non-``arrival_s`` columns are preserved verbatim, so downstream runners
(``run_full_joint_512.py``) keep working with their existing hash metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("empty trace file")
        for required in ("arrival_s", "output_tokens"):
            if required not in reader.fieldnames:
                raise ValueError(f"source trace lacks required column {required}")
        return list(reader), reader.fieldnames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-tok-s", type=float, required=True,
        help="target offered output-token rate (tokens/s) after rescaling",
    )
    parser.add_argument(
        "--basis-capacity-tok-s", type=float, default=None,
        help="informational: measured sustained device capacity used to pick target",
    )
    parser.add_argument(
        "--basis-fraction", type=float, default=None,
        help="informational: fraction of capacity the target represents",
    )
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()

    if args.target_tok_s <= 0:
        parser.error("--target-tok-s must be positive")

    rows, fieldnames = read_rows(args.source)
    arrivals = [float(row["arrival_s"]) for row in rows]
    if not arrivals or abs(arrivals[0]) > 1e-9 or any(
        left > right for left, right in zip(arrivals, arrivals[1:])
    ):
        raise ValueError("arrivals must start at zero and be nondecreasing")
    if len(rows) < 2 or arrivals[-1] <= 0:
        raise ValueError("trace needs more than one arrival with a positive span")

    output_tokens = [int(row["output_tokens"]) for row in rows]
    if any(value <= 0 for value in output_tokens):
        raise ValueError("all output_tokens must be positive")
    total_output = sum(output_tokens)
    native_span_s = arrivals[-1] - arrivals[0]
    native_offered_tok_s = total_output / native_span_s
    target_span_s = total_output / args.target_tok_s
    scale = target_span_s / native_span_s
    scaled_arrivals = [arrival * scale for arrival in arrivals]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, arrival_s in zip(rows, scaled_arrivals):
            out = dict(row)
            out["arrival_s"] = f"{arrival_s:.9f}"
            writer.writerow(out)

    realized_offered_tok_s = total_output / (scaled_arrivals[-1] - scaled_arrivals[0])
    source_hash = sha256_file(args.source)
    trace_hash = sha256_file(args.output)
    metadata = {
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "trace_sha256": trace_hash,
        "request_count": len(rows),
        "semantics": {
            "prefill": "already completed; every arrival is decode-ready",
            "timestamp_transform": "affine scale after subtracting first timestamp",
            "token_lengths_modified": False,
            "request_set": "complete source request set, order unchanged",
        },
        "native_time_span_s": native_span_s,
        "native_offered_tok_s": native_offered_tok_s,
        "target_offered_tok_s": args.target_tok_s,
        "realized_offered_tok_s": realized_offered_tok_s,
        "time_scale": scale,
        "basis_capacity_tok_s": args.basis_capacity_tok_s,
        "basis_fraction": args.basis_fraction,
        "total_output_tokens": total_output,
        "note": (
            "same user population replayed at the pace of a single-replica "
            "device; relative inter-arrival and token-length structure kept"
        ),
    }
    metadata_path = args.metadata_output or Path(f"{args.output}.metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())