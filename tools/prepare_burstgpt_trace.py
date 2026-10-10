#!/usr/bin/env python3
"""Prepare small, reproducible BurstGPT traces for ElasticAFD.

The source dataset is never silently reinterpreted as a full production mix.
This tool filters one explicitly named length class, selects the densest
consecutive window *within that conditioned point process*, and optionally
rescales only the time axis to a target offered request rate. Token lengths
and relative inter-arrival structure are preserved exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SOURCE_URL = (
    "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/"
    "data/BurstGPT_1.csv"
)
SOURCE_SHA256 = "46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a"
SOURCE_LICENSE = "CC-BY-4.0"


@dataclass(frozen=True)
class TraceClass:
    name: str
    input_min: int
    input_max: int | None
    output_min: int
    output_max: int | None = None

    def contains(self, input_tokens: int, output_tokens: int) -> bool:
        return (
            input_tokens >= self.input_min
            and (self.input_max is None or input_tokens <= self.input_max)
            and output_tokens >= self.output_min
            and (self.output_max is None or output_tokens <= self.output_max)
        )


TRACE_CLASSES = {
    "short_input_long_output": TraceClass(
        "short_input_long_output",
        input_min=1,
        input_max=2048,
        output_min=1024,
    ),
    "long_input_long_output": TraceClass(
        "long_input_long_output",
        input_min=4096,
        input_max=None,
        output_min=1024,
    ),
}


@dataclass(frozen=True)
class SourceRecord:
    source_row: int
    timestamp_s: float
    model: str
    input_tokens: int
    output_tokens: int
    log_type: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: Path) -> tuple[list[SourceRecord], int, int]:
    records: list[SourceRecord] = []
    total_rows = 0
    failed_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Timestamp",
            "Model",
            "Request tokens",
            "Response tokens",
            "Log Type",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "BurstGPT source is missing columns: "
                + ", ".join(sorted(missing))
            )
        for source_row, row in enumerate(reader, start=1):
            total_rows += 1
            input_tokens = int(row["Request tokens"])
            output_tokens = int(row["Response tokens"])
            if input_tokens <= 0 or output_tokens <= 0:
                failed_rows += 1
                continue
            records.append(
                SourceRecord(
                    source_row=source_row,
                    timestamp_s=float(row["Timestamp"]),
                    model=row["Model"],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    log_type=row["Log Type"],
                )
            )
    records.sort(key=lambda item: (item.timestamp_s, item.source_row))
    return records, total_rows, failed_rows


def densest_window(
    records: Sequence[SourceRecord], count: int
) -> tuple[SourceRecord, ...]:
    if count <= 1:
        raise ValueError("count must be greater than one")
    if len(records) < count:
        raise ValueError(
            f"only {len(records)} records match the class; cannot select {count}"
        )
    start = min(
        range(len(records) - count + 1),
        key=lambda index: (
            records[index + count - 1].timestamp_s
            - records[index].timestamp_s,
            records[index].timestamp_s,
            records[index].source_row,
        ),
    )
    return tuple(records[start : start + count])


def coefficient_of_variation(values: Iterable[float]) -> float | None:
    materialized = tuple(values)
    if not materialized:
        return None
    mean = statistics.fmean(materialized)
    if mean == 0.0:
        return None
    return statistics.pstdev(materialized) / mean


def length_summary(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def prepare_trace(
    source: Path,
    output: Path,
    *,
    trace_class: TraceClass,
    count: int,
    target_rps: float | None,
    expected_sha256: str = SOURCE_SHA256,
    metadata_output: Path | None = None,
) -> dict[str, object]:
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "BurstGPT source SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    if target_rps is not None and target_rps <= 0.0:
        raise ValueError("target_rps must be positive")

    successful, total_rows, failed_rows = read_source(source)
    qualified = [
        record
        for record in successful
        if trace_class.contains(record.input_tokens, record.output_tokens)
    ]
    selected = densest_window(qualified, count)
    first_timestamp = selected[0].timestamp_s
    original_span_s = selected[-1].timestamp_s - first_timestamp
    if original_span_s <= 0.0:
        raise ValueError("selected window has a non-positive timestamp span")

    target_span_s = (
        original_span_s
        if target_rps is None
        else (len(selected) - 1) / target_rps
    )
    time_scale = target_span_s / original_span_s
    arrivals_s = [
        (record.timestamp_s - first_timestamp) * time_scale
        for record in selected
    ]
    interarrivals_s = [
        right - left for left, right in zip(arrivals_s, arrivals_s[1:])
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "request_id",
            "arrival_s",
            "input_tokens",
            "output_tokens",
            "source_timestamp_s",
            "source_row",
            "source_model",
            "source_log_type",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, arrival_s in zip(selected, arrivals_s):
            writer.writerow(
                {
                    "request_id": f"burstgpt1-row-{record.source_row}",
                    "arrival_s": f"{arrival_s:.9f}",
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "source_timestamp_s": f"{record.timestamp_s:.9f}",
                    "source_row": record.source_row,
                    "source_model": record.model,
                    "source_log_type": record.log_type,
                }
            )

    metadata: dict[str, object] = {
        "format_version": 1,
        "source": {
            "name": "BurstGPT_1",
            "url": SOURCE_URL,
            "sha256": observed_sha256,
            "license": SOURCE_LICENSE,
        },
        "semantics": {
            "prefill": "already completed; every arrival is decode-ready",
            "selection_population": "successful requests only",
            "selection": "minimum-span consecutive window in conditioned class",
            "timestamp_transform": "affine scale after subtracting first timestamp",
            "token_lengths": "unchanged from source",
        },
        "trace_class": asdict(trace_class),
        "source_rows": total_rows,
        "source_failed_or_zero_token_rows": failed_rows,
        "source_successful_rows": len(successful),
        "qualified_rows": len(qualified),
        "selected_rows": len(selected),
        "selected_source_row_first": selected[0].source_row,
        "selected_source_row_last": selected[-1].source_row,
        "selected_source_timestamp_first_s": selected[0].timestamp_s,
        "selected_source_timestamp_last_s": selected[-1].timestamp_s,
        "original_arrival_span_s": original_span_s,
        "target_rps": target_rps,
        "output_arrival_span_s": target_span_s,
        "realized_offered_rps": (len(selected) - 1) / target_span_s,
        "time_scale": time_scale,
        "interarrival_cv": coefficient_of_variation(interarrivals_s),
        "input_tokens": length_summary(
            [record.input_tokens for record in selected]
        ),
        "output_tokens": length_summary(
            [record.output_tokens for record in selected]
        ),
        "output_csv": str(output),
    }
    metadata_path = metadata_output or Path(f"{output}.metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--class-name", choices=tuple(TRACE_CLASSES), required=True
    )
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--target-rps", type=float)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--expected-sha256", default=SOURCE_SHA256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = prepare_trace(
        args.source,
        args.output,
        trace_class=TRACE_CLASSES[args.class_name],
        count=args.count,
        target_rps=args.target_rps,
        expected_sha256=args.expected_sha256,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
