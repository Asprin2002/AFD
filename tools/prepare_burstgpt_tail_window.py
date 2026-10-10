#!/usr/bin/env python3
"""Extract one unfiltered, contiguous high-output BurstGPT service window.

This is an explicitly conditioned tail stress case, separate from the three
unconditioned service windows in run_full_joint_trace.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_burstgpt_trace import (
    SOURCE_SHA256, SOURCE_URL, read_source, sha256_file,
)


def prepare(source: Path, output: Path, *, count: int, quantile: float,
            max_span_s: float) -> None:
    actual = sha256_file(source)
    if actual != SOURCE_SHA256:
        raise ValueError(f"BurstGPT source hash mismatch: {actual}")
    records, source_rows, failed = read_source(source)
    if count < 2 or not 0 < quantile < 1 or max_span_s <= 0:
        raise ValueError("invalid count, quantile, or maximum span")
    n = count
    running = sum(record.output_tokens for record in records[:n])
    candidates: list[tuple[int, int]] = []
    for index in range(len(records) - n + 1):
        if records[index + n - 1].timestamp_s - records[index].timestamp_s <= max_span_s:
            candidates.append((running, index))
        if index + n < len(records):
            running += records[index + n].output_tokens - records[index].output_tokens
    if not candidates:
        raise ValueError("no eligible contiguous window")
    totals = sorted(output_total for output_total, _ in candidates)
    threshold = totals[int(quantile * (len(totals) - 1))]
    selected_index = next(
        index for output_total, index in candidates if output_total >= threshold
    )
    window = records[selected_index:selected_index + n]
    first = window[0].timestamp_s
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "request_id", "arrival_s", "input_tokens", "output_tokens",
        ))
        writer.writeheader()
        for record in window:
            writer.writerow({
                "request_id": f"burstgpt-{record.source_row}",
                "arrival_s": record.timestamp_s - first,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
            })
    metadata = {
        "source_url": SOURCE_URL,
        "source_sha256": actual,
        "source_rows": source_rows,
        "source_positive_rows": len(records),
        "source_failed_or_zero_rows": failed,
        "selection": "first contiguous positive-token window at rolling-output quantile",
        "rolling_output_quantile": quantile,
        "rolling_output_threshold_tokens": threshold,
        "maximum_native_span_s": max_span_s,
        "eligible_window_count": len(candidates),
        "window_count": n,
        "source_row_range": [window[0].source_row, window[-1].source_row],
        "native_time_span_s": window[-1].timestamp_s - first,
        "length_filter": "none beyond positive input/output tokens",
        "time_rescaled": False,
        "mean_input_tokens": sum(r.input_tokens for r in window) / n,
        "mean_output_tokens": sum(r.output_tokens for r in window) / n,
        "long_output_count": sum(r.output_tokens >= 1024 for r in window),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {output}: rows={metadata['source_row_range']} "
          f"mean_output={metadata['mean_output_tokens']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--max-span-s", type=float, default=300)
    args = parser.parse_args()
    prepare(args.source, args.output, count=args.count,
            quantile=args.quantile, max_span_s=args.max_span_s)


if __name__ == "__main__":
    main()
