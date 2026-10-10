#!/usr/bin/env python3
"""Prepare reviewable, unscaled 1024-request long-service trace candidates.

This script only selects and describes workload inputs. It does not run any
simulator or policy comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCES = {
    "qwen-thinking": {
        "url": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_thinking_blksz_16.jsonl",
        "sha256": "41ac36d9d1b54d084eeb6b05e9d142ba9d02f8bf79cfc9fff418d8ab7cdee906",
        "timestamp_unit": "seconds",
        "timestamp_to_s": 1.0,
    },
    "qwen-coder": {
        "url": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_coder_blksz_16.jsonl",
        "sha256": "3d74974cebb7bccb69b1ca6ea210a44d9a3033443c0271ed015327284dea1aff",
        "timestamp_unit": "seconds",
        "timestamp_to_s": 1.0,
    },
    "mooncake-conversation": {
        "url": "https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/conversation_trace.jsonl",
        "sha256": "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df",
        "timestamp_unit": "milliseconds",
        "timestamp_to_s": 0.001,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], p: float) -> int:
    return sorted(values)[int(p * (len(values) - 1))]


def prepare(name: str, source: Path, output_dir: Path, count: int) -> dict:
    spec = SOURCES[name]
    observed_hash = sha256_file(source)
    if observed_hash != spec["sha256"]:
        raise ValueError(f"unexpected {name} source checksum: {observed_hash}")
    rows = []
    with source.open("r", encoding="utf-8") as stream:
        for source_line, line in enumerate(stream, 1):
            record = json.loads(line)
            input_tokens = int(record["input_length"])
            output_tokens = int(record["output_length"])
            if input_tokens <= 0 or output_tokens <= 0:
                raise ValueError(f"nonpositive token count at {name} line {source_line}")
            rows.append((
                float(record["timestamp"]) * spec["timestamp_to_s"],
                input_tokens,
                output_tokens,
                source_line,
            ))
    if len(rows) < count:
        raise ValueError(f"{name} has fewer than {count} requests")
    if any(a[0] > b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{name} is not in timestamp order")
    start = (len(rows) - count) // 2
    chosen = rows[start:start + count]
    first_timestamp = chosen[0][0]
    arrivals = [row[0] - first_timestamp for row in chosen]
    inputs = [row[1] for row in chosen]
    outputs = [row[2] for row in chosen]
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    span = arrivals[-1]
    bins = {int(arrival // 10) for arrival in arrivals}
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{name}-center-{count}.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "request_id", "arrival_s", "input_tokens", "output_tokens",
            "source_line", "source_timestamp_s",
        ))
        writer.writeheader()
        for row, arrival in zip(chosen, arrivals):
            writer.writerow({
                "request_id": f"{name}-{row[3]}",
                "arrival_s": f"{arrival:.9f}",
                "input_tokens": row[1],
                "output_tokens": row[2],
                "source_line": row[3],
                "source_timestamp_s": f"{row[0]:.9f}",
            })
    metadata = {
        "source_name": name,
        "source_url": spec["url"],
        "source_sha256": observed_hash,
        "trace_sha256": sha256_file(output),
        "original_timestamp_unit": spec["timestamp_unit"],
        "selection": "central consecutive records by original row order",
        "length_filter": "none; all source rows have positive lengths",
        "time_rescaled": False,
        "arrival_semantics_for_proposed_replay": "decode-ready; prefill not modeled",
        "source_total_requests": len(rows),
        "source_line_first": chosen[0][3],
        "source_line_last": chosen[-1][3],
        "request_count": count,
        "native_time_span_s": span,
        "offered_rps": (count - 1) / span,
        "input_tokens": {
            "mean": statistics.fmean(inputs),
            "p50": percentile(inputs, 0.5),
            "p95": percentile(inputs, 0.95),
            "max": max(inputs),
        },
        "output_tokens": {
            "mean": statistics.fmean(outputs),
            "p50": percentile(outputs, 0.5),
            "p95": percentile(outputs, 0.95),
            "max": max(outputs),
        },
        "max_total_tokens_in_one_request": max(a + b for a, b in zip(inputs, outputs)),
        "input_ge_4096_count": sum(value >= 4096 for value in inputs),
        "output_ge_1024_count": sum(value >= 1024 for value in outputs),
        "both_thresholds_count": sum(
            a >= 4096 and b >= 1024 for a, b in zip(inputs, outputs)
        ),
        "max_interarrival_s": max(gaps),
        "occupied_10s_bins": len(bins),
        "total_10s_bins": math.ceil(span / 10),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-thinking", type=Path, required=True)
    parser.add_argument("--qwen-coder", type=Path, required=True)
    parser.add_argument("--mooncake-conversation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    args = parser.parse_args()
    if args.count < 65:
        parser.error("candidate count must exceed the proposed batch cap of 64")
    inputs = {
        "qwen-thinking": args.qwen_thinking,
        "qwen-coder": args.qwen_coder,
        "mooncake-conversation": args.mooncake_conversation,
    }
    for name, source in inputs.items():
        metadata = prepare(name, source, args.output_dir, args.count)
        print(name, metadata["source_line_first"], metadata["source_line_last"],
              f"{metadata['offered_rps']:.3f} RPS")


if __name__ == "__main__":
    main()
