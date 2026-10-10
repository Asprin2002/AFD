#!/usr/bin/env python3
"""Select a bounded, capacity-screened 512-request Qwen Coder pilot trace.

The selected requests remain consecutive, with native lengths and arrival gaps.
The static capacity screen only makes a 64-request batch possible; it does not
assert that the replay will reach or sustain 64 active requests.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from itertools import accumulate
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elasticafd.dense_offload import DenseOffloadCost
from tools.prepare_burstgpt_trace import sha256_file
from tools.run_full_joint_trace import config_for, legal_plans


SOURCE_URL = (
    "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/"
    "blob/main/qwen_coder_blksz_16.jsonl"
)
SOURCE_SHA256 = "3d74974cebb7bccb69b1ca6ea210a44d9a3033443c0271ed015327284dea1aff"
COUNT = 512
BATCH_CAP = 64
MAX_SINGLE_OUTPUT = 8192
LONG_INPUT = 4096
LONG_OUTPUT = 1024


def rolling_max(values: list[int], width: int) -> list[int]:
    queue: deque[int] = deque()
    maxima = []
    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if queue[0] <= index - width:
            queue.popleft()
        if index >= width - 1:
            maxima.append(values[queue[0]])
    return maxima


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[int(fraction * (len(values) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observed = sha256_file(args.source)
    if observed != SOURCE_SHA256:
        raise ValueError(f"Qwen Coder source checksum mismatch: {observed}")

    records = []
    with args.source.open("r", encoding="utf-8") as stream:
        for source_line, line in enumerate(stream, 1):
            record = json.loads(line)
            timestamp = float(record["timestamp"])
            input_tokens = int(record["input_length"])
            output_tokens = int(record["output_length"])
            if input_tokens <= 0 or output_tokens <= 0:
                raise ValueError(f"nonpositive length at line {source_line}")
            records.append((timestamp, input_tokens, output_tokens, source_line))
    if len(records) < COUNT or any(
        left[0] > right[0] for left, right in zip(records, records[1:])
    ):
        raise ValueError("source has insufficient records or unordered arrivals")

    config = config_for(args.output, max_batch_size=BATCH_CAP, segment_setup_us=1.0)
    cost = DenseOffloadCost(
        config, legal_plans("joint"), allow_conditional_ffn=True
    )
    kv_bytes = config.model.kv_bytes_per_token
    pim_token_cap = math.floor(cost.pim_kv_capacity_bytes / kv_bytes)
    combined_token_cap = math.floor(
        (cost.pim_kv_capacity_bytes + cost.gpu_kv_capacity_bytes) / kv_bytes
    )

    timestamps = [row[0] for row in records]
    inputs = [row[1] for row in records]
    outputs = [row[2] for row in records]
    input_prefix = [0, *accumulate(inputs)]
    output_prefix = [0, *accumulate(outputs)]
    long_prefix = [0, *accumulate(
        int(a >= LONG_INPUT and b >= LONG_OUTPUT)
        for a, b in zip(inputs, outputs)
    )]
    group_input = [
        input_prefix[index + BATCH_CAP] - input_prefix[index]
        for index in range(len(records) - BATCH_CAP + 1)
    ]
    group_total = [
        group_input[index]
        + output_prefix[index + BATCH_CAP] - output_prefix[index]
        for index in range(len(group_input))
    ]
    groups_per_window = COUNT - BATCH_CAP + 1
    max_group_input = rolling_max(group_input, groups_per_window)
    max_group_total = rolling_max(group_total, groups_per_window)
    max_output = rolling_max(outputs, COUNT)

    candidates = []
    for start in range(len(records) - COUNT + 1):
        end = start + COUNT
        span = timestamps[end - 1] - timestamps[start]
        if (span <= 0 or max_output[start] > MAX_SINGLE_OUTPUT
                or max_group_input[start] > pim_token_cap
                or max_group_total[start] > combined_token_cap):
            continue
        long_count = long_prefix[end] - long_prefix[start]
        mean_output = (output_prefix[end] - output_prefix[start]) / COUNT
        offered_rps = (COUNT - 1) / span
        candidates.append((
            (long_count, mean_output, offered_rps, -start), start
        ))
    if not candidates:
        raise RuntimeError("no 512-request window passes the pilot screen")
    _, start = max(candidates)
    selected = records[start:start + COUNT]
    first_timestamp = selected[0][0]
    arrivals = [row[0] - first_timestamp for row in selected]
    selected_inputs = [row[1] for row in selected]
    selected_outputs = [row[2] for row in selected]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "request_id", "arrival_s", "input_tokens", "output_tokens",
            "source_line", "source_timestamp_s",
        ))
        writer.writeheader()
        for row, arrival in zip(selected, arrivals):
            writer.writerow({
                "request_id": f"qwen-coder-{row[3]}",
                "arrival_s": f"{arrival:.9f}",
                "input_tokens": row[1],
                "output_tokens": row[2],
                "source_line": row[3],
                "source_timestamp_s": f"{row[0]:.9f}",
            })

    metadata = {
        "source_url": SOURCE_URL,
        "source_sha256": observed,
        "trace_sha256": sha256_file(args.output),
        "selection": (
            "among all consecutive 512-record windows satisfying the static "
            "64-record KV capacity screen and max output <= 8192, maximize "
            "the number with input >= 4096 and output >= 1024; then mean "
            "output, offered RPS, and earliest source position"
        ),
        "source_total_requests": len(records),
        "source_line_first": selected[0][3],
        "source_line_last": selected[-1][3],
        "request_count": COUNT,
        "native_time_span_s": arrivals[-1],
        "offered_rps": (COUNT - 1) / arrivals[-1],
        "time_rescaled": False,
        "token_lengths_modified": False,
        "max_batch_size_for_capacity_screen": BATCH_CAP,
        "max_single_output_tokens_for_pilot": MAX_SINGLE_OUTPUT,
        "pim_kv_token_capacity": pim_token_cap,
        "combined_kv_token_capacity": combined_token_cap,
        "max_64_request_input_tokens": max_group_input[start],
        "max_64_request_input_plus_output_tokens": max_group_total[start],
        "input_tokens": {
            "mean": statistics.fmean(selected_inputs),
            "p50": percentile(selected_inputs, 0.5),
            "p95": percentile(selected_inputs, 0.95),
            "max": max(selected_inputs),
        },
        "output_tokens": {
            "mean": statistics.fmean(selected_outputs),
            "p50": percentile(selected_outputs, 0.5),
            "p95": percentile(selected_outputs, 0.95),
            "max": max(selected_outputs),
            "sum": sum(selected_outputs),
        },
        "both_input_output_ge_1024": sum(
            a >= 1024 and b >= 1024
            for a, b in zip(selected_inputs, selected_outputs)
        ),
        "both_input_ge_4096_output_ge_1024": sum(
            a >= LONG_INPUT and b >= LONG_OUTPUT
            for a, b in zip(selected_inputs, selected_outputs)
        ),
        "max_interarrival_s": max(
            right - left for left, right in zip(arrivals, arrivals[1:])
        ),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: source lines {selected[0][3]}–{selected[-1][3]}, "
        f"{arrivals[-1]:.3f}s, {metadata['output_tokens']['sum']} output tokens"
    )


if __name__ == "__main__":
    main()
