"""Static-batch and trace-serving workload drivers."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import WorkloadConfig
from .state import RequestState
from .trace_profiles import sample_profile_lengths


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    arrival_us: float
    input_tokens: int
    output_tokens: int

    def to_state(self) -> RequestState:
        return RequestState(
            request_id=self.request_id,
            arrival_us=self.arrival_us,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


def static_batch_requests(config: WorkloadConfig) -> list[WorkloadRequest]:
    assert config.batch_size is not None
    assert config.input_tokens is not None
    assert config.output_tokens is not None
    return [
        WorkloadRequest(
            request_id=f"static-{index}",
            arrival_us=0.0,
            input_tokens=config.input_tokens,
            output_tokens=config.output_tokens,
        )
        for index in range(config.batch_size)
    ]


def _first(row: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _request_from_mapping(row: Mapping[str, Any], index: int) -> WorkloadRequest:
    request_id = str(_first(row, ("request_id", "id"), f"trace-{index}"))
    if _first(row, ("arrival_us",)) is not None:
        arrival_us = float(row["arrival_us"])
    elif _first(row, ("arrival_ms",)) is not None:
        arrival_us = float(row["arrival_ms"]) * 1e3
    else:
        arrival_us = float(_first(row, ("arrival_s", "arrival_time"), 0.0)) * 1e6
    input_tokens = int(_first(row, ("input_tokens", "prompt_tokens", "input_length")))
    output_tokens = int(
        _first(row, ("output_tokens", "decode_tokens", "output_length"))
    )
    return WorkloadRequest(request_id, arrival_us, input_tokens, output_tokens)


def _read_trace(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix in (".jsonl", ".ndjson"):
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, Mapping):
            value = value.get("requests", [])
        if not isinstance(value, list):
            raise TypeError("JSON trace must be a list or contain a requests list")
        return value
    raise ValueError(f"unsupported trace format {suffix}; use CSV, JSON, or JSONL")


def _sample_length(rng: random.Random, raw: Any, default: int) -> int:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return max(1, int(raw))
    values = dict(raw)
    mean = float(values.get("mean", default))
    std = float(values.get("std", 0.0))
    minimum = int(values.get("min", 1))
    maximum = int(values.get("max", max(minimum, int(mean + 4 * std))))
    distribution = values.get("distribution", "normal")
    if distribution == "normal":
        sample = rng.gauss(mean, std)
    elif distribution == "lognormal":
        # Here mean/std parameterize the underlying normal distribution. This
        # explicit convention avoids pretending to infer distribution moments.
        sample = rng.lognormvariate(mean, std)
    else:
        raise ValueError("synthetic length distribution must be normal or lognormal")
    return max(minimum, min(maximum, int(round(sample))))


def _synthetic_requests(raw: Mapping[str, Any]) -> list[WorkloadRequest]:
    values = dict(raw)
    count = int(values.get("count", 100))
    arrival_rate_rps = float(values.get("arrival_rate_rps", 1.0))
    seed = int(values.get("seed", 0))
    profile_name = values.get("profile")
    classes_raw = values.get("classes")
    arrival_pattern = str(values.get("arrival_pattern", "poisson"))
    burst_size = int(values.get("burst_size", 4))
    input_config = values.get("input_tokens", 128)
    output_config = values.get("output_tokens", 1024)
    if count <= 0:
        raise ValueError("synthetic count must be positive")
    if arrival_rate_rps < 0:
        raise ValueError("synthetic arrival_rate_rps cannot be negative")
    if arrival_pattern not in ("poisson", "periodic", "burst"):
        raise ValueError(
            "synthetic arrival_pattern must be poisson, periodic, or burst"
        )
    if arrival_pattern == "burst" and burst_size <= 0:
        raise ValueError("synthetic burst_size must be positive")
    if profile_name is not None and classes_raw is not None:
        raise ValueError("synthetic profile and classes are mutually exclusive")

    classes: tuple[tuple[float, Any, Any], ...] = ()
    if classes_raw is not None:
        if not isinstance(classes_raw, list) or not classes_raw:
            raise ValueError("synthetic classes must be a non-empty list")
        parsed: list[tuple[float, Any, Any]] = []
        for index, raw_class in enumerate(classes_raw):
            if not isinstance(raw_class, Mapping):
                raise TypeError(f"synthetic class {index} must be a mapping")
            class_values = dict(raw_class)
            weight = float(class_values.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"synthetic class {index} weight must be positive")
            if "input_tokens" not in class_values or "output_tokens" not in class_values:
                raise ValueError(
                    f"synthetic class {index} requires input_tokens and output_tokens"
                )
            parsed.append(
                (weight, class_values["input_tokens"], class_values["output_tokens"])
            )
        classes = tuple(parsed)

    rng = random.Random(seed)
    profile_lengths = (
        sample_profile_lengths(str(profile_name), count, seed)
        if profile_name is not None
        else None
    )
    now_us = 0.0
    requests: list[WorkloadRequest] = []
    for index in range(count):
        if index and arrival_rate_rps > 0:
            if arrival_pattern == "poisson":
                now_us += rng.expovariate(arrival_rate_rps) * 1e6
            elif arrival_pattern == "periodic":
                now_us += 1e6 / arrival_rate_rps
            elif index % burst_size == 0:
                # Bursts have the requested long-run average offered rate,
                # while all requests within a burst share a timestamp.
                now_us += burst_size * 1e6 / arrival_rate_rps

        if classes:
            draw = rng.random() * sum(item[0] for item in classes)
            cumulative = 0.0
            selected_class = classes[-1]
            for item in classes:
                cumulative += item[0]
                if draw < cumulative:
                    selected_class = item
                    break
            _, selected_input, selected_output = selected_class
        else:
            selected_input = input_config
            selected_output = output_config
        requests.append(
            WorkloadRequest(
                request_id=f"synthetic-{index}",
                arrival_us=now_us,
                input_tokens=(
                    profile_lengths[index][0]
                    if profile_lengths is not None
                    else _sample_length(rng, selected_input, 128)
                ),
                output_tokens=(
                    profile_lengths[index][1]
                    if profile_lengths is not None
                    else _sample_length(rng, selected_output, 1024)
                ),
            )
        )
    return requests


def trace_serving_requests(
    config: WorkloadConfig, base_dir: Path | None = None
) -> list[WorkloadRequest]:
    rows: list[Mapping[str, Any]] = []
    if config.trace_file:
        path = Path(config.trace_file)
        if not path.is_absolute():
            path = (base_dir or Path.cwd()) / path
        rows.extend(_read_trace(path.resolve()))
    rows.extend(config.requests)

    if rows:
        requests = [_request_from_mapping(row, index) for index, row in enumerate(rows)]
    elif config.synthetic:
        requests = _synthetic_requests(config.synthetic)
    else:
        raise ValueError(
            "trace_serving requires trace_file, inline requests, or synthetic config"
        )
    requests.sort(key=lambda request: (request.arrival_us, request.request_id))
    if config.max_requests is not None:
        requests = requests[: config.max_requests]
    ids = [request.request_id for request in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("request IDs in the workload must be unique")
    return requests


def closed_batch_requests(
    config: WorkloadConfig, base_dir: Path | None = None
) -> list[WorkloadRequest]:
    requests = trace_serving_requests(config, base_dir)
    if any(request.arrival_us != 0.0 for request in requests):
        raise ValueError("closed_batch requires every request to arrive at time zero")
    return requests


def build_workload(
    config: WorkloadConfig, base_dir: Path | None = None
) -> list[WorkloadRequest]:
    if config.type == "static_batch":
        return static_batch_requests(config)
    if config.type == "closed_batch":
        return closed_batch_requests(config, base_dir)
    return trace_serving_requests(config, base_dir)
