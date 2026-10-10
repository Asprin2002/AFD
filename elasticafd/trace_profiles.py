"""Workload statistics reported by CHIME Table III.

These profiles are an offline, version-stable fallback. They generate positive
heavy-tailed lengths from lognormal distributions parameterized by the reported
arithmetic mean and population standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from math import exp, log, sqrt
import random
from statistics import fmean, pstdev


@dataclass(frozen=True)
class LengthMoments:
    mean: float
    std: float


@dataclass(frozen=True)
class TraceProfile:
    name: str
    input_tokens: LengthMoments
    output_tokens: LengthMoments
    source: str


PAPER_TRACE_PROFILES: dict[str, TraceProfile] = {
    "openr1": TraceProfile(
        name="OpenR1-Math-220k",
        input_tokens=LengthMoments(96.0, 75.1),
        output_tokens=LengthMoments(12684.1, 8464.6),
        source="CHIME Table III",
    ),
    "openthoughts": TraceProfile(
        name="OpenThoughts-114k-math",
        input_tokens=LengthMoments(89.4, 66.7),
        output_tokens=LengthMoments(6366.7, 4662.9),
        source="CHIME Table III / generated_token_count",
    ),
    "dolphin": TraceProfile(
        name="Dolphin-r1",
        input_tokens=LengthMoments(201.9, 563.0),
        output_tokens=LengthMoments(3926.2, 4216.0),
        source="CHIME Table III",
    ),
    "dolphin_short": TraceProfile(
        name="Dolphin-short",
        input_tokens=LengthMoments(20.19, 56.30),
        output_tokens=LengthMoments(392.62, 421.60),
        source="CHIME: Dolphin-r1 lengths divided by 10",
    ),
}


def normalized_profile_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def get_trace_profile(name: str) -> TraceProfile:
    wanted = normalized_profile_name(name)
    for key, profile in PAPER_TRACE_PROFILES.items():
        if wanted in (normalized_profile_name(key), normalized_profile_name(profile.name)):
            return profile
    raise KeyError(
        f"unknown trace profile {name!r}; choices: "
        + ", ".join(sorted(PAPER_TRACE_PROFILES))
    )


def sample_lognormal_moments(
    rng: random.Random, moments: LengthMoments, minimum: int = 1
) -> int:
    if moments.std <= 0:
        return max(minimum, int(round(moments.mean)))
    variance_ratio = (moments.std / moments.mean) ** 2
    sigma2 = log(1.0 + variance_ratio)
    sigma = sqrt(sigma2)
    mu = log(moments.mean) - sigma2 / 2.0
    return max(minimum, int(round(rng.lognormvariate(mu, sigma))))


def sample_profile_lengths(
    profile_name: str, count: int, seed: int
) -> list[tuple[int, int]]:
    if count <= 0:
        raise ValueError("trace profile count must be positive")
    profile = get_trace_profile(profile_name)
    rng = random.Random(seed)

    def finite_sample(moments: LengthMoments) -> list[int]:
        if count == 1 or moments.std <= 0:
            return [max(1, int(round(moments.mean)))] * count
        zscores = [rng.gauss(0.0, 1.0) for _ in range(count)]
        target_cv = moments.std / moments.mean
        low, high = 0.0, 8.0
        for _ in range(80):
            sigma = (low + high) / 2.0
            peak = max(sigma * z for z in zscores)
            values = [exp(sigma * z - peak) for z in zscores]
            cv = pstdev(values) / fmean(values)
            if cv < target_cv:
                low = sigma
            else:
                high = sigma
        sigma = (low + high) / 2.0
        peak = max(sigma * z for z in zscores)
        values = [exp(sigma * z - peak) for z in zscores]
        scale = moments.mean / fmean(values)
        return [max(1, int(round(scale * value))) for value in values]

    inputs = finite_sample(profile.input_tokens)
    outputs = finite_sample(profile.output_tokens)
    return list(zip(inputs, outputs))


class ConditionalRemainingLength:
    """Empirical E[remaining length | a request has survived so far].

    The estimator only observes the number of tokens already generated. It
    never sees a request's target output length. This makes it a usable online
    baseline between a fixed horizon and an oracle predictor.
    """

    def __init__(self, profile_name: str, count: int = 10000, seed: int = 0) -> None:
        if count <= 0:
            raise ValueError("prediction sample count must be positive")
        self.profile_name = get_trace_profile(profile_name).name
        self.lengths = sorted(
            output for _, output in sample_profile_lengths(profile_name, count, seed)
        )
        self.suffix_sum = [0] * (len(self.lengths) + 1)
        for index in range(len(self.lengths) - 1, -1, -1):
            self.suffix_sum[index] = self.suffix_sum[index + 1] + self.lengths[index]

    def expected_remaining(self, generated_tokens: int) -> float | None:
        if generated_tokens < 0:
            raise ValueError("generated_tokens cannot be negative")
        # Active requests satisfy output_length > generated_tokens.
        index = bisect_right(self.lengths, generated_tokens)
        survivors = len(self.lengths) - index
        if survivors == 0:
            # The observation lies beyond the empirical support. Returning None
            # lets the policy fall back to its conservative fixed horizon.
            return None
        return (
            self.suffix_sum[index] - generated_tokens * survivors
        ) / survivors

    def expected_capped_remaining(
        self, generated_tokens: int, horizon: int
    ) -> float | None:
        """Return E[min(remaining, horizon) | request is still active]."""

        if horizon <= 0:
            raise ValueError("horizon must be positive")
        begin = bisect_right(self.lengths, generated_tokens)
        survivors = len(self.lengths) - begin
        if survivors == 0:
            return None
        end = bisect_right(self.lengths, generated_tokens + horizon)
        finishing_count = end - begin
        finishing_sum = self.suffix_sum[begin] - self.suffix_sum[end]
        long_count = len(self.lengths) - end
        capped_sum = (
            finishing_sum
            - generated_tokens * finishing_count
            + horizon * long_count
        )
        return capped_sum / survivors
