"""Numerically stable partial-softmax reference semantics.

The timing simulator does not execute model tensors, so this module provides a
small executable specification for the values exchanged by split Attention.
It is intentionally dependency-free and is used as a correctness gate before
performance experiments are allowed to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PartialSoftmaxState:
    """Stable softmax state for one KV partition.

    ``weighted_value`` stores the unnormalised vector ``u``.  Empty partitions
    use ``max_score=-inf``, ``denominator=0`` and a zero vector.
    """

    max_score: float
    denominator: float
    weighted_value: tuple[float, ...]

    @property
    def is_empty(self) -> bool:
        return self.denominator == 0.0


def partial_softmax(
    scores: Sequence[float],
    values: Sequence[Sequence[float]],
    *,
    value_dim: int | None = None,
) -> PartialSoftmaxState:
    """Compute ``(max, denominator, weighted value)`` for one partition."""

    if len(scores) != len(values):
        raise ValueError("scores and values must have the same token count")
    if value_dim is not None and value_dim < 0:
        raise ValueError("value_dim cannot be negative")
    if not scores:
        dimension = 0 if value_dim is None else value_dim
        return PartialSoftmaxState(float("-inf"), 0.0, (0.0,) * dimension)

    dimension = len(values[0])
    if value_dim is not None and dimension != value_dim:
        raise ValueError("value vectors do not match value_dim")
    if any(len(vector) != dimension for vector in values):
        raise ValueError("all value vectors must have the same dimension")
    if any(not isfinite(float(score)) for score in scores):
        raise ValueError("partial_softmax requires finite scores")

    maximum = max(float(score) for score in scores)
    weights = [exp(float(score) - maximum) for score in scores]
    denominator = sum(weights)
    weighted = tuple(
        sum(weight * float(vector[index]) for weight, vector in zip(weights, values))
        for index in range(dimension)
    )
    return PartialSoftmaxState(maximum, denominator, weighted)


def merge_partial_softmax(
    states: Iterable[PartialSoftmaxState],
) -> PartialSoftmaxState:
    """Merge any number of partition states without reconstructing logits."""

    materialized = tuple(states)
    if not materialized:
        return PartialSoftmaxState(float("-inf"), 0.0, ())
    dimensions = {len(state.weighted_value) for state in materialized}
    if len(dimensions) != 1:
        raise ValueError("partial states must use one value dimension")
    if any(state.denominator < 0.0 for state in materialized):
        raise ValueError("partial denominator cannot be negative")

    active = tuple(state for state in materialized if not state.is_empty)
    dimension = dimensions.pop()
    if not active:
        return PartialSoftmaxState(float("-inf"), 0.0, (0.0,) * dimension)
    if any(not isfinite(state.max_score) for state in active):
        raise ValueError("non-empty partial states require finite maxima")

    maximum = max(state.max_score for state in active)
    scales = [exp(state.max_score - maximum) for state in active]
    denominator = sum(
        scale * state.denominator for scale, state in zip(scales, active)
    )
    weighted = tuple(
        sum(
            scale * state.weighted_value[index]
            for scale, state in zip(scales, active)
        )
        for index in range(dimension)
    )
    return PartialSoftmaxState(maximum, denominator, weighted)


def normalize_partial_softmax(state: PartialSoftmaxState) -> tuple[float, ...]:
    """Return the normalized Attention vector represented by ``state``."""

    if state.denominator <= 0.0:
        raise ValueError("cannot normalize an empty partial-softmax state")
    return tuple(value / state.denominator for value in state.weighted_value)


def stable_attention(
    scores: Sequence[float], values: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    """Reference unsplit Attention used by numerical equivalence tests."""

    return normalize_partial_softmax(partial_softmax(scores, values))
