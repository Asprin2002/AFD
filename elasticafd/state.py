"""Runtime request state and segmented KV-cache placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class Location(str, Enum):
    GPU = "gpu"
    PIM = "pim"

    @property
    def other(self) -> "Location":
        return Location.PIM if self is Location.GPU else Location.GPU


@dataclass
class KVSegment:
    location: Location
    start_token: int
    end_token: int

    @property
    def tokens(self) -> int:
        return self.end_token - self.start_token


@dataclass
class RequestState:
    request_id: str
    arrival_us: float
    input_tokens: int
    output_tokens: int
    gpu_replica: int = 0
    generated_tokens: int = 0
    tail_location: Location = Location.PIM
    segments: list[KVSegment] = field(default_factory=list)
    admitted_us: float | None = None
    first_token_us: float | None = None
    finish_us: float | None = None
    last_switch_iteration: int = -10**18
    switch_count: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens <= 0:
            raise ValueError(f"request {self.request_id}: input_tokens must be positive")
        if self.output_tokens <= 0:
            raise ValueError(f"request {self.request_id}: output_tokens must be positive")
        if self.gpu_replica < 0:
            raise ValueError(f"request {self.request_id}: gpu_replica cannot be negative")
        if not self.segments:
            self.segments.append(KVSegment(Location.PIM, 0, self.input_tokens))

    @property
    def context_tokens(self) -> int:
        return self.input_tokens + self.generated_tokens

    @property
    def remaining_tokens(self) -> int:
        return self.output_tokens - self.generated_tokens

    @property
    def is_finished(self) -> bool:
        return self.generated_tokens >= self.output_tokens

    def tokens_at(self, location: Location) -> int:
        return sum(segment.tokens for segment in self.segments if segment.location is location)

    def segment_lengths_at(self, location: Location) -> tuple[int, ...]:
        return tuple(
            segment.tokens for segment in self.segments if segment.location is location
        )

    @property
    def tail_segment_tokens(self) -> int:
        if self.segments and self.segments[-1].location is self.tail_location:
            return self.segments[-1].tokens
        return 0

    def switch_tail(self, location: Location, iteration: int) -> bool:
        if location is self.tail_location:
            return False
        self.tail_location = location
        self.last_switch_iteration = iteration
        self.switch_count += 1
        return True

    def append_generated_token(self) -> None:
        token_index = self.context_tokens
        if self.segments and self.segments[-1].location is self.tail_location:
            self.segments[-1].end_token += 1
        else:
            self.segments.append(
                KVSegment(self.tail_location, token_index, token_index + 1)
            )
        self.generated_tokens += 1
        if sum(segment.tokens for segment in self.segments) != self.context_tokens:
            raise AssertionError("KV segment lengths no longer match request context")


@dataclass(frozen=True)
class RequestView:
    """Read-only placement snapshot consumed by the cost model.

    A projected view may include hypothetical future tokens.  Keeping it
    separate from RequestState prevents the online policy search from mutating
    simulation state while evaluating alternatives.
    """

    request_id: str
    pim_segment_lengths: tuple[int, ...]
    gpu_segment_lengths: tuple[int, ...]
    tail_location: Location
    gpu_replica: int = 0

    @property
    def pim_tokens(self) -> int:
        return sum(self.pim_segment_lengths)

    @property
    def gpu_tokens(self) -> int:
        return sum(self.gpu_segment_lengths)

    @property
    def context_tokens(self) -> int:
        return self.pim_tokens + self.gpu_tokens

    @property
    def is_split(self) -> bool:
        return self.pim_tokens > 0 and self.gpu_tokens > 0


def make_request_view(
    request: RequestState,
    *,
    projected_tokens: int = 0,
    projected_location: Location | None = None,
) -> RequestView:
    if projected_tokens < 0:
        raise ValueError("projected_tokens cannot be negative")
    location = projected_location or request.tail_location
    pim = list(request.segment_lengths_at(Location.PIM))
    gpu = list(request.segment_lengths_at(Location.GPU))

    if projected_tokens:
        target = pim if location is Location.PIM else gpu
        # The global last segment can only be extended if it is at the projected
        # tail location; otherwise a new physical segment is created.
        if request.segments and request.segments[-1].location is location and target:
            target[-1] += projected_tokens
        else:
            target.append(projected_tokens)

    return RequestView(
        request_id=request.request_id,
        pim_segment_lengths=tuple(pim),
        gpu_segment_lengths=tuple(gpu),
        tail_location=location,
        gpu_replica=request.gpu_replica,
    )


def make_request_view_with_runs(
    request: RequestState,
    projected_runs: Sequence[tuple[Location, int]],
) -> RequestView:
    """Project several immutable future KV runs without mutating ``request``.

    Runs are appended in decode order. Adjacent runs on the same device are
    coalesced exactly as :meth:`RequestState.append_generated_token` would do.
    This is used by lifetime planning, where the next decision block has a
    chosen device while later, still-undecided blocks use a reference device.
    """

    pim = list(request.segment_lengths_at(Location.PIM))
    gpu = list(request.segment_lengths_at(Location.GPU))
    last_location = (
        request.segments[-1].location if request.segments else request.tail_location
    )
    projected_tail = request.tail_location

    for location, tokens in projected_runs:
        if tokens < 0:
            raise ValueError("projected run length cannot be negative")
        if tokens == 0:
            continue
        target = pim if location is Location.PIM else gpu
        if last_location is location and target:
            target[-1] += tokens
        else:
            target.append(tokens)
        last_location = location
        projected_tail = location

    return RequestView(
        request_id=request.request_id,
        pim_segment_lengths=tuple(pim),
        gpu_segment_lengths=tuple(gpu),
        tail_location=projected_tail,
        gpu_replica=request.gpu_replica,
    )


def views_for_requests(requests: Iterable[RequestState]) -> list[RequestView]:
    return [make_request_view(request) for request in requests]
