"""Event-scored, full-lifecycle Attention-only planning for a closed batch.

The offline policy searches one homogeneous count/start/end window over the
*whole* known decode lifetime. It is a bounded search, not an exact oracle.
The online-style policy reconsiders future KV every 64 generated tokens.
Both use the same device-contiguous event cost and never move historical KV.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from elasticafd.dense_offload import DenseOffloadCost, DensePlan
from elasticafd.state import Location
from tools.attention_strategy import Schedule, all_pim, sampled_blocks, views, window


CONTROL = DensePlan()


@dataclass(frozen=True)
class SearchResult:
    schedules: tuple[Schedule, ...]
    coarse_evaluations: int
    final_evaluations: int


def event_lifecycle_score(model: DenseOffloadCost, schedule: Schedule,
                          input_tokens: int, output_tokens: int,
                          stride: int) -> float:
    if not model.fits(views(schedule, input_tokens, output_tokens)):
        return inf
    total = 0.0
    for weight, generated in sampled_blocks(output_tokens, stride, (schedule,)):
        state = views(schedule, input_tokens, generated)
        if not model.fits(state):
            return inf
        total += weight * model.measure(state, CONTROL).cost.total_us
    return total


def _count_grid(batch: int) -> tuple[int, ...]:
    values = {1, batch}
    value = 2
    while value < batch:
        values.add(value)
        value *= 2
    values.update(max(1, round(batch * part / 4)) for part in (1, 2, 3))
    return tuple(sorted(value for value in values if value <= batch))


def _capacity_end(model: DenseOffloadCost, batch: int, count: int,
                  start: int, input_tokens: int, output_tokens: int) -> int:
    low, high = start, output_tokens
    while low < high:
        mid = (low + high + 1) // 2
        if model.fits(views(window(batch, count, start, mid, output_tokens),
                            input_tokens, output_tokens)):
            low = mid
        else:
            high = mid - 1
    return low


def offline_event_windows(model: DenseOffloadCost, batch: int,
                          input_tokens: int, output_tokens: int,
                          *, final_stride: int = 64,
                          finalists: int = 6) -> SearchResult:
    """Plan before decode using known B/I/O and event-scored lifetime cost.

    A sparse global count/start/end lattice is searched first, including the
    capacity-clipped end for every early-start count. Coordinate refinement
    then explores count and both cut positions. Exact sampled event replay
    ranks the finalists, control, and any supplied greedy schedule elsewhere.
    """
    coarse_stride = max(64, (output_tokens + 5) // 6)
    minimum = min(128, max(1, output_tokens // 4))
    starts = tuple(sorted({0, output_tokens // 4,
                           output_tokens // 2, 3 * output_tokens // 4}))
    score_cache: dict[tuple[tuple[tuple[int, Location], ...], ...], float] = {}

    def score(candidate: Schedule) -> float:
        cached = score_cache.get(candidate.cuts)
        if cached is None:
            cached = event_lifecycle_score(
                model, candidate, input_tokens, output_tokens, coarse_stride)
            score_cache[candidate.cuts] = cached
        return cached

    seeds: dict[str, Schedule] = {}
    early_capacity: list[Schedule] = []
    for count in _count_grid(batch):
        for start in starts:
            if start >= output_tokens:
                continue
            max_end = _capacity_end(model, batch, count, start,
                                    input_tokens, output_tokens)
            if max_end - start < minimum:
                continue
            for fraction in (0.5, 1.0):
                end = min(output_tokens, start + round((max_end - start) * fraction))
                if end - start < minimum:
                    continue
                candidate = window(batch, count, start, end, output_tokens)
                seeds[candidate.name] = candidate
                if start == 0 and fraction == 1.0:
                    early_capacity.append(candidate)
    ranked = sorted(((score(item), item) for item in seeds.values()),
                    key=lambda pair: (pair[0], pair[1].name))
    ranked = [item for value, item in ranked if value < inf]
    # Early, capacity-limited windows must survive proxy/quantization ranking:
    # this is precisely where the prior resource-score shortlist failed.
    selected = {item.name: item for item in ranked[:finalists]}
    for item in early_capacity:
        selected[item.name] = item

    refined = dict(selected)
    for seed in ranked[:min(3, finalists)]:
        current = seed
        best = score(seed)
        token_step = max(1, output_tokens // 8)
        count_step = max(1, batch // 4)
        while token_step or count_step:
            candidates = []
            labels = current.name.split("_")
            count, start, end = (int(labels[-3][1:]), int(labels[-2][1:]),
                                 int(labels[-1][1:]))
            for dc in (-count_step, 0, count_step):
                next_count = count + dc
                if not 1 <= next_count <= batch:
                    continue
                for ds in (-token_step, 0, token_step):
                    next_start = start + ds
                    if not 0 <= next_start < output_tokens:
                        continue
                    for de in (-token_step, 0, token_step):
                        next_end = end + de
                        if (next_end > output_tokens or
                                next_end - next_start < minimum):
                            continue
                        candidates.append(window(batch, next_count, next_start,
                                                 next_end, output_tokens))
            improved = False
            for candidate in candidates:
                value = score(candidate)
                if (value, candidate.name) < (best, current.name):
                    current, best, improved = candidate, value, True
            if not improved:
                token_step //= 2
                count_step //= 2
            elif token_step == 0 and count_step == 0:
                break
        refined[current.name] = current

    final = sorted(
        ((event_lifecycle_score(model, item, input_tokens, output_tokens,
                                final_stride), item)
         for item in refined.values()),
        key=lambda pair: (pair[0], pair[1].name),
    )
    return SearchResult(tuple(item for value, item in final if value < inf),
                        len(score_cache), len(refined))


def event_horizon_greedy(model: DenseOffloadCost, batch: int,
                         input_tokens: int, output_tokens: int,
                         *, decision_interval: int = 64,
                         horizon: int = 256,
                         min_residence: int = 128,
                         threshold: float = 0.002) -> Schedule:
    """Every 64 tokens, greedily choose one exchangeable future-KV group cut.

    Uses only the current state and a short forecast, not an offline prebuilt
    schedule. Output length is known, as stipulated by this experiment.
    """
    cuts: list[list[tuple[int, Location]]] = [[] for _ in range(batch)]
    last_switch = [-10**9] * batch
    locations = [Location.PIM] * batch
    for token in range(0, output_tokens, decision_interval):
        forecast = min(output_tokens, token + horizon)
        current = Schedule("current", tuple(tuple(part) for part in cuts))
        current_us = model.measure(views(current, input_tokens, forecast),
                                   CONTROL).cost.total_us
        best_us, best_schedule, best_indices = current_us, None, ()
        # Identical histories are exchangeable; try group sizes, not all 2^B
        # assignments. At most one group decision is committed per epoch.
        groups: dict[tuple[Location, tuple[tuple[int, Location], ...]], list[int]] = {}
        for index in range(batch):
            if token - last_switch[index] >= min_residence:
                groups.setdefault((locations[index], tuple(cuts[index])), []).append(index)
        for (location, _), indices in groups.items():
            counts = {1, len(indices)}
            n = 2
            while n < len(indices):
                counts.add(n)
                n *= 2
            counts.update(max(1, round(len(indices) * part / 4))
                          for part in (1, 2, 3))
            for count in sorted(counts):
                selected = tuple(indices[:count])
                trial = [list(part) for part in cuts]
                for index in selected:
                    trial[index].append((token, location.other))
                candidate = Schedule("candidate", tuple(tuple(part) for part in trial))
                projected = views(candidate, input_tokens, forecast)
                if not model.fits(projected) or not model.fits(
                    views(candidate, input_tokens, output_tokens)
                ):
                    continue
                value = model.measure(projected, CONTROL).cost.total_us
                if value < best_us:
                    best_us, best_schedule, best_indices = value, candidate, selected
        if (best_schedule is not None and
                (current_us - best_us) / max(current_us, 1e-9) >= threshold):
            cuts = [list(part) for part in best_schedule.cuts]
            for index in best_indices:
                locations[index] = locations[index].other
                last_switch[index] = token
    return Schedule("event_horizon_greedy",
                    tuple(tuple(part) for part in cuts))
