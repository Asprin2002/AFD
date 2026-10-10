"""Future-only Attention placement search for the device-contiguous pipeline.

The offline search is a bounded homogeneous window planner, not an exact
unrestricted oracle. Both it and the horizon policy are verified by the same
event DAG as the all-PIM control. Historical KV segments never move.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from elasticafd.dense_offload import DenseOffloadCost, DensePlan
from elasticafd.state import Location, RequestView
CONTROL = DensePlan()


@dataclass(frozen=True)
class Schedule:
    name: str
    # Each request has ordered (decode-token boundary, future-KV location) cuts.
    cuts: tuple[tuple[tuple[int, Location], ...], ...]
    planning_score_us: float = 0.0


def all_pim(batch: int) -> Schedule:
    return Schedule("control", tuple(() for _ in range(batch)))


def sampled_blocks(output_tokens: int, stride: int,
                   schedules: tuple[Schedule, ...]) -> list[tuple[int, int]]:
    boundaries = set(range(0, output_tokens, stride)) | {output_tokens}
    boundaries.update(token for schedule in schedules for cuts in schedule.cuts
                      for token, _ in cuts if 0 <= token <= output_tokens)
    ordered = sorted(boundaries)
    return [(right - left, (left + right - 1) // 2)
            for left, right in zip(ordered, ordered[1:]) if right > left]


def views(schedule: Schedule, input_tokens: int, generated: int) -> list[RequestView]:
    result = []
    for index, cuts in enumerate(schedule.cuts):
        pim = [input_tokens]
        gpu: list[int] = []
        previous = 0
        location = Location.PIM
        for boundary, target in cuts:
            if boundary > generated:
                break
            if boundary > previous:
                length = boundary - previous
                segments = pim if location is Location.PIM else gpu
                # Only adjacent runs on the same device may be coalesced.
                if location is Location.PIM and previous == 0:
                    segments[-1] += length
                else:
                    segments.append(length)
            previous = boundary
            location = target
        if generated > previous:
            length = generated - previous
            segments = pim if location is Location.PIM else gpu
            if location is Location.PIM and previous == 0:
                segments[-1] += length
            else:
                segments.append(length)
        result.append(RequestView(f"r{index}", tuple(pim), tuple(gpu), location))
    return result


def window(batch: int, count: int, start: int, end: int,
           output_tokens: int) -> Schedule:
    cuts = tuple(
        ((start, Location.GPU), (end, Location.PIM))
        if index < count and end < output_tokens else
        ((start, Location.GPU),) if index < count else ()
        for index in range(batch)
    )
    return Schedule(f"offline_window_n{count}_s{start}_e{end}", cuts)


def _score(model: DenseOffloadCost, schedule: Schedule, input_tokens: int,
           output_tokens: int, stride: int) -> float:
    if not model.fits(views(schedule, input_tokens, output_tokens)):
        return inf
    total = 0.0
    for weight, generated in sampled_blocks(output_tokens, stride, (schedule,)):
        state = views(schedule, input_tokens, generated)
        if not model.fits(state):
            return inf
        total += weight * model.resource_score(state, CONTROL)
    return total


def horizon_greedy(model: DenseOffloadCost, batch: int, input_tokens: int,
                   output_tokens: int, *, decision_interval: int = 64,
                   horizon: int = 256, min_residence: int = 128,
                   max_switches: int = 8, threshold: float = .002) -> Schedule:
    cuts: list[list[tuple[int, Location]]] = [[] for _ in range(batch)]
    last_switch = [-10**9] * batch
    locations = [Location.PIM] * batch
    for token in range(0, output_tokens, decision_interval):
        forecast = min(output_tokens, token + horizon)
        current = Schedule("current", tuple(tuple(part) for part in cuts))
        current_score = model.resource_score(views(current, input_tokens, forecast), CONTROL)
        best_score = current_score
        best_index = -1
        best_schedule: Schedule | None = None
        # Greedily accept individual tail-location changes; repeat at this
        # decision boundary until no profitable eligible change remains.
        for _ in range(min(max_switches, batch)):
            for index in range(batch):
                if token - last_switch[index] < min_residence:
                    continue
                trial = [list(part) for part in cuts]
                trial[index].append((token, locations[index].other))
                candidate = Schedule("candidate", tuple(tuple(part) for part in trial))
                projected = views(candidate, input_tokens, forecast)
                if not model.fits(projected) or not model.fits(
                    views(candidate, input_tokens, output_tokens)
                ):
                    continue
                score = model.resource_score(projected, CONTROL)
                if score < best_score:
                    best_score, best_index, best_schedule = score, index, candidate
            if best_schedule is None or (current_score - best_score) / max(current_score, 1e-9) < threshold:
                break
            cuts = [list(part) for part in best_schedule.cuts]
            locations[best_index] = locations[best_index].other
            last_switch[best_index] = token
            current_score = best_score
            best_score = current_score
            best_index, best_schedule = -1, None
    result = Schedule("horizon_greedy", tuple(tuple(part) for part in cuts))
    return Schedule(result.name, result.cuts,
                    _score(model, result, input_tokens, output_tokens, decision_interval))


def offline_windows(model: DenseOffloadCost, batch: int, input_tokens: int,
                    output_tokens: int, *, search_stride: int = 128,
                    grid_points: int = 5, finalists: int = 8,
                    min_residence: int = 128) -> list[Schedule]:
    knots = sorted({round(i * output_tokens / (grid_points - 1))
                    for i in range(grid_points)})
    scored: list[Schedule] = []
    for count in range(1, batch + 1):
        for start in knots[:-1]:
            for end in knots[1:]:
                if end - start < min_residence or end <= start:
                    continue
                candidate = window(batch, count, start, end, output_tokens)
                score = _score(model, candidate, input_tokens, output_tokens, search_stride)
                if score < inf:
                    scored.append(Schedule(candidate.name, candidate.cuts, score))
    scored.sort(key=lambda item: item.planning_score_us)
    refined = {item.name: item for item in scored[:finalists]}
    for seed in scored[:min(4, finalists)]:
        # Local coordinate refinement gives token-level cut points without
        # enumerating every possible (start, end) pair.
        label = seed.name.split("_")
        count, start, end = (int(label[-3][1:]), int(label[-2][1:]),
                             int(label[-1][1:]))
        best = seed
        step = max(1, output_tokens // (grid_points - 1) // 2)
        while step:
            improved = True
            while improved:
                improved = False
                for next_count in {count - 1, count, count + 1}:
                    if not 1 <= next_count <= batch:
                        continue
                    for next_start in {start - step, start, start + step}:
                        for next_end in {end - step, end, end + step}:
                            if (next_start < 0 or next_end > output_tokens
                                    or next_end - next_start < min_residence):
                                continue
                            candidate = window(batch, next_count, next_start,
                                               next_end, output_tokens)
                            score = _score(model, candidate, input_tokens,
                                           output_tokens, search_stride)
                            if score < best.planning_score_us:
                                best = Schedule(candidate.name, candidate.cuts, score)
                                count, start, end = next_count, next_start, next_end
                                improved = True
            step //= 2
        refined[best.name] = best
    return sorted(refined.values(), key=lambda item: item.planning_score_us)[:finalists]
