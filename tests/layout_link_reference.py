"""Frozen phase-two link scheduler for differential regression tests.

Kept independent of the optimized reservation/reflow paths so tests compare
packet times and readiness, including intermediate event-list order.
"""
from __future__ import annotations
from collections import defaultdict
from elasticafd.config import ProjectConfig
from elasticafd.layout_aware import (LinkCalendar, LinkEvent, _us,
                                     _pending_preparations, _truncate_inflight_preparation)

def _reserve_link(calendar: LinkCalendar, full_duplex: bool, direction: str,
                  duration_us: float, earliest_us: float, kind: str,
                  job_id: int | None = None, size: int = 0) -> tuple[float, float]:
    """Place a packet in the first gap on its physical link timeline."""
    if duration_us <= 0:
        return earliest_us, 0.0
    start = earliest_us
    # The common half-duplex preparation stream appends packets in time
    # order. Avoid sorting the entire calendar for every 4 MiB packet.
    if (calendar.events and not full_duplex
            and calendar.events[-1].end_us <= earliest_us):
        calendar.events.append(LinkEvent(start, start + duration_us, direction,
                                         kind, job_id, size))
        return start, 0.0
    for event in sorted(calendar.events, key=lambda item: item.start_us):
        if full_duplex and event.direction != direction:
            continue
        if event.end_us <= start:
            continue
        if event.start_us >= start + duration_us:
            break
        start = event.end_us
    wait = sum(max(0.0, min(start, event.end_us) - max(earliest_us, event.start_us))
               for event in calendar.events if event.kind == "preparation"
               and (not full_duplex or event.direction == direction))
    calendar.events.append(LinkEvent(start, start + duration_us, direction,
                                     kind, job_id, size))
    return start, wait

def _reflow_preparations(calendar: LinkCalendar, config: ProjectConfig,
                         cutoff_us: float) -> None:
    """Foreground keeps priority over unsent background packets."""
    preserved = [event for event in calendar.events
                 if event.kind == "foreground" or event.start_us < cutoff_us]
    calendar.events = preserved
    retained_by_job: dict[int, list[LinkEvent]] = defaultdict(list)
    for event in preserved:
        if event.kind == "preparation":
            retained_by_job[event.job_id].append(event)
    prior_end: dict[str, float] = defaultdict(lambda: cutoff_us)
    # Completed jobs remain available for copy readiness and statistics, but
    # cannot delay packets starting at or after the pruning time. Preserve
    # the original all-job traversal for an explicitly retrospective query.
    jobs = (calendar.jobs if cutoff_us < calendar._pruned_through_us
            else _pending_preparations(calendar))
    for job in jobs.values():
        lane = job.direction if config.system.link.full_duplex else "shared"
        retained = retained_by_job[job.job_id]
        sent = job.completed_bytes + sum(event.bytes for event in retained)
        last_end = max((event.end_us for event in retained),
                       default=job.completed_end_us)
        if sent == job.size:
            job.ready_us = last_end + job.conversion_us
            prior_end[lane] = max(prior_end[lane], last_end)
            continue
        earliest = max(cutoff_us, job.earliest_us, prior_end[lane], last_end)
        size = job.size - sent
        bandwidth = (config.system.link.host_to_pim_bytes_s
                     if job.direction == "ga" else
                     config.system.link.pim_to_host_bytes_s)
        startup = (config.system.link.transfer_latency_us
                   * config.simulation.layout_launch_scale if sent == 0 else 0.0)
        duration = startup + _us(size / bandwidth)
        start, _ = _reserve_link(calendar, config.system.link.full_duplex,
                                 job.direction, duration, earliest,
                                 "preparation", job.job_id, size)
        calendar.events[-1].startup_us = startup
        earliest = start + duration
        job.ready_us = earliest + job.conversion_us
        prior_end[lane] = earliest

def _book_boundary(calendar: LinkCalendar, config: ProjectConfig,
                   start_us: float, ga_us: float, ag_us: float
                   ) -> tuple[float, float, float]:
    finish = start_us
    busy = wait = 0.0
    if (not _pending_preparations(calendar)
            and start_us >= calendar._pruned_through_us):
        # No background stream exists, so neither preemption nor rebuilding
        # its queue can change anything. Preserve the same packet reservations.
        for direction, duration in (("ga", ga_us), ("ag", ag_us)):
            if duration <= 0:
                continue
            earliest = start_us if config.system.link.full_duplex else finish
            packet_start, packet_wait = _reserve_link(
                calendar, config.system.link.full_duplex, direction, duration,
                earliest, "foreground")
            finish = max(finish, packet_start + duration)
            busy += duration
            wait += packet_wait
        return finish, busy, wait
    # The stage model knows direction-level aggregate payloads, not the
    # individual operator packet release times. GA then AG is a fixed tie-break.
    for direction, duration in (("ga", ga_us), ("ag", ag_us)):
        if duration <= 0:
            continue
        earliest = start_us if config.system.link.full_duplex else finish
        cutoff = earliest
        _truncate_inflight_preparation(calendar, config, cutoff, direction)
        calendar.events = [event for event in calendar.events
                           if event.kind == "foreground" or event.start_us < cutoff]
        packet_start, packet_wait = _reserve_link(
            calendar, config.system.link.full_duplex, direction, duration,
            earliest, "foreground")
        finish = max(finish, packet_start + duration)
        busy += duration
        wait += packet_wait
        _reflow_preparations(calendar, config, cutoff)
    return finish, busy, wait
