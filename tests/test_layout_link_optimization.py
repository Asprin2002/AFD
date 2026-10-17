from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from random import Random
import unittest
from unittest.mock import patch

import elasticafd.layout_aware as layout
from elasticafd.layout_cost_cache import reuse_costs
from elasticafd.workload import WorkloadRequest
from tests import layout_link_reference as reference
from tests.test_layout_aware import tiny_config
from tests.test_layout_cost_cache import world_with_requests


def assert_world_equal(test, actual, expected):
    for name, value in vars(actual).items():
        if name != 'catalog':  # PrefixCatalog intentionally has identity equality.
            test.assertEqual(value, getattr(expected, name), name)


class LinkOptimizationTests(unittest.TestCase):
    def test_mixed_packets_match_frozen_scheduler_after_every_operation(self):
        # Include nonzero startup/conversion, both lanes, packet boundaries,
        # pruning, gaps, future foreground reservations and completed jobs.
        for duplex in (False, True):
            for seed in range(12):
                rng = Random(seed)
                cfg = tiny_config()
                cfg = replace(cfg, system=replace(cfg.system, link=replace(
                    cfg.system.link, full_duplex=duplex, host_to_pim_gbps=0.003,
                    pim_to_host_gbps=0.002, transfer_latency_us=0.125)),
                    simulation=replace(cfg.simulation, layout_preparation_packet_bytes=4))
                world = layout.LayoutWorld(cfg, layout.PrefixCatalog(()))
                old = deepcopy(world.link_calendar)
                now = 0.0
                for index in range(80):
                    if index % 9 == 0:
                        world.clock_us = now
                        layout._schedule_preparation(world, rng.randrange(1, 40),
                                                     rng.choice((layout.GPU, layout.PIM)),
                                                     rng.choice((0, 0.25, 3)))
                        # Queue creation is unchanged in semantics. Start the
                        # independent booking comparison from the same queue.
                        job_id = world.link_calendar.next_job_id - 1
                        job = deepcopy(world.link_calendar.jobs[job_id])
                        old.jobs[job_id] = job
                        old.next_job_id += 1
                        layout._prune_link(old, now)
                        full = cfg.system.link.full_duplex
                        earliest = max(job.earliest_us, max((e.end_us for e in old.events
                            if e.kind == 'preparation' and (not full or e.direction == job.direction)),
                            default=job.earliest_us))
                        bandwidth = (cfg.system.link.host_to_pim_bytes_s if job.direction == 'ga'
                                     else cfg.system.link.pim_to_host_bytes_s)
                        startup = cfg.system.link.transfer_latency_us * cfg.simulation.layout_launch_scale
                        duration = startup + layout._us(job.size / bandwidth)
                        start, _ = reference._reserve_link(old, full, job.direction, duration,
                                                          earliest, 'preparation', job_id, job.size)
                        old.events[-1].startup_us = startup
                        job.ready_us = start + duration + job.conversion_us
                        if old._pending_jobs is not None:
                            old._pending_jobs[job_id] = job
                            old._known_jobs = len(old.jobs)
                    if index % 5 == 0:
                        for calendar in (world.link_calendar, old):
                            layout._prune_link(calendar, now)
                    ga, ag = rng.choice((0, 0.25, 2, 9)), rng.choice((0, 0.125, 4, 11))
                    expected = reference._book_boundary(old, cfg, now, ga, ag)
                    actual = layout._book_boundary(world.link_calendar, cfg, now, ga, ag)
                    self.assertEqual(actual, expected, (duplex, seed, index))
                    self.assertEqual(world.link_calendar, old, (duplex, seed, index))
                    now = actual[0] + rng.choice((0, 0.125, 5))

    def test_idle_fast_path_preserves_existing_events_and_duplex_overlap(self):
        for duplex in (False, True):
            cfg = tiny_config()
            cfg = replace(cfg, system=replace(cfg.system, link=replace(cfg.system.link,
                                                                      full_duplex=duplex)))
            for events in ([], [layout.LinkEvent(1, 3, 'ag', 'foreground')],
                           [layout.LinkEvent(10, 20, 'ga', 'foreground'),
                            layout.LinkEvent(0, 1, 'ag', 'foreground')]):
                for now in (0, 3, 12, 25):
                    old = layout.LinkCalendar(events=deepcopy(events))
                    new = deepcopy(old)
                    self.assertEqual(layout._book_boundary(new, cfg, now, 2, 3),
                                     reference._book_boundary(old, cfg, now, 2, 3))
                    self.assertEqual(new, old)

    def test_full_forecast_with_copy_matches_original_link_scheduler(self):
        for duplex in (False, True):
            world = world_with_requests(shared=True, layers=10)
            world.config = replace(world.config, system=replace(world.config.system,
                link=replace(world.config.system.link, full_duplex=duplex)))
            ids = tuple(world.requests['a'].block_ids[layer][0]
                        for layer in range(world.config.model.layers))
            layout.apply_action(world, layout.LayoutAction('copy', ('a', 'b', 'c'), ids, layout.GPU),
                                baseline_us=1)
            old = deepcopy(world)
            remaining = {'a': 4, 'b': 7, 'c': 9}
            with patch.object(layout, '_book_boundary', reference._book_boundary):
                expected = layout._forecast_total(old, 12, remaining)
            actual = layout._forecast_total(world, 12, remaining)
            self.assertEqual(actual, expected)
            assert_world_equal(self, world, old)


class EvaluationInputTests(unittest.TestCase):
    def test_input_reuse_is_local_to_one_evaluation(self):
        world = world_with_requests(shared=True, sparse=True, layers=10)
        with reuse_costs():
            for index in range(4):
                for request in world.requests.values():
                    request.future_writer = layout.GPU if index % 2 else layout.PIM
                    world.append_token(request)
                world.clock_us += 1
                if index == 2:
                    world.retire('a')
                    world.admit(WorkloadRequest('new', 0, 11, 20), 0)
                expected_world = deepcopy(world)
                with patch.object(layout, 'current_cost_cache', return_value=None):
                    expected = layout.evaluate_step(expected_world, offset=index, book_link=True)
                actual = layout.evaluate_step(world, offset=index, book_link=True)
                self.assertEqual(actual, expected)
                assert_world_equal(self, world, expected_world)

    def test_layer_layout_metadata_tracks_configuration_changes(self):
        world = world_with_requests(layers=10)
        requests = list(world.requests.values())
        with reuse_costs() as cache:
            for shards in (1, 2, 4):
                world.config = replace(world.config, simulation=replace(world.config.simulation,
                                                                       layout_dense_ranksets=shards))
                for layer in (0, 1, 9):
                    self.assertEqual(layout._stage_cost(world, requests, 0, layer=layer),
                        layout._compute_stage_cost(world, requests, 0, layer=layer))
            self.assertEqual(len(cache.layer_layouts), 3)


if __name__ == '__main__':
    unittest.main()
