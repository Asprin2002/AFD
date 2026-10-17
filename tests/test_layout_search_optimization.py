from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from elasticafd.layout_aware import (
    GPU, PIM, OPS, LayoutAction, LayoutWorld, LinkCalendar, LinkEvent,
    PimInterval, PrefixCatalog, _book_boundary, _forecast_total, _prune_link,
    _schedule_preparation, apply_action,
    finalize_preparations,
)
from elasticafd.simulator import run_simulation
from elasticafd.specs import GiB
from elasticafd.workload import WorkloadRequest
from tests.test_layout_aware import tiny_config


class CapacityAccountingTests(unittest.TestCase):
    def assert_matches_scan(self, world):
        totals = world._capacity_totals
        self.assertIsNotNone(totals)
        cached_kv, cached_usage = world.kv_usage(), world.usage()
        world.check_capacity()
        cached_peaks = (world.peak_gpu_bytes, world.peak_pim_bytes,
                        world.peak_gpu_kv_bytes, world.peak_pim_kv_bytes)
        world._capacity_totals = None
        try:
            self.assertEqual(cached_kv, world.kv_usage())
            self.assertEqual(cached_usage, world.usage())
            world.check_capacity()
            self.assertEqual(cached_peaks, (world.peak_gpu_bytes, world.peak_pim_bytes,
                                           world.peak_gpu_kv_bytes, world.peak_pim_kv_bytes))
        finally:
            world._capacity_totals = totals

    def test_shared_copies_weights_writers_and_retirement(self):
        cfg = tiny_config(shared=True)
        cfg = replace(cfg, model=replace(cfg.model, data_parallel=2, tensor_parallel=4),
                      system=replace(cfg.system,
                                     gpu=replace(cfg.system.gpu, kv_block_tokens=4),
                                     pim=replace(cfg.system.pim, kv_block_tokens=3)),
                      simulation=replace(cfg.simulation, layout_dense_ranksets=4))
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        world._start_capacity_accounting()
        for name, replica in (("a", 0), ("b", 0), ("c", 1)):
            world.admit(WorkloadRequest(name, 0, 8, 50, "common"), replica)
            self.assert_matches_scan(world)
        keys = tuple(world.requests["a"].block_ids[layer][0]
                     for layer in range(cfg.model.layers))
        apply_action(world, LayoutAction("copy", ("a", "b"), keys, GPU), baseline_us=1)
        # The destination copy counts while the old one is still present.
        self.assert_matches_scan(world)
        world.clock_us = max(world.blocks[key].owner_ready_us for key in keys)
        finalize_preparations(world)
        self.assert_matches_scan(world)
        apply_action(world, LayoutAction("future", ("a",), target=GPU), baseline_us=1)
        apply_action(world, LayoutAction("interval", ("c",), interval=PimInterval(1, 3)),
                     baseline_us=1)
        self.assert_matches_scan(world)
        for _ in range(7):
            for request in world.requests.values():
                world.append_token(request)
                self.assert_matches_scan(world)
        for name in ("a", "b", "c"):
            world.retire(name)
            self.assert_matches_scan(world)

    def test_cached_shared_prefix_release_and_eviction(self):
        cfg = tiny_config(shared=True)
        cfg = replace(cfg, simulation=replace(cfg.simulation, prefix_cache_budget_gib=1))
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        world.admit(WorkloadRequest("a", 0, 8, 2, "common"), 0)
        world._start_capacity_accounting()
        world.retire("a")
        self.assert_matches_scan(world)
        self.assertGreater(sum(world.kv_usage()[1]), 0)
        world.config = replace(cfg, simulation=replace(cfg.simulation, prefix_cache_budget_gib=0))
        world.evict_cache()
        self.assert_matches_scan(world)
        self.assertEqual(sum(world.kv_usage()[1]), 0)

    def test_capacity_failure_is_seen_at_the_same_append(self):
        cfg = tiny_config()
        token_bytes = cfg.model.kv_bytes_per_token_per_layer
        cfg = replace(cfg, system=replace(cfg.system, pim=replace(
            cfg.system.pim, capacity_gib=9 * token_bytes * cfg.system.pim.ranksets / GiB)),
            simulation=replace(cfg.simulation, policy=replace(
                cfg.simulation.policy, planning_horizon_tokens=1)))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        reference = deepcopy(world)
        world._start_capacity_accounting()
        for candidate in (world, reference):
            with self.assertRaisesRegex(MemoryError, "PIM rankset"):
                candidate.append_token(candidate.requests["a"])
        self.assertEqual(world.kv_usage(), reference.kv_usage())
        self.assertEqual(world.requests["a"].generated, reference.requests["a"].generated)

    def test_copied_forecast_has_independent_counters(self):
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        world._start_capacity_accounting()
        original = world.kv_usage()
        trial = deepcopy(world)
        trial.append_token(trial.requests["a"])
        self.assertEqual(world.kv_usage(), original)
        self.assertNotEqual(trial.kv_usage(), original)
        self.assert_matches_scan(trial)

    def test_forecast_releases_counters_on_failure(self):
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        with patch("elasticafd.layout_aware.evaluate_step", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                _forecast_total(world, 2, {"a": 2})
        self.assertIsNone(world._capacity_totals)
        # Public physical objects remain directly editable outside a forecast.
        world.blocks[world.requests["a"].block_ids[0][0]].tokens += 3
        self.assertGreater(sum(world.kv_usage()[1]), 16 * world.kv_bytes)

    def test_all_decisions_and_outputs_match_scanning_forecasts(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                cfg = tiny_config(shared=True, sparse=sparse, policy="layout_greedy")
                cached = run_simulation(cfg)
                with patch.object(LayoutWorld, "_start_capacity_accounting", return_value=None):
                    scanned = run_simulation(cfg)
                self.assertEqual(cached.requests, scanned.requests)
                self.assertEqual(cached.switches, scanned.switches)
                self.assertEqual(cached.iterations, scanned.iterations)
                for key in cached.summary:
                    if key != "policy_search_wall_time_s":
                        self.assertEqual(cached.summary[key], scanned.summary[key], key)


class RepeatedWorkTests(unittest.TestCase):
    def test_completed_jobs_keep_readiness_without_being_rescheduled(self):
        cfg = tiny_config()
        cfg = replace(cfg, system=replace(cfg.system, link=replace(
            cfg.system.link, host_to_pim_gbps=0.001, pim_to_host_gbps=0.001)),
            simulation=replace(cfg.simulation, layout_preparation_packet_bytes=4))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        calendar = world.link_calendar
        _schedule_preparation(world, 8, PIM, 3)
        _schedule_preparation(world, 8, PIM)
        _schedule_preparation(world, 8, GPU)
        self.assertEqual(_book_boundary(calendar, cfg, 3, 2, 0), (6, 2, 1))
        self.assertEqual([job.ready_us for job in calendar.jobs.values()], [13, 18, 26])
        _prune_link(calendar, 10)
        self.assertEqual(_book_boundary(calendar, cfg, 10, 0, 2), (12, 2, 0))
        self.assertEqual([job.ready_us for job in calendar.jobs.values()], [13, 20, 28])
        _prune_link(calendar, 20)
        self.assertEqual(_book_boundary(calendar, cfg, 20, 2, 0), (22, 2, 0))
        self.assertEqual([job.ready_us for job in calendar.jobs.values()], [13, 20, 30])
        _prune_link(calendar, 30)
        self.assertEqual(len(calendar.jobs), 3)
        with patch("elasticafd.layout_aware._reflow_preparations",
                   side_effect=AssertionError("all DMA has completed")):
            self.assertEqual(_book_boundary(calendar, cfg, 30, 1, 0), (31, 1, 0))
        # A new preparation must re-enter the queue after the cache is empty.
        world.clock_us = 31
        _schedule_preparation(world, 8, PIM)
        self.assertEqual(_book_boundary(calendar, cfg, 32, 1, 0), (36, 1, 3))
        self.assertEqual(calendar.jobs[3].ready_us, 40)

    def test_completed_opposite_lane_does_not_delay_remaining_full_duplex_job(self):
        cfg = tiny_config()
        cfg = replace(cfg, system=replace(cfg.system, link=replace(
            cfg.system.link, full_duplex=True,
            host_to_pim_gbps=0.001, pim_to_host_gbps=0.001)),
            simulation=replace(cfg.simulation, layout_preparation_packet_bytes=4))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        calendar = world.link_calendar
        _schedule_preparation(world, 8, PIM, 3)
        _schedule_preparation(world, 8, PIM)
        _schedule_preparation(world, 8, GPU)
        self.assertEqual(_book_boundary(calendar, cfg, 3, 2, 0), (6, 2, 1))
        self.assertEqual([job.ready_us for job in calendar.jobs.values()], [13, 18, 8])
        _prune_link(calendar, 10)
        self.assertEqual(_book_boundary(calendar, cfg, 10, 0, 2), (12, 2, 0))
        self.assertEqual([job.ready_us for job in calendar.jobs.values()], [13, 18, 8])
        _prune_link(calendar, 18)
        with patch("elasticafd.layout_aware._reflow_preparations",
                   side_effect=AssertionError("both lanes have completed")):
            self.assertEqual(_book_boundary(calendar, cfg, 18, 1, 0), (19, 1, 0))

    def test_interval_membership_matches_expanded_operator_coordinates(self):
        intervals = (PimInterval(), *PimInterval().alternatives())
        for interval in intervals:
            for layer in (0, 1, 2, 95):
                coordinates = {index for anchor in (layer - 1, layer, layer + 1)
                               for index in range(5 * anchor + interval.left,
                                                  5 * anchor + interval.right + 1)}
                for position, op in enumerate(OPS):
                    self.assertEqual(interval.contains(layer, op), 5 * layer + position in coordinates)

    def test_foreground_only_keeps_existing_reservations(self):
        for full_duplex, expected in ((False, 9), (True, 4)):
            cfg = tiny_config()
            cfg = replace(cfg, system=replace(cfg.system, link=replace(
                cfg.system.link, full_duplex=full_duplex)))
            calendar = LinkCalendar(events=[LinkEvent(3, 6, "ag", "foreground")])
            with patch("elasticafd.layout_aware._reflow_preparations",
                       side_effect=AssertionError("no background work exists")):
                finish, busy, waited = _book_boundary(calendar, cfg, 2, 2, 1)
            self.assertEqual((finish, busy, waited), (expected, 3, 0))
            self.assertEqual((calendar.events[0].start_us, calendar.events[0].end_us), (3, 6))


if __name__ == "__main__":
    unittest.main()
