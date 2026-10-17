from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

import elasticafd.layout_aware as layout
from elasticafd.layout_cost_cache import current_cost_cache, reuse_costs
from elasticafd.workload import WorkloadRequest
from elasticafd.simulator import run_simulation
from tests.test_layout_aware import tiny_config


def world_with_requests(*, shared=False, sparse=False, shards=1, replicas=1, layers=6):
    cfg = tiny_config(shared=shared, sparse=sparse, policy="layout_greedy")
    cfg = replace(cfg, model=replace(cfg.model, layers=layers, data_parallel=replicas,
                                    tensor_parallel=8 // replicas),
                  simulation=replace(cfg.simulation, layout_dense_ranksets=shards))
    world = layout.LayoutWorld(cfg, layout.PrefixCatalog(cfg.workload.prefix_blocks))
    for index, name in enumerate(("a", "b", "c")):
        world.admit(WorkloadRequest(name, 0, 8 + 2 * index, 20,
                                    "common" if shared else None), index % replicas)
    return world


class ExactCostReuseTests(unittest.TestCase):
    def assert_stage_matches(self, world, requests=None, *, now=0, offset=0):
        requests = requests if requests is not None else list(world.requests.values())
        for layer in range(world.config.model.layers):
            expected = layout._compute_stage_cost(world, requests, now, offset, layer=layer)
            self.assertEqual(layout._stage_cost(world, requests, now, offset, layer=layer), expected)

    def test_kernel_reuse_requires_equal_settings_and_arguments(self):
        cfg = tiny_config()
        other = replace(cfg, system=replace(cfg.system, gpu=replace(
            cfg.system.gpu, compute_efficiency=0.1, hbm_bandwidth_tbps=0.1)))
        with reuse_costs() as cache:
            for config in (cfg, deepcopy(cfg), other):
                for m, presence in ((1, 1), (1.5, 0.7), (0, 0)):
                    for layer in (0, 1, 95):
                        for device in (layout.GPU, layout.PIM):
                            actual = layout._linear_us(config, device, "ffn", m,
                                                       layer=layer, presence=presence)
                            expected = layout._compute_linear_us(config, device, "ffn", m,
                                                                  layer=layer, presence=presence)
                            self.assertEqual(actual, expected)
            self.assertEqual(len(cache.configs), 2)
            self.assertGreater(cache.counts["linear_hits"], cache.counts["linear_misses"])

    def test_equivalent_middle_layers_reuse_cost_with_distinct_physical_ids(self):
        world = world_with_requests(shared=True, layers=10)
        requests = list(world.requests.values())
        with reuse_costs() as cache:
            first = layout._stage_cost(world, requests, 0, layer=1)
            misses = cache.counts["stage_misses"]
            second = layout._stage_cost(world, requests, 0, layer=5)
            self.assertEqual(first, second)
            self.assertEqual(cache.counts["stage_misses"], misses)
            self.assertEqual(cache.counts["stage_hits"], 1)

    def test_sharing_is_physical_identity_not_equal_lengths(self):
        shared = world_with_requests(shared=True)
        private = world_with_requests()
        with reuse_costs() as cache:
            a = layout._stage_cost(shared, list(shared.requests.values()), 0, layer=1)
            b = layout._stage_cost(private, list(private.requests.values()), 0, layer=1)
            self.assertNotEqual(a.pim_scan_bytes, b.pim_scan_bytes)
            self.assert_stage_matches(shared)
            self.assert_stage_matches(private)
            self.assertGreater(cache.counts["stage_hits"], 0)

    def test_sparse_overlap_offsets_and_generation_progress(self):
        world = world_with_requests(shared=True, sparse=True)
        world.config = replace(world.config, simulation=replace(
            world.config.simulation, recent_blocks=8, layout_survival_shape=2))
        with reuse_costs():
            for generated in (0, 1, 4):
                for request in world.requests.values():
                    while request.generated < generated:
                        world.append_token(request)
                for offset in (0, 1, 3, 10):
                    self.assert_stage_matches(world, offset=offset)

    def test_future_private_identity_is_separate_from_shared_history(self):
        cfg = tiny_config(shared=True)
        world = layout.LayoutWorld(cfg, layout.PrefixCatalog(cfg.workload.prefix_blocks))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 4, 20, "common"), 0)
        a, b = world.requests.values()
        with reuse_costs():
            for requests in ([a, a], [a, b]):
                expected = layout._compute_stage_cost(world, requests, 0, 2, layer=0)
                self.assertEqual(layout._stage_cost(world, requests, 0, 2, layer=0), expected)

    def test_partial_layer_copy_and_readiness_change_do_not_reuse_old_owner(self):
        world = world_with_requests()
        key = world.requests["a"].block_ids[1][0]
        block = world.blocks[key]
        world.add_copy(block, layout.GPU, 10)
        block.pending_owner = layout.GPU
        block.owner_ready_us = 10
        with reuse_costs():
            for now in (0, 9, 10, 11):
                self.assert_stage_matches(world, now=now, offset=1)
            first = layout._stage_cost(world, [world.requests["a"]], 0, layer=1)
            moved = layout._stage_cost(world, [world.requests["a"]], 10, layer=1)
            other_layer = layout._stage_cost(world, [world.requests["a"]], 10, layer=5)
            self.assertNotEqual(first, moved)
            self.assertNotEqual(moved, other_layer)

    def test_rankset_and_pending_interval_changes_are_cost_inputs(self):
        world = world_with_requests(shards=4)
        with reuse_costs():
            self.assert_stage_matches(world)
            key = world.requests["b"].block_ids[1][0]
            before = layout._stage_cost(world, list(world.requests.values()), 0, layer=1)
            world.blocks[key].copies[layout.PIM].rankset = 3
            after = layout._stage_cost(world, list(world.requests.values()), 0, layer=1)
            self.assertNotEqual(before, after)
            request = world.requests["a"]
            request.pending_interval = layout.PimInterval(1, 3)
            request.interval_ready_us = 10
            for now in (0, 10):
                self.assert_stage_matches(world, now=now)

    def test_first_last_dense_shards_and_replicas_match_raw_costs(self):
        with reuse_costs():
            for shards in (1, 2, 4):
                for replicas in (1, 2):
                    world = world_with_requests(shared=True, shards=shards, replicas=replicas)
                    for request in world.requests.values():
                        request.interval = layout.PimInterval(-1, 1)
                    self.assert_stage_matches(world)
                    for request in world.requests.values():
                        request.interval = layout.PimInterval(1, 4)
                    self.assert_stage_matches(world)

    def test_unchanged_subbatch_can_be_reused_when_another_request_changes(self):
        world = world_with_requests()
        b = [world.requests["b"]]
        with reuse_costs() as cache:
            original = layout._stage_cost(world, b, 0, offset=2, layer=1)
            misses = cache.counts["stage_misses"]
            world.requests["a"].future_writer = layout.GPU
            self.assertEqual(layout._stage_cost(world, b, 0, offset=2, layer=1), original)
            self.assertEqual(cache.counts["stage_misses"], misses)
            self.assert_stage_matches(world, offset=2)

    def test_link_is_replayed_for_every_layer_even_with_cost_hits(self):
        world = world_with_requests(shared=True)
        world.config = replace(world.config, system=replace(world.config.system,
            link=replace(world.config.system.link, host_to_pim_gbps=0.001)))
        layout._schedule_preparation(world, 256, layout.PIM)
        expected_world = deepcopy(world)
        expected = layout.evaluate_step(expected_world, book_link=True)
        with reuse_costs() as cache:
            # Fill costs, without reserving foreground packets on the world.
            layout.evaluate_step(world)
            with patch.object(layout, "_book_boundary", wraps=layout._book_boundary) as book:
                actual = layout.evaluate_step(world, book_link=True)
            self.assertEqual(book.call_count, 2 * world.config.model.layers)
            self.assertGreater(cache.counts["stage_hits"], 0)
        self.assertEqual(actual, expected)
        self.assertEqual(asdict(world.link_calendar), asdict(expected_world.link_calendar))

    def test_search_matches_all_uncached_candidate_scores(self):
        for sparse, shards in ((False, 4), (True, 1)):
            world = world_with_requests(shared=True, sparse=sparse, shards=shards)
            # Exercise a setting where an interval move can win.
            world.config = replace(world.config, system=replace(world.config.system,
                gpu=replace(world.config.system.gpu, hbm_bandwidth_tbps=0.001),
                link=replace(world.config.system.link, host_to_pim_gbps=100000,
                             pim_to_host_gbps=100000)),
                simulation=replace(world.config.simulation, layout_pim_dense_scale=0.01))
            expected_screen, actual_screen, stats = [], [], {}
            expected = layout._choose_action(deepcopy(world), screening=expected_screen)
            actual = layout.choose_action(world, screening=actual_screen, reuse_statistics=stats)
            self.assertIsNotNone(expected[0])
            self.assertEqual(actual, expected)
            self.assertEqual(actual_screen, expected_screen)
            self.assertGreater(stats["stage_hits"], 0)
            self.assertIsNone(current_cost_cache())

    def test_online_shared_arrivals_and_sparse_windows_match_uncached_execution(self):
        for sparse, shards in ((False, 1), (False, 4), (True, 1)):
            cfg = tiny_config(shared=True, sparse=sparse, policy="layout_greedy")
            cfg = replace(cfg, model=replace(cfg.model, layers=6),
                workload=replace(cfg.workload, type="trace_serving", requests=(
                    {"request_id": "a", "arrival_us": 0, "input_tokens": 16,
                     "output_tokens": 4, "prefix_leaf_id": "common"},
                    {"request_id": "b", "arrival_us": 0, "input_tokens": 20,
                     "output_tokens": 7, "prefix_leaf_id": "common"},
                    {"request_id": "c", "arrival_us": 0.3, "input_tokens": 24,
                     "output_tokens": 9, "prefix_leaf_id": "common"})),
                simulation=replace(cfg.simulation, max_batch_size=3, layout_dense_ranksets=shards,
                    layout_max_candidates=4, layout_forecast_scenarios=2,
                    policy=replace(cfg.simulation.policy, planning_horizon_tokens=4,
                                   decision_interval=4, event_driven_replan=True)))
            expected = None
            with patch.object(layout, "choose_action", layout._choose_action):
                expected = run_simulation(cfg)
            actual = run_simulation(cfg)
            self.assertEqual(actual.requests, expected.requests)
            self.assertEqual(actual.switches, expected.switches)
            self.assertEqual(actual.iterations, expected.iterations)
            for key in actual.summary:
                if key != "policy_search_wall_time_s":
                    self.assertEqual(actual.summary[key], expected.summary[key], (sparse, shards, key))

    def test_scope_resets_after_exception_and_nested_scopes_are_independent(self):
        self.assertIsNone(current_cost_cache())
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with reuse_costs() as outer:
                with reuse_costs() as inner:
                    self.assertIs(current_cost_cache(), inner)
                self.assertIs(current_cost_cache(), outer)
                raise RuntimeError("stop")
        self.assertIsNone(current_cost_cache())

    def test_cache_size_is_bounded_when_data_keeps_growing(self):
        world = world_with_requests()
        with reuse_costs(stage_limit=2, kernel_limit=2) as cache:
            for _ in range(10):
                self.assert_stage_matches(world)
                for request in world.requests.values():
                    world.append_token(request)
                for kind, bucket in cache.values.items():
                    self.assertLessEqual(len(bucket), cache.limits[kind])
            self.assertGreater(cache.counts["stage_resets"], 0)


if __name__ == "__main__":
    unittest.main()
