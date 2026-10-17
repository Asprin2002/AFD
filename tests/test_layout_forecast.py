from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from math import nextafter, inf
from random import Random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import elasticafd.layout_aware as layout
from elasticafd.layout_cost_cache import reuse_costs
from elasticafd.layout_forecast import build_forecast_plan, evaluate_forecasts, ForecastPruned
from tests.test_layout_cost_cache import world_with_requests
from tools.layout_search_comparison import compare_screening
from elasticafd.specs import GiB


class SharedForecastTests(unittest.TestCase):
    def expected(self, world, scenarios, horizon):
        return tuple(layout._forecast_total(deepcopy(world), horizon, remaining) for remaining in scenarios)

    def test_identical_paths_execute_once_and_keep_first_step(self):
        world = world_with_requests(layers=6)
        scenarios = tuple(dict.fromkeys(world.requests, 8) for _ in range(4))
        plan = build_forecast_plan(world, 8, scenarios)
        expected = self.expected(world, scenarios, 8)
        stats = {}
        seed = deepcopy(world)
        result = evaluate_forecasts(seed, plan, statistics=stats)
        self.assertEqual(result.totals, expected)
        self.assertEqual(result.first_step_us, layout.evaluate_step(world).total_us)
        self.assertEqual(stats['forecast_step_evaluations'], 8)
        self.assertEqual(stats['forecast_prefix_reused_steps'], 24)
        self.assertIsNone(seed._capacity_totals)

    def test_branch_before_different_exits_matches_independent_replays(self):
        scenarios = ({'a': 2, 'b': 6, 'c': 8}, {'a': 4, 'b': 5, 'c': 8},
                     {'a': 2, 'b': 7, 'c': 8}, {'a': 4, 'b': 5, 'c': 8})
        for sparse in (False, True):
            for duplex in (False, True):
                world = world_with_requests(shared=True, sparse=sparse, shards=4, replicas=2)
                world.config = replace(world.config, system=replace(world.config.system,
                    link=replace(world.config.system.link, full_duplex=duplex,
                                 host_to_pim_gbps=0.001, pim_to_host_gbps=0.002,
                                 transfer_latency_us=0.25)),
                    simulation=replace(world.config.simulation, layout_preparation_packet_bytes=4))
                world.clock_us = 12.125
                keys = tuple(world.requests['a'].block_ids[layer][0]
                             for layer in range(world.config.model.layers))
                layout.apply_action(world, layout.LayoutAction('copy', ('a', 'b', 'c'), keys, layout.GPU),
                                    baseline_us=1)
                layout.apply_action(world, layout.LayoutAction('interval', ('b',), interval=layout.PimInterval(1, 3)),
                                    baseline_us=1)
                plan = build_forecast_plan(world, 8, scenarios)
                expected = self.expected(world, scenarios, 8)
                for limit in (0, 1, 8):
                    stats = {}
                    seed = deepcopy(world)
                    with reuse_costs():
                        result = evaluate_forecasts(seed, plan, snapshot_limit=limit, statistics=stats)
                    self.assertEqual(result.totals, expected, (sparse, duplex, limit))
                    self.assertEqual(result.first_step_us, layout.evaluate_step(world).total_us)
                    self.assertLessEqual(stats.get('forecast_peak_checkpoints', 0), limit)
                    self.assertIsNone(seed._capacity_totals)
                    self.assertEqual(stats['forecast_step_evaluations'] + stats['forecast_prefix_reused_steps'],
                                     plan.nominal_steps)

    def test_request_order_and_append_retire_order_are_preserved(self):
        world = world_with_requests(layers=2)
        scenarios = ({'a': 1, 'b': 2, 'c': 3},)
        plan = build_forecast_plan(world, 3, scenarios)
        events = []
        append, retire = layout.LayoutWorld.append_token, layout.LayoutWorld.retire
        def appended(instance, request):
            events.append(('append', request.source.request_id))
            return append(instance, request)
        def retired(instance, key):
            events.append(('retire', key))
            return retire(instance, key)
        with patch.object(layout.LayoutWorld, 'append_token', appended), patch.object(layout.LayoutWorld, 'retire', retired):
            evaluate_forecasts(deepcopy(world), plan)
        self.assertEqual(events[:4], [('append', 'a'), ('retire', 'a'), ('append', 'b'), ('append', 'c')])

    def test_original_scenario_order_is_preserved(self):
        world = world_with_requests(layers=2)
        scenarios = ({'a': 1, 'b': 3, 'c': 3}, {'a': 2, 'b': 3, 'c': 3},
                     {'a': 1, 'b': 3, 'c': 3})
        plan = build_forecast_plan(world, 3, scenarios)
        original = layout.LayoutWorld.retire
        exits = []
        def retired(instance, key):
            if key == 'a':
                exits.append(instance.requests[key].generated)
            return original(instance, key)
        with patch.object(layout.LayoutWorld, 'retire', retired):
            evaluate_forecasts(deepcopy(world), plan)
        self.assertEqual(exits, [1, 2, 1])

    def test_one_logical_plan_is_reused_by_different_candidates(self):
        world = world_with_requests(shared=True, layers=6)
        scenarios = ({'a': 1, 'b': 4, 'c': 6}, {'a': 3, 'b': 2, 'c': 6})
        plan = build_forecast_plan(world, 6, scenarios)
        for side in (layout.GPU, layout.PIM):
            candidate = deepcopy(world)
            layout.apply_action(candidate, layout.LayoutAction('future', ('a',), target=side), baseline_us=1)
            expected = self.expected(candidate, scenarios, 6)
            self.assertEqual(evaluate_forecasts(candidate, plan).totals, expected)
        self.assertIsNone(world._capacity_totals)
        self.assertEqual(tuple(world.requests), ('a', 'b', 'c'))

    def test_plan_ignores_hidden_output_lengths_and_checks_context(self):
        world = world_with_requests(layers=2)
        scenarios = ({'a': 2, 'b': 3, 'c': 4},)
        first = build_forecast_plan(world, 4, scenarios)
        for request in world.requests.values():
            request.source = replace(request.source, output_tokens=100000)
        self.assertEqual(first, build_forecast_plan(world, 4, scenarios))
        world.append_token(world.requests['a'])
        with self.assertRaisesRegex(ValueError, 'contexts'):
            evaluate_forecasts(world, first)

    def test_failure_releases_capacity_scope_and_does_not_touch_real_world(self):
        world = world_with_requests(layers=2)
        plan = build_forecast_plan(world, 4, ({'a': 1, 'b': 3, 'c': 4}, {'a': 2, 'b': 3, 'c': 4}))
        seed = deepcopy(world)
        with patch.object(layout, 'evaluate_step', side_effect=RuntimeError('stop')):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                evaluate_forecasts(seed, plan)
        self.assertIsNone(seed._capacity_totals)
        self.assertEqual(tuple(world.requests), ('a', 'b', 'c'))
        self.assertTrue(all(request.generated == 0 for request in world.requests.values()))

    def test_sixty_four_paths_fall_back_at_checkpoint_limit(self):
        world = world_with_requests(layers=2)
        rng = Random(125)
        scenarios = tuple({key: rng.randrange(1, 17) for key in world.requests} for _ in range(64))
        plan = build_forecast_plan(world, 16, scenarios)
        expected = self.expected(world, scenarios, 16)
        stats = {}
        actual = evaluate_forecasts(deepcopy(world), plan, snapshot_limit=2, statistics=stats)
        self.assertEqual(actual.totals, expected)
        self.assertEqual(stats['forecast_peak_checkpoints'], 2)
        self.assertEqual(stats['forecast_step_evaluations'] + stats['forecast_prefix_reused_steps'], plan.nominal_steps)

    def test_branch_keeps_capacity_failure_at_original_append(self):
        world = world_with_requests(layers=2)
        token_bytes = world.kv_bytes
        world.config = replace(world.config, system=replace(world.config.system,
            pim=replace(world.config.system.pim, capacity_gib=31 * token_bytes * world.config.system.pim.ranksets / GiB)),
            simulation=replace(world.config.simulation, layout_growth_reserve_tokens=0))
        scenarios = ({'a': 1, 'b': 1, 'c': 1}, {'a': 3, 'b': 3, 'c': 3})
        plan = build_forecast_plan(world, 3, scenarios)
        with self.assertRaisesRegex(MemoryError, 'PIM rankset'):
            self.expected(world, scenarios, 3)
        seed = deepcopy(world)
        with self.assertRaisesRegex(MemoryError, 'PIM rankset'):
            evaluate_forecasts(seed, plan)
        self.assertIsNone(seed._capacity_totals)


class BoundedForecastTests(unittest.TestCase):
    def test_bound_is_an_upper_gain_and_stops_certain_loser(self):
        world = world_with_requests(layers=2)
        plan = build_forecast_plan(world, 32, tuple(dict.fromkeys(world.requests, 32) for _ in range(4)))
        stats = {}
        seed = deepcopy(world)
        with patch.object(layout, 'evaluate_step', return_value=SimpleNamespace(total_us=100.0)):
            with self.assertRaises(ForecastPruned) as stopped:
                evaluate_forecasts(seed, plan, baseline_us=32, statistics=stats)
        proof = stopped.exception.proof
        self.assertGreaterEqual(proof['gain_upper_us'], 32 - 3200)
        self.assertLess(proof['gain_upper_us'], 0)
        self.assertEqual(stats['forecast_pruned_candidates'], 1)
        self.assertLess(stats['forecast_step_evaluations'], 32)
        self.assertGreater(stats['forecast_pruned_steps'], 0)
        self.assertIsNone(seed._capacity_totals)

    def test_zero_and_tie_band_scores_are_not_discarded(self):
        world = world_with_requests(layers=2)
        plan = build_forecast_plan(world, 4, tuple(dict.fromkeys(world.requests, 4) for _ in range(2)))
        with patch.object(layout, 'evaluate_step', return_value=SimpleNamespace(total_us=1.0)):
            for baseline in (4.0, nextafter(4.0, -inf), 4.0 - 1e-8):
                result = evaluate_forecasts(deepcopy(world), plan, baseline_us=baseline,
                                            tie_margin_us=1e-6, check_interval=1)
                self.assertEqual(result.totals, (4.0, 4.0))

    def test_large_cost_cancellation_keeps_rounding_margin_before_subtraction(self):
        import elasticafd.layout_forecast as forecast
        world = world_with_requests(layers=2)
        plan = build_forecast_plan(world, 1, tuple(dict.fromkeys(world.requests, 1) for _ in range(4)))
        original_mean = forecast.fmean
        def rounded_mean(values):
            result = original_mean(values)
            # Exercise conservative handling of intermediate mean rounding.
            return nextafter(nextafter(result, inf), inf) if isinstance(values, list) else result
        with patch.object(layout, 'evaluate_step', return_value=SimpleNamespace(total_us=1e12)), \
                patch.object(forecast, 'fmean', rounded_mean):
            result = evaluate_forecasts(deepcopy(world), plan, baseline_us=1e12,
                                        incumbent_gain_us=0, tie_margin_us=1e-9, check_interval=1)
        self.assertEqual(result.mean_us, 1e12)

    def test_small_first_move_is_preserved_for_second_round_combination(self):
        world = world_with_requests(layers=2)
        world.config = replace(world.config, simulation=replace(world.config.simulation,
            policy=replace(world.config.simulation.policy, planning_horizon_tokens=10,
                           relative_improvement_threshold=0.05)))
        actions = [layout.LayoutAction('future', (key,), target=layout.GPU) for key in ('a', 'b')]
        def step(instance, *args, **kwargs):
            count = sum(request.future_writer == layout.GPU for request in instance.requests.values())
            return SimpleNamespace(total_us=(10.0, 9.8, 1.0)[min(count, 2)])
        with patch.object(layout, 'evaluate_step', step), patch.object(layout, 'generate_candidates', return_value=actions), \
                patch.object(layout, '_forecast_scenarios', return_value=tuple(dict.fromkeys(world.requests, 10) for _ in range(4))):
            expected_screen, actual_screen = [], []
            expected = layout.choose_action(world, forecast_mode='reference', screening=expected_screen)
            actual = layout.choose_action(world, screening=actual_screen)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0].kind, 'bundle')
        self.assertLess(expected_screen[0]['best_gain_us'], 5)
        check = compare_screening(expected_screen, actual_screen, baseline_us=actual[2])
        self.assertTrue(check['compatible'], check)

    def test_real_search_matches_reference_scores_and_selection(self):
        for sparse, shards in ((False, 4), (True, 1)):
            world = world_with_requests(shared=True, sparse=sparse, shards=shards)
            world.config = replace(world.config, simulation=replace(world.config.simulation,
                layout_forecast_scenarios=4, policy=replace(world.config.simulation.policy,
                                                          planning_horizon_tokens=8)))
            expected_screen = []
            expected = layout.choose_action(world, forecast_mode='reference', screening=expected_screen)
            for mode in ('shared', 'bounded'):
                screen, stats = [], {}
                actual = layout.choose_action(world, forecast_mode=mode, screening=screen, reuse_statistics=stats)
                self.assertEqual(actual, expected)
                check = compare_screening(expected_screen, screen, baseline_us=actual[2])
                self.assertTrue(check['compatible'], check)
                if mode == 'shared':
                    self.assertTrue(check['exact_match'])
                self.assertEqual(stats['forecast_plan_builds'], 1)

    def test_service_guard_is_preserved(self):
        world = world_with_requests(layers=2)
        world.config = replace(world.config, simulation=replace(world.config.simulation,
            slo_tpot_ms=1.0, policy=replace(world.config.simulation.policy, planning_horizon_tokens=10)))
        action = layout.LayoutAction('future', ('a',), target=layout.GPU)
        def step(instance, *args, **kwargs):
            moved = instance.requests['a'].future_writer == layout.GPU
            duration = (2000 if instance.requests['a'].generated == 0 else 10) if moved else 500
            return SimpleNamespace(total_us=duration)
        before, after = [], []
        with patch.object(layout, 'evaluate_step', step), patch.object(layout, 'generate_candidates', return_value=[action]), \
                patch.object(layout, '_forecast_scenarios', return_value=tuple(dict.fromkeys(world.requests, 10) for _ in range(4))):
            expected = layout.choose_action(world, forecast_mode='reference', screening=before)
            actual = layout.choose_action(world, screening=after)
        self.assertEqual(actual, expected)
        self.assertIsNone(actual[0])
        self.assertEqual(before[0]['candidate_outcomes'][0]['status'], 'service_guard')
        self.assertTrue(compare_screening(before, after, baseline_us=actual[2])['compatible'])

    def test_near_tie_can_select_lower_gain_with_less_preparation(self):
        world = world_with_requests(layers=2)
        world.config = replace(world.config, simulation=replace(world.config.simulation,
            layout_error_margin_us=0.05, policy=replace(world.config.simulation.policy, planning_horizon_tokens=10)))
        actions = [layout.LayoutAction('future', (key,), target=layout.GPU) for key in ('a', 'b')]
        original = layout.apply_action
        def apply(instance, action, **kwargs):
            result = original(instance, action, **kwargs)
            instance.preparation_bytes = 100 if instance.requests['a'].future_writer == layout.GPU else 0
            return result
        def step(instance, *args, **kwargs):
            duration = (9.0 if instance.requests['a'].future_writer == layout.GPU else
                        9.002 if instance.requests['b'].future_writer == layout.GPU else 10.0)
            return SimpleNamespace(total_us=duration)
        def generate(instance, *, allow_pending_extension=False, **kwargs):
            return [] if allow_pending_extension else actions
        with patch.object(layout, 'evaluate_step', step), patch.object(layout, 'apply_action', apply), \
                patch.object(layout, 'generate_candidates', generate), patch.object(layout, '_forecast_scenarios',
                    return_value=tuple(dict.fromkeys(world.requests, 10) for _ in range(4))):
            expected = layout.choose_action(world, forecast_mode='reference')
            actual = layout.choose_action(world)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0].request_ids, ('b',))


class BoundComparisonTests(unittest.TestCase):
    def test_forged_bound_or_changed_round_selection_is_detected(self):
        old = [{'best_gain_us': 1, 'candidate_outcomes': [
            {'kind': 'future', 'request_ids': ['a'], 'status': 'scored', 'gain_us': 2}]}]
        new = [{'best_gain_us': 1, 'candidate_outcomes': [
            {'kind': 'future', 'request_ids': ['a'], 'status': 'bound_pruned',
             'gain_upper_us': 0, 'partial_mean_us': 10, 'incumbent_gain_us': 1, 'tie_margin_us': 0.1}]}]
        check = compare_screening(old, new, baseline_us=10)
        self.assertFalse(check['compatible'])
        self.assertTrue(any('reference score exceeds' in item for item in check['differences']))
        new[0]['best_gain_us'] = 2
        self.assertTrue(any('selection' in item for item in compare_screening(old, new)['differences']))


if __name__ == '__main__':
    unittest.main()
