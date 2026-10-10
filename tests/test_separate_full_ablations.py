from __future__ import annotations

import unittest

from elasticafd.state import Location
from tools.attention_strategy import (horizon_greedy, offline_windows,
                                      sampled_blocks, views, window)
from tools.run_contiguous_ablation import config
from elasticafd.dense_offload import DenseOffloadCost, DensePlan
from tools.run_separate_full_ablations import (
    Candidate, candidates, evaluate_shape, request_views,
)


class SeparateFullAblationTests(unittest.TestCase):
    def test_attention_changes_only_future_kv(self):
        option = Candidate("half_mid", gpu_requests=2, switch_after=4)
        before = request_views(4, 128, 3, option)
        after = request_views(4, 128, 6, option)
        self.assertTrue(all(view.gpu_tokens == 0 for view in before))
        self.assertEqual([view.gpu_tokens for view in after], [2, 2, 0, 0])
        self.assertEqual([view.pim_tokens for view in after], [132, 132, 134, 134])
        self.assertEqual(after[0].tail_location, Location.GPU)
        self.assertEqual(after[-1].tail_location, Location.PIM)

    def test_attention_window_keeps_historical_kv_in_place(self):
        schedule = window(2, 1, 2, 6, 8)
        before = views(schedule, 128, 1)
        during = views(schedule, 128, 4)
        after = views(schedule, 128, 7)
        self.assertEqual(before[0].pim_tokens, 129)
        self.assertEqual(before[0].gpu_tokens, 0)
        self.assertEqual(during[0].pim_tokens, 130)
        self.assertEqual(during[0].gpu_tokens, 2)
        self.assertEqual(after[0].pim_segment_lengths, (130, 1))
        self.assertEqual(after[0].gpu_segment_lengths, (4,))
        self.assertEqual(after[0].tail_location, Location.PIM)
        self.assertEqual(after[1].pim_tokens, 135)

    def test_attention_searches_cut_points(self):
        model = DenseOffloadCost(config(2), (DensePlan(),))
        greedy = horizon_greedy(model, 2, 128, 4, decision_interval=2,
                                horizon=2, min_residence=2)
        finalists = offline_windows(model, 2, 128, 4, search_stride=2,
                                   min_residence=1, grid_points=3)
        self.assertEqual(greedy.name, "horizon_greedy")
        self.assertTrue(all(model.fits(views(item, 128, 4)) for item in finalists))

    def test_sampling_never_crosses_a_switch_boundary(self):
        schedule = window(2, 1, 3, 7, 10)
        self.assertEqual(sampled_blocks(10, 5, (schedule,)),
                         [(3, 1), (2, 3), (2, 5), (3, 8)])

    def test_fixed_subbatch_membership_survives_attention_switches(self):
        model = DenseOffloadCost(config(8, scheduler="fixed"), (DensePlan(),))
        schedule = window(8, 7, 2, 6, 10)
        expected = None
        backend = model.backends[(0.0, 0.0)]
        for age in (0, 1, 2, 4, 5, 6, 8, 10):
            state = views(schedule, 128, age)
            memberships = tuple(
                frozenset(view.request_id for view in part)
                for part in model.partition_views(state)
            )
            self.assertEqual(memberships, tuple(
                frozenset(view.request_id for view in part)
                for part in backend.base.partition_views(state)
            ))
            if expected is None:
                expected = memberships
            self.assertEqual(memberships, expected)

    def test_f_grid_never_splits_attention(self):
        options = candidates("f")
        self.assertEqual(len(options), 11)
        self.assertTrue(all(option.gpu_requests == 0 for option in options))
        self.assertTrue(all(option.switch_after == 0 for option in options))
        self.assertEqual(sum(option.conditional_ffn for option in options), 4)

    def test_both_families_use_event_costs_and_same_baseline(self):
        options = dict(
            stride=2,
            dense_compute_scale=1.0, dense_bandwidth_scale=1.0,
        )
        attention = evaluate_shape("attention", 2, 128, 2, **options)
        dense = evaluate_shape("f", 2, 128, 2, **options)
        self.assertEqual(attention[0]["baseline_total_us"],
                         dense[0]["baseline_total_us"])
        self.assertTrue(all(row["status"] == "completed" for row in attention))
        self.assertTrue(all(row["status"] == "completed" for row in dense))
        self.assertTrue(all(row["gpu_kv_requests"] == 0 for row in dense))
        self.assertTrue(all(row["ffn_pim_fraction"] == 0 for row in attention))
        self.assertEqual([row["candidate"] for row in attention],
                         ["control", "horizon_greedy", "offline_window",
                          "offline_best_of"])
        self.assertLessEqual(attention[-1]["candidate_total_us"],
                             attention[1]["candidate_total_us"])
        self.assertLessEqual(attention[-1]["candidate_total_us"],
                             attention[2]["candidate_total_us"])


if __name__ == "__main__":
    unittest.main()
