from __future__ import annotations

import unittest

from tools.run_full_af_validation import (
    legal_beam_case,
    monotonic_writer_views,
    sampled_event_case,
    screen_case,
    selected_event_cases,
)


class FullAFValidationTests(unittest.TestCase):
    def test_analytic_joint_screen_contains_every_ablation(self):
        row = screen_case(2, 128, 16, 8)
        self.assertEqual(row["status"], "completed")
        self.assertGreaterEqual(row["joint_oracle_speedup"], 1.0)
        self.assertGreaterEqual(row["joint_vs_best_single"], 1.0)
        self.assertGreaterEqual(row["joint_dynamic_vs_joint_static"], 1.0)
        self.assertEqual(
            set(row["joint_static_plan"]), {"qkv", "o_proj", "ffn"}
        )

    def test_event_selection_includes_single_large_and_negative_control(self):
        rows = [
            {
                "batch_size": batch,
                "input_tokens": 128,
                "output_tokens": output,
                "status": "completed",
                "joint_oracle_speedup": speedup,
                "joint_vs_best_single": speedup - 1.0,
                "joint_dynamic_vs_joint_static": 1.0,
            }
            for batch, output, speedup in (
                (1, 4096, 1.05),
                (8, 8192, 1.40),
                (16, 1024, 1.001),
                (32, 4096, 1.10),
            )
        ]
        selected = selected_event_cases(rows, 1)
        batches = {row["batch_size"] for row in selected}
        self.assertTrue({1, 8, 32} <= batches)
        self.assertIn(16, batches)

    def test_sampled_event_candidate_contains_single_family_paths(self):
        analytic = screen_case(2, 128, 16, 8)
        row = sampled_event_case(analytic, 100.0)
        self.assertGreaterEqual(row["event_candidate_speedup"], 1.0)
        self.assertGreaterEqual(row["event_candidate_vs_best_single"], 1.0)
        self.assertTrue(row["event_candidate_timeline"])

    def test_monotonic_views_never_rewrite_history(self):
        views = monotonic_writer_views(3, 100, 12, (0, 8, -1))
        self.assertEqual((views[0].pim_tokens, views[0].gpu_tokens), (100, 12))
        self.assertEqual((views[1].pim_tokens, views[1].gpu_tokens), (108, 4))
        self.assertEqual((views[2].pim_tokens, views[2].gpu_tokens), (112, 0))

    def test_legal_beam_includes_static_fallback(self):
        analytic = screen_case(2, 128, 16, 8)
        row = legal_beam_case(
            analytic,
            beam_width=4,
            plan_shortlist=2,
            switch_execution_us=100.0,
            min_operator_residence_tokens=8,
        )
        self.assertGreaterEqual(row["beam_vs_static"], 1.0)
        writer_counts = [entry["gpu_writers"] for entry in row["timeline"]]
        self.assertEqual(writer_counts, sorted(writer_counts))


if __name__ == "__main__":
    unittest.main()
