from __future__ import annotations

import unittest

from elasticafd.experiments import (
    _benefit_classification,
    dynamic_operator_offload_rows,
    future_only_benefit_rows,
    future_only_composition_rows,
    homogeneous_lifecycle_rows,
    joint_af_event_replay_rows,
    joint_af_heterogeneous_oracle_rows,
    joint_af_small_oracle_rows,
    open_arrival_slo_rows,
    static_operator_mapping_rows,
)


class BenefitClassificationTests(unittest.TestCase):
    def test_zero_budget_is_explicitly_classified(self) -> None:
        label = _benefit_classification(
            baseline_status="completed",
            elastic_status="completed",
            speedup=1.0,
            gpu_budget_gib=0.0,
            peak_gpu_kv_gib=0.0,
            switches=0,
            baseline_pim_bottleneck_fraction=1.0,
            output_tokens=1024,
            decision_interval=64,
        )
        self.assertEqual(label, "neutral_zero_gpu_kv_budget")

    def test_small_paired_run_uses_identical_workload(self) -> None:
        rows = future_only_benefit_rows(
            model_names=("gpt175b",),
            batch_sizes=(1,),
            input_lengths=(128,),
            output_lengths=(2,),
            gpu_kv_budgets_gib=(0.0,),
            decision_interval=1,
            planning_horizon_tokens=2,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["baseline_status"], "completed")
        self.assertEqual(row["elastic_status"], "completed")
        self.assertEqual(row["speedup_vs_chime"], 1.0)
        self.assertEqual(row["classification"], "neutral_zero_gpu_kv_budget")

    def test_composition_map_holds_final_context_constant(self) -> None:
        rows = future_only_composition_rows(
            model_names=("gpt175b",),
            batch_sizes=(1,),
            total_contexts=(8,),
            output_lengths=(2, 6),
            gpu_kv_budgets_gib=(0.0,),
            decision_interval=1,
            planning_horizon_tokens=2,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [int(row["input_tokens"]) + int(row["output_tokens"]) for row in rows],
            [8, 8],
        )
        self.assertEqual([row["final_context_tokens"] for row in rows], [8, 8])

    def test_homogeneous_lifecycle_map_is_paired_and_preplanned(self) -> None:
        rows = homogeneous_lifecycle_rows(
            model_names=("gpt175b",),
            batch_sizes=(2,),
            input_lengths=(128,),
            output_lengths=(4,),
            gpu_kv_budgets_gib=(None,),
            mode_grid_points=3,
            quadrature_samples=2,
            exact_finalists=2,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["baseline_status"], "completed")
        self.assertEqual(row["offline_mode_status"], "completed")
        self.assertGreaterEqual(float(row["offline_mode_speedup_vs_chime"]), 1.0)
        self.assertFalse(row["historical_kv_migration"])
        self.assertEqual(row["offline_mode_capacity_switches"], 0)

    def test_homogeneous_lifecycle_can_replay_on_event_model(self) -> None:
        rows = homogeneous_lifecycle_rows(
            model_names=("gpt175b",),
            batch_sizes=(2,),
            input_lengths=(128,),
            output_lengths=(2,),
            gpu_kv_budgets_gib=(None,),
            mode_grid_points=3,
            quadrature_samples=2,
            exact_finalists=2,
            pipeline_model="rankset_event",
            policy_pipeline_model="analytic",
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["pipeline_model"], "rankset_event")
        self.assertEqual(row["policy_pipeline_model"], "analytic")
        self.assertEqual(row["baseline_status"], "completed")
        self.assertEqual(row["offline_mode_status"], "completed")

    def test_open_slo_comparison_uses_one_paired_trace(self) -> None:
        rows = open_arrival_slo_rows(
            request_count=3,
            seeds=(7,),
            arrival_rates_rps=(1.0,),
            max_batch_size=2,
            input_mean=64,
            input_std=0,
            input_min=64,
            input_max=64,
            output_mean=4,
            output_std=0,
            output_min=4,
            output_max=4,
            slo_queue_ms=1000,
            slo_tpot_ms=1000,
            decision_interval=2,
            beam_depth=1,
            beam_width=1,
            beam_request_candidates=1,
            beam_evaluation_stride_tokens=1,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            len({row["realized_offered_rate_rps"] for row in rows}), 1
        )
        self.assertEqual(
            [row["future_arrivals_visible"] for row in rows],
            [False, False, True],
        )

    def test_open_slo_can_add_paired_event_driven_variants(self) -> None:
        rows = open_arrival_slo_rows(
            request_count=2,
            seeds=(7,),
            arrival_rates_rps=(1.0,),
            max_batch_size=2,
            input_mean=16,
            input_std=0,
            input_min=16,
            input_max=16,
            output_mean=2,
            output_std=0,
            output_min=2,
            output_max=2,
            slo_queue_ms=1000,
            slo_tpot_ms=1000,
            decision_interval=1,
            beam_depth=1,
            beam_width=1,
            beam_request_candidates=1,
            beam_evaluation_stride_tokens=1,
            include_event_driven_variant=True,
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            sum(bool(row["event_driven_replan"]) for row in rows), 2
        )
        self.assertEqual(
            len({row["realized_offered_rate_rps"] for row in rows}), 1
        )

    def test_dynamic_af_upper_bound_dominates_best_static(self) -> None:
        row = dynamic_operator_offload_rows(
            model_name="gpt175b",
            system_name="chime_a100",
            subbatch_sizes=(1,),
            input_lengths=(128,),
            output_lengths=(8,),
            fractions=(0.0, 0.5, 1.0),
            decision_interval=2,
            min_residence_tokens=2,
        )[0]
        self.assertGreaterEqual(float(row["best_static_speedup"]), 1.0)
        self.assertGreaterEqual(
            float(row["dynamic_vs_best_static"]), 1.0 - 1e-12
        )
        self.assertTrue(row["constrained_dynamic_schedule"])

    def test_dynamic_af_supports_fine_long_horizon(self) -> None:
        row = dynamic_operator_offload_rows(
            model_name="gpt175b",
            system_name="chime_a100",
            subbatch_sizes=(2,),
            input_lengths=(2048,),
            output_lengths=(1100,),
            fractions=(0.0, 1.0),
            decision_interval=1,
            min_residence_tokens=0,
        )[0]
        self.assertGreaterEqual(
            float(row["dynamic_vs_best_static"]), 1.0 - 1e-12
        )

    def test_static_operator_map_is_paired_to_chime(self) -> None:
        rows = static_operator_mapping_rows(
            model_names=("gpt175b",),
            mappings=("chime", "attn_pim_control", "gpu_only"),
            batch_sizes=(1,),
            context_lengths=(128,),
            output_tokens=1,
        )
        self.assertEqual(len(rows), 3)
        baseline = next(row for row in rows if row["operator_mapping"] == "chime")
        control = next(
            row for row in rows if row["operator_mapping"] == "attn_pim_control"
        )
        self.assertEqual(baseline["speedup_vs_chime"], 1.0)
        self.assertEqual(control["speedup_vs_attn_pim_control"], 1.0)

    def test_joint_small_exact_dominates_ablations(self) -> None:
        row = joint_af_small_oracle_rows(
            model_names=("gpt175b",),
            batch_sizes=(2,),
            input_lengths=(2048,),
            output_lengths=(4,),
            fractions=(0.0, 0.5, 1.0),
            decision_interval=2,
            min_residence_tokens=2,
            max_cost_evaluations=10_000,
        )[0]
        self.assertGreaterEqual(float(row["joint_vs_best_single"]), 1.0)
        self.assertGreaterEqual(
            float(row["joint_dynamic_vs_best_static"]), 1.0
        )
        self.assertGreaterEqual(float(row["greedy_slowdown_vs_exact"]), 1.0)
        self.assertFalse(row["historical_kv_migration"])

    def test_joint_heterogeneous_profiles_hold_total_output_constant(self) -> None:
        rows = joint_af_heterogeneous_oracle_rows(
            model_names=("gpt175b",),
            profile_names=("balanced", "staircase"),
            input_lengths=(128,),
            fractions=(0.0,),
            decision_interval=256,
            min_residence_tokens=64,
            max_cost_evaluations=10_000,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [int(row["total_output_tokens"]) for row in rows],
            [640, 640],
        )
        self.assertEqual(
            [int(row["completion_waves"]) for row in rows],
            [1, 4],
        )

    def test_joint_event_replay_reports_bounded_candidate_scope(self) -> None:
        source = {
            "batch_size": 1,
            "input_tokens": 128,
            "output_tokens": 2,
            "o_proj_fractions": "0",
            "decision_interval": 2,
            "min_residence_tokens": 2,
            "max_kv_switches_per_decision": 1,
            "o_proj_only_schedule": "0:f=0,gpu=-",
            "joint_exact_schedule": "0:f=0,gpu=-",
            "joint_greedy_schedule": "0:f=0,gpu=-",
        }
        row = joint_af_event_replay_rows(
            (source,), evaluation_strides=(2,)
        )[0]

        self.assertEqual(
            row["cost_model"], "dependency_aware_joint_operator_event"
        )
        self.assertIn(
            "resource_runtime_kv_schedules",
            str(row["candidate_oracle_scope"]),
        )
        self.assertFalse(row["historical_kv_migration"])

    def test_joint_event_replay_can_compare_split_aware_oproj(self) -> None:
        source = {
            "batch_size": 1,
            "input_tokens": 128,
            "output_tokens": 2,
            "o_proj_fractions": "0|1",
            "decision_interval": 2,
            "min_residence_tokens": 2,
            "max_kv_switches_per_decision": 1,
            "o_proj_only_schedule": "0:f=1,gpu=-",
            "joint_exact_schedule": "0:f=1,gpu=r0",
            "joint_greedy_schedule": "0:f=1,gpu=r0",
        }
        row = joint_af_event_replay_rows(
            (source,),
            evaluation_strides=(1,),
            include_split_aware_oproj=True,
        )[0]

        self.assertIn("split_aware_best_candidate_us", row)
        self.assertAlmostEqual(
            float(row["split_aware_best_oproj_dynamic_us"]),
            float(row["event_best_oproj_dynamic_us"]),
            places=9,
        )
        self.assertIn(
            "split_or_gpu_origin_requests_keep_full_oproj_on_gpu",
            str(row["split_aware_dataflow"]),
        )

    def test_joint_event_replay_can_compare_owner_coupled_merge(self) -> None:
        source = {
            "batch_size": 1,
            "input_tokens": 128,
            "output_tokens": 2,
            "o_proj_fractions": "0|1",
            "decision_interval": 2,
            "min_residence_tokens": 2,
            "max_kv_switches_per_decision": 1,
            "o_proj_only_schedule": "0:f=1,gpu=-",
            "joint_exact_schedule": "0:f=1,gpu=r0",
            "joint_greedy_schedule": "0:f=1,gpu=r0",
        }
        row = joint_af_event_replay_rows(
            (source,),
            evaluation_strides=(1,),
            include_owner_coupled_merge=True,
            owner_coupled_pim_merge_cost_scale=0.0,
        )[0]

        self.assertIn("owner_coupled_best_candidate_us", row)
        self.assertAlmostEqual(
            float(row["owner_coupled_best_oproj_dynamic_us"]),
            float(row["event_best_oproj_dynamic_us"]),
            places=9,
        )
        self.assertIn(
            "pim_merge_when_any_pim_oproj_fraction",
            str(row["owner_coupled_dataflow"]),
        )


if __name__ == "__main__":
    unittest.main()
