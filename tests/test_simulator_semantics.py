from __future__ import annotations

from copy import deepcopy
import unittest
from dataclasses import replace

from elasticafd.config import project_config_from_dict
from elasticafd.simulator import SimulationCapacityError, run_simulation


def tiny_project(*, allow_gpu_prompt_fallback: bool = False):
    return project_config_from_dict(
        {
            "model": {
                "name": "tiny-dp2",
                "parameter_count_b": 0.0,
                "layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "ffn_hidden_size": 8,
                "tensor_parallel": 1,
                "data_parallel": 2,
            },
            "hardware": {
                "preset": "chime_a100",
                "gpu": {
                    "count": 2,
                    "capacity_gib": 1.0,
                    "hbm_bandwidth_tbps": 1.0,
                    "fp16_tflops": 1.0,
                },
                "pim": {"capacity_gib": 1.0},
            },
            "workload": {
                "type": "static_batch",
                "batch_size": 4,
                "input_tokens": 4,
                "output_tokens": 1,
            },
            "simulation": {
                "max_batch_size": 4,
                "record_every": 1,
                "allow_gpu_prompt_fallback": allow_gpu_prompt_fallback,
                "policy": "chime",
            },
        }
    )


class SimulatorSemanticTests(unittest.TestCase):
    def test_closed_batch_cannot_silently_run_in_capacity_waves(self) -> None:
        raw = {
            "model": {
                "name": "tiny-atomic-admission",
                "parameter_count_b": 0.0,
                "layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "ffn_hidden_size": 8,
                "tensor_parallel": 1,
                "data_parallel": 1,
            },
            "hardware": {
                "preset": "chime_a100",
                "gpu": {"count": 1, "capacity_gib": 1.0},
                # Each four-token prompt fits, but both prompts do not.
                "pim": {"capacity_gib": 1e-7},
            },
            "workload": {
                "type": "static_batch",
                "batch_size": 2,
                "input_tokens": 4,
                "output_tokens": 1,
            },
            "simulation": {"max_batch_size": 2, "policy": "chime"},
        }
        with self.assertRaisesRegex(
            SimulationCapacityError, "admitted atomically"
        ):
            run_simulation(project_config_from_dict(raw))

    def test_forced_fraction_moves_only_selected_future_kv(self) -> None:
        config = tiny_project()
        config = replace(
            config,
            workload=replace(config.workload, output_tokens=2),
            simulation=replace(
                config.simulation,
                policy=replace(
                    config.simulation.policy,
                    name="elastic",
                    optimizer="forced_fraction",
                    forced_gpu_fraction=0.5,
                ),
            ),
        )
        result = run_simulation(config)
        selected = [row for row in result.requests if row["gpu_tokens_final"]]
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["pim_tokens_final"] == 4 for row in selected))
        self.assertTrue(all(row["gpu_tokens_final"] == 2 for row in selected))
        self.assertEqual(result.summary["forced_gpu_request_count"], 2)
        self.assertEqual(result.summary["tail_switches"], 2)

    def test_forced_zero_fraction_matches_chime(self) -> None:
        baseline = run_simulation(tiny_project())
        config = tiny_project()
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                policy=replace(
                    config.simulation.policy,
                    name="elastic",
                    optimizer="forced_fraction",
                    forced_gpu_fraction=0.0,
                ),
            ),
        )
        forced = run_simulation(config)
        self.assertAlmostEqual(
            baseline.summary["simulated_time_s"],
            forced.summary["simulated_time_s"],
        )
        self.assertEqual(forced.summary["tail_switches"], 0)

    def test_online_static_fraction_assigns_only_future_kv_once(self) -> None:
        raw = {
            "model": {
                "name": "tiny-online-static",
                "parameter_count_b": 0.0,
                "layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "ffn_hidden_size": 8,
                "tensor_parallel": 1,
                "data_parallel": 1,
            },
            "hardware": {
                "preset": "chime_a100",
                "gpu": {"count": 1, "capacity_gib": 1.0},
                "pim": {"capacity_gib": 1.0},
            },
            "workload": {
                "type": "trace_serving",
                "requests": (
                    {"request_id": "a", "arrival_us": 0, "input_tokens": 4, "output_tokens": 2},
                    {"request_id": "b", "arrival_us": 0, "input_tokens": 4, "output_tokens": 2},
                ),
            },
            "simulation": {
                "max_batch_size": 2,
                "record_every": 1,
                "policy": {
                    "name": "elastic",
                    "optimizer": "online_static_fraction",
                    "forced_gpu_fraction": 0.5,
                    "event_driven_replan": True,
                },
            },
        }

        result = run_simulation(project_config_from_dict(raw))

        gpu = [row for row in result.requests if row["gpu_tokens_final"]]
        self.assertEqual(len(gpu), 1)
        self.assertEqual(gpu[0]["pim_tokens_final"], 4)
        self.assertEqual(gpu[0]["gpu_tokens_final"], 2)
        self.assertEqual(result.summary["online_static_gpu_requests"], 1)
        self.assertEqual(result.summary["tail_switches"], 1)

    def test_af_mechanism_metrics_are_reported(self) -> None:
        result = run_simulation(tiny_project())
        for field in (
            "af_ta_work_s",
            "af_tf_work_s",
            "af_ideal_stage_s",
            "af_iteration_cost_s",
            "af_imbalance_s",
            "af_balance_ratio",
            "h2d_transfer_gib",
            "d2h_transfer_gib",
            "pim_kv_scan_gib",
            "gpu_kv_scan_gib",
        ):
            self.assertIn(field, result.summary)
        self.assertGreaterEqual(result.summary["af_balance_ratio"], 0.0)
        self.assertLessEqual(result.summary["af_balance_ratio"], 1.0)
        self.assertGreaterEqual(
            result.summary["af_actual_minus_ideal_s"], -1e-12
        )

        event_config = replace(
            tiny_project(),
            simulation=replace(tiny_project().simulation, pipeline_model="rankset_event"),
        )
        event = run_simulation(event_config)
        self.assertGreaterEqual(event.summary["af_actual_minus_ideal_s"], -1e-12)

    def test_zero_split_overhead_is_an_optimistic_bound(self) -> None:
        config = tiny_project()
        config = replace(
            config,
            workload=replace(config.workload, output_tokens=3),
            simulation=replace(
                config.simulation,
                policy=replace(
                    config.simulation.policy,
                    name="elastic",
                    optimizer="forced_fraction",
                    forced_gpu_fraction=0.5,
                ),
            ),
        )
        charged = run_simulation(config)
        ideal = run_simulation(
            replace(
                config,
                simulation=replace(config.simulation, split_overhead_scale=0.0),
            )
        )
        self.assertLessEqual(
            ideal.summary["simulated_time_s"],
            charged.summary["simulated_time_s"],
        )
        self.assertLessEqual(
            ideal.summary["d2h_transfer_gib"],
            charged.summary["d2h_transfer_gib"],
        )

    def test_static_batch_is_pinned_round_robin_to_dp_replicas(self) -> None:
        result = run_simulation(tiny_project())
        self.assertEqual(
            [row["gpu_replica"] for row in result.requests],
            [0, 1, 0, 1],
        )
        self.assertEqual(result.summary["tail_switches"], 0)
        self.assertEqual(result.summary["peak_gpu_kv_gib"], 0.0)

    def test_gpu_prompt_fallback_is_disabled_by_default(self) -> None:
        hardware = {
            "preset": "chime_a100",
            "gpu": {
                "count": 2,
                "capacity_gib": 1.0,
                "hbm_bandwidth_tbps": 1.0,
                "fp16_tflops": 1.0,
            },
            "pim": {"capacity_gib": 1e-9},
        }
        base = {
            "model": {
                "name": "tiny-dp2",
                "parameter_count_b": 0.0,
                "layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "ffn_hidden_size": 8,
                "tensor_parallel": 1,
                "data_parallel": 2,
            },
            "hardware": hardware,
            "workload": {
                "type": "static_batch",
                "batch_size": 1,
                "input_tokens": 10,
                "output_tokens": 1,
            },
            "simulation": {
                "max_batch_size": 1,
                "record_every": 1,
                "policy": {"name": "elastic"},
            },
        }
        with self.assertRaises(SimulationCapacityError):
            run_simulation(project_config_from_dict(base))

        base["simulation"]["allow_gpu_prompt_fallback"] = True
        result = run_simulation(project_config_from_dict(base))
        self.assertEqual(result.requests[0]["pim_tokens_final"], 0)
        self.assertEqual(result.requests[0]["gpu_tokens_final"], 11)

    def test_queue_and_slo_metrics_are_reported(self) -> None:
        config = tiny_project()
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                slo_queue_ms=1.0,
                slo_tpot_ms=1e9,
                slo_attainment_target=0.99,
            ),
        )
        result = run_simulation(config)
        self.assertEqual(result.summary["latency_ms"]["queue_p95"], 0.0)
        self.assertTrue(result.summary["slo"]["evaluated"])
        self.assertEqual(result.summary["slo"]["attainment"], 1.0)
        self.assertTrue(result.summary["slo"]["target_met"])
        self.assertTrue(all(row["meets_slo"] for row in result.requests))

    def test_event_driven_replan_runs_on_initial_admission(self) -> None:
        config = tiny_project()
        config = replace(
            config,
            simulation=replace(
                config.simulation,
                policy=replace(
                    config.simulation.policy,
                    event_driven_replan=True,
                ),
            ),
        )
        result = run_simulation(config)
        self.assertTrue(result.summary["policy_event_driven_replan"])
        self.assertEqual(result.summary["event_policy_decisions"], 1)

    def test_policy_can_select_with_analytic_cost_and_execute_event_dag(self) -> None:
        config = tiny_project()
        config = replace(
            config,
            workload=replace(config.workload, output_tokens=2),
            simulation=replace(
                config.simulation,
                pipeline_model="rankset_event",
                policy_pipeline_model="analytic",
                policy=replace(
                    config.simulation.policy,
                    name="elastic",
                    prediction="oracle",
                ),
            ),
        )

        result = run_simulation(config)

        self.assertEqual(result.summary["pipeline_model"], "rankset_event")
        self.assertEqual(result.summary["policy_pipeline_model"], "analytic")

    def test_switch_execution_cost_is_charged_to_simulated_time(self) -> None:
        raw = {
            "model": {
                "name": "tiny-switch-cost",
                "parameter_count_b": 0.0,
                "layers": 1,
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_kv_heads": 4,
                "ffn_hidden_size": 32,
                "tensor_parallel": 1,
                "data_parallel": 1,
            },
            "hardware": {
                "preset": "chime_a100",
                "gpu": {
                    "count": 1,
                    "capacity_gib": 1.0,
                    "hbm_bandwidth_tbps": 1.0,
                    "fp16_tflops": 1.0,
                },
                "pim": {
                    "capacity_gib": 1.0,
                    "aggregate_bandwidth_tbps": 0.000001,
                },
            },
            "workload": {
                "type": "static_batch",
                "batch_size": 1,
                "input_tokens": 1024,
                "output_tokens": 8,
            },
            "simulation": {
                "max_batch_size": 1,
                "record_every": 1,
                "policy": {
                    "name": "elastic",
                    "optimizer": "lifetime_greedy",
                    "prediction": "oracle",
                    "decision_interval": 4,
                    "lifetime_evaluation_stride_tokens": 1,
                    "min_segment_tokens": 0,
                    "min_switch_interval": 0,
                    "relative_improvement_threshold": 0.0,
                    "max_switches_per_decision": 1,
                    "switch_penalty_us": 0.0,
                },
            },
        }
        free = run_simulation(project_config_from_dict(raw))
        charged_raw = deepcopy(raw)
        charged_raw["simulation"]["policy"]["switch_execution_us"] = 17.0
        charged = run_simulation(project_config_from_dict(charged_raw))
        self.assertEqual(free.summary["tail_switches"], 1)
        self.assertAlmostEqual(
            charged.summary["simulated_time_s"]
            - free.summary["simulated_time_s"],
            17e-6,
        )
        self.assertEqual(charged.summary["switch_execution_time_s"], 17e-6)
        self.assertEqual(charged.switches[0]["execution_overhead_us"], 17.0)


if __name__ == "__main__":
    unittest.main()
