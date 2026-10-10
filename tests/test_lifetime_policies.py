from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from elasticafd.config import project_config_from_dict
from elasticafd.policy import (
    LifetimeGreedyPlacementPolicy,
    RollingBeamPlacementPolicy,
)
from elasticafd.simulator import Simulator, run_simulation
from elasticafd.state import (
    Location,
    RequestState,
    make_request_view_with_runs,
)


def tiny_policy_project(optimizer: str):
    return project_config_from_dict(
        {
            "model": {
                "name": "tiny-policy",
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
                "pim": {"capacity_gib": 1.0},
            },
            "workload": {
                "type": "static_batch",
                "batch_size": 3,
                "input_tokens": 64,
                "output_tokens": 8,
            },
            "simulation": {
                "max_batch_size": 3,
                "record_every": 100,
                "policy": {
                    "name": "elastic",
                    "optimizer": optimizer,
                    "prediction": "oracle",
                    "decision_interval": 4,
                    "planning_horizon_tokens": 4,
                    "lifetime_evaluation_stride_tokens": 1,
                    "min_segment_tokens": 1,
                    "min_switch_interval": 4,
                    "relative_improvement_threshold": 0.0,
                    "max_switches_per_decision": 3,
                    "exact_max_candidates": 3,
                    "exact_max_cost_evaluations": 10000,
                },
            },
        }
    )


class LifetimeProjectionTests(unittest.TestCase):
    def test_multiple_projected_runs_do_not_mutate_history(self) -> None:
        request = RequestState("r", 0.0, input_tokens=4, output_tokens=8)
        view = make_request_view_with_runs(
            request,
            ((Location.GPU, 2), (Location.PIM, 3)),
        )
        self.assertEqual(view.pim_segment_lengths, (4, 3))
        self.assertEqual(view.gpu_segment_lengths, (2,))
        self.assertEqual(view.tail_location, Location.PIM)
        self.assertEqual([(segment.location, segment.tokens) for segment in request.segments], [(Location.PIM, 4)])

    def test_lifetime_greedy_can_cross_a_same_state_group_threshold(self) -> None:
        policy = Simulator(tiny_policy_project("lifetime_greedy")).policy
        self.assertIsInstance(policy, LifetimeGreedyPlacementPolicy)
        requests = [
            RequestState(f"r-{index}", 0.0, input_tokens=64, output_tokens=8)
            for index in range(3)
        ]

        # A singleton improves by only 0.05%, below the configured 0.2%
        # hysteresis, while a two-request action improves by 10%.
        policy.config = type(policy.config)(
            **{
                **policy.config.__dict__,
                "relative_improvement_threshold": 0.002,
                "max_switches_per_decision": 2,
            }
        )
        policy._objective = lambda _requests, overrides: (
            100.0
            if not overrides
            else 99.95
            if len(overrides) == 1
            else 90.0
        )

        decisions = policy.decide(requests, iteration=0)

        self.assertEqual(len(decisions), 2)
        self.assertTrue(
            all(
                decision.reason == "lifetime_value_per_byte_time_group"
                for decision in decisions
            )
        )

    def test_source_bottleneck_guard_rejects_wrong_way_switch(self) -> None:
        policy = Simulator(tiny_policy_project("lifetime_greedy")).policy
        self.assertIsInstance(policy, LifetimeGreedyPlacementPolicy)
        policy.config = replace(
            policy.config,
            require_source_bottleneck=True,
            relative_improvement_threshold=0.0,
        )
        requests = [RequestState("r", 0.0, input_tokens=64, output_tokens=8)]
        policy._objective = lambda _requests, overrides: 90.0 if overrides else 100.0
        policy.cost_model.iteration_cost = lambda _views: SimpleNamespace(
            pim_core_busy_us=10.0,
            link_busy_us=1.0,
            gpu_busy_us=20.0,
        )

        decisions = policy.decide(requests, iteration=0)

        self.assertEqual(decisions, [])


class ExactPolicyTests(unittest.TestCase):
    def test_global_exact_is_not_slower_than_feasible_greedy_schedules(self) -> None:
        summaries = {
            optimizer: run_simulation(tiny_policy_project(optimizer)).summary
            for optimizer in (
                "horizon_greedy",
                "lifetime_greedy",
                "rolling_beam",
                "global_exact",
            )
        }
        exact_time = float(summaries["global_exact"]["simulated_time_s"])
        self.assertLessEqual(
            exact_time,
            float(summaries["horizon_greedy"]["simulated_time_s"]) + 1e-12,
        )
        self.assertLessEqual(
            exact_time,
            float(summaries["lifetime_greedy"]["simulated_time_s"]) + 1e-12,
        )
        self.assertLessEqual(
            exact_time,
            float(summaries["rolling_beam"]["simulated_time_s"]) + 1e-12,
        )
        self.assertEqual(
            summaries["global_exact"]["policy_optimizer"], "global_exact"
        )

    def test_global_exact_rejects_future_arrivals(self) -> None:
        raw = {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "trace_serving",
                "requests": (
                    {
                        "request_id": "r",
                        "arrival_ms": 0.0,
                        "input_tokens": 8,
                        "output_tokens": 8,
                    },
                ),
            },
            "simulation": {
                "policy": {
                    "name": "elastic",
                    "optimizer": "global_exact",
                    "prediction": "oracle",
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "static_batch or closed_batch"):
            project_config_from_dict(raw)

    def test_homogeneous_modes_is_bounded_by_chime(self) -> None:
        summaries = {
            "homogeneous_modes": run_simulation(
                tiny_policy_project("homogeneous_modes")
            ).summary
        }
        mode_time = float(summaries["homogeneous_modes"]["simulated_time_s"])
        chime = project_config_from_dict(
            {
                "model": {
                    "name": "tiny-policy",
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
                    "pim": {"capacity_gib": 1.0},
                },
                "workload": {
                    "type": "static_batch",
                    "batch_size": 3,
                    "input_tokens": 64,
                    "output_tokens": 8,
                },
                "simulation": {"max_batch_size": 3, "policy": "chime"},
            }
        )
        chime_time = float(run_simulation(chime).summary["simulated_time_s"])
        self.assertLessEqual(mode_time, chime_time + 1e-12)
        self.assertIn("homogeneous_mode_count", summaries["homogeneous_modes"])

    def test_homogeneous_modes_rejects_heterogeneous_closed_batch(self) -> None:
        raw = {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "closed_batch",
                "requests": (
                    {
                        "request_id": "short",
                        "arrival_ms": 0.0,
                        "input_tokens": 64,
                        "output_tokens": 8,
                    },
                    {
                        "request_id": "long",
                        "arrival_ms": 0.0,
                        "input_tokens": 64,
                        "output_tokens": 16,
                    },
                ),
            },
            "simulation": {
                "max_batch_size": 2,
                "policy": {
                    "name": "elastic",
                    "optimizer": "homogeneous_modes",
                    "prediction": "oracle",
                    "max_switches_per_decision": 2,
                },
            },
        }
        with self.assertRaisesRegex(RuntimeError, "homogeneous batch"):
            run_simulation(project_config_from_dict(raw))


class BalanceCurveTests(unittest.TestCase):
    @staticmethod
    def _project(batch_size: int, input_tokens: int, output_tokens: int):
        return project_config_from_dict(
            {
                "model": "gpt175b",
                "hardware": "chime_a100",
                "workload": {
                    "type": "static_batch",
                    "batch_size": batch_size,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "simulation": {
                    "max_batch_size": batch_size,
                    "record_every": output_tokens,
                    "policy": {
                        "name": "elastic",
                        "optimizer": "balance_curve",
                        "prediction": "oracle",
                        "decision_interval": 64,
                        "min_segment_tokens": 64,
                        "min_switch_interval": 64,
                        "max_switches_per_decision": batch_size,
                    },
                },
            }
        )

    def test_closed_form_plan_preserves_prompt_kv(self) -> None:
        result = run_simulation(self._project(16, 5200, 256))
        self.assertGreater(len(result.switches), 0)
        self.assertEqual(result.summary["balance_curve_schedule_searches"], 0)
        for request in result.requests:
            self.assertTrue(str(request["segment_layout"]).startswith("pim:0-5200"))

    def test_no_benefit_case_stays_at_chime(self) -> None:
        balance = run_simulation(self._project(4, 5200, 256))
        raw = {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "static_batch",
                "batch_size": 4,
                "input_tokens": 5200,
                "output_tokens": 256,
            },
            "simulation": {"max_batch_size": 4, "policy": "chime"},
        }
        chime = run_simulation(project_config_from_dict(raw))
        self.assertEqual(len(balance.switches), 0)
        self.assertAlmostEqual(
            float(balance.summary["simulated_time_s"]),
            float(chime.summary["simulated_time_s"]),
            places=12,
        )

    def test_balance_curve_rejects_non_static_workloads(self) -> None:
        raw = {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "closed_batch",
                "requests": (
                    {
                        "request_id": "r",
                        "arrival_ms": 0.0,
                        "input_tokens": 64,
                        "output_tokens": 64,
                    },
                ),
            },
            "simulation": {
                "policy": {
                    "name": "elastic",
                    "optimizer": "balance_curve",
                    "prediction": "oracle",
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "only for static_batch"):
            project_config_from_dict(raw)


class RollingBeamTests(unittest.TestCase):
    def test_actions_can_switch_an_entire_same_location_group(self) -> None:
        policy = Simulator(tiny_policy_project("rolling_beam")).policy
        self.assertIsInstance(policy, RollingBeamPlacementPolicy)
        requests = [
            RequestState(f"r-{index}", 0.0, input_tokens=64, output_tokens=8)
            for index in range(4)
        ]
        for request in requests[:2]:
            request.switch_tail(Location.GPU, 0)
            request.append_generated_token()
        actions = policy._candidate_actions(requests, iteration=4)
        self.assertTrue(
            any(
                all(targets[request.request_id] is Location.GPU for request in requests)
                for targets in actions
            )
        )

    def test_future_arrival_search_admits_known_pending_request(self) -> None:
        raw = {
            "model": {
                "name": "tiny-future",
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
                "pim": {"capacity_gib": 1.0},
            },
            "workload": {
                "type": "trace_serving",
                "requests": (
                    {
                        "request_id": "active",
                        "arrival_us": 0.0,
                        "input_tokens": 64,
                        "output_tokens": 8,
                    },
                    {
                        "request_id": "future",
                        "arrival_us": 0.01,
                        "input_tokens": 64,
                        "output_tokens": 8,
                    },
                ),
            },
            "simulation": {
                "max_batch_size": 2,
                "policy": {
                    "name": "elastic",
                    "optimizer": "rolling_beam",
                    "prediction": "oracle",
                    "beam_include_future_arrivals": True,
                    "decision_interval": 4,
                    "beam_depth": 2,
                    "beam_width": 2,
                    "beam_request_candidates": 1,
                    "beam_evaluation_stride_tokens": 1,
                    "min_segment_tokens": 1,
                    "min_switch_interval": 1,
                    "relative_improvement_threshold": 0.0,
                },
            },
        }
        simulator = Simulator(project_config_from_dict(raw))
        simulator._admit_ready()
        policy = simulator.policy
        self.assertIsInstance(policy, RollingBeamPlacementPolicy)
        policy.update_runtime_context(
            clock_us=simulator.clock_us,
            pending=simulator.pending,
            max_batch_size=2,
            initial_location=Location.PIM,
        )
        stay = {
            request.request_id: request.tail_location
            for request in simulator.active
        }
        block = policy._simulate_block(
            simulator.active,
            iteration=0,
            targets=stay,
            pending=simulator.pending,
            clock_us=simulator.clock_us,
        )
        self.assertIsNotNone(block)
        assert block is not None
        survivors = block[1]
        self.assertEqual(
            {request.request_id for request in survivors},
            {"active", "future"},
        )
        future = next(
            request for request in survivors if request.request_id == "future"
        )
        self.assertEqual(future.segments[0].location, Location.PIM)

    def test_future_arrival_oracle_rejects_closed_workload(self) -> None:
        config = tiny_policy_project("rolling_beam")
        raw = {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "static_batch",
                "batch_size": 1,
                "input_tokens": 8,
                "output_tokens": 8,
            },
            "simulation": {
                "policy": {
                    "name": "elastic",
                    "optimizer": "rolling_beam",
                    "prediction": "oracle",
                    "beam_include_future_arrivals": True,
                }
            },
        }
        self.assertEqual(config.workload.type, "static_batch")
        with self.assertRaisesRegex(ValueError, "trace_serving"):
            project_config_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
