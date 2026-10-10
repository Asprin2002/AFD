from __future__ import annotations

import unittest

from elasticafd.capacity import views_fit_capacity
from elasticafd.cost import ChimeCostModel
from elasticafd.joint import (
    JointAFEventModel,
    JointAFResourceModel,
    JointAFScheduler,
    JointAction,
    parse_joint_schedule,
)
from elasticafd.specs import (
    GpuSpec,
    MODEL_PRESETS,
    ModelSpec,
    SYSTEM_PRESETS,
    SystemSpec,
)
from elasticafd.state import Location, RequestState, RequestView


def tiny_model(*, data_parallel: int = 1) -> ModelSpec:
    return ModelSpec(
        name="tiny",
        parameter_count_b=0.0,
        layers=1,
        hidden_size=4,
        num_attention_heads=1,
        num_kv_heads=1,
        ffn_hidden_size=8,
        tensor_parallel=1,
        data_parallel=data_parallel,
    )


class ImmutableSegmentTests(unittest.TestCase):
    def test_switch_changes_only_future_kv(self) -> None:
        request = RequestState("r0", 0.0, input_tokens=10, output_tokens=4)
        request.append_generated_token()
        self.assertTrue(request.switch_tail(Location.GPU, iteration=1))
        request.append_generated_token()

        self.assertEqual(request.tokens_at(Location.PIM), 11)
        self.assertEqual(request.tokens_at(Location.GPU), 1)
        self.assertEqual(
            [(segment.location, segment.start_token, segment.end_token)
             for segment in request.segments],
            [
                (Location.PIM, 0, 11),
                (Location.GPU, 11, 12),
            ],
        )

    def test_joint_action_keeps_prompt_segment_immutable(self) -> None:
        request = RequestState("joint", 0.0, input_tokens=16, output_tokens=4)
        cost = JointAFResourceModel(
            MODEL_PRESETS["gpt175b"],
            SYSTEM_PRESETS["chime_a100"],
            max_o_proj_pim_fraction=1.0,
        )
        scheduler = JointAFScheduler(
            cost,
            fractions=(0.0, 1.0),
            decision_interval=2,
            min_kv_segment_tokens=0,
            min_kv_switch_interval=0,
            min_operator_residence_tokens=0,
        )
        action = JointAction.from_mapping({"joint": Location.GPU}, 1.0)
        block = scheduler._simulate_block(
            [request], 0, 0.0, -10**18, action
        )
        self.assertIsNotNone(block)
        assert block is not None
        projected = block.survivors[0]
        self.assertEqual(projected.segments[0].location, Location.PIM)
        self.assertEqual(projected.segments[0].tokens, 16)
        self.assertEqual(projected.segments[1].location, Location.GPU)
        self.assertEqual(projected.segments[1].tokens, 2)

    def test_joint_static_creation_is_not_limited_by_runtime_switch_cap(self) -> None:
        requests = [
            RequestState(f"r{index}", 0.0, input_tokens=16, output_tokens=1)
            for index in range(2)
        ]
        cost = JointAFResourceModel(
            MODEL_PRESETS["gpt175b"],
            SYSTEM_PRESETS["chime_a100"],
            max_o_proj_pim_fraction=0.0,
        )
        scheduler = JointAFScheduler(
            cost,
            fractions=(0.0,),
            max_kv_switches_per_decision=0,
        )
        result = scheduler.run_best_static(requests, allow_gpu_kv=True)

        # Homogeneous requests on one data-parallel replica have three unique
        # counts: zero, one, or two future tails on GPU. The runtime cap and
        # residence requirement must not remove them at request creation.
        self.assertEqual(result.cost_evaluations, 3)

    def test_joint_dynamic_initial_placement_is_not_a_runtime_switch(self) -> None:
        requests = [
            RequestState(f"r{index}", 0.0, input_tokens=16, output_tokens=2)
            for index in range(2)
        ]
        cost = JointAFResourceModel(
            MODEL_PRESETS["gpt175b"],
            SYSTEM_PRESETS["chime_a100"],
            max_o_proj_pim_fraction=0.0,
        )
        scheduler = JointAFScheduler(
            cost,
            fractions=(0.0,),
            decision_interval=1,
            min_kv_segment_tokens=64,
            max_kv_switches_per_decision=0,
        )
        actions = scheduler._actions(
            requests,
            0,
            0.0,
            -10**18,
            allow_gpu_kv=True,
            allow_operator=False,
            coordinate_only=False,
        )
        all_gpu = next(
            action
            for action in actions
            if all(
                target is Location.GPU
                for _, target in action.tail_targets
            )
        )
        block = scheduler._simulate_block(
            requests, 0, 0.0, -10**18, all_gpu
        )
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block.kv_switches, 0)
        self.assertEqual(
            [request.last_switch_iteration for request in block.survivors],
            [0, 0],
        )
        self.assertEqual(
            [request.tokens_at(Location.GPU) for request in block.survivors],
            [1, 1],
        )


class JointEventModelTests(unittest.TestCase):
    def test_f0_matches_generic_attention_pim_control(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        views = [
            RequestView("r0", (16,), (), Location.PIM),
            RequestView("r1", (24,), (), Location.PIM),
        ]
        control = ChimeCostModel(
            model,
            system,
            operator_mapping="attn_pim_control",
        ).iteration_cost(views)
        joint = JointAFEventModel(
            model, system, max_o_proj_pim_fraction=0.0
        ).iteration_cost(views, 0.0)

        self.assertAlmostEqual(joint.total_us, control.total_us, places=9)
        self.assertAlmostEqual(joint.gpu_busy_us, control.gpu_busy_us, places=9)
        self.assertAlmostEqual(
            joint.pim_busy_us, control.pim_core_busy_us, places=9
        )
        self.assertAlmostEqual(
            joint.link_busy_us, control.link_busy_us, places=9
        )

    def test_pim_oproj_waits_for_local_attention(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        views = [RequestView("r0", (16,), (), Location.PIM)]
        cost = JointAFEventModel(
            model, system, max_o_proj_pim_fraction=1.0
        ).iteration_cost(views, 1.0)
        attention = next(
            event for event in cost.events if event.kind == "pim_attention"
        )
        o_proj = next(
            event for event in cost.events if event.kind == "pim_o_proj"
        )

        self.assertGreaterEqual(o_proj.start_us, attention.end_us)
        self.assertTrue(set(attention.resources) & set(o_proj.resources))

    def test_split_aware_oproj_keeps_split_request_projection_on_gpu(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        views = [RequestView("r0", (16,), (4,), Location.GPU)]
        cost = JointAFEventModel(
            model,
            system,
            max_o_proj_pim_fraction=1.0,
            o_proj_routing="split_aware_requests",
        ).iteration_cost(views, 1.0)

        kinds = [event.kind for event in cost.events]
        self.assertIn("merge", kinds)
        self.assertIn("gpu_o_proj", kinds)
        self.assertNotIn("o_proj_h2d", kinds)
        self.assertNotIn("pim_o_proj", kinds)
        self.assertEqual(
            cost.pipeline_model, "dependency_event_split_aware_oproj"
        )

    def test_split_aware_f0_matches_uniform_routing(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        views = [
            RequestView("pim", (16,), (), Location.PIM),
            RequestView("split", (16,), (4,), Location.GPU),
        ]
        uniform = JointAFEventModel(
            model, system, max_o_proj_pim_fraction=1.0
        ).iteration_cost(views, 0.0)
        split_aware = JointAFEventModel(
            model,
            system,
            max_o_proj_pim_fraction=1.0,
            o_proj_routing="split_aware_requests",
        ).iteration_cost(views, 0.0)

        self.assertAlmostEqual(split_aware.total_us, uniform.total_us, places=9)
        self.assertAlmostEqual(
            split_aware.gpu_o_proj_us, uniform.gpu_o_proj_us, places=9
        )

    def test_pim_oproj_can_own_split_softmax_merge(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        views = [RequestView("r0", (16,), (4,), Location.GPU)]
        cost = JointAFEventModel(
            model,
            system,
            max_o_proj_pim_fraction=1.0,
            split_merge_owner="pim_when_pim_oproj",
        ).iteration_cost(views, 1.0)

        kinds = [event.kind for event in cost.events]
        self.assertIn("gpu_partial_h2d", kinds)
        self.assertIn("pim_merge", kinds)
        self.assertIn("pim_o_proj", kinds)
        self.assertIn("o_proj_d2h", kinds)
        self.assertNotIn("pim_attention_d2h", kinds)
        self.assertNotIn("merge", kinds)
        self.assertNotIn("o_proj_h2d", kinds)
        merge = next(event for event in cost.events if event.kind == "pim_merge")
        attention = next(
            event for event in cost.events if event.kind == "pim_attention"
        )
        o_proj = next(
            event for event in cost.events if event.kind == "pim_o_proj"
        )
        self.assertGreaterEqual(merge.start_us, attention.end_us)
        self.assertGreaterEqual(o_proj.start_us, merge.end_us)
        self.assertEqual(
            cost.pipeline_model, "dependency_event_owner_coupled_merge"
        )

    def test_owner_coupled_merge_f0_matches_chime_dataflow(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        views = [RequestView("r0", (16,), (4,), Location.GPU)]
        gpu_merge = JointAFEventModel(
            model, system, max_o_proj_pim_fraction=1.0
        ).iteration_cost(views, 0.0)
        owner_coupled = JointAFEventModel(
            model,
            system,
            max_o_proj_pim_fraction=1.0,
            split_merge_owner="pim_when_pim_oproj",
        ).iteration_cost(views, 0.0)

        self.assertAlmostEqual(
            owner_coupled.total_us, gpu_merge.total_us, places=9
        )
        self.assertIn("merge", [event.kind for event in owner_coupled.events])
        self.assertNotIn(
            "pim_merge", [event.kind for event in owner_coupled.events]
        )

    def test_partial_oproj_returns_pim_merge_for_gpu_slice(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        views = [RequestView("r0", (16,), (4,), Location.GPU)]
        cost = JointAFEventModel(
            model,
            system,
            max_o_proj_pim_fraction=1.0,
            split_merge_owner="pim_when_pim_oproj",
        ).iteration_cost(views, 0.5)

        kinds = [event.kind for event in cost.events]
        self.assertIn("pim_merge_d2h", kinds)
        self.assertIn("gpu_o_proj", kinds)
        self.assertIn("pim_o_proj", kinds)

    def test_stride_one_replay_matches_fixed_schedule(self) -> None:
        model = tiny_model()
        system = SystemSpec(gpu=GpuSpec(count=1, capacity_gib=1.0))
        requests = [RequestState("r0", 0.0, 16, 3)]
        cost = JointAFEventModel(
            model, system, max_o_proj_pim_fraction=1.0
        )
        scheduler = JointAFScheduler(
            cost,
            fractions=(0.0, 1.0),
            decision_interval=1,
            min_kv_segment_tokens=0,
            min_kv_switch_interval=0,
            min_operator_residence_tokens=0,
        )
        schedule = parse_joint_schedule("0:f=1,gpu=-", ("r0",))
        replay = scheduler.replay_sampled(
            requests, schedule, evaluation_stride_tokens=1
        )
        fixed_us = scheduler._simulate_fixed(requests, schedule[0][1])

        self.assertIsNotNone(fixed_us)
        assert fixed_us is not None
        self.assertAlmostEqual(replay.total_us, fixed_us, places=9)


class ReplicaCapacityTests(unittest.TestCase):
    def test_one_replica_cannot_borrow_anothers_capacity(self) -> None:
        model = tiny_model(data_parallel=2)
        system = SystemSpec(gpu=GpuSpec(count=2, capacity_gib=1.0))
        same_replica = (
            RequestView("a", (), (4,), Location.GPU, gpu_replica=0),
            RequestView("b", (), (4,), Location.GPU, gpu_replica=0),
        )
        balanced = (
            same_replica[0],
            RequestView("b", (), (4,), Location.GPU, gpu_replica=1),
        )

        # Each request uses 4 tokens * 16 B/token = 64 B. Aggregate GPU
        # capacity is 200 B, but a replica is capped at 100 B.
        self.assertFalse(
            views_fit_capacity(
                same_replica,
                model,
                system,
                pim_capacity_bytes=1_000,
                gpu_replica_capacity_bytes=100,
            )
        )
        self.assertTrue(
            views_fit_capacity(
                balanced,
                model,
                system,
                pim_capacity_bytes=1_000,
                gpu_replica_capacity_bytes=100,
            )
        )

    def test_gpu_preset_must_match_tp_times_dp(self) -> None:
        model = tiny_model(data_parallel=2)
        system = SystemSpec(gpu=GpuSpec(count=4, capacity_gib=4.0))
        with self.assertRaisesRegex(ValueError, "partial-system deployment"):
            system.gpu_kv_capacity_bytes(model)

    def test_explicit_gpu_kv_budget_caps_physical_free_memory(self) -> None:
        model = tiny_model(data_parallel=2)
        system = SystemSpec(
            gpu=GpuSpec(
                count=2,
                capacity_gib=2.0,
                kv_capacity_limit_gib=0.25,
            )
        )
        self.assertAlmostEqual(
            system.gpu_kv_capacity_bytes(model) / (1 << 30), 0.25
        )


if __name__ == "__main__":
    unittest.main()
