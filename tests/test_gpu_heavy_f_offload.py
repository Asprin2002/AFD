from __future__ import annotations

from dataclasses import replace
import unittest

from elasticafd.mapping import ExecutionTarget, OPERATOR_MAPPINGS
from elasticafd.state import RequestState, views_for_requests
from tools.run_gpu_heavy_f_offload import (
    ALL_F_OPERATORS,
    CONTROL,
    F_MAPPINGS,
    FactorialFCostModel,
    base_config,
)


class GpuHeavyFOffloadTests(unittest.TestCase):
    def test_factorial_family_contains_every_dense_placement_once(self):
        placements = set()
        for name in F_MAPPINGS:
            mapping = OPERATOR_MAPPINGS[name]
            self.assertIs(mapping.attention, ExecutionTarget.PIM)
            placements.add((mapping.qkv, mapping.o_proj, mapping.ffn))
        self.assertEqual(len(placements), 8)

    def test_dense_sensitivity_does_not_change_attention(self):
        config = base_config(4)
        config = replace(config, model=replace(config.model, layers=2))
        baseline = FactorialFCostModel(
            config,
            CONTROL,
            resident_operators=ALL_F_OPERATORS,
        )
        scaled = FactorialFCostModel(
            config,
            CONTROL,
            resident_operators=ALL_F_OPERATORS,
            pim_dense_bandwidth_scale=2.0,
            pim_dense_compute_scale=4.0,
        )
        views = views_for_requests(
            [RequestState(f"r{i}", 0.0, 512, 1) for i in range(4)]
        )
        self.assertEqual(baseline.pim_layer_cost(views), scaled.pim_layer_cost(views))
        base_dense = baseline.pim_matrix_layer_cost("ffn", 4)
        scaled_dense = scaled.pim_matrix_layer_cost("ffn", 4)
        self.assertAlmostEqual(scaled_dense.memory_us, base_dense.memory_us / 2)
        self.assertAlmostEqual(scaled_dense.compute_us, base_dense.compute_us / 4)

    def test_all_open_keeps_gpu_weights_and_reserves_pim_copy(self):
        config = base_config(4)
        config = replace(config, model=replace(config.model, layers=2))
        model = FactorialFCostModel(
            config,
            "f_all_pim",
            resident_operators=ALL_F_OPERATORS,
        )
        self.assertGreater(model.pim_persistent_weight_bytes, 0)
        self.assertEqual(model.gpu_offloaded_weight_bytes, 0)
        self.assertEqual(
            model.gpu_kv_capacity_bytes,
            config.system.gpu_kv_capacity_bytes(config.model),
        )
        cost = model.iteration_cost(
            views_for_requests(
                [RequestState(f"r{i}", 0.0, 128, 1) for i in range(4)]
            )
        )
        kinds = {event.kind for event in cost.events}
        self.assertTrue({"pim_qkv", "pim_o_proj", "pim_ffn"} <= kinds)


if __name__ == "__main__":
    unittest.main()
