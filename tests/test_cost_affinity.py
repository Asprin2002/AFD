from __future__ import annotations

import unittest

from elasticafd.cost import ChimeCostModel
from elasticafd.specs import GpuSpec, SystemSpec
from elasticafd.state import Location, RequestView

from tests.test_state_and_capacity import tiny_model


class CostAffinityTests(unittest.TestCase):
    def test_dense_latency_observes_fixed_replica_assignment(self) -> None:
        model = tiny_model(data_parallel=2)
        system = SystemSpec(
            gpu=GpuSpec(
                count=2,
                capacity_gib=2.0,
                hbm_bandwidth_tbps=1_000_000.0,
                fp16_tflops=0.000001,
                dense_hbm_efficiency=1.0,
                compute_efficiency=1.0,
            )
        )
        cost_model = ChimeCostModel(model, system)
        balanced = (
            RequestView("a", (1,), (), Location.PIM, gpu_replica=0),
            RequestView("b", (1,), (), Location.PIM, gpu_replica=1),
        )
        unbalanced = (
            RequestView("a", (1,), (), Location.PIM, gpu_replica=0),
            RequestView("b", (1,), (), Location.PIM, gpu_replica=0),
        )

        balanced_us = cost_model.subbatch_layer_cost(balanced).dense.total_us
        unbalanced_us = cost_model.subbatch_layer_cost(unbalanced).dense.total_us
        self.assertGreater(unbalanced_us, balanced_us)

    def test_lpt_partition_preserves_replica_balance(self) -> None:
        model = tiny_model(data_parallel=2)
        cost_model = ChimeCostModel(
            model, SystemSpec(gpu=GpuSpec(count=2, capacity_gib=2.0))
        )
        views = tuple(
            RequestView(
                f"r{index}",
                (100 - index,),
                (),
                Location.PIM,
                gpu_replica=index % 2,
            )
            for index in range(8)
        )

        left, right = cost_model.partition_views(views)
        for replica in range(2):
            left_count = sum(view.gpu_replica == replica for view in left)
            right_count = sum(view.gpu_replica == replica for view in right)
            self.assertLessEqual(abs(left_count - right_count), 1)

    def test_event_cost_cache_reuses_request_id_equivalent_states(self) -> None:
        model = tiny_model(data_parallel=1)
        cost_model = ChimeCostModel(
            model,
            SystemSpec(gpu=GpuSpec(count=1, capacity_gib=2.0)),
            pipeline_model="rankset_event",
        )
        first = (
            RequestView("a", (100,), (20,), Location.GPU),
            RequestView("b", (120,), (), Location.PIM),
        )
        renamed_and_reordered = (
            RequestView("other-b", (120,), (), Location.PIM),
            RequestView("other-a", (100,), (20,), Location.GPU),
        )

        first_cost = cost_model.iteration_cost(first)
        second_cost = cost_model.iteration_cost(renamed_and_reordered)

        self.assertIs(first_cost, second_cost)
        self.assertEqual(cost_model.event_cost_cache_stats["misses"], 1)
        self.assertEqual(cost_model.event_cost_cache_stats["hits"], 1)

if __name__ == "__main__":
    unittest.main()
