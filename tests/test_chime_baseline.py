from __future__ import annotations

import unittest

from elasticafd.chime_baseline import ChimeDecodeBaseline
from elasticafd.state import Location, RequestView
from tools.run_contiguous_ablation import config
from tools.run_chime_baseline_light import Case, chime_model, measure_case


def pim_view(index: int, tokens: int) -> RequestView:
    return RequestView(f"r{index}", (tokens,), (), Location.PIM)


class ChimeDecodeBaselineTests(unittest.TestCase):
    def test_mha_n1_rounds_cover_requests_and_balance_group_size(self) -> None:
        model = chime_model(Case("test", "attention", 5, 128, 2, 2, 0, 2))
        state = [pim_view(index, tokens) for index, tokens in
                 enumerate((1000, 900, 300, 200, 100))]
        groups = model.partition_views(state)
        self.assertEqual([len(group) for group in groups], [3, 2])
        self.assertEqual(
            {view.request_id for group in groups for view in group},
            {view.request_id for view in state},
        )
        self.assertNotEqual(groups[0][0].request_id, groups[1][0].request_id)
        self.assertEqual(groups, model.partition_views(tuple(reversed(state))))

    def test_baseline_has_ordered_device_visits_without_overtake(self) -> None:
        model = chime_model(Case("test", "attention", 2, 128, 2, 1, 0, 2))
        self.assertIsInstance(model, ChimeDecodeBaseline)
        self.assertEqual(model.subbatch_progression, "device_contiguous")
        events = model.iteration_cost((pim_view(0, 128), pim_view(1, 130))).events
        by_id = {event.task_id: event for event in events}
        for visit in range(2 * model.model.layers + 1):
            self.assertGreaterEqual(
                by_id[f"visit:{visit+1}:start"].start_us,
                by_id[f"visit:{visit}:done"].end_us,
            )

    def test_candidate_configuration_cannot_change_baseline(self) -> None:
        case = Case("test", "f", 2, 128, 2)
        baseline = chime_model(case)
        before = baseline.iteration_cost((pim_view(0, 128), pim_view(1, 128)))
        from elasticafd.dense_offload import DenseOffloadCost, DensePlan
        candidate = DenseOffloadCost(
            config(2, progression="device_contiguous", scheduler="fixed"),
            (DensePlan(), DensePlan(1, 0, 1)),
            allow_conditional_ffn=True,
        )
        candidate.measure((pim_view(0, 128), pim_view(1, 128)),
                          DensePlan(1, 0, 1))
        after = baseline.iteration_cost((pim_view(0, 128), pim_view(1, 128)))
        self.assertEqual(before.total_us, after.total_us)

    def test_candidate_cut_boundaries_cannot_change_baseline_sampling(self) -> None:
        plain = Case("plain", "attention", 2, 128, 5, 1, 0, 5)
        extra = Case("extra", "attention", 2, 128, 5, 1, 0, 5,
                     dynamic=((0, 3),))
        first = measure_case(plain, 4, 1, 1)[0]
        second = measure_case(extra, 4, 1, 1)[0]
        self.assertEqual(first["chime_baseline_total_us"],
                         second["chime_baseline_total_us"])

    def test_rejects_non_chime_gpu_kv(self) -> None:
        model = chime_model(Case("test", "attention", 2, 128, 2, 1, 0, 2))
        with self.assertRaises(ValueError):
            model.partition_views((RequestView("r0", (128,), (1,), Location.GPU),))


if __name__ == "__main__":
    unittest.main()
