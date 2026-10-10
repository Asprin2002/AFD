from __future__ import annotations

import unittest

from elasticafd.cost import ChimeCostModel
from elasticafd.specs import MODEL_PRESETS, SYSTEM_PRESETS
from elasticafd.state import RequestState, views_for_requests


class ChimeCalibrationTests(unittest.TestCase):
    def test_fig17_in_sample_ranges(self) -> None:
        model = MODEL_PRESETS["gpt175b"]
        system = SYSTEM_PRESETS["chime_a100"]
        cost_model = ChimeCostModel(model, system, "tile")
        views = views_for_requests(
            [RequestState(f"r{i}", 0.0, 5200, 1) for i in range(12)]
        )
        cost = cost_model.iteration_cost(views)
        half = cost.subbatches[0]

        self.assertTrue(120 <= half.dense.qkv_us <= 150)
        self.assertTrue(370 <= half.dense.proj_ff_us <= 440)
        self.assertTrue(430 <= half.pim.core_us <= 530)
        self.assertTrue(7.2 <= half.pim.h2d_us <= 7.4)
        self.assertTrue(2.4 <= half.pim.d2h_us <= 2.5)


if __name__ == "__main__":
    unittest.main()

