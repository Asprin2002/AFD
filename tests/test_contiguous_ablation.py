from __future__ import annotations

import unittest

from elasticafd.state import Location
from tools.run_contiguous_ablation import blocks, views


class ContiguousAblationTests(unittest.TestCase):
    def test_delayed_future_kv_does_not_move_history(self):
        before = views(4, 100, 3, 10, "gpu_half_mid")
        after = views(4, 100, 7, 10, "gpu_half_mid")
        self.assertEqual((before[0].pim_tokens, before[0].gpu_tokens), (103, 0))
        self.assertEqual((after[0].pim_tokens, after[0].gpu_tokens), (105, 2))
        self.assertEqual(after[0].tail_location, Location.GPU)
        self.assertEqual((after[2].pim_tokens, after[2].gpu_tokens), (107, 0))

    def test_sample_blocks_cover_every_output_position_once(self):
        sampled = tuple(blocks(10, 4))
        self.assertEqual(sampled, ((4, 1), (4, 5), (2, 8)))
        self.assertEqual(sum(weight for weight, _ in sampled), 10)


if __name__ == "__main__":
    unittest.main()
