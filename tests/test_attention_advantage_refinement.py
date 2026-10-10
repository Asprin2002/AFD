from __future__ import annotations

import unittest

from tools.run_attention_advantage_refinement import (
    Point,
    coarse_points,
    refinement_points,
)


class AttentionAdvantageRefinementTests(unittest.TestCase):
    def test_coarse_grid_is_cartesian_and_adds_unique_anchors(self) -> None:
        points = coarse_points((16, 32), (2048,), (4096,))
        self.assertEqual(len(points), 4)
        self.assertIn(Point(16, 2048, 4096), points)
        self.assertIn(Point(32, 2048, 4096), points)
        self.assertIn(Point(4, 5200, 4096), points)
        self.assertIn(Point(32, 128, 4096), points)

    def test_refinement_is_local_positive_and_excludes_existing(self) -> None:
        seed = Point(32, 2048, 4096)
        points = refinement_points(
            (seed,),
            (seed,),
            batch_step=2,
            input_step=512,
            output_step=512,
        )
        self.assertEqual(len(points), 26)
        self.assertNotIn(seed, points)
        self.assertIn(Point(30, 1536, 3584), points)
        self.assertIn(Point(34, 2560, 4608), points)
        self.assertTrue(
            all(
                point.batch_size > 0
                and point.input_tokens > 0
                and point.output_tokens > 0
                for point in points
            )
        )


if __name__ == "__main__":
    unittest.main()
