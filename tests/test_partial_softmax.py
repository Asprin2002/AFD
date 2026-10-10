from __future__ import annotations

import math
import unittest

from elasticafd.partial_softmax import (
    merge_partial_softmax,
    normalize_partial_softmax,
    partial_softmax,
    stable_attention,
)


class PartialSoftmaxTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, left, right, places=12) -> None:
        self.assertEqual(len(left), len(right))
        for actual, expected in zip(left, right):
            self.assertAlmostEqual(actual, expected, places=places)

    def test_hand_calculated_two_partition_merge(self) -> None:
        scores = [0.0, math.log(2.0), math.log(3.0)]
        values = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        left = partial_softmax(scores[:1], values[:1])
        right = partial_softmax(scores[1:], values[1:])
        output = normalize_partial_softmax(merge_partial_softmax((left, right)))
        self.assertVectorAlmostEqual(output, (4.0 / 6.0, 5.0 / 6.0))

    def test_split_matches_unsplit_for_extreme_finite_logits(self) -> None:
        scores = [-10000.0, -9999.0, 0.0, 9999.0, 10000.0]
        values = [
            (1.0, -1.0, 0.5),
            (2.0, 0.0, -0.5),
            (-3.0, 1.0, 2.0),
            (4.0, 2.0, 1.0),
            (5.0, -2.0, 3.0),
        ]
        expected = stable_attention(scores, values)
        partitions = (
            partial_softmax(scores[:2], values[:2]),
            partial_softmax(scores[2:4], values[2:4]),
            partial_softmax(scores[4:], values[4:]),
        )
        actual = normalize_partial_softmax(merge_partial_softmax(partitions))
        self.assertVectorAlmostEqual(actual, expected)

    def test_empty_partition_is_identity(self) -> None:
        active = partial_softmax([1.0, 2.0], [(2.0, 4.0), (6.0, 8.0)])
        empty = partial_softmax([], [], value_dim=2)
        merged = merge_partial_softmax((empty, active, empty))
        self.assertVectorAlmostEqual(
            normalize_partial_softmax(merged),
            normalize_partial_softmax(active),
        )
        self.assertEqual(empty.max_score, float("-inf"))
        self.assertEqual(empty.denominator, 0.0)

    def test_shape_errors_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            partial_softmax([1.0], [])
        with self.assertRaises(ValueError):
            partial_softmax([1.0, 2.0], [(1.0,), (1.0, 2.0)])
        with self.assertRaises(ValueError):
            merge_partial_softmax(
                (
                    partial_softmax([1.0], [(1.0,)]),
                    partial_softmax([2.0], [(1.0, 2.0)]),
                )
            )


if __name__ == "__main__":
    unittest.main()
