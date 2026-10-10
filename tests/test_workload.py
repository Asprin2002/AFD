from __future__ import annotations

import unittest

from elasticafd.config import WorkloadConfig
from elasticafd.workload import trace_serving_requests


class SyntheticWorkloadTests(unittest.TestCase):
    def test_weighted_classes_only_emit_declared_length_pairs(self) -> None:
        config = WorkloadConfig(
            type="trace_serving",
            synthetic={
                "count": 100,
                "arrival_rate_rps": 1.0,
                "seed": 7,
                "classes": [
                    {"weight": 1, "input_tokens": 32, "output_tokens": 64},
                    {"weight": 1, "input_tokens": 128, "output_tokens": 256},
                ],
            },
        )

        requests = trace_serving_requests(config)
        pairs = {(request.input_tokens, request.output_tokens) for request in requests}

        self.assertEqual(pairs, {(32, 64), (128, 256)})

    def test_burst_arrivals_preserve_requested_long_run_rate(self) -> None:
        config = WorkloadConfig(
            type="trace_serving",
            synthetic={
                "count": 8,
                "arrival_rate_rps": 2.0,
                "arrival_pattern": "burst",
                "burst_size": 4,
                "input_tokens": 32,
                "output_tokens": 64,
            },
        )

        requests = trace_serving_requests(config)

        self.assertEqual(
            [request.arrival_us for request in requests],
            [0.0, 0.0, 0.0, 0.0, 2e6, 2e6, 2e6, 2e6],
        )

    def test_profile_and_classes_cannot_be_combined(self) -> None:
        config = WorkloadConfig(
            type="trace_serving",
            synthetic={
                "count": 2,
                "profile": "openr1",
                "classes": [
                    {"input_tokens": 32, "output_tokens": 64},
                ],
            },
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            trace_serving_requests(config)


if __name__ == "__main__":
    unittest.main()
