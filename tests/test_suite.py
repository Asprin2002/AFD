from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from elasticafd.suite import (
    SuiteCase,
    _input_columns,
    comparison_rows,
    load_suite,
    run_cases,
    select_cases,
    validate_cases,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class WorkloadSuiteDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_suite(
            REPOSITORY_ROOT / "configs" / "suites" / "workloads-v1.yaml"
        )

    def test_e0_full_suite_has_expected_scenario_coverage(self) -> None:
        cases = select_cases(self.suite, stage="e0", tier="full")
        self.assertEqual(len(cases), 71)
        counts = {
            scenario: sum(case.scenario == scenario for case in cases)
            for scenario in {case.scenario for case in cases}
        }
        self.assertEqual(counts["offline_homogeneous"], 39)
        self.assertEqual(counts["offline_heterogeneous"], 4)
        self.assertEqual(counts["workstation"], 8)
        self.assertEqual(counts["online_synthetic"], 20)

    def test_tiers_are_cumulative(self) -> None:
        smoke = select_cases(self.suite, stage="e0", tier="smoke")
        core = select_cases(self.suite, stage="e0", tier="core")
        full = select_cases(self.suite, stage="e0", tier="full")
        self.assertEqual((len(smoke), len(core), len(full)), (4, 36, 71))
        self.assertTrue({case.case_id for case in smoke} < {case.case_id for case in core})
        self.assertTrue({case.case_id for case in core} < {case.case_id for case in full})

    def test_shards_are_disjoint_and_cover_selection(self) -> None:
        all_cases = select_cases(self.suite, stage="e0", tier="core")
        shards = [
            select_cases(self.suite, stage="e0", tier="core", shard=(index, 4))
            for index in range(4)
        ]
        all_ids = {case.case_id for case in all_cases}
        shard_ids = [{case.case_id for case in shard} for shard in shards]
        self.assertEqual(set().union(*shard_ids), all_ids)
        for left in range(len(shard_ids)):
            for right in range(left + 1, len(shard_ids)):
                self.assertFalse(shard_ids[left] & shard_ids[right])

    def test_e1_is_paired_and_has_offline_global_variant(self) -> None:
        cases = select_cases(self.suite, stage="e1", tier="core")
        self.assertEqual(len(cases), 89)
        self.assertEqual(sum(case.variant == "chime" for case in cases), 36)
        self.assertEqual(
            sum(case.variant == "future_kv_lifetime" for case in cases), 36
        )
        self.assertEqual(
            sum(case.variant == "future_kv_balance_curve" for case in cases), 17
        )
        example = "off-homo-anchor-b16-i5200-o4096"
        pair = [case for case in cases if case.workload_id == example]
        self.assertEqual(len(pair), 3)
        self.assertEqual(
            len({str(case.config["workload"]) for case in pair}), 1
        )
        lifetime = next(
            case for case in pair if case.variant == "future_kv_lifetime"
        )
        self.assertEqual(
            lifetime.config["simulation"]["policy"]["prediction"], "oracle"
        )
        self.assertNotIn(
            "kv_capacity_limit_gib", lifetime.config["hardware"]["gpu"]
        )

    def test_e2_is_representative_event_replay_with_paired_overhead(self) -> None:
        smoke = select_cases(self.suite, stage="e2", tier="smoke")
        core = select_cases(self.suite, stage="e2", tier="core")
        self.assertEqual((len(smoke), len(core)), (4, 26))
        self.assertTrue(
            all(
                case.config["simulation"]["pipeline_model"]
                == "rankset_event"
                for case in core
            )
        )
        self.assertEqual(
            sum(case.variant == "future_kv_lifetime_event" for case in core),
            9,
        )
        self.assertEqual(
            sum(case.variant == "future_kv_mode_event" for case in core),
            2,
        )

        overhead = [
            case for case in core if case.workload_id.endswith("-overhead")
        ]
        self.assertEqual(len(overhead), 6)
        grouped = {}
        for case in overhead:
            grouped.setdefault(case.workload_id, []).append(case)
        self.assertEqual({len(pair) for pair in grouped.values()}, {2})
        for pair in grouped.values():
            self.assertEqual(
                {case.config["hardware"]["gpu"]["segment_setup_us"] for case in pair},
                {1.0},
            )
            self.assertEqual(
                {case.config["hardware"]["pim"]["segment_setup_us"] for case in pair},
                {1.0},
            )


class WorkloadSuiteRunnerTests(unittest.TestCase):
    def test_checkpoint_resume_and_validation(self) -> None:
        config = {
            "model": "gpt175b",
            "hardware": "chime_a100",
            "workload": {
                "type": "static_batch",
                "batch_size": 1,
                "input_tokens": 16,
                "output_tokens": 2,
            },
            "simulation": {
                "max_batch_size": 1,
                "fidelity": "roofline",
                "pipeline_model": "analytic",
                "record_every": 1,
                "operator_mapping": "chime",
                "policy": "chime",
            },
        }
        case = SuiteCase(
            suite_name="test-suite",
            suite_version=1,
            source_path=REPOSITORY_ROOT / "test-suite.yaml",
            stage="e0",
            case_id="tiny-case",
            tier="smoke",
            scenario="unit_test",
            description="tiny checkpoint test",
            tags=("test",),
            config=config,
            config_sha256="test-digest",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = run_cases((case,), output)
            self.assertEqual(first[0]["status"], "completed")
            self.assertFalse(first[0]["cache_hit"])
            second = run_cases((case,), output, resume=True)
            self.assertEqual(second[0]["status"], "completed")
            self.assertTrue(second[0]["cache_hit"])
            rows, issues = validate_cases((case,), output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(issues, [])
            self.assertTrue((output / "tiny-case" / "summary.json").is_file())
            self.assertTrue((output / "tiny-case" / "case.json").is_file())

    def test_comparison_uses_goodput_online_and_makespan_offline(self) -> None:
        def row(workload_id: str, variant: str, scenario: str, time: float):
            return {
                "workload_id": workload_id,
                "variant": variant,
                "scenario": scenario,
                "status": "completed",
                "simulated_time_s": time,
                "throughput_tokens_s": 100 / time,
                "mean_tpot_ms": time,
                "p95_e2e_from_decode_ready_ms": time,
                "goodput_rps": 1 / time if scenario.startswith("online") else "",
                "slo_attainment": 1.0 if scenario.startswith("online") else "",
                "case_id": f"{variant}-{workload_id}",
            }

        rows = [
            row("offline", "chime", "offline_homogeneous", 2.0),
            row("offline", "future", "offline_homogeneous", 1.0),
            row("online", "chime", "online_synthetic", 2.0),
            row("online", "future", "online_synthetic", 1.0),
        ]
        compared = comparison_rows(rows)
        by_id = {item["workload_id"]: item for item in compared}
        self.assertEqual(by_id["offline"]["primary_metric"], "makespan")
        self.assertEqual(by_id["online"]["primary_metric"], "goodput_rps")
        self.assertEqual(by_id["offline"]["primary_speedup"], 2.0)
        self.assertEqual(by_id["online"]["primary_speedup"], 2.0)

    def test_trace_file_metadata_overrides_unused_synthetic_template(self) -> None:
        case = SuiteCase(
            suite_name="test-suite",
            suite_version=1,
            source_path=REPOSITORY_ROOT / "test-suite.yaml",
            stage="public",
            case_id="public-trace",
            tier="smoke",
            scenario="public_trace_burst",
            description="",
            tags=(),
            config={
                "model": "gpt175b",
                "hardware": "chime_a100",
                "workload": {
                    "type": "trace_serving",
                    "trace_file": "public.csv",
                    "synthetic": {"arrival_rate_rps": 99.0, "count": 1},
                },
                "simulation": {},
            },
            config_sha256="test-digest",
        )

        columns = _input_columns(case)

        self.assertEqual(columns["trace_file"], "public.csv")
        self.assertNotIn("arrival_rate_rps", columns)
        self.assertNotIn("input_request_count", columns)

    def test_trace_serving_uses_goodput_outside_online_named_scenarios(self) -> None:
        def row(variant: str, time: float, offered: float):
            return {
                "workload_id": "public",
                "variant": variant,
                "scenario": "public_trace_burst",
                "workload_type": "trace_serving",
                "status": "completed",
                "simulated_time_s": time,
                "throughput_tokens_s": 100 / time,
                "mean_tpot_ms": time,
                "p95_e2e_from_decode_ready_ms": time,
                "goodput_rps": 1 / time,
                "trace_offered_rate_rps": offered,
                "slo_attainment": 1.0,
                "case_id": f"{variant}-public",
            }

        compared = comparison_rows(
            (row("chime", 2.0, 0.5), row("future", 1.0, 0.5))
        )[0]

        self.assertEqual(compared["primary_metric"], "goodput_rps")
        self.assertEqual(compared["primary_speedup"], 2.0)
        self.assertEqual(compared["arrival_rate_rps"], 0.5)


if __name__ == "__main__":
    unittest.main()
