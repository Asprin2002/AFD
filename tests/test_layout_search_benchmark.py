from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PressureBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trace = self.root/'trace.csv'

    def run_benchmark(self, output, *extra):
        script = Path(__file__).resolve().parents[1]/'tools/benchmark_layout_search.py'
        return subprocess.run([sys.executable, '-B', str(script), '--trace', str(self.trace),
            '--start', '0', '--active-requests', '1', '--horizon', '1', '--candidates', '1',
            '--scenarios', '1', '--prior-mean-tokens', '10', '--repeat', '1',
            '--output-dir', str(self.root/output), *extra], capture_output=True, text=True)

    def write_trace(self, tokens):
        with self.trace.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=('request_id', 'arrival_s', 'input_tokens', 'output_tokens'))
            writer.writeheader()
            writer.writerow({'request_id': 'a', 'arrival_s': 17, 'input_tokens': tokens, 'output_tokens': 2})

    def test_completed_sample_can_be_compared_with_larger_reference_run(self):
        self.write_trace(8)
        before = self.run_benchmark('before', '--repeat', '2')
        self.assertEqual(before.returncode, 0, before.stderr)
        after = self.run_benchmark('after', '--reference', str(self.root/'before'))
        self.assertEqual(after.returncode, 0, after.stderr)
        report = json.loads((self.root/'after/timings.json').read_text())
        self.assertEqual(report['comparison'], {'exact_match': True, 'differences': [],
                                               'compared_searches': 1, 'reference_searches': 2})

    def test_capacity_failure_is_reported_without_running_an_infeasible_search(self):
        self.write_trace(10**9)
        result = self.run_benchmark('rejected')
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads((self.root/'rejected/decisions.jsonl').read_text())
        self.assertEqual(row['status'], 'capacity_rejected')
        self.assertEqual(row['admitted_before_rejection'], 0)
        self.assertNotIn('wall_s', row)
        report = json.loads((self.root/'rejected/timings.json').read_text())
        self.assertEqual(report['cases']['1']['completed_searches'], 0)
        self.assertIsNone(report['cases']['1']['median_s'])

    def test_missing_reference_sample_is_an_error(self):
        self.write_trace(8)
        before = self.run_benchmark('before')
        self.assertEqual(before.returncode, 0, before.stderr)
        (self.root/'before/decisions.jsonl').write_text('')
        after = self.run_benchmark('after', '--reference', str(self.root/'before'))
        self.assertNotEqual(after.returncode, 0)
        report = json.loads((self.root/'after/timings.json').read_text())
        self.assertFalse(report['comparison']['exact_match'])
        self.assertIn('missing reference', report['comparison']['differences'][0])


if __name__ == '__main__':
    unittest.main()
