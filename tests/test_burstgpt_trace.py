from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.prepare_burstgpt_trace import TRACE_CLASSES, prepare_trace


class BurstGPTTracePreparationTests(unittest.TestCase):
    def _source(self, directory: Path) -> tuple[Path, str]:
        path = directory / "source.csv"
        payload = (
            "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
            "0,ChatGPT,64,1024,1088,API log\n"
            "50,ChatGPT,0,0,0,API log\n"
            "100,GPT-4,128,1200,1328,Conversation log\n"
            "101,GPT-4,256,1400,1656,Conversation log\n"
            "102,ChatGPT,512,1600,2112,API log\n"
            "200,ChatGPT,1024,1800,2824,API log\n"
        )
        path.write_text(payload, encoding="utf-8")
        return path, hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def test_densest_conditioned_window_preserves_lengths_and_scales_time(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source, digest = self._source(directory)
            output = directory / "prepared.csv"

            metadata = prepare_trace(
                source,
                output,
                trace_class=TRACE_CLASSES["short_input_long_output"],
                count=3,
                target_rps=2.0,
                expected_sha256=digest,
            )

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["source_row"]) for row in rows], [3, 4, 5])
            self.assertEqual(
                [int(row["input_tokens"]) for row in rows], [128, 256, 512]
            )
            self.assertEqual(
                [int(row["output_tokens"]) for row in rows], [1200, 1400, 1600]
            )
            self.assertEqual(
                [float(row["arrival_s"]) for row in rows], [0.0, 0.5, 1.0]
            )
            self.assertEqual(metadata["source_failed_or_zero_token_rows"], 1)
            self.assertEqual(metadata["qualified_rows"], 5)
            self.assertEqual(metadata["realized_offered_rps"], 2.0)

            metadata_path = Path(f"{output}.metadata.json")
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["source"]["sha256"], digest)

    def test_source_hash_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source, _ = self._source(directory)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_trace(
                    source,
                    directory / "prepared.csv",
                    trace_class=TRACE_CLASSES["short_input_long_output"],
                    count=3,
                    target_rps=None,
                    expected_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
