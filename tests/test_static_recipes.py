"""M0 generator guarantees: budget, determinism, capacity preflight, invariants.

The static recipe corpus is the runnable grid for the M1-M3 matrix; these tests
lock down the *generator's* executable semantics (independent of the matrix
runner): recipe counts, seeded determinism, per-shape invariants, and the
all-PIM capacity preflight used to keep the corpus runnable.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from tools import prepare_static_recipes as gen
from tools.prepare_static_recipes import (
    build_heterogeneous,
    capacity_feasible,
    discover,
    generate,
    load_recipe,
    round_output,
)

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "workloads" / "static"


class RecipeGenerationTests(unittest.TestCase):
    def test_recipe_counts(self):
        manifest = generate(WORKLOADS, dry=True)
        self.assertEqual(len(manifest["files"]), 35)  # 17 anchors + 18 hetero
        anchors = [k for k in manifest["files"]
                   if manifest["files"][k]["kind"] == "anchor"]
        self.assertEqual(17, len(anchors))
        hetero = [k for k in manifest["files"]
                  if manifest["files"][k]["kind"] == "heterogeneous"]
        self.assertEqual(18, len(hetero))

    def test_every_recipe_loaded_and_preflight_capable(self):
        if not WORKLOADS.joinpath("manifest.json").exists():
            self.skipTest("static corpus not generated yet")
        paths = discover(WORKLOADS)
        self.assertEqual(35, len(paths))
        for path in paths:
            specs = load_recipe(path)
            self.assertGreaterEqual(len(specs), 1)
            self.assertTrue(
                capacity_feasible(specs),
                f"recipe {path} should pass the all-PIM capacity preflight",
            )

    def test_generator_deterministic_per_seed(self):
        for profile in gen.PROFILES:
            for batch in gen.BATCHES:
                for seed in gen.SEEDS:
                    a = build_heterogeneous(profile, batch, seed)
                    b = build_heterogeneous(profile, batch, seed)
                    self.assertEqual(
                        [(s.input_tokens, s.output_tokens) for s in a],
                        [(s.input_tokens, s.output_tokens) for s in b],
                    )

    def test_long_input_profile_invariants(self):
        specs = build_heterogeneous("long_in", 16, 20260922)
        self.assertEqual(16, len(specs))
        for spec in specs:
            self.assertIn(spec.input_tokens, (1024, 2048, 4096, 8192, 12288))
            self.assertEqual(0, spec.output_tokens % gen.DECISION_STRIDE)
            self.assertLessEqual(round_output(1), spec.output_tokens)
            self.assertGreaterEqual(spec.output_tokens, 512)
            self.assertLessEqual(spec.output_tokens, 2048)

    def test_mixed_profile_invariants(self):
        specs = build_heterogeneous("mixed", 8, 20260923)
        self.assertEqual(8, len(specs))
        for spec in specs:
            # Small cluster: short context + short output.
            short = spec.input_tokens <= 2048
            long_ctx = spec.input_tokens >= 4096
            self.assertTrue(short or long_ctx, spec)
            if spec.input_tokens <= 2048:
                self.assertLessEqual(spec.output_tokens, 512)
            self.assertEqual(0, spec.output_tokens % gen.DECISION_STRIDE)

    def test_long_context_first_sorting(self):
        specs = build_heterogeneous("long_in", 32, 20260923)
        inputs = [spec.input_tokens for spec in specs]
        self.assertEqual(inputs, sorted(inputs, reverse=True))

    def test_written_recipe_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus"
            manifest = generate(out, dry=False)
            name = "mixed_b16_s20260922"
            path = out / "heterogeneous" / "mixed" / "b16_s20260922.csv"
            self.assertTrue(path.exists())
            specs = load_recipe(path)
            expected = build_heterogeneous("mixed", 16, 20260922)
            self.assertEqual(
                [(s.request_id, s.input_tokens, s.output_tokens)
                 for s in specs],
                [(s.request_id, s.input_tokens, s.output_tokens)
                 for s in expected],
            )
            meta = json.loads(path.with_suffix(".metadata.json").read_text())
            self.assertEqual("mixed", meta["profile"])
            self.assertEqual(20260922, meta["seed"])
            self.assertTrue(meta["capacity_preflight"])
            self.assertEqual(manifest["generator_sha256"], meta["generator_sha256"])

    def test_round_output_sticks_to_stride(self):
        for value in (1, 63, 64, 65, 200, 2048):
            self.assertEqual(0, round_output(value) % gen.DECISION_STRIDE)
            self.assertGreaterEqual(round_output(value), value)
        self.assertEqual(gen.DECISION_STRIDE, round_output(1))

    def test_anchor_menu_covers_regions(self):
        shapes = {shape: gen.HOMOGENEOUS_ANCHORS.count(shape)
                  for shape in set(gen.HOMOGENEOUS_ANCHORS)}
        self.assertIn((16, 5200, 8192), shapes)   # headline 1.5074x
        self.assertIn((32, 2048, 8192), shapes)   # 1.4847x / 1.4001x
        self.assertIn((8, 8192, 8192), shapes)    # 1.2694x
        self.assertIn((2, 128, 256), shapes)      # no-benefit region
        # Anchor batch values span the no-benefit 2/4 region to the headrooms.
        self.assertEqual({c[0] for c in gen.HOMOGENEOUS_ANCHORS},
                         {2, 4, 8, 16, 32})


if __name__ == "__main__":
    unittest.main()