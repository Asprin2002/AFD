"""Collect completed adaptive-shape pairs without mixing preflight or old runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .run_layout_shape_matrix import SHAPES, policy_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="layout-shape-matrix-v3-adaptive")
    parser.add_argument("--output", default="results/layout-shape-matrix-v3-adaptive-summary.csv")
    args = parser.parse_args()
    root = Path(args.results_root)
    rows = []
    missing = []
    code_hashes: set[str] = set()
    for case_id, batch, input_tokens, output_tokens in SHAPES:
        case_dir = root / f"{args.prefix}-{case_id}"
        pair_path = case_dir / "paired.csv"
        if not pair_path.exists():
            missing.append(case_id)
            continue
        with pair_path.open(encoding="utf-8", newline="") as handle:
            pair = list(csv.DictReader(handle))
        if len(pair) != 1 or pair[0]["case_id"] != case_id:
            raise ValueError(f"invalid paired result: {pair_path}")
        row = pair[0]
        if (row["preflight"] != "False" or row["fixed_status"] != "completed"
                or row["afd_status"] != "completed"):
            raise ValueError(f"incomplete result: {case_id}")
        if (int(row["horizon"]), int(row["decision_interval"])) != policy_window(
                batch, input_tokens):
            raise ValueError(f"wrong H/Delta: {case_id}")
        if (int(row["fixed_tokens"]) != batch * output_tokens
                or int(row["afd_tokens"]) != batch * output_tokens):
            raise ValueError(f"paired token mismatch: {case_id}")
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["window_uses_target_output_tokens"]:
            raise ValueError(f"target output leaked into policy window: {case_id}")
        code_hashes.add(manifest["layout_code_sha256"])
        summary = json.loads((case_dir / case_id / "layout_greedy" / "summary.json")
                             .read_text(encoding="utf-8"))
        row["cost_model_revision"] = summary["cost_model_revision"]
        row["layout_code_sha256"] = manifest["layout_code_sha256"]
        covered = {request_id for decision in summary["candidate_screening"]
                   for screening in decision["rounds"]
                   for request_id in screening["evaluated_request_ids"]}
        row["candidate_covered_requests"] = len(covered)
        row["candidate_coverage_fraction"] = len(covered) / batch
        row["policy_decisions"] = summary["policy_decisions"]
        simulation = json.loads((
            case_dir / case_id / "layout_greedy" / "config.json")
            .read_text(encoding="utf-8"))["simulation"]
        row["layout_survival_mean_tokens"] = simulation["layout_survival_mean_tokens"]
        row["layout_survival_shape"] = simulation.get("layout_survival_shape", "")
        row["survival_model"] = summary.get(
            "output_length_prior", {}).get("family", "legacy_progress_subtraction")
        rows.append(row)
    if len(code_hashes) > 1:
        raise ValueError("different layout code hashes across cases")
    if rows:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)}/{len(SHAPES)} completed: {output}")
    print("missing: " + (", ".join(missing) if missing else "none"))


if __name__ == "__main__":
    main()
