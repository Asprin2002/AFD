#!/usr/bin/env python3
"""M4 digest: turn a matrix run into the dimensional report.

Reads ``matrix.csv`` and ``schedules.json`` produced by
``run_heterogeneous_joint_matrix.py`` and prints, for every scenario:

  - control baseline and the real-cost speedups of ``attn`` / ``f_only`` /
    ``joint`` / pilots (best across formations), with switch and copy detail;
  - the formation axis: when ``alignment`` changes ``joint`` vs ``lpt`` (the
    plan-aware pairing trade-off);
  - the M3 economics axis: ``joint`` scheduled under the real cost model vs the
    free cost model — whether the cost model itself changed the chosen timing;
  - M1-B: the best ``joint`` vs the independent ``joint_static_reference`` and
    any recorded violations (segmented losing to best-static, fresh-engine
    replay mismatch);
  - infeasible cells, recorded verbatim.

Writes ``summary.json`` and ``HETEROGENEOUS_JOINT_SUMMARY.md`` next to the
matrix.  The intended consumption path is the human review of the delivered
run, not a population estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import inf
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_ARM = "real"


def load_rows(matrix: Path) -> list[dict]:
    with matrix.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def total(row: dict) -> float:
    value = float(row["total_us"])
    return inf if value >= float("inf") or not row["feasible"] == "1" else value


def best(rows: list[dict], strategy: str, *, search_arm: str = "real",
         formation: str | None = None) -> dict | None:
    key = [r for r in rows
           if r["strategy"] == strategy and r["search_arm"] == search_arm
           and r["report_arm"] == REPORT_ARM
           and (formation is None or r["formation"] == formation)]
    key = [r for r in key if r["feasible"] == "1"]
    return min(key, key=total) if key else None


def speedup(control_us: float, row: dict | None) -> float | None:
    if row is None or not control_us or control_us < 0:
        return None
    value = total(row)
    if value >= inf:
        return None
    return control_us / value


def build_report(rows: list[dict], meta: dict, out: Path) -> dict:
    scenarios = sorted({r["scenario"] for r in rows})
    formations = sorted({r["formation"] for r in rows})
    entries: list[dict] = []
    infeasible_cells: list[dict] = []
    for scenario in scenarios:
        r = [x for x in rows if x["scenario"] == scenario]
        control = best(r, "control")
        control_us = total(control) if control else None
        entry: dict = {
            "scenario": scenario,
            "batch": int(r[0]["batch"]),
            "horizon": int(r[0]["horizon"]),
            "control_total_us": control_us,
        }
        for strategy in ("attn", "f_only", "joint", "ffn_pilot", "qkv_pilot"):
            best_row = best(r, strategy, search_arm="real")
            entry[f"real_{strategy}_total_us"] = (
                total(best_row) if best_row else None)
            entry[f"real_{strategy}_speedup"] = speedup(control_us, best_row)
            entry[f"real_{strategy}_kv_switches"] = (
                int(best_row["kv_switches"]) if best_row else None)
            entry[f"real_{strategy}_operator_switches"] = (
                int(best_row["operator_switches"]) if best_row else None)
            entry[f"real_{strategy}_peak_gib"] = (
                float(best_row["peak_pim_resident_gib"]) if best_row else None)
        const_row = best(r, "best_constant_menu", search_arm="-")
        entry["real_best_constant_menu_total_us"] = (
            total(const_row) if const_row else None)
        entry["real_best_constant_menu_speedup"] = speedup(control_us, const_row)
        # Cost-economics axis: joint scheduled under free vs real.
        joint_free = best(r, "joint", search_arm="free")
        entry["free_joint_total_us"] = total(joint_free) if joint_free else None
        entry["free_joint_speedup"] = speedup(control_us, joint_free)
        entry["free_joint_kv_switches"] = (
            int(joint_free["kv_switches"]) if joint_free else None)
        # Formation axis on joint.
        formation_joint: dict[str, float] = {}
        for formation in formations:
            frow = best(r, "joint", search_arm="real", formation=formation)
            formation_joint[formation] = total(frow) if frow else None
        entry["formation_joint_total_us"] = formation_joint
        # M1-B reference.
        ref = best(r, "joint_static_reference", search_arm="-")
        entry["reference_total_us"] = total(ref) if ref else None
        entry["reference_partial"] = None
        entry["reference_acceptable"] = None
        mine = best(r, "joint", search_arm="real")
        if ref is not None and mine is not None and total(ref) < inf:
            entry["reference_acceptable"] = (
                total(mine) <= total(ref) + 1e-6)
            entry["reference_delta_us"] = total(mine) - total(ref)
        for row in r:
            if row.get("reference_meta"):
                try:
                    meta = json.loads(row["reference_meta"])
                except (json.JSONDecodeError, TypeError):
                    meta = None
                if meta:
                    entry["reference_partial"] = bool(meta["reference_partial"])
                    entry["reference_evaluations"] = meta["reference_evaluations"]
                    entry["reference_space_exact"] = bool(meta["reference_space_exact"])
        entries.append(entry)
        for cell in r:
            if cell["feasible"] != "1":
                infeasible_cells.append({
                    "scenario": scenario,
                    "formation": cell["formation"],
                    "strategy": cell["strategy"],
                    "search_arm": cell["search_arm"],
                })
    return {
        "scenarios": entries,
        "infeasible_cells": infeasible_cells,
        "violations": meta.get("violations", []),
        "runner_code_sha256": meta.get("runner_code_sha256"),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 异质联合静态矩阵 —— 汇总 (summary.json => 本文件)  ",
        "",
        "表格按场景一行;速度比 = control_total / 该行 total_us。",
        "",
        "| 场景 | B | L | control(ms) | attn | f_only | joint | best_const | ffn_pilot | qkv_pilot | free/joint | refΔ (us) | ref部分? |",
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | -- |",
    ]
    for entry in report["scenarios"]:
        control_us = entry.get("control_total_us")

        def cell(key: str) -> str:
            value = entry.get(key)
            if value is None:
                return "—"
            if key.startswith(("real_", "free_")):
                return f"{value:.3f}x"
            if key == "reference_delta_us":
                return f"{value:+.0f}us"
            return str(value)

        control_text = f"{control_us / 1e3:.1f}" if control_us else "—"
        partial = entry.get("reference_partial")
        partial_text = "partial" if partial else ("全" if partial is not None else "—")
        lines.append(
            f"| {entry['scenario']} | {entry['batch']} | {entry['horizon']} "
            f"| {control_text} "
            f"| {cell('real_attn_speedup')} | {cell('real_f_only_speedup')} "
            f"| {cell('real_joint_speedup')} "
            f"| {cell('real_best_constant_menu_speedup')} "
            f"| {cell('real_ffn_pilot_speedup')} "
            f"| {cell('real_qkv_pilot_speedup')} | {cell('free_joint_speedup')} "
            f"| {cell('reference_delta_us')} | {partial_text}"
            f" |",
        )
    lines += [
        "",
        "## 不可行单元",
        "",
    ]
    if report["infeasible_cells"]:
        for cell in report["infeasible_cells"]:
            lines.append(f"- {cell['scenario']} / {cell['formation']} / "
                         f"{cell['strategy']} (arm {cell['search_arm']})")
    else:
        lines.append("无。所有搜索出的计划在原价格下可行。")
    lines += ["", "## M1-B 验证 violations", ""]
    if report["violations"]:
        for violation in report["violations"]:
            lines.append(f"- {violation}")
    else:
        lines.append("无(每次 segmented ≤ 独立 best_static,且 fresh-engine replay 一致)。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "results/heterogeneous-joint-matrix/matrix.csv")
    parser.add_argument("--meta", type=Path, default=None)
    args = parser.parse_args()
    matrix = args.matrix.resolve()
    meta_path = args.meta or matrix.with_name("schedules.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = load_rows(matrix)
    if not rows:
        print("matrix.csv is empty", file=sys.stderr)
        return 1
    report = build_report(rows, meta, matrix.parent)
    (matrix.parent / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (matrix.parent / "HETEROGENEOUS_JOINT_SUMMARY.md").write_text(
        render_markdown(report), encoding="utf-8")
    # Compact stdout.
    for entry in report["scenarios"]:
        print(f"{entry['scenario']}: control={entry['control_total_us']/1e3:.1f}ms "
              f"attn={entry.get('real_attn_speedup')} "
              f"f_only={entry.get('real_f_only_speedup')} "
              f"joint={entry.get('real_joint_speedup')} "
              f"ffn_pilot={entry.get('real_ffn_pilot_speedup')} "
              f"qkv_pilot={entry.get('real_qkv_pilot_speedup')} "
              f"free_joint={entry.get('free_joint_speedup')} "
              f"ref_delta={entry.get('reference_delta_us')}")
    print(f"infeasible={len(report['infeasible_cells'])} "
          f"violations={len(report['violations'])}")
    print(f"wrote {matrix.parent / 'summary.json'} and "
          f"{matrix.parent / 'HETEROGENEOUS_JOINT_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())