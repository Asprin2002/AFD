"""Steady-state fixed-layout 1/4-rankset sensitivity on one decode shape.

This is a mechanism experiment with explicit PIM dense placement, not a
hardware measurement or an AFD policy-speedup result. Weight preparation is
reported separately from the decode-loop time.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from elasticafd.config import WorkloadConfig, load_config
from elasticafd.layout_aware import (LayoutWorld, PimInterval, PrefixCatalog,
                                     _prepare_interval, evaluate_step)
from elasticafd.specs import GiB
from elasticafd.workload import WorkloadRequest


INTERVALS = {
    "AF": PimInterval(1, 1),
    "QKV_Attention": PimInterval(0, 1),
    "Attention_Output": PimInterval(1, 3),
    "Attention_Output_FFN": PimInterval(1, 4),
}


def run(config_path: str, batch: int, input_tokens: int,
        output_tokens: int) -> list[dict[str, object]]:
    base = load_config(config_path)
    rows = tuple({"request_id": f"r{index}", "arrival_us": 0,
                  "input_tokens": input_tokens, "output_tokens": output_tokens}
                 for index in range(batch))
    results = []
    for ranksets in (1, 4):
        config = replace(base,
                         workload=WorkloadConfig("closed_batch", requests=rows),
                         simulation=replace(base.simulation, layout_dense_ranksets=ranksets,
                                            max_batch_size=batch))
        for name, interval in INTERVALS.items():
            world = LayoutWorld(config, PrefixCatalog(()))
            for row in rows:
                world.admit(WorkloadRequest(row["request_id"], 0,
                                            input_tokens, output_tokens), 0)
            prepared_us = 0.0
            if name != "AF":
                prepared_us = _prepare_interval(world, interval, {0})
                for request in world.requests.values():
                    request.interval = interval
            world.check_capacity()
            decode_start = max((copy.ready_us for copy in world.weights.values()), default=0.0)
            world.clock_us = decode_start
            cross_bytes = cross_us = boundary_us = pim_us = gpu_us = 0.0
            for _ in range(output_tokens):
                cost = evaluate_step(world, book_link=True)
                world.clock_us += cost.total_us
                cross_bytes += cost.cross_group_bytes
                cross_us += cost.cross_group_us
                boundary_us += cost.boundary_us
                pim_us += cost.pim_busy_us
                gpu_us += cost.gpu_busy_us
                for request in tuple(world.requests.values()):
                    world.append_token(request)
            decode_us = world.clock_us - decode_start
            results.append({
                "model_revision": "layout-v4-link-calendar",
                "calibration": "uncalibrated mechanism estimate",
                "batch": batch, "input_tokens": input_tokens,
                "output_tokens_per_request": output_tokens,
                "layout_dense_ranksets": ranksets, "interval": name,
                "decoded_tokens": batch * output_tokens,
                "steady_state_decode_ms": decode_us / 1000,
                "steady_state_tokens_s": batch * output_tokens / (decode_us / 1e6),
                "pim_busy_ms": pim_us / 1000, "gpu_busy_ms": gpu_us / 1000,
                "boundary_ms": boundary_us / 1000,
                "cross_group_bytes": cross_bytes,
                "cross_group_ms": cross_us / 1000,
                "prepared_weight_gib": world.preparation_bytes / GiB,
                "prepared_weight_transfer_ms": prepared_us / 1000,
                "peak_pim_kv_weight_reserve_gib": world.peak_pim_bytes / GiB,
                "max_rankset_occupancy_gib": max(world.usage()[1]) / GiB,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/layout-aware-long-context.yaml")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--output", default="results/layout-dense-sharding-b8-i8192-o16.json")
    args = parser.parse_args()
    if min(args.batch, args.input_tokens, args.output_tokens) <= 0:
        raise ValueError("batch and token counts must be positive")
    rows = run(args.config, args.batch, args.input_tokens, args.output_tokens)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
