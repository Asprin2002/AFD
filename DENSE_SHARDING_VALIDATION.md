# Fixed PIM dense sharding sensitivity (layout-v3-dense-shards)

These are **simulated, uncalibrated model estimates**, not hardware timings.
The independent variable is `simulation.layout_dense_ranksets = 1 or 4`;
Attention stays on its layer-home rankset. The cost model includes full input
replication, conditional output collection, serial rank-bus traffic, and
physical weight shards. The 1-group case is the default layout.

## Forced fixed-layout mechanism check

Run `python3 -B -m tools.run_layout_dense_sharding --batch 8
--input-tokens 8192 --output-tokens 16`. This uses GPT-175B, eight private
requests, 96 layers, and 128 decoded tokens. All weights are prepared before
the measured decode loop; their external transfer bytes/time are reported
separately. The interval is fixed for this sensitivity experiment, so this
table does not measure the dynamic policy's choices.

| PIM interval | 1 group decode (ms) | 4 groups decode (ms) | 4-group rank-bus (ms) | Prepared weights (GiB) |
|---|---:|---:|---:|---:|
| Attention only (AF) | 1723.363 | 1723.363 | 0 | 0 |
| QKV + Attention | 5142.832 | 2577.652 | 3.871 | 81 |
| Attention + output projection | 2859.231 | 2005.093 | 2.212 | 27 |
| Attention + output projection + FFN | 11994.007 | 4299.846 | 13.271 | 243 |

The numbers above are direct simulator outputs in
`results/layout-dense-sharding-b8-i8192-o16.json`; the 4-versus-1 ratios
are arithmetic comparisons of those outputs. Four groups substantially
reduce PIM dense time in these forced layouts, but none beats the AF row
under the current provisional kernel parameters. Total PIM weight capacity
and external preparation bytes are unchanged between 1 and 4. For this
96-layer round-robin mapping, the maximum occupied rankset is also unchanged
because the single-group weights were already balanced across the four
ranksets by layer.

## Full policy path

The paired `layout_fixed`/`layout_greedy` run used a private
B16/I5200/O256 closed batch, 128-token prediction window, 128-token replan
interval, and the same backend and provisional parameters in both policies.
Every run completed 4096 output tokens. Direct simulator outputs are in
`results/layout-dense-policy-b16-i5200-o256.json`.

| Dense groups | Fixed AF (token/s) | Greedy AFD (token/s) | Greedy cross-group bytes |
|---:|---:|---:|---:|
| 1 | 114.702 | 130.066 | 0 |
| 4 | 114.702 | 130.066 | 0 |

The approximately 13.4% greedy-versus-fixed throughput difference is an
arithmetic ratio of model outputs. Greedy made two historical-copy/future
writer actions, while the four-group run recorded no cross-group traffic.
Consequently this shape confirms AFD's KV-placement effect in the present
model but does **not** demonstrate a gain from dense sharding. A shorter
B8/I8192/O32 policy check also completed all 256 tokens under both layouts
and selected no action; it is recorded separately in
`results/layout-dense-policy-b8-i8192-o32.json`.

## Model limits

The GPU-PIM link and rank bus are distinct resources. Rank-bus collectives
are serialized and cannot overlap with PIM compute in this estimate. The
weight DMA is assumed to route disjoint shards directly to their destination
groups, so the external link sends the full weight exactly once and the rank
bus does not repeat that delivery. Hardware routing, collective effective
bandwidth, nonzero synchronization/startup time, and bank contention with
background weight writes still need measurement. The separate foreground
and background GPU-PIM link accounting is an existing model limitation and
is not resolved by this sharding experiment.

## B16/I4096/O4096 repeat

The same fixed-interval sensitivity was rerun with 16 private requests,
4096 prompt tokens, and 4096 output tokens each. All eight arms completed
65,536 decoded tokens. These are direct model outputs from
`results/layout-dense-sharding-b16-i4096-o4096.json`; weights are prepared
before the measured decode loop and their preparation time is separate.

| PIM interval | 1 group decode (s) | 4 groups decode (s) | 4-group rank-bus (s) | Prepared weights (GiB) |
|---|---:|---:|---:|---:|
| Attention only (AF) | 659.073 | 659.073 | 0 | 0 |
| QKV + Attention | 2412.666 | 1099.294 | 1.982 | 81 |
| Attention + output projection | 1243.462 | 806.143 | 1.132 | 27 |
| Attention + output projection + FFN | 5920.468 | 1981.057 | 6.795 | 243 |

Four-way sharding improves every forced PIM dense interval but none beats
the fixed AF interval with these provisional parameters. Because the 96
layers are evenly distributed across four ranksets, the measured total PIM
capacity and maximum rankset occupancy were the same for 1 and 4 groups
within each interval. This is a result for this balanced mapping, not a
general guarantee that per-rankset capacity is unchanged.

The full online policy pair on this shape used the same backend, a
128-token prediction horizon, a 128-token replan interval, and a 12-candidate
limit in both rankset settings. Each of the four arms completed 65,536 output
tokens. Direct summaries and raw action records are under
`results/layout-dense-policy-b16-i4096-o4096/`.

| Dense groups | Fixed AF (token/s) | Greedy AFD (token/s) | Greedy versus fixed | Greedy cross-group bytes |
|---:|---:|---:|---:|---:|
| 1 | 99.437 | 124.354 | +25.06% | 0 |
| 4 | 99.437 | 124.354 | +25.06% | 0 |

The two greedy runs selected the same six request actions at decode steps
768, 1024, 1792, 2176, 3200, and 3840. They copied historical KV and
changed future KV writes to GPU; every raw action retained interval `1:1`
(PIM Attention only). Thus four-group dense sharding did not enter the
selected execution path. The summary field `interval_actions = 6` counts
these `combined` actions even though their interval did not change; use the
raw `switches.csv` to determine actual operator placement. Both greedy
runs prepared 164.25 GiB of physical KV data, not PIM dense weights.

The Python policy search took about 1708 seconds wall time for 1 group and
1652 seconds for 4 groups across 32 planning decisions. These search times
are explicitly excluded from the model's throughput. The +25.06% figure is
therefore a same-backend **cost-model scheduling benefit**, not a claim that
this Python implementation can make these decisions within a live decode
interval or that the hardware achieves this speedup.
