# Layout-aware Attention scheduling model

This opt-in backend implements the mechanism in
`Attention动态调度建模与设计收敛.pdf`. It is a **model estimate**, not a calibrated
prediction of PIM dense or softmax kernels. Existing experiments and result
files use the unchanged `legacy` backend by default.

## Run and compare

```bash
python3 -m elasticafd simulate configs/layout-aware-shared.yaml --output-dir /tmp/layout-greedy
python3 -m elasticafd simulate configs/layout-aware-shared.yaml --policy layout_fixed --output-dir /tmp/layout-fixed
python3 -m tools.run_layout_scenarios --config configs/layout-aware-shared.yaml --output /tmp/layout-scenarios.csv
python3 -m tools.run_layout_sensitivity --config configs/layout-aware-shared.yaml --output /tmp/layout-sensitivity.csv
python3 -m tools.run_layout_dense_sharding --batch 8 --input-tokens 8192 --output-tokens 16 --output /tmp/layout-dense-sharding.json
```

The fixed and greedy policies are evaluated with the same new cost backend.
Do not divide a new-model latency by a legacy-model latency to claim a policy
speedup. The sensitivity command varies one provisional scale at a time and
records the fixed and greedy result for each value.
`configs/layout-aware-long-context.yaml` is a synthetic PIM-heavy stress case
for the same commands; it is not a trace-derived workload.

## Input and state

`simulation.cost_backend: layout_aware` requires `policy.name` to be
`layout_fixed` or `layout_greedy`. `attention_selector` is `dense` or
`recent_blocks`; the latter uses the last `recent_blocks` logical windows of
`pim.kv_block_tokens` tokens visible at the current step, even when a physical
KV segment spans many such windows. All heads of one request use the same selection.
The selector never reads future masks or arrivals.

A JSON trace may contain both `prefix_blocks` and `requests`:

```json
{
  "prefix_blocks": [
    {"id": "root", "parent_id": null, "tokens": 64},
    {"id": "branch", "parent_id": "root", "tokens": 32}
  ],
  "requests": [
    {"request_id": "a", "arrival_us": 0, "input_tokens": 128,
     "output_tokens": 16, "prefix_leaf_id": "branch"}
  ]
}
```

The same fields may be inline in YAML. Prefix IDs assert actual shared KV
identity; matching lengths do not imply sharing. A request's remaining prompt
and all generated tokens are private. Shared physical blocks are replicated
per data-parallel replica. A completed request releases its references;
unreferenced PIM prefix blocks may remain only within
`prefix_cache_budget_gib`, and cached blocks produce no decode work. Arrival
time means *after prefill*; prefill latency is outside the model.

The state records per-layer block token ranges, references, physical copies,
rankset, layout, readiness, and one compute owner. A historical copy is
allocated before it is ready and switches the owner for all references only
at a safe decode boundary. Missing PIM weights are similarly prepared once
per replica/layer/operator. Capacity includes active copies, weights,
`layout_temporary_gib`, and a configurable future-token reserve.

## Timing and policy

Each request has a repeated PIM interval in actual operator order: QKV,
Attention, Merge, output projection, FFN. Its left edge may include the
preceding layer's FFN, so the interval can cross a layer boundary. The default
fixed AF interval contains Attention only. If a request's selected KV spans
both devices, Merge is an explicit call using the exact partial-softmax state
specified by `elasticafd.partial_softmax`; impossible QKV/merge dependencies
are rejected. Complete FFNs are assigned per request.

Linear calls aggregate same-layer, same-operator, same-replica ready requests
and use GEMM FLOPs `2MKN`. Attention calls aggregate references to the *same
physical block within one subbatch*. Selected intervals are split at every
access boundary: an interval accessed by `M` queries contributes FLOPs
`4 h_q M L d_h`, while its physical KV bytes are counted once under an
optimistic local-read reuse assumption. Thus partially overlapping sparse
windows share only their intersection; separate subbatches each pay for their
own read. Forecast intervals use the probability that at least one request
accesses each slice. This assumes a fused execution path capable of reusing
each KV tile across the participating queries and is not kernel-validated. PIM
work is summed within each physical rankset for one operator call; independently
executable ranksets finish at their maximum for that call. Dependent operators
and layer cycles are added in execution order. Device calls include roofline weight and
activation reads, while weight preparation and KV copying are charged
separately.
GPU bandwidth and throughput are divided among data-parallel replicas;
independent replicas execute concurrently, while each replica keeps its
own KV capacity limit.

### Fixed PIM dense sharding (`layout-v4-link-calendar`, inherited from v3)

`simulation.layout_dense_ranksets` defaults to **1**. Values 2 and 4 are
separate fixed-layout sensitivity cases, not a free scheduling action. The
number must be between 1 and the hardware's rankset count. QKV, output
projection and FFN matrices are split along their output-neuron (N) axis;
this parameter is independent of the legacy backend's
`compute_ranksets_per_layer` setting. The group for a layer starts at
`layer % ranksets` and continues cyclically.
The QKV output is laid out as contiguous Q, K, V columns. Attention KV
blocks stay on their original single layer-home rankset, and changing this
dense-layout parameter leaves the Attention kernel estimate unchanged.

For each shard, GEMM FLOPs, weights, output writes, and local FFN
intermediate traffic use that shard's N width. **Every shard reads the full
input activation.** The operator call completes when its slowest shard
finishes; dependent operator calls and layers remain serial. The same
total weight bytes are copied from GPU to PIM once per replica/layer/operator
and are distributed as disjoint shards over ranksets. Accordingly total PIM
weight capacity and GPU-PIM preparation bytes do not fall by 4; the peak
occupancy of an individual rankset can fall. The model assumes the GPU-PIM
DMA routes each packed weight shard directly to its destination rankset,
without an additional PIM-internal weight redistribution. That routing is a
layout assumption to validate on hardware.

For activations, the existing GPU-PIM boundary packet delivers a full input
once to the layer-home rankset when a dense operator follows GPU work. The
home then broadcasts it to the other dense shards. A PIM Attention or Merge
also produces its input at home, so output projection broadcasts it there.
QKV gathers only the Q columns needed by local PIM Attention and the K/V
columns needed for local future KV writes. The exact remote-column count
follows the contiguous QKV shard boundaries. If Q or the following dense
output goes straight to GPU, its shards stream through the already charged
GPU-PIM packet; no PIM-internal concatenation is charged for that output.
If PIM output projection feeds PIM FFN, its output is all-gathered so every
FFN shard has the full input. FFN up/gate shards perform local elementwise
work, then all-gather the intermediate activation before the N-sharded down
projection. A PIM FFN feeding the next layer's PIM QKV likewise delivers a
full output to all destination groups, including groups that differ between
layers. The first layer's PIM QKV input is charged as a GPU-to-PIM packet
after prefill.

The rank bus is treated as one serial resource across the participating
ranksets: broadcast/all-gather moves `(R-1) * tensor_bytes`, and transfers
between different groups add their actual bytes. Each packed internal
boundary pays `layout_rankset_sync_us` once when nonzero. Rank-bus time is
added after the per-operator group maxima, so this estimate assumes no
compute/communication overlap. The bus bandwidth and synchronization are
not calibrated for these collectives. In particular, background PIM weight
writes competing with foreground KV reads, routing overhead, and packet
startup on the GPU-PIM link remain outside this sharding estimate. Summaries
report `layout_dense_ranksets`, `cross_group_bytes`, and `cross_group_us`.

The paired stage objective is `max(G0, A1) + max(G1, A0) + E0 + E1`
per layer cycle, summed over layers.
Transfers are packed by layer, stage, and direction. Foreground boundary
packets and background KV/weight preparations now occupy one link calendar:
half-duplex shares one timeline, while full-duplex has one timeline per
direction. The model reserves foreground packets at the stage start; an
in-flight background packet may finish, then foreground has priority before
unsent background packets. Background transfers yield at
`layout_preparation_packet_bytes` (default 4 MiB), an explicitly provisional
preemption granularity. A preparation pays the configured transfer startup
once, not once per scheduling packet. Its optional weight conversion begins
after the DMA and does not occupy the GPU-PIM link. Foreground events are
persisted only in real simulation steps; forecasts book them on a copied
calendar. Copy and interval readiness are updated if foreground traffic
pushes their DMA completion later.

The stage-level approximation uses aggregate directional transfer durations,
not individual operator packet release times. If the last boundary packet
finishes at F after stage start T, exposed boundary time is
`max(0, F - T - layout_boundary_overlap_fraction * abs(G - P))`.
Stage time is at least both `max(G,P) + layout_boundary_scale * exposed` and
`F-T`, so a sensitivity scale cannot make a phase finish before required data
arrives. With no preparation contention and scale 1, this reduces to the
previous boundary expression. The overlap fraction describes computation
overlap, a different mechanism from link occupancy; it remains provisional
and should be swept. Its default is 0.25. The link's per-transfer startup
`transfer_latency_us` defaults to zero and needs calibration; the existing
expected-presence multiplier on foreground startup is retained.

Summary fields `link_busy_us`, `preparation_link_busy_us`, and
`foreground_wait_us` respectively report booked foreground link occupancy,
background occupancy elapsed within the simulated run, and foreground time
waiting for in-flight background packets. Link contention does not cover
PIM-bank contention between background weight writes and foreground KV reads.
That physical mechanism needs a separately named and calibrated model.
The old `layout_preparation_exposed_fraction` configuration was removed;
configurations still containing it fail strict parsing.

The GPU/PIM bottleneck phase label is assigned from per-layer critical-path
time: each layer credits `max(G,P)` to its slower side, splitting exact ties.
`mixed` means the two sides receive equal total credit. Separate layer-count
fractions report how often each side is slower; raw GPU/PIM busy times retain
their original meaning. Boundary time is reported separately.

When the hardware preset has no PIM softmax throughput, this backend uses
the explicitly provisional `layout_pim_softmax_gelements_s_per_rankset: 50`
estimate. `layout_pim_dense_scale` multiplies each provisional PIM dense
shard time. These parameters support sensitivity analysis, not claims about
the real kernels.
`layout_weight_conversion_us_per_gib` adds an optional PIM weight-layout
preparation cost; its zero default is an optimistic bound and must be changed
when a layout conversion measurement is available.
`layout_rankset_sync_us` exposes synchronization for cross-group dense
collectives and for custom Attention KV placement spanning groups.

`layout_greedy` evaluates future-KV writes, existing or copied history, and
neighboring interval boundaries. It forms request groups from explicit
prefix identity and also proposes shared-block copies once per physical
object with all active references. A cheap screen sees every active request
group, estimates relevance to both phase bottlenecks and the busiest PIM
rankset, discounts preparation bytes and low remaining life, then rotates
ties across replans. It allocates full evaluations across phase and action
kind, with at most two greedy rounds and `layout_max_candidates` candidates
per round. The cheap score only orders proposals; each selected action is
reevaluated with the full two-stage cost, capacity and readiness checks.
The summary includes `policy_candidate_revision: phase-screen-v3` and per
decision `candidate_screening` counts for eligible, evaluated and truncated
groups, evaluated request IDs, and candidate scores or rejection reasons. The policy reevaluates each full
combined action relative to the current state over a bounded horizon.
`policy.decision_interval` sets the replanning cadence (Delta), while
`policy.planning_horizon_tokens` sets the complete forecast window (H); neither
silently caps the other. The larger of H and Delta sets the minimum KV growth
reservation checked when admitting a request or preparing an action. Larger H
increases Python search work; `layout_max_candidates` and two greedy rounds
bound the number of candidate trajectories. The summary records both values
and the measured `policy_search_wall_time_s` separately from simulated time.
Candidate scoring samples complete remaining-output-length scenarios from a
Weibull total-length prior conditioned on each request's observed generated
progress (`layout_survival_mean_tokens`, `layout_survival_shape`). The default
`layout_forecast_scenarios: 4` uses reproducible stratified conditional
quantiles and `policy.prediction_seed`; every candidate and the fixed-layout
baseline replay the same sampled lengths. At each predicted step, requests
are either fully active or retired, so subbatches, shared references, KV
growth, capacity, preparations, and link use advance on a coherent path.
The score is the mean complete-window latency difference across scenarios.
The planner sees neither actual target output lengths nor future arrivals.
Shape 1 is the exponential prior and does not treat an already long-running
request as about to finish. Expected active-token counts remain a cheap
candidate-screening heuristic and part of the approximate service guard;
they do not determine final candidate latency. If the current layout fails
the deterministic growth-reserve check, capacity recovery takes precedence
because an overflowing baseline has no valid future latency. More scenarios
reduce sampling noise but increase Python search time roughly proportionally.
Actual target output lengths are used only by the simulator to end requests.
The new backend requires the default `policy.prediction: fixed` and
`policy.optimizer: horizon_greedy`; legacy oracle/profile policy flags are
rejected instead of being silently ignored. Preparations can overlap old
execution; a new owner or interval becomes effective only when its data is
ready. A
relative improvement threshold and `layout_error_margin_us` guard changes.
Admissions and periodic checkpoints trigger replanning; completions can
trigger it when `policy.event_driven_replan` is enabled. The
`layout_min_replan_steps` gate (default 4) limits how often these triggers
invoke the bounded search.
When configured, TPOT and end-to-end SLO thresholds also reject an action
whose short-window estimate would cross a threshold that the current mapping
meets. This is a screening guard, not a tail-latency guarantee.
The summary labels `layout-v4-link-calendar`, reports physical KV and preparation metrics,
and lists uncalibrated parameters. The provisional roofline omits real
shape-dependent kernel profiles, allocator fragmentation beyond block
rounding, prefill interference, and a measured pipeline startup/drain model.
`policy_search_wall_time_s` reports the actual Python planning time separately;
it is not included in the simulated decode latency, so a slow large-trace
search should not be interpreted as an online-ready implementation.
