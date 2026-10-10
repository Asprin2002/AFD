# Current state after cleanup

Date: 2026-09-22

## Research question

The project studies whether immutable future-KV placement and legal contiguous
operator placement can reduce the longer GPU/PIM stage in a CHIME-style decode
pipeline. The goal is to identify effective and ineffective workload regions,
not to assume that migration always helps.

## Canonical model

- GPT-175B, 96 layers, FP16, TP=8, DP=1.
- Eight A100 GPUs: 640 GiB aggregate HBM, 16.3 TB/s aggregate bandwidth.
- DIMM-PIM: 2048 GiB, 13 TB/s aggregate bandwidth, four ranksets; one rankset
  serves a layer by default.
- Shared half-duplex GPU/PIM link: 60 GB/s in either direction.
- Arrivals are prefill-complete and decode-ready. Prompt KV initially resides
  on PIM.
- Runtime reserve, DMA fixed latency, and allocator-fragmentation defaults are
  optimistic and require later sensitivity analysis.

## Pipeline and legality

The active focused model is `device_contiguous`. Each sub-batch alternates
GPU and PIM visits without multi-layer lead. A logical layer may be cut into a
GPU prefix, PIM middle, and GPU suffix; adjacent GPU suffix/prefix work is one
continuous visit, so the whole model follows `G-P-G-P`.

Historical KV never migrates. GPU/PIM local Attention results use stable
partial-softmax statistics. The current device-contiguous graph performs the
merge on PIM. PIM FFN requires full PIM O-projection, and PIM QKV is legal only
when Attention/new KV remain on PIM. These constraints avoid device ping-pong.

The event model schedules dependencies and mutual exclusion for GPU, every PIM
rankset, and the link. It does not yet model finite activation buffers.

## Baseline

`elasticafd/chime_baseline.py` implements a separate CHIME-style decode
baseline: all KV/Attention on PIM, QKV/O-projection/FFN on GPU, and N=1
predictive request pairing for GPT-175B MHA. The paper does not completely
specify decode-only admission/order, and this implementation uses the shared
simulator cost predictor rather than CHIME's runtime-trained random forest.
It must not be described as an official artifact reproduction.

## Trusted static results

The final Attention-only grid contains batch sizes 1/2/4/8/16/32, input
lengths 128/2048/5200/8192, and output lengths 256/1024/4096/8192. Of 96
shapes, 95 have a capacity-feasible all-PIM baseline.

| B / input / output | Strategy | Speedup |
| --- | --- | ---: |
| 16 / 5200 / 8192 | bounded offline KV window | 1.5074x |
| 32 / 2048 / 4096 | bounded offline KV window | 1.4847x |
| 32 / 2048 / 8192 | bounded offline KV window | 1.4001x |
| 16 / 5200 / 4096 | 64-token replanning | 1.3742x |
| 8 / 8192 / 8192 | bounded offline KV window | 1.2694x |

Thirty-five feasible shapes exceed 1.01x, 18 exceed 1.10x, and 11 exceed
1.20x. Benefits concentrate in large-batch, long-context, long-output cases
where all-PIM Attention is the long stage. Batch 2 and 4 have no positive case
in this grid; the best batch-1 speedup is 1.0739x. Selected independent-baseline
rechecks reproduce the important anchor results. B4/input5200/output4096 gives
0.9808x, confirming that GPU Attention can hurt when PIM pressure is too low.

The static dense-placement sweep is conditional on PIM dense capability. The
best selected independent-baseline results are:

| B / input / output | Placement | Warm / cold speedup |
| --- | --- | ---: |
| 2 / 128 / 256 | PIM QKV + full O-projection, no FFN | 1.4295x / 1.2949x |
| 2 / 128 / 4096 | PIM QKV + 50% O-projection, no FFN | 1.4012x / 1.3935x |

The largest gains therefore come from QKV/O-projection, not FFN. Plans with
full O-projection plus 12.5% FFN are around 1.20x in representative B=2 cases.
PIM FFN/GELU/residual/norm remain uncalibrated conditional capabilities.

## Online results

The 32-request BurstGPT pilot shows that causal joint scheduling can match or
slightly underperform F-only when unnecessary future-KV moves add GPU
Attention to an already GPU-bound trace. These windows are mechanism pilots,
not population estimates.

The newest run uses 512 consecutive Qwen Coder requests, max batch 64, FCFS,
and decode-ready arrivals. It compares the independent CHIME-style baseline
with joint scheduling:

| Metric | CHIME-style | Joint | Change |
| --- | ---: | ---: | ---: |
| simulated time | 3796.54 s | 3457.61 s | 1.098x speedup |
| throughput | 106.42 token/s | 116.85 token/s | +9.8% |
| mean E2E | 1520.30 s | 1457.14 s | -4.15% |
| P95 E2E | 3112.20 s | 2851.15 s | -8.39% |
| mean TPOT | 524.35 ms | 513.10 ms | -2.15% |

The trace is severely overloaded: 512 requests arrive over about 90 seconds,
while draining takes roughly 3400--3700 seconds. It is a capacity/drain test,
not a stable-service latency result. Joint performs 451 KV-writer switches and
only nine operator switches, reserves about 324 GiB of PIM dense weights, and
shifts bottleneck phases strongly toward GPU. Its baseline uses CHIME pairing,
whereas joint uses LPT, so the observed difference combines placement and
sub-batch-formation effects.

### Low-load alignment-formation rerun (observed 2026-09-28)

All policy arms now use the alignment (CHIME-style) sub-batch formation from the
heterogeneous static matrix; LPT is retired. The overloaded native trace was
rescaled (`tools/rescale_online_load.py`) to ~77 tokens/s offered (0.7x the
measured ~110 tokens/s capacity) into `qwen-coder-stable-512.csv`, giving a
single-replica stable-service regime (average batch 7.6). Three arms on the same
trace and code hash (`results/full-joint-qwen-coder-stable-alignment/`):

| metric | chime_baseline | control(alignment) | joint(alignment) |
| --- | ---: | ---: | ---: |
| throughput (tok/s) | 72.18 | 72.18 | 72.53 |
| E2E mean (s) | 84.75 | 84.65 | 83.69 |
| E2E p95 (s) | 261.1 | 260.8 | 256.8 |
| mean TPOT (ms) | 107.42 | 107.27 | 106.74 |
| GPU path util | 0.971 | 0.972 | 0.971 |
| KV / op switches | 0 / 0 | 0 / 0 | 976 / 358 |

Same-formation attribution: joint improves over control by only 1.15% E2E mean /
1.51% p95 / +0.49% throughput, and pays 976 KV plus 358 operator switches.
Mechanism: GPU path utilization is already 0.97 at this load, so migrating
future-KV to GPU adds GPU-attention work onto a saturated GPU -- the
"capacity-headroom" multiplier in the benefit formula is near zero. This is the
honest corner of the benefit window: big gains (batch >= 16) belong to
higher-concurrency/long-output regimes, not to the stable regime a single-replica
device can serve. Smoke on the 32-request short-long-r85 trace still shows
alignment == LPT (formation axis degenerate on near-homogeneous loads).

## Next required experiment

Before expanding the workload matrix:

1. Run `attention` and `f_all` on the same 512-request trace and current code
   hash, completing the four-mode attribution with the existing CHIME baseline
   and joint results.
2. Add a same-scheduler diagnostic to separate placement gains from CHIME
   pairing versus LPT formation. Keep the independent CHIME baseline as the
   formal comparison.
3. Replay the same trace at stable, near-saturation, and overload arrival rates.
4. Investigate the 451 KV switches using longer residence/horizon and a
   source-bottleneck guard, then validate on a different trace.
5. Sweep PIM dense/merge cost, weight-copy policy, link bandwidth, and segment
   overhead before treating dense migration as a hardware conclusion.

## M0–M4 static-matrix pipeline (results observed 2026-09-24)

A closed-static-batch joint-planning pipeline. M0–M4 deliver code + 37 unit
tests; the flagship recheck of the four polluted cells was executed on the CLI
and the numbers below are **observed run outputs**, not mechanism predictions.

- M0 `tools/prepare_static_recipes.py`: 17 trusted-grid homogeneous anchors +
  3 hetero profiles (long_in/long_out/mixed) x seeds {20260922, 20260923} x
  batch {8,16,32} = 35 recipes under workloads/static, all capacity-preflighted.
- M1-A `elasticafd/segmented.py`: `DCSchedule` (operator plan_timeline +
  per-request future-KV cuts); `state_views` with heterogeneous clipping; XOR
  legality; `eval_schedule` with running-peak reserve-max capacity;
  `search_operator_timeline` / `search_kv_timeline` (epoch-greedy, lookahead
  256) / `search_joint`; plus the feasibility closure `_kv_closure` (2026-09-24):
  the greedy short-window acceptance cannot see the monotone GPU-KV capacity
  cliff near the end of a long-output lifetime, so it can over-migrate and
  return a full-lifetime-infeasible schedule; the closure re-feasibilizes by
  binary-searching the largest "return to PIM" boundary for the crashing
  block's GPU-tail requests. Covered by `test_kv_closure_*` regressions.
- M1-B+M2+M3 `tools/run_heterogeneous_joint_matrix.py`: strategy x formation
  {lpt, alignment} x cost arm {real, free} matrix; independent JointAF
  best-static oracle replayed on the same engine; fresh-engine replay guard.
  2026-09-24: the three M1-B guards now use an always-real `guard_pricing` so
  they run under `--arms free` alone (previously `KeyError: 'real'`), and
  `--report-arms` decouples re-valuing each searched schedule under a different
  cost model from searching under it.
- M4 `tools/analyze_heterogeneous_joint_matrix.py`: digests matrix.csv into per
  scenario speedups, formation axis, economics axis, M1-B deltas, violations.

Observed run (`results/heterogeneous-joint-matrix-recheck{,/-extra,-freeb32}`
merged into one matrix; runner sha recorded in schedules.json). Every cell
feasible=1, violations=0, fresh-engine replay exact. Number = joint/attn
(identical everywhere; kv_switches>0, operator_switches=0), speedup =
control_total / total_us:

| scenario | B/L | joint | speedup | vs best_const | refΔ (us) | free==real |
| --- | --: | --: | --: | --: | --: | --: |
| anchor_b16_i5200_o8192 | 16/8192 | 1193.4s | 1.503x | < | −357,295,207 | identical |
| anchor_b32_i2048_o1024 | 32/1024 | 112.9s | 1.131x | < | −11,148,061 | identical |
| anchor_b32_i2048_o4096 | 32/4096 | 566.4s | 1.417x | < | −175,383,096 | identical |
| anchor_b32_i2048_o8192 | 32/8192 | 1773.0s | 1.345x | < | −367,914,604 | identical |
| anchor_b32_i5200_o4096 | 32/4096 | 1097.7s | 1.277x | < | −243,541,146 | identical |

Read-outs (all observed):
- b16/i5200/o8192 1.503x reproduces the trusted bounded-offline KV-window
  1.5074x; the closure is what rescues that ~1.5x from infeasible.
- Gain is all KV-placement: attn==joint and ffn_pilot==qkv_pilot==1.000x, so
  PIM-FFN/PIM-QKV operator plans add nothing here; the value is in where
  future-KV is written, consistent with the online joint results.
- Economics axis: the free-searched schedule re-valued at real prices equals
  the real-searched schedule on every cell (same total, same switches). The
  search demonstrably reads `pricing` in its acceptance criterion, so this is
  not vacuous: in this large-batch region each migration's compute benefit
  exceeds the 100us/switch penalty by orders of magnitude, so the optimal
  timing is cost-model-insensitive. Not claimed for thin-margin cells.
- Honest gap kept on record: pilot b8/i8192/o8192 joint 1133.6s loses to its
  own best_static 958.3s (+18%); a genuine "segmentation is not always best"
  case, recorded as a violation and left for future work, not hidden.

Execution is documented in `HETEROGENEOUS_JOINT_RUN.md` (smoke, recommended
pilot, full options, and the 2026-09-24 closure/report-arms additions).

## Cleanup decision

Legacy stage reports, wrong/free-wavefront result sets, superseded checkpoints,
generated document previews, packaging caches, and unreferenced result-summary
scripts were deleted on 2026-09-22. No active source module or regression test
was removed. The retained code passed 131 tests immediately before cleanup.
