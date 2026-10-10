# Parameter provenance and model limits

The default preset is intended to reconstruct the CHIME Attention-FC
Disaggregated system described in:

> CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated
> Inference with DIMM-PIM, ISCA 2026, DOI 10.1109/ISCA66397.2026.00056.

The paper PDF and original simulator are not present in this workspace. This
document therefore distinguishes imported prototype assumptions from
independent validation. Numerical agreement with an in-sample figure is not an
independent reproduction.

## Default system preset

| Field | Default | Status |
|---|---:|---|
| GPU count | 8 | Imported CHIME prototype assumption |
| Aggregate GPU capacity | 640 GiB | Imported assumption; binary GiB accounting |
| Aggregate GPU HBM bandwidth | 16.3 TB/s | Paper-derived in the prototype |
| Aggregate FP16 throughput | 2496 TFLOP/s | Paper-derived in the prototype |
| Dense HBM efficiency | 0.4115 | Fitted to Fig. 17 dense phases |
| PIM capacity | 2048 GiB | Imported CHIME prototype assumption |
| Aggregate PIM bandwidth | 13.0 TB/s | Paper-derived in the prototype |
| Ranksets | 4 | Paper-derived in the prototype |
| Ranksets used by one layer | 1 | Fig. 9 interpretation in the prototype |
| GPU-PIM link bandwidth | 60 GB/s each direction | Inferred from Fig. 17 transfers |
| Runtime/activation reserve | 0 GiB | Optimistic calibration default; must be swept |
| Explicit GPU KV budget | unset | Experiment cap applied after weights/reserve |
| KV allocation block | 1 token | No published block size; sensitivity parameter |
| Attention tile | 1 token | No published tile size; sensitivity parameter |

Bandwidth uses decimal GB/TB per second; memory capacity uses binary GiB.

## Calibration scope

`python -m elasticafd calibrate` reconstructs one representative GPT-175B
point with batch 12 and context 5.2K. It checks the imported Fig. 17 ranges:

- GPU QKV: 120--150 us per layer and half-batch.
- GPU projection + FFN: 370--440 us.
- PIM attention core: 430--530 us.
- GPU-to-PIM query/new-KV transfer: 7.2--7.4 us.
- PIM-to-GPU output transfer: 2.4--2.5 us.

The exact request-length composition of the paper point is not published in
the imported artifact. These assertions catch unit and topology regressions,
but they must not be described as out-of-sample validation.

## Capacity semantics

- Model weights are replicated once per data-parallel replica and sharded
  within its tensor-parallel group.
- Runtime reserve is an aggregate system quantity and is divided uniformly
  across DP replicas.
- Every request is pinned to one DP replica at admission.
- A replica cannot borrow free KV capacity or HBM bandwidth from another.
- CHIME and default elastic runs place prompt KV on PIM. GPU prompt fallback is
  disabled unless `simulation.allow_gpu_prompt_fallback: true` is explicit.

## Deliberately unmodeled

- Prompt prefill latency and prefill/decode interference.
- Network queuing beyond the modeled GPU-PIM link.
- Energy and static power.
- Numerical accuracy of partial-softmax merging.
- Real allocator metadata and fragmentation beyond token-block rounding.
- Prefix-cache sharing, eviction, reuse, and shared-prefix compute fusion.
- SLO-aware placement optimization; Stage 2C evaluates SLOs after a
  makespan-oriented future-arrival oracle but does not optimize them directly.
- Long steady-state traces with warmup/measurement/cooldown windows.

## Stage 3 joint-AF upper bound

`joint-af-small` and `joint-af-heterogeneous` reuse the calibrated CHIME
attention and GPU dense costs, but PIM O_proj is an optimistic roofline rather
than a paper-validated kernel. They reserve a persistent O_proj weight replica
in PIM, keep the GPU copy, and charge the initial copy with the selected
system's host-to-PIM link bandwidth. For the default GPT-175B/CHIME preset this
is 27 GiB over 60 GB/s, or about 0.483 s.

`joint-af-small` and `joint-af-heterogeneous` use
`max(GPU busy, PIM busy, link busy)` after summing the relevant operations.
Operator dependencies and PIM bank conflicts are not represented, so those
speedups are screening upper bounds.

`joint-af-event-replay` adds explicit QKV/attention/merge/O_proj/FFN edges,
GPU/PIM/link calendars, PIM rankset conflicts, and all activation transfers.
Its `f=0`, all-PIM-KV path is regression-tested against the generic
`attn_pim_control` event graph. It still inherits the unvalidated PIM O_proj
roofline and uses a bounded candidate oracle rather than global runtime-KV
exact search, so it is dependency-aware but not a calibrated end-to-end
hardware prediction.
