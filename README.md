# ElasticAFD

ElasticAFD is a decode-only simulator for studying GPU + DIMM-PIM scheduling.
The legacy experiment compares a
CHIME-style decode baseline with future-KV Attention splitting, conditional
dense-operator placement, and their online joint policy.

An opt-in `layout_aware` backend now implements physical KV references,
historical KV copies, per-request contiguous PIM intervals, and the two-stage
joint objective in `Attention动态调度建模与设计收敛.pdf`. See
[LAYOUT_AWARE_MODEL.md](LAYOUT_AWARE_MODEL.md) for its inputs, assumptions, and
paired experiment commands. The existing results below use the legacy model.
The current layout-aware cost revision is `layout-v4-link-calendar`; its full
shape matrix has not been rerun after the link-calendar and bottleneck-statistic
changes. Earlier layout-aware shape results retain their original model
revision and are only historical diagnostics.

## Read this first

The authoritative project handoff is [CURRENT_STATE.md](CURRENT_STATE.md).
It records the current execution semantics, trusted results, known limitations,
and the next experiment. Do not infer the current design from old stage names
or recreate deleted legacy results.

Supporting documents:

- [PARAMETER_PROVENANCE.md](PARAMETER_PROVENANCE.md): hardware and calibration
  provenance.
- [ATTENTION_FULL_EVENT_REPLAN_REPORT.md](ATTENTION_FULL_EVENT_REPLAN_REPORT.md):
  final static Attention-only sweep under the corrected pipeline.
- [FULL_JOINT_TRACE_REPORT.md](FULL_JOINT_TRACE_REPORT.md): small online trace
  pilot with four strategy modes.
- [FULL_JOINT_512_RUN.md](FULL_JOINT_512_RUN.md): latest 512-request online run.
- [LONG_SERVICE_TRACE_CANDIDATES.md](LONG_SERVICE_TRACE_CANDIDATES.md): trace
  selection rationale.

## Legacy execution semantics

- Arrivals are decode-ready. Prefill latency is not modeled.
- Historical KV never migrates in the legacy model. A policy changes only the destination of future
  KV and the device that computes Attention over that KV.
- The focused pipeline uses device-contiguous GPU/PIM visits. A logical layer
  has a GPU prefix, PIM middle, and GPU suffix; the suffix joins the next
  layer's prefix, producing an overall `G-P-G-P` alternation rather than
  unconstrained multi-layer run-ahead.
- GPU, each PIM rankset, and the shared half-duplex link are explicit mutually
  exclusive resources in the event model.
- Split Attention uses numerically stable partial-softmax statistics. In the
  device-contiguous experiment, the merge is PIM-owned to preserve the
  contiguous middle segment.
- PIM QKV/O-projection/FFN timings are conditional roofline estimates, not
  calibrated CHIME kernels. FFN on PIM requires complete PIM O-projection in
  the current legal two-cut path.
- `ChimeDecodeBaseline` has an independent N=1 predictive pairing routine. It
  is a paper-aligned decode specialization, not the official CHIME artifact or
  its runtime-trained predictor.

## Active results

Only the following result families are retained:

- `results/attention-full-event-replan-fixed-v3.csv`: final 96-shape static
  Attention sweep.
- `results/attention-event-replan-fixed-anchor-v3.csv`: denser anchor replay.
- `results/f-full-v1.csv`: conditional static dense-placement sweep.
- `results/chime-attention-device-n1-v2.csv`: selected Attention shapes against
  the independent CHIME-style baseline.
- `results/chime-f-device-n1-v2.csv`: selected dense-placement shapes against
  the independent CHIME-style baseline.
- `results/full-joint-burstgpt-v1/` and
  `results/full-joint-burstgpt-tail-v1/`: 32-request online pilots.
- `results/full-joint-qwen-coder-512-chime-baseline/`: latest paired 512-request
  CHIME-style baseline and joint run.

Each retained CSV has adjacent metadata where applicable. JSON runs contain
the model-code and trace hashes needed to reject stale comparisons.

## Current entry points

- `tools/run_separate_full_ablations.py`: static Attention/F sweeps.
- `tools/run_chime_baseline_light.py`: selected independent-baseline rechecks.
- `tools/run_full_joint_trace.py`: four-mode 32-request trace pilot.
- `tools/run_full_joint_tail.py`: conditioned tail trace pilot.
- `tools/run_full_joint_512.py`: latest continuous 512-request runner.
- `tools/prepare_burstgpt_trace.py`, `tools/prepare_burstgpt_tail_window.py`,
  `tools/prepare_long_service_candidates.py`, and
  `tools/prepare_qwen_coder_512.py`: reproducible trace preparation.

Other retained experiment helpers are imported by current tools or regression
tests. They are not the recommended starting point for new experiments.

## Install and verify

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -q
```

The cleanup baseline on 2026-09-22 was 131 passing tests. Before the
layout-aware addition, the full suite passed 168 tests; the current suite
passed 201 tests on 2026-10-12. Source files are tracked in the GitHub
repository `Asprin2002/AFD`; generated `results/` remain ignored by Git.
Result metadata hashes and explicit checkpoints identify experiment outputs.

## Interpretation rules

- Do not mix results produced by `free_wavefront`, old operator-family phases,
  and the current `device_contiguous` pipeline.
- A `1.5x` speedup means one third less execution time, not a 50% latency drop.
- `pim_busy` is normally the busiest single rankset, not total PIM work.
- A resource lower-bound gap is not an exact pipeline-bubble measurement.
- Warm dense-offload timing excludes the initial PIM weight copy; cold timing
  includes it.
- The legacy simulator has no explicit finite activation-buffer/backpressure
  model and does not include prefill, prefix caching, or calibrated PIM FFN.
