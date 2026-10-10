# Qwen Coder 512-request joint-strategy pilot

This pilot compares the new decode-only CHIME-style baseline with the full
joint policy on one continuous request stream. The new baseline uses the
`ChimeDecodeBaseline` predictor-based N=1 two-sub-batch formation for MHA,
the CHIME operator/KV mapping, and ordered GPU/PIM visits. It is a
paper-aligned simulator specialization, not the official CHIME runtime.

## Input

`workloads/public/long_service_candidates/qwen-coder-capacity-512.csv` contains
512 consecutive Qwen Coder production requests, source lines 33,343–33,854.
Arrival times span 89.958 seconds at a native offered rate of 5.680 requests
per second. No arrivals or token lengths were scaled. Average input/output
lengths are 4,998/789 tokens; 113 requests have both lengths at least 1,024.
There are 404,016 output tokens total. The maximum single output is 6,010.

The selection rule and exact source/trace checksums are in the adjacent
`.metadata.json`. `tools/prepare_qwen_coder_512.py` regenerates the same CSV
from the public [Qwen Bailian source](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon).

## Execution

From the project root:

```bash
python3 -B tools/run_full_joint_512.py --dry-run
python3 -B -u tools/run_full_joint_512.py \
  --trace workloads/public/long_service_candidates/qwen-coder-capacity-512.csv \
  --output-dir results/full-joint-qwen-coder-512-chime-baseline \
  --modes chime_baseline joint
```

The runner fixes `max_batch_size=64`, FCFS admission, rankset-event timing,
and `device_contiguous` execution. The CHIME-style baseline keeps prompt and
future KV plus Attention on PIM, QKV/O-projection/FFN on GPU, and forms two
sub-batches each decode iteration with the new N=1 predictive pairing model.
The joint mode uses LPT sub-batches and searches legal resident
QKV/O-projection/FFN plans together with future GPU/PIM KV targets, with
64-iteration and admission/completion replanning. Thus the two modes share
trace, hardware, admission, and event timing, while each uses its own
intended sub-batch formation.

The run writes `chime_baseline.json` and `joint.json` in the output directory. Each
contains the full configuration, source-code and trace checksums, request
results, sampled iterations, switch events, and latency/resource summary.
Once both modes finish, `paired-summary.json` records the warm latency
comparison and actual batch sizes.
The runner checks the new baseline's type, placement, and sub-batch coverage
for batch sizes 1, 16, and 64. This does not validate predicted latency
against CHIME's official artifact.
Results are saved after each completed mode. Repeating the command skips a
completed matching checkpoint; a changed simulator or trace raises a stale
checkpoint error rather than mixing versions. The previous
`results/full-joint-qwen-coder-512/control.json` used a different baseline and
was removed during the 2026-09-22 cleanup. The current runner reads only the
independent CHIME-style baseline directory documented above.

For attribution after this paired pilot, run the same command with
`--modes attention f_all`; those modes use the joint strategy's trace and
configuration.

## Interpretation

This is a decode-ready, decode-only replay of request submissions from a Qwen
service on the current GPT-175B/CHIME model. Prefill timing, CHIME's
runtime-trained RFR predictor, and source prefix reuse are not represented.
A cap of 64 does not guarantee 64 active requests;
the outputs must be checked for actual batch size and queueing. Despite the
smaller request count, 404,016 output tokens make token-exact simulation
potentially time consuming.
