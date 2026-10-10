# Full joint strategy trace pilot

## Question and execution model

This pilot replays one request stream while allowing the policy to choose both
future KV placement and resident dense-operator placement at each causal
decision. The operator menu contains only paths legal under the current
`device_contiguous` GPU→PIM→GPU progression. Four paired modes are run on each
trace: fixed all-PIM placement, Attention/KV placement only, F-capable operator placement
only, and the joint policy. The policy chooses from the legal joint action menu
using the same event-based cost model that evaluates the candidate schedule.
The all-PIM control uses CHIME's operator mapping but shares this simulator's
new sub-batch schedule; it is not a validated reproduction of the published
CHIME scheduler. The comparisons below are same-scheduler mechanism results.

## Trace source and selection

Source: [BurstGPT public repository](https://github.com/HPMLL/BurstGPT),
`data/BurstGPT_1.csv`. The downloaded source SHA-256 is
`46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a`.
It has 1,429,737 records; 1,404,294 have positive input and output token
lengths. The three main windows retain the source request order, interarrival
times, input lengths, and output lengths. No time scaling or length filter was
applied after excluding zero-token or failed rows. For each request-rate
quantile among eligible hours (10th, 50th, 90th), the central 32 consecutive
positive-token requests were selected.

| Window | Native span | Mean input tokens | Mean output tokens | Mean active batch |
| --- | ---: | ---: | ---: | ---: |
| Quiet | 933 s | 649.9 | 143.2 | 1.09 |
| Typical | 446 s | 475.0 | 166.5 | 1.46–1.48 |
| Busy | 13 s | 202.1 | 7.0 | 2.29 |

The input format is `request_id,arrival_s,input_tokens,output_tokens`, with
`arrival_s` relative to the first selected submission. Files and selection
metadata are in `workloads/public/burstgpt/representative-v1/`.

A separate tail stress window selects the first contiguous 32-request window
at the 99th percentile of rolling output-token sum among windows spanning at
most 300 native seconds. Its span is 37 seconds; mean input and output lengths
are 852.6 and 1198.4 tokens. This outcome-conditioned window is a mechanism
stress test, not an unbiased service sample.

## Main results

All four modes within each real-trace window use the same source-code hash,
`bd8adba924a5c271…`. Every run completed all 32 requests and conserved the
output-token count. Numbers below are warm-replay mean end-to-end latency.

| Window | Control | Attention only | F capable only | Joint | Joint KV / operator switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quiet | 8,860 ms | 8,860 ms | 8,313 ms | 8,313 ms | 0 / 8 |
| Typical | 13,528 ms | 13,551 ms | 11,607 ms | 11,842 ms | 2 / 19 |
| Busy | 716 ms | 716 ms | 600 ms | 600 ms | 0 / 2 |
| Tail stress | 133,061 ms | 133,061 ms | 129,411 ms | 129,411 ms | 0 / 10 |

The sampled real windows are predominantly GPU-bound. The joint policy
therefore mostly chooses F migration. In the typical window, its two KV moves
increase mean latency relative to F-only; causal local choices do not guarantee
the best whole-trace outcome. The busy window's mean improves, while its p95
end-to-end latency rises from 727 ms to 761 ms. Full metrics and decisions are
in `results/full-joint-burstgpt-v1/` and
`results/full-joint-burstgpt-tail-v1/`.

## Interpretation limits

- BurstGPT includes both ChatGPT and GPT-4 calls. This experiment maps their
  recorded lengths and times onto a single GPT-175B hardware model, so the
  replay is a service-shaped input, not a reconstruction of the original
  service's measured latency.
- Submission times are treated as decode-ready arrivals. Prefill is assumed
  complete and prompt KV is initially on PIM.
- The PIM dense-operator cost is a conditional roofline estimate rather than
  a measured CHIME kernel. The roughly 5.8-second initial weight copy in modes
  with dense offload is reported separately and excluded from warm latency.
- Three 32-request windows and one conditioned tail window are a pilot, not a
  statistical estimate for the whole `BurstGPT_1` source file.
- A synthetic batch-16, input-5200, output-1024 mechanism check showed the
  Attention branch selecting 16 GPU-KV switches. Its four modes ran across a
  concurrent shared-source edit, so the check is excluded from paired
  quantitative comparisons.

## Reproduction

The preparation and replay entry point is `tools/run_full_joint_trace.py`.
It validates the source checksum, records the model-code checksum with each
run, and rejects stale result checkpoints. `tools/prepare_burstgpt_tail_window.py`
and `tools/run_full_joint_tail.py` produce the separate tail stress results.
The shared simulator source changed during a separate task after the real-trace
replays completed. Their saved checksums identify the exact version used;
replaying with the current source requires fresh result files.
