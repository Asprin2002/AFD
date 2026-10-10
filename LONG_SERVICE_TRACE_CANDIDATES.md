# Long-service trace candidates for review

These are workload inputs only. No strategy replay has been run on them.

## Selection

Each candidate is the central **1,024 consecutive requests** in its public
source file. Every source record in the interval is retained in original order.
Only the first timestamp is subtracted to make arrivals start at zero; there
is no time scaling, token-length filtering, or request duplication. The
proposed continuous-batching cap is 64, so new requests continue arriving
after the initial batch is full. Actual active batch size will also be limited
by KV capacity and completion/admission behavior.

| Candidate | Source lines | Arrival span | Offered rate | Input tokens mean / p50 / p95 | Output tokens mean / p50 / p95 | Both input and output ≥1K | Largest gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen Thinking | 4,895–5,918 | 584.446 s | 1.750/s | 4,432 / 3,682 / 15,222 | 4,508 / 1,799 / 20,269 | 605/1,024 | 4.246 s |
| Qwen Coder | 20,994–22,017 | 179.756 s | 5.691/s | 5,339 / 3,834 / 13,289 | 802 / 496 / 2,377 | 208/1,024 | 1.391 s |
| Mooncake Conversation | 5,504–6,527 | 303.000 s | 3.376/s | 11,117 / 6,356 / 34,370 | 353 / 356 / 716 | 18/1,024 | 3.001 s |

Every 10-second bin throughout each selected interval contains arrivals.
The individual candidate CSVs and JSON metadata are in
`workloads/public/long_service_candidates/`.

The Qwen Thinking trace is the clearest real-input **K-input/K-output** case;
its source is a reasoning-service trace with long-form generations. Qwen
Coder supplies a denser mixed workload and more plausible opportunities for
repeated admission up to a 64-request cap. Mooncake Conversation isolates
the long-context Attention/KV pressure with mostly short outputs.

Sources: [Qwen Bailian anonymous production traces](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)
and [Mooncake FAST'25 traces](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md).
Qwen timestamps are seconds; Mooncake timestamps are milliseconds. The
preparation script checks SHA-256 hashes of the exact downloaded source files.

## Static capacity screen before replay

Under the current GPT-175B/CHIME settings, a full-precision KV token occupies
4.5 MiB over 96 layers. The joint operator menu reserves 324 GiB of PIM for
dense weights, leaving approximately 1,724 GiB of PIM KV capacity and 314 GiB
of GPU KV capacity. These figures come from the current code's capacity
accounting with a batch cap of 64; they are not measured runtime occupancy.

For every consecutive group of 64 records inside each selected trace, the
following are the **largest hypothetical simultaneous KV footprints**:

| Candidate | Inputs only, all 64 on PIM | Input + complete output, PIM + GPU |
| --- | ---: | ---: |
| Qwen Thinking | 1,897 GiB | 3,846 GiB |
| Qwen Coder | 1,964 GiB | 2,152 GiB |
| Mooncake Conversation | 5,034 GiB | 5,126 GiB |

Thus a configured `max_batch_size=64` does not imply 64 simultaneously active
requests for all windows. Qwen Coder is closest to that physical regime;
Mooncake is primarily an admission/capacity stress case. The replay should
report actual batch size, queue length, and capacity-delayed admissions rather
than assume that the cap was reached.

## Interpretation decisions before running

- The source services use Qwen or Kimi models. Feeding their lengths and
  arrivals into GPT-175B is a **counterfactual hardware workload**, not a
  replay of the source model's measured latency or its native context limit.
- The existing simulator interprets each arrival as decode-ready and does not
  model prefill time. For these long inputs, that assumption materially changes
  timing. It must be retained as an explicit decode-only experiment or revised
  before an end-to-end service claim.
- Source prefix-cache hashes exist, but the current simulator does not model
  prefix reuse. Treating every request's entire input as resident KV may
  overestimate capacity pressure where prefixes are shared.
- The Qwen Thinking window includes an individual output as long as 60,766
  tokens; the Mooncake window includes an input as long as 124,847 tokens.
  Neither length has been clipped. The proposed model must accept those
  context lengths, or a separately declared length cap must be selected.

Preparation command:

```bash
python3 -B tools/prepare_long_service_candidates.py \
  --qwen-thinking /private/tmp/AFD_qwen_thinking.jsonl \
  --qwen-coder /private/tmp/AFD_qwen_coder.jsonl \
  --mooncake-conversation /private/tmp/AFD_mooncake_conversation.jsonl \
  --output-dir workloads/public/long_service_candidates \
  --count 1024
```
