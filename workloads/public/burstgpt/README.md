# BurstGPT conditioned pilot traces

These files are small, deterministic derivatives of `BurstGPT_1.csv`. The
original 48.4 MiB trace is not vendored here.

## Source and license

- Dataset: BurstGPT, a real-world Azure OpenAI serving trace.
- Official repository: <https://github.com/HPMLL/BurstGPT>
- Source file: <https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv>
- Source SHA-256: `46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a`
- License: CC BY 4.0.
- Paper: Yuxin Wang et al., “BurstGPT: A Real-World Workload Dataset to
  Optimize LLM Serving Systems,” KDD 2025,
  <https://doi.org/10.1145/3711896.3737413>.

## Selection semantics

Only successful rows with positive request and response token counts are
eligible. Two conditioned classes are defined before selection:

- `short_input_long_output`: `1 <= input_tokens <= 2048` and
  `output_tokens >= 1024`;
- `long_input_long_output`: `input_tokens >= 4096` and
  `output_tokens >= 1024`.

For each class, the preparation tool selects the minimum-time-span consecutive
window of 32 requests in the class-conditioned point process. This deliberately
creates a burst stress subset. It is **not** an unbiased sample of the complete
BurstGPT length distribution.

The source contains 1,429,737 rows: 1,404,294 successful rows and 25,443 zero-
token/failed rows. The short/long conditioned populations contain 32,302 and 64
rows respectively. The chosen short window has 32 arrivals across two adjacent
seconds; the chosen long window spans 544,538 source seconds.

Token lengths and the relative inter-arrival structure are unchanged. Scaled
files apply one affine multiplier to timestamps after subtracting the first
timestamp. Every arrival is interpreted by this project as prefill-complete and
decode-ready; BurstGPT itself does not make that transformation.

## Files

- `*-native.csv`: selected source timestamps shifted to start at zero.
- `*-saturated.csv`: 1000-rps time scale used only for finite-trace CHIME drain
  calibration.
- `*-r85.csv` and `*-r110.csv`: workload-specific time scales derived from 0.85
  and 1.10 times the corresponding saturated CHIME completion rate.
- `*.metadata.json`: complete filter, source-row, scaling, length, and
  inter-arrival metadata for the adjacent CSV.

The saturated analytic CHIME completion rates are 0.0642365 req/s for the
short-input subset and 0.0417634 req/s for the long-input subset. These are
finite 32-request drain rates, not measured steady-state service capacities.

## Reproduction

Download the exact source to a temporary location, verify its hash, then run:

```bash
python3 tools/prepare_burstgpt_trace.py \
  --source /private/tmp/BurstGPT_1.csv \
  --output workloads/public/burstgpt/long-long-r85.csv \
  --class-name long_input_long_output \
  --count 32 \
  --target-rps 0.035498875815 \
  --metadata-output workloads/public/burstgpt/long-long-r85.metadata.json
```

The preparation tool rejects a source whose SHA-256 differs from the value
above.
