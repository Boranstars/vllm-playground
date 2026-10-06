# Week 1 experiments

## Prompt length

**Question:** How does prompt/context length affect TTFT and decode performance?

**Controlled:** Qwen/Qwen3-0.6B, concurrency 1, maximum output 64 tokens.

**Variable:** Prompt length.

**Method:** Five repetitions per size, randomized order, unique per-request prefix, one uncounted warmup; medians reduce sensitivity to individual outliers.

**Observed in the latest clean run:** Median actual prompt lengths were approximately 77, 178, 588, 1130, and 2230 tokens. Median TTFT was 14, 15, 34, 62, and 133 ms. Median decode throughput was 161.356, 159.907, 153.606, 143.946, and 137.298 tokens/s.

**Interpretation:** Longer prompts add Prefill work and strongly increase TTFT. Decode also slows somewhat because each new query attends over more historical KV. In this workload, TTFT is much more sensitive to prompt length than TPOT.

**Caveats:** These are client-observed measurements over localhost, on one RTX 4060 Laptop GPU, with the small Qwen3-0.6B model. This is an educational experiment, not a hardware benchmark score.

## Output length

**Question:** How does output length affect E2E latency and Decode performance when prompt workload is approximately fixed?

**Controlled:**
- model = Qwen/Qwen3-0.6B
- concurrency = 1
- prompt ≈ 512 tokens

**Variable:** Output length / max_tokens.

**Method:**
- 5 repetitions
- randomized order
- unique early request prefix
- fixed base prompt body
- one uncounted warmup
- medians

**Results:** Tested output limits of 16, 64, 128, 256, and 512 tokens. Actual prompt lengths stayed clustered around 575–580 tokens. TTFT remained approximately constant. E2E latency increased approximately linearly with generated length. Client-observed TPOT was generally around 6–6.5 ms/token, with raw outliers—especially for shorter outputs—across repeated runs.

**Interpretation:** Longer outputs mainly add Decode steps and increase total E2E latency. Current data does not support a strong monotonic increase in TPOT with output length.

**Caveats:** Client-observed measurements over localhost with Qwen/Qwen3-0.6B on a single RTX 4060 Laptop GPU.
