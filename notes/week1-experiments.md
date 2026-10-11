# Week 1 experiments

Selected result figures are preserved in `pics/` with descriptive names and embedded
below. Each section identifies its source run; original CSV/PNG outputs remain in
gitignored `results/`. These copies let the notes retain their figures in Git.

## Prompt length

**Question:** How does prompt/context length affect TTFT and decode performance?

**Controlled:** Qwen/Qwen3-0.6B, concurrency 1, maximum output 64 tokens.

**Variable:** Prompt length.

**Method:** Five repetitions per size, randomized order, unique per-request prefix, one uncounted warmup; medians reduce sensitivity to individual outliers.

**Illustrated run:** `20261006-222435`, from `results/prompt_length-20261006-222435.csv`. This is the only archived prompt-length run available in `results/`: all five sizes have five complete samples, with 64 output tokens and finish reason `length` throughout. The summary below uses this CSV so its numbers and figures refer to the same run.

**Observed in the illustrated run:** Median actual prompt lengths were approximately 77, 180, 588, 1135, and 2229 tokens. Median TTFT was 11.8, 14.6, 31.4, 55.3, and 120.4 ms. Median decode throughput was 172.182, 171.345, 167.733, 159.760, and 140.928 tokens/s.

**Interpretation:** Longer prompts add Prefill work and strongly increase TTFT. Decode also slows somewhat because each new query attends over more historical KV. In this workload, TTFT is much more sensitive to prompt length than TPOT.

**Figures:** Raw samples and group medians from the illustrated run.

![Prompt length versus TTFT: raw samples and group medians](pics/prompt-length-vs-ttft.png)

![Prompt length versus TPOT: raw samples and group medians](pics/prompt-length-vs-tpot.png)

**Caveats:** These are client-observed measurements over localhost, on one RTX 4060 Laptop GPU, with the small Qwen3-0.6B model. This is an educational experiment, not a hardware benchmark score.

## Output length

**Question:** How does output length affect E2E latency and Decode performance when prompt workload is approximately fixed?

**Controlled:**

- model = Qwen/Qwen3-0.6B
- concurrency = 1
- fixed prompt body; actual prompt ≈ 577 tokens

**Variable:** Output length / max_tokens.

**Method:**

- 5 repetitions
- randomized order
- unique early request prefix
- fixed base prompt body
- one uncounted warmup
- medians

**Illustrated run:** `20261006-225619`, from `results/output_length-20261006-225619.csv`. Three archived output-length runs were checked; all have five complete samples at each of the five output limits and finish reason `length` throughout. The latest complete run is used for both figures, retaining all 25 samples rather than selecting individual points or combining runs.

**Results:** Tested output limits of 16, 64, 128, 256, and 512 tokens; actual outputs reached each limit. In the illustrated run, actual prompt lengths stayed clustered around 574–581 tokens. TTFT medians remained approximately constant at 32.5–34.7 ms. E2E latency increased approximately linearly with generated length. Client-observed TPOT medians were approximately 6.3–6.6 ms/token, with visible raw variation. The earlier runs also show variation and occasional outliers, particularly for shorter outputs.

**Interpretation:** Longer outputs mainly add Decode steps and increase total E2E latency. Current data does not support a strong monotonic increase in TPOT with output length.

**Figures:** Raw samples and group medians from the illustrated run.

![Output length versus end-to-end latency: raw samples and group medians](pics/output-length-vs-e2e.png)

![Output length versus TPOT: raw samples and group medians](pics/output-length-vs-tpot.png)

**Caveats:** Client-observed measurements over localhost with Qwen/Qwen3-0.6B on a single RTX 4060 Laptop GPU.

## Concurrency

**Question:** Under a fixed homogeneous workload, how does client concurrency affect aggregate generation throughput, TTFT / E2E latency, engine Running / Waiting, and the KV-cache working set? Where do throughput gains begin to diminish?

**Controlled:**

- Fedora, Python 3.12, vLLM 0.31.0, one RTX 4060 Laptop GPU (~8 GB VRAM).
- Qwen/Qwen3-0.6B, `--gpu-memory-utilization 0.75`, `--max-model-len 4096`; other scheduler/batching settings kept at the baseline.
- Fixed instruction and body reused from the output-length experiment, with a unique UUID at the beginning of every request.
- `temperature=0`, `max_tokens=128`, streaming requests over localhost; Prefix Cache remained enabled.

**Variable:** Client concurrency = 1, 2, 4, 8, 16, 32, tested in ascending order.

**Method:** One closed-loop sweep: three excluded global warmup requests, then approximately 2*C warmup completions per level. Each completed request was immediately replaced to maintain approximately C in flight; warmup transitioned continuously into measurement. Collected exactly 128 latency samples per level whose request starts were at or after t0. After the cutoff, stopped replenishing, took t1, drained remaining requests, and waited for Running=Waiting=0 before advancing. See [the formal protocol](concurrency-experiment.md).

Generation throughput uses server Counter deltas divided by the t0/t1 observation interval. Server-window work includes boundary-crossing requests and is a different population from the 128 client samples. Gauges were sampled approximately every 0.2 s, with arithmetic sample mean and peak. Client TTFT ends at the first non-empty content chunk; E2E ends after consuming the full stream; TPOT divides the post-first-content duration by completion_tokens-1. Latency p50/p95 use nearest rank.

**Run:** `20261010-231439-7f9be14b`, 2026-10-10, local time Asia/Shanghai. Source files: `results/concurrency-20261010-231439-7f9be14b-{requests,levels,metrics}.csv` (generated files remain gitignored). The earlier C=1/2, four-sample smoke test was only an integration check and is excluded from these findings.

**Results:** All six levels collected 128 valid samples, 768 total. Actual prompt-token medians were 577 at every level, with an overall range of 570–581. Every measured response generated 128 tokens and finished with `length`. All levels recorded zero local-cache-hit tokens, external-transfer tokens, and preemptions; logical prompt-token deltas matched local-compute deltas.

| Concurrency | Generation tokens/s | TTFT p50 / p95 (ms) | E2E p50 / p95 (s) | TPOT p50 / p95 (ms/token) |
|---:|---:|---:|---:|---:|
| 1 | 158.7 | 33.2 / 36.4 | 0.797 / 0.848 | 6.02 / 6.39 |
| 2 | 288.5 | 41.8 / 58.7 | 0.883 / 0.887 | 6.65 / 6.68 |
| 4 | 500.8 | 107.5 / 109.3 | 1.022 / 1.024 | 7.20 / 7.73 |
| 8 | 784.9 | 132.9 / 178.5 | 1.304 / 1.306 | 9.21 / 9.90 |
| 16 | 1157.0 | 213.5 / 287.4 | 1.765 / 1.795 | 12.23 / 13.43 |
| 32 | 1415.2 | 277.4 / 379.1 | 2.884 / 2.971 | 20.49 / 21.56 |

| Concurrency | Running mean / peak | Waiting mean / peak | KV block-pool usage mean / peak (%) |
|---:|---:|---:|---:|
| 1 | 0.95 / 1 | 0 / 0 | 1.66 / 1.94 |
| 2 | 1.94 / 2 | 0 / 0 | 3.39 / 3.84 |
| 4 | 3.38 / 4 | 0 / 0 | 5.84 / 7.43 |
| 8 | 7.51 / 8 | 0 / 0 | 13.03 / 15.29 |
| 16 | 14.90 / 16 | 0.013 / 1 | 25.78 / 30.41 |
| 32 | 30.11 / 32 | 0 / 0 | 52.21 / 60.39 |

The sole sampled Waiting=1 observation occurred at C=16 with reason `capacity`; deferred waiting stayed zero. This reason alone does not establish KV exhaustion. Running is not GPU batch size, and KV usage describes the vLLM block pool, not total GPU VRAM.

**Interpretation:** Increasing concurrency raised aggregate generation throughput while slowing individual requests. For successive concurrency doublings, throughput gains decreased from 81.7% to 73.6%, 56.7%, 47.4%, and 22.3%. From C=16 to 32, throughput increased 22.3%, but E2E p50 increased 63.4% and TPOT p50 increased 67.6%; Decode slowdown is an important part of the additional latency cost.

The 16–32 range is a candidate region of diminishing returns under this workload, not a proven exact saturation point: throughput still increased at C=32. Running and KV usage increased, but sampled Waiting showed no sustained backlog and no preemption was recorded. The data does not establish KV-capacity exhaustion or identify a specific GPU bottleneck. E2E p95 generally remained close to p50, suggesting broad request slowdown rather than a pronounced extreme tail in this run.

**Plots:** Generation throughput, TTFT p50/p95, and E2E p50/p95 are the original primary plots. TPOT p50/p95, KV usage mean/peak, and Running mean/peak with a client-concurrency reference line were added post hoc as diagnostic / learning plots, using the same recorded levels CSV without new inference or independent evidence.

All six figures below use the formal run `20261010-231439-7f9be14b` identified above.

**Primary plots:**

![Concurrency versus aggregate generation throughput](pics/concurrency-vs-generation-throughput.png)

![Concurrency versus TTFT p50 and p95](pics/concurrency-vs-ttft.png)

![Concurrency versus end-to-end latency p50 and p95](pics/concurrency-vs-e2e.png)

**Post-hoc diagnostic / learning plots:**

![Concurrency versus TPOT p50 and p95](pics/concurrency-vs-tpot.png)

![Concurrency versus KV block-pool usage mean and peak](pics/concurrency-vs-kv-usage.png)

![Concurrency versus Running requests mean and peak, with client concurrency reference](pics/concurrency-vs-running.png)

**Caveats:** One ascending sweep on one machine, with no repeated runs or confidence intervals. GPU temperature, frequency, power, and background activity were not strictly controlled; no formal GPU profiling was collected. Measurement windows ranged from about 104 s at C=1 to 14.5 s at C=32. The 128 requests share runtime/scheduling conditions and are not 128 independent replications. Gauge sampling can miss brief waits/peaks, and snapshot timing/exporter refresh introduce boundary uncertainty. Fixed-concurrency closed-loop traffic does not establish behavior under a fixed external arrival rate. Findings apply to this model, workload, baseline configuration, and hardware.

## Mixed Prefill–Decode interference

**Question:** When a fixed cohort is already decoding, does injecting a new Prefill workload produce a temporary, client-observed output stall? How does the response change between a small prompt and a prompt exceeding one scheduler token budget?

**Controlled:**

- Fedora, Python 3.12, vLLM 0.31.0, one RTX 4060 Laptop GPU (~8 GB VRAM).
- Qwen/Qwen3-0.6B, `--gpu-memory-utilization 0.75`, `--max-model-len 4096`; the serving script and scheduler settings were unchanged.
- Baseline configuration: Chunked Prefill enabled, scheduler token budget 2048, max sequences 256, FCFS, AsyncScheduler, Prefix Cache enabled.
- Eight background requests submitted once per trial, without replacement. Each used the output-length/concurrency prompt material with a unique UUID prefix, approximately 577 actual prompt tokens, `temperature=0`, and `max_tokens=512`.
- Foreground requests used `temperature=0`, `max_tokens=1`, and a new early UUID prefix, limiting sustained extra Decode contention after Prefill.

**Variable:** Baseline = no foreground; Short = one foreground prompt targeting approximately 577 tokens; Long = one foreground prompt targeting approximately 3072 tokens, with actual usage required to exceed 2048.

**Method:** One excluded warmup cohort with C=8 and output limit 64. Each recorded trial waited until all eight background streams had accumulated at least 64 generated token IDs, then recorded the trigger and each request's actual progress. Baseline used that trigger as its alignment time. Short/Long aligned to the client timestamp immediately before the foreground API call. All requests completed before engine idle was checked and the next trial began.

The formal schedule contained 10 blocks, each with exactly one Baseline, Short, and Long trial in a seeded random order (`seed=20261011`), for 30 trials total. Blocks spread each condition throughout the run to reduce systematic thermal/time drift. See [the experiment protocol](prefill-interference-experiment.md).

Every parsed streaming event was recorded in memory with a client monotonic timestamp and incremental `token_ids`; final usage validated both prompt counts and total generated IDs. A token-bearing event can contain multiple tokens. The primary observable is therefore **client-observed token-bearing event gap**, not GPU-side per-token Decode latency. This experiment did not use average TPOT as its primary metric.

For each background request, gap = the difference between successive token-bearing event receipt times. The later event determines window membership: pre `[-0.5,0)` s and post `[0,+0.5]` s. Each request's stall ratio = post maximum gap / pre median gap; each trial's ratio = the median of its eight request ratios. Both windows required at least two gaps per request. The pre-gap and post-peak columns below are condition medians of trial-level summaries, each itself the median across eight request values; the reported stall ratio is not their quotient.

**Run:** `20261011-213351-40137cab`, 2026-10-11, local time Asia/Shanghai. Sources: `results/prefill-interference-20261011-213351-40137cab-{events.jsonl,trials.csv,metrics.csv,analysis.csv}`. The earlier three-trial smoke run `20261011-211120-78a1b24d` checked integration and is excluded from these findings. All figures below use the formal run, with timeline improvements generated offline from its unchanged raw events.

**Results:** All 30 trials completed successfully and passed analysis validity, with 10 trials per condition. Each background reached the trigger exactly once; recorded progress ranged from 64 to 66 tokens. All 240 backgrounds generated 512 tokens: 239 finished with `length`, one with `stop`. All 20 foregrounds generated one token and finished with `length`. Background actual prompt lengths ranged from 572 to 581 tokens. Short foregrounds ranged from 575 to 580; Long foregrounds ranged from 3068 to 3074, all above the 2048-token scheduler budget.

Raw-event replay validated the generated-ID/usage match and event cumulative counts for all 260 recorded requests; prompt-ID counts also matched usage. Backgrounds produced 122,880 token-bearing events, all singleton in this run: 1-token events 100%, 2-token and >=3-token events 0%, maximum delta=1. Including foregrounds gives 122,900 token-bearing events, within 123,420 total parsed SSE events. This observation does not guarantee singleton delivery in future runs; multi-token handling was tested synthetically.

| Condition | Trials | Pre median gap (ms) | Post peak gap summary (ms) | Trial stall ratio median (min–max) | Foreground TTFT median (min–max, ms) |
|---|---:|---:|---:|---:|---:|
| Baseline | 10 | 8.509 | 11.045 | 1.297 (1.218–1.638) | — |
| Short | 10 | 8.513 | 32.657 | 3.834 (3.635–4.002) | 50.517 (48.805–52.119) |
| Long | 10 | 8.511 | 107.704 | 12.576 (10.192–12.897) | 199.691 (198.423–234.192) |

All trials had zero local-cache-hit tokens, external KV-transfer tokens, and preemptions. Across the run, logical prompt tokens equalled local-compute tokens at 174,971; generation tokens totalled 122,900. Engine Gauges were sampled approximately every 0.2 s, with 849 samples total. Running peaks for Baseline/Short/Long were 8/8/9, Waiting/capacity/deferred stayed zero, and the largest sampled KV block-pool fraction was 23.55%. Every trial's final idle observation had Running=Waiting=KV usage=0. Short's brief extra request can escape 0.2 s sampling; its sampled Running peak of 8 does not imply the foreground was absent.

**Interpretation:** Within this run, both injected workloads coincided with larger transient background stream gaps than the no-injection control, with Long producing larger gaps than Short. Background intervals returned to their usual short cadence after the injection region; the effect is localized rather than summarized only by whole-request average TPOT.

For the post-hoc stall-only diagnostic, each request used its own pre median and a strict threshold `gap > 2 × pre_median_gap`. Actual thresholds ranged from 16.82 to 21.39 ms. No Baseline request exceeded its threshold within the plotted `[-0.5,+1.0]` s window. Every Short background had one post-alignment interval exceeding its threshold, and every Long background had two. Short trial B8/T22 additionally had a pre-alignment anomaly shared across all eight requests; it is retained rather than attributed to the later injection. The multiple Long intervals are consistent with a Prefill workload spanning token-budget iterations, but client events do not identify GPU scheduler iteration boundaries.

**Plots:** The full-interval timeline preserves every observed gap as `[previous event receipt, next event receipt]`, with separate Baseline/Short/Long panels, one row per trial, and eight background tracks per row. B denotes randomized block (1–10), T denotes overall execution order (1–30). Length and color encode the complete gap duration, and ticks mark actual event receipts. There is no temporal binning, smoothing, or cross-trial aggregation; adjacent normal gaps can visually join into a continuous track.

The stall-only timeline was added post hoc for readability, using the per-request 2× threshold above and the same true interval endpoints. Its threshold is a visualization rule, not a statistical significance test or a universal definition of stall. The stall-ratio and foreground-TTFT plots retain one point per trial plus condition medians. All four figures are derived from the same formal run; no additional inference was needed to produce the revised timelines.

**Primary views:**

![Mixed Prefill–Decode interference: client-observed full event-gap intervals for all trials and backgrounds](pics/prefill-interference-full-interval-timeline.png)

![Mixed Prefill–Decode interference: trial stall ratios with condition medians](pics/prefill-interference-stall-ratio.png)

![Mixed Prefill–Decode interference: Short and Long foreground TTFT per trial](pics/prefill-interference-foreground-ttft.png)

**Post-hoc diagnostic / learning view:**

![Mixed Prefill–Decode interference: only event-gap intervals greater than twice each request's pre median](pics/prefill-interference-stall-only-timeline.png)

**Caveats:** One 30-trial run on one machine. Blocks reduce condition/order confounding but do not strictly control temperature, GPU clocks, power, or background activity. Eight streams within a trial share serving conditions and are not eight independent experiment replications. Client receipt times also include HTTP/SDK/event-loop/OS effects; even singleton events do not expose GPU generation timestamps. Foreground request-start alignment is not the GPU Prefill start. Gauge sampling cannot resolve these short pauses, and a gap crossing a window boundary is assigned by its later event. Chunked Prefill was enabled throughout: there is no on/off control, so these results cannot establish a causal TPOT improvement from Chunked Prefill, exact GPU stall durations, or the scheduler cause of each observed gap.
