1. 实验如何设计？

```text
Question
-------
Does injecting Prefill work into a stable Decode workload
produce a transient, client-observed output stall in existing
Decode streams, and how does the observed gap structure change
when the uncached Prefill exceeds one scheduler token budget?

Server
------
Qwen/Qwen3-0.6B, RTX 4060 Laptop (~8 GB)
vLLM 0.31.0
gpu_memory_utilization = 0.75
max_model_len = 4096
max_num_batched_tokens = 2048
effective scheduler token budget = 2048
max_num_seqs = 256
chunked prefill = enabled (current default)
AsyncScheduler, FCFS, Prefix Cache enabled
serving script and scheduler settings unchanged

Background
----------
8 concurrent fixed requests, submitted once; no replenishment
prompt ≈ 577 actual tokens
max_tokens = 512
temperature = 0
unique UUID prefix at the beginning of the user prompt

Injection trigger
-----------------
Trigger exactly once after all 8 background requests have
cumulative generated token-ID counts >=64; no fixed sleep.
Record trigger_ns and each request's actual count at trigger.
A multi-token event may cross the threshold, e.g. 62 + 3 = 65.

Conditions
----------
Baseline: no foreground request
Short:    ≈577-token prompt, max_tokens=1
Long:     ≈3072-token prompt, max_tokens=1
All requests: temperature=0, fresh early UUID prefix.
Server-returned prompt IDs / final usage confirm actual lengths.
Background/Short: 550–600; Long: 3000–3200 and strictly >2048.
Invalid lengths fail the trial; do not push Long toward 4096.
Foreground output=1 limits sustained extra Decode contention.

Alignment
---------
Baseline: trigger_ns (pseudo-injection; no foreground call).
Short/Long: foreground_request_start_ns, recorded immediately
before the foreground API call after trigger; retain both times.
Request start is a client timestamp, not GPU Prefill start.

Primary measurement
-------------------
Client-observed token-bearing SSE event receipt timestamps.
Per background, keep events with delta_token_count >0:
event_gap_i = received_ns(i) - received_ns(previous such event).
An SSE event can contain multiple generated token IDs.
Never split/interpolate it into fabricated per-token timestamps.
Singleton-to-singleton intervals are a diagnostic only;
neither event gaps nor singleton intervals are GPU-side ITL.

Streaming validation
--------------------
stream=True, include_usage=True, return_token_ids=True.
Keep role/empty/token-bearing/finish/usage events in memory;
write records after the trial, without per-event disk I/O.
Concatenated generated IDs must match final usage completion_tokens.
Missing usage or mismatched counts fail the request/trial.
Report 1-token, 2-token, >=3-token event proportions.

Analysis and visualization
--------------------------
Visualization: [-0.5 s, +1.0 s] relative to alignment.
Stall summary: pre [-0.5 s, 0), post [0, +0.5 s].
The later event's receipt time determines window membership.
Each request: pre_median_gap = median(pre gaps);
              post_peak_gap = max(post gaps);
              request_stall_ratio = post_peak_gap / pre_median_gap.
Each trial: median of the 8 request_stall_ratios.
Trial pre/post summaries are medians of the 8 respective values,
not a global max; their quotient need not equal the trial ratio.
Require >=2 gaps per window per request and positive pre median;
otherwise explicitly mark the trial analysis-invalid.

Full timeline: each gap is its real [t_previous, t_current]
receipt interval; separate condition panels and 8 tracks per trial.
Stall-only timeline: retain intervals strictly exceeding
2 x that request's pre_median_gap; normal intervals are omitted.
No binning, smoothing, cross-trial aggregation or merging gaps.
B = block index; T = overall chronological trial index.
The 2x threshold is a post-hoc visualization rule, not a
significance test, and does not change the primary stall ratio.
Stall-ratio and foreground-TTFT plots keep one point per trial.

Trials
------
Smoke: 1 block, 1 x each condition; integration checks only.
Formal: 10 blocks, each containing exactly Baseline/Short/Long
once in a seeded random order within that block (seed=20261011).
Total: 30 trials, 10 per condition.

Controls
--------
Unique early prefixes to suppress Prefix Cache reuse;
verify per-trial local_cache_hit delta and warn if nonzero.
Unique prefixes alone do not guarantee zero cache reuse.
Same background workload
Same injection trigger
Same foreground output for Short/Long
Same server configuration
Excluded warmup before recorded trials: C=8, short prompt, output=64.
Wait for all streams to complete, then engine Running=Waiting=0
and KV usage near zero before the next trial; enforce timeouts.
Prometheus sampling ~0.2 s is auxiliary, not the stall measurement.

Interpretation limits
---------------------
Baseline controls for natural stream-delivery jitter at the same
Decode progress; Short/Long compare Prefill workload scales.
Client receipt times include HTTP/SDK/event-loop/OS effects.
Observed intervals do not expose GPU scheduler iteration boundaries.
Chunked Prefill stays enabled throughout; this experiment cannot
prove a causal Chunked Prefill improvement in TPOT.
```

当前已完成 smoke 和正式 run `20261011-213351-40137cab`；正式 run 的 30 个
trial 均成功且 analysis-valid，所有请求的 generated IDs 均与 final usage 匹配，
local-cache-hit delta 均为 0。新版区间图和 stall-only 图由原始结果离线生成。
完整配置和测量规则见 [实验 protocol](prefill-interference-experiment.md)，
正式结果、解释与四张图见 [Week 1 实验记录](week1-experiments.md#mixed-prefilldecode-interference)。
