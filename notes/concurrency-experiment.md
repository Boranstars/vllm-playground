# Concurrency experiment 1: homogeneous closed-loop sweep

**Status:** Protocol and implementation, with post-hoc diagnostic plots.
Conceptual notes belong in `concurrency-schedule.md`; completed observations and conclusions belong in
`week1-experiments.md` only after a real run.

**Question:** Under a fixed homogeneous workload, how does increasing client
concurrency affect aggregate generation throughput, TTFT / E2E latency, engine
Running / Waiting, and the KV-cache working set? In what region does the serving
system approach saturation / queueing?

**Hypotheses:**
1. More in-flight requests at low concurrency provide more batching opportunities
   and increase aggregate generation throughput.
2. Throughput gains diminish as the GPU approaches saturation.
3. Beyond that region, latency (especially the tail) may worsen with little
   additional throughput.
4. Running requests and KV-cache usage generally increase with concurrency.
5. Waiting or preemption may appear under pressure; neither is assumed in advance.
6. We seek a saturation region / knee, not an exact universal concurrency value.

**Controlled:** Fedora, Python 3.12, vLLM 0.31.0, RTX 4060 Laptop GPU (~8 GB).
Server: `Qwen/Qwen3-0.6B --gpu-memory-utilization 0.75 --max-model-len 4096`.
Keep scheduler, batching, token budget, chunked prefill, prefix cache, and maximum
sequence settings at the current baseline. Use an otherwise idle, dedicated
server: Counter deltas include all traffic for model `Qwen/Qwen3-0.6B`, engine `0`.
The runner validates metrics, not the running server's launch arguments/version;
the operator must verify those from the server startup log.

**Workload:** Copy the fixed instruction and six repetitions of the paragraph
from `output_length.py` verbatim (importing that script would execute it).
Every prompt begins `request-<32 hex UUID>\n`, before the shared instruction/body.
Only this ID changes. Prior output-length runs placed this workload around
575–580 actual prompt tokens; the target here is approximately 550–600, verified
from returned usage, never inferred from characters. Use `temperature=0` and
`max_tokens=128`. Record actual output tokens and finish reason; do not assume
every response reaches 128. Unique early prefixes minimize Prefix Cache reuse
without disabling the server cache. Inspect `local_cache_hit` deltas for residual
contamination; a unique ID is not a guarantee of zero hits.

**Variable:** Ascending concurrency levels `1, 2, 4, 8, 16, 32`. Do not include
64 in this sweep; review whether KV pressure mixes with compute saturation first.
Ascending order is intentional; thermal/runtime drift remains a limitation.

## Method and boundaries

Use one asynchronous client and persistent connections. Validate all required
metrics before sending requests and wait for Running=Waiting=0. Send three
sequential global warmup requests using the same workload, exclude them from
all results, then wait for idle again.

For each concurrency C:
1. Fill C in-flight slots.
2. On each completion, immediately replace that request (closed loop).
3. After approximately 2*C warmup completions, request the t0 metrics snapshot
   while continuing to process completions and replenish slots. Never drain or
   pause replenishment for this transition.
4. t0 is the local monotonic timestamp immediately after a successful metrics
   HTTP response has been fully received. Validate its samples before using it.
   Continue at C; only requests whose actual `request_start >= t0` enter the
   measured latency sample set. Earlier requests can finish across t0 and their
   work still contributes to server Counters.
5. Collect exactly 128 eligible completions. When several tasks are ready,
   process them by stream-end time (request sequence breaks ties).
6. Upon the 128th eligible completion, stop replenishing immediately and fetch
   the t1 snapshot promptly. t1 uses the same successful-response timestamp
   convention as t0. Other in-flight requests continue during the fetch.
7. Stop Gauge sampling after t1; retain only samples timestamped in [t0, t1].
8. Drain remaining requests naturally, without cancellation, then poll until
   Running=Waiting=0 before advancing to the next level.

There is no barrier between waves: each freed slot is replaced independently.
The server window and the 128 client requests are different populations. Work
crossing either boundary legitimately contributes to Counter deltas. Do not
force server token/request deltas to match the latency sample set. HTTP snapshot
latency, server metric refresh cadence, and event-loop dispatch introduce boundary
uncertainty; the timestamp is an observation time, not an atomic engine timestamp.

**Client timings:** `request_start` immediately precedes the API call; TTFT ends
at the first **non-empty `delta.content`**, ignoring role-only/empty chunks.
`stream_end` is after consuming the full stream, including final usage.
Token counts come from final vLLM/OpenAI usage.

```text
TTFT = first_nonempty_content_time - request_start
E2E = stream_end - request_start
TPOT = (stream_end - first_nonempty_content_time) / (completion_tokens - 1)
decode tokens/s = (completion_tokens - 1) / (stream_end - first_nonempty_content_time)
```

If content is absent, TTFT/TPOT/decode rate are undefined (blank CSV cells).
TPOT/decode rate are also undefined for <=1 completion token. Report valid sample
counts for each latency distribution; exclude undefined values from percentiles.
Use nearest-rank p50 and p95: sort n valid values, take the 1-based element
`ceil(p*n)` for p=0.50/0.95. No interpolation or p99. Token medians use the usual
middle value / average of two middle values, plus min/max.

## Server metrics

Parse Prometheus labels by name, not order; resolve exactly one sample per
required label selector for engine `0` and this model. Missing, ambiguous,
nonfinite, or negative samples fail clearly. No Prometheus deployment/dependency.

**Counters:** Take t1 minus t0 for `generation_tokens_total`,
`prompt_tokens_total`, `prompt_tokens_by_source_total` with sources
`local_compute`, `local_cache_hit`, `external_kv_transfer`,
`num_preemptions_total`, and `request_success_total` by `finished_reason`.
Require the verified reasons `stop`, `length`, `abort`, `error`, `repetition`;
preserve any additional reason series too. Reject Counter decreases or changing
reason sets rather than reporting a valid level after a server reset/schema change.

```text
measurement duration = t1 - t0
generation throughput = delta generation_tokens_total / duration
logical prompt throughput = delta prompt_tokens_total / duration
local Prefill compute throughput = delta local_compute / duration
```

`prompt_tokens_total` includes cached tokens. `local_compute` counts locally
computed prompt tokens; `local_cache_hit` diagnoses Prefix Cache contamination.
Record raw external-transfer and preemption deltas too. Preserve all
`request_success_total` reason deltas separately: its name does not make aborts
or errors successes. Preemption count alone does not establish its cause.

**Gauges:** Sample approximately every 0.2 s during [t0, t1], also retaining the
Gauges in the two boundary snapshots. Record relative monotonic timestamp,
Running, Waiting, capacity-waiting, deferred-waiting, and `kv_cache_usage_perc`.
Mean is the arithmetic mean of retained observations (not time weighted); peak
is their maximum. Sampling can miss brief peaks and reflects exporter cadence.
Never subtract endpoint Gauges. Running is **not GPU batch size**;
`reason="capacity"` is **not specifically KV exhaustion**; KV usage is the
fraction of the vLLM KV block pool occupied, **not total GPU VRAM**. CSV retains
the raw 0–1 KV fraction; console output converts it to percent.

GPU utilization / `nvidia-smi` may be observed manually but is not a formal metric
and has no collection infrastructure in this experiment.

## Output, failure handling, and manual execution

One timestamp plus short UUID identifies the whole sweep. Standard-library CSVs
under ignored `results/` contain:
- `concurrency-<run-id>-requests.csv`: exactly the eligible client samples,
  including sequence/ID, actual token counts, finish reason, timings, and start/end
  offsets from t0. Sequence numbers count all submissions within a level, so gaps
  due to warmup/boundary requests are expected.
- `concurrency-<run-id>-levels.csv`: Counter deltas/rates, client distributions
  and valid counts, finish-reason counts, Gauge mean/peak and sample count.
Finish-reason maps are JSON objects inside standard CSV cells.
- `concurrency-<run-id>-metrics.csv`: retained raw Gauge time series.

Flush each completed level to all three files. If a level fails, report the
request ID/sequence and error (or metrics diagnostic) on stderr and exit nonzero;
do not replace failed data silently. Previously completed levels remain available;
the failed level is not written as a valid result. Requests have a 180 s overall
deadline, metrics HTTP calls a 5 s timeout, and idle waits a 60 s deadline.
Disable client automatic retries. On failure stop replenishing, drain submitted
requests, and report drain errors as well. Normal draining never cancels requests;
timeouts/user interruption are exceptional exits. No subsequent level runs after
an error. Small sample CLI overrides are smoke tests, not formal results.

```bash
./scripts/serve_qwen3_0_6b.sh
# In another terminal, from the repo root:
uv run --no-sync python experiments/concurrency.py --levels 1 2 --samples-per-level 4
uv run --no-sync python experiments/concurrency.py
uv run --no-sync python experiments/plot_concurrency.py results/concurrency-<run-id>-levels.csv
```

The launch script uses the existing uv environment without synchronization,
preserving the manually installed vLLM/OpenAI packages. Tiny runs still include
three global warmups, 2*C per-level warmup completions, and boundary/drain work.
The plotter reads only the existing levels CSV and saves six separate PNGs beside
it, using the tested concurrency levels as discrete sweep points.

**Primary plots (specified before the experiment):**

1. Generation throughput versus concurrency: `-throughput.png`.
2. TTFT p50/p95 versus concurrency, in seconds: `-ttft.png`.
3. E2E p50/p95 versus concurrency, in seconds: `-e2e.png`.

**Post-hoc diagnostic / learning plots (added after reviewing the existing run):**

4. TPOT p50/p95 versus concurrency, converted from seconds/token to ms/token:
   `-tpot.png`.
5. KV-cache usage mean/peak versus concurrency, converted from the raw fraction
   to percent of the vLLM KV block pool: `-kv-usage.png`. This is not total VRAM use.
6. Running requests mean/peak versus concurrency, with a dashed `y = concurrency`
   reference line: `-running.png`. The reference compares observed engine Running
   against client concurrency; Running is not GPU batch size.

Diagnostic plots reuse the already recorded latency percentiles and Gauge
mean/peak statistics. They add no measurements, change no CSV values or protocol
boundaries, and require no new inference sweep. Waiting and preemption remain in
the CSV/summary, without separate plots. Review raw token distributions, cache
hits, finish reasons, and state summaries before interpreting the knee. Record
actual conclusions only after running the experiment.

## Exact CSV schemas

Headers below are in file order; line breaks in the levels header are only for
readability. All durations use seconds. Request IDs are hex strings. Token counts
are actual usage counts; Counter deltas and Gauge samples are parsed numeric
values. JSON count-map keys are finish reasons; no reasons are combined into a
generic successful-request total. The `warmup_completions` field counts processed
completions before the t0 snapshot becomes available, including those processed
while the scrape was in progress.

`concurrency-<run-id>-requests.csv`:
```text
concurrency,request_sequence,request_id,actual_prompt_tokens,completion_tokens,finish_reason,ttft_s,e2e_s,tpot_s_per_token,decode_tokens_per_s,request_start_relative_s,request_end_relative_s
```

`concurrency-<run-id>-levels.csv`:
```text
concurrency,measurement_duration_s,measured_request_count,
generation_tokens_delta,prompt_tokens_delta,local_compute_prompt_tokens_delta,
local_cache_hit_tokens_delta,external_kv_transfer_tokens_delta,preemptions_delta,
generation_tokens_per_s,prompt_tokens_per_s,local_compute_prompt_tokens_per_s,
ttft_valid_count,ttft_p50_s,ttft_p95_s,e2e_valid_count,e2e_p50_s,e2e_p95_s,
tpot_valid_count,tpot_p50_s_per_token,tpot_p95_s_per_token,
actual_prompt_tokens_median,actual_prompt_tokens_min,actual_prompt_tokens_max,
completion_tokens_median,completion_tokens_min,completion_tokens_max,
client_finish_reason_counts,server_finish_reason_deltas,gauge_sample_count,
running_mean,running_peak,waiting_mean,waiting_peak,
waiting_capacity_mean,waiting_capacity_peak,waiting_deferred_mean,waiting_deferred_peak,
kv_cache_usage_perc_mean,kv_cache_usage_perc_peak,
model,max_tokens,temperature,gauge_interval_s,warmup_completions
```

`concurrency-<run-id>-metrics.csv`:
```text
concurrency,sample_index,relative_time_s,running,waiting,waiting_capacity,waiting_deferred,kv_cache_usage_perc
```
