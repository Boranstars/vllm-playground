"""Experiment 1: homogeneous, fixed-concurrency closed-loop serving."""

import argparse
import asyncio
import csv
import json
import math
import statistics
import sys
import time
import uuid
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from common import (
    COUNTER_SELECTORS,
    GAUGE_SELECTORS,
    get_metrics_snapshot,
    measure_streaming_request_async,
)


MODEL = "Qwen/Qwen3-0.6B"
BASE_URL = "http://127.0.0.1:8000/v1"
METRICS_URL = "http://127.0.0.1:8000/metrics"
LEVELS = [1, 2, 4, 8, 16, 32]
MAX_TOKENS = 128
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Verbatim output_length.py workload; importing that script would run its experiment.
FIXED_INSTRUCTION = "请基于以下材料，持续生成连贯、详细的说明和例子，尽可能达到允许的最大输出长度。"
FIXED_PARAGRAPH = (
    "A language model processes a sequence one token at a time. For each token, "
    "it projects hidden states into keys and values for every attention layer. "
    "These tensors represent information from the preceding context. Keeping them "
    "in memory lets later decoding steps attend to earlier tokens without "
    "recomputing their projections. The cache grows as new tokens are generated, "
    "so memory use depends on sequence length, layer count, head dimensions, and "
    "the chosen data type."
)
FIXED_BODY = "\n".join([FIXED_PARAGRAPH] * 6)

REQUEST_COLUMNS = [
    "concurrency", "request_sequence", "request_id", "actual_prompt_tokens",
    "completion_tokens", "finish_reason", "ttft_s", "e2e_s",
    "tpot_s_per_token", "decode_tokens_per_s", "request_start_relative_s",
    "request_end_relative_s",
]
METRIC_COLUMNS = ["concurrency", "sample_index", "relative_time_s", *GAUGE_SELECTORS]
LATENCIES = {
    "ttft": ("ttft_s", "s"),
    "e2e": ("e2e_s", "s"),
    "tpot": ("tpot_s_per_token", "s_per_token"),
}
LEVEL_COLUMNS = [
    "concurrency", "measurement_duration_s", "measured_request_count",
    *[f"{key}_delta" for key in COUNTER_SELECTORS],
    "generation_tokens_per_s", "prompt_tokens_per_s", "local_compute_prompt_tokens_per_s",
    *[column for name, (_, unit) in LATENCIES.items()
      for column in (f"{name}_valid_count", f"{name}_p50_{unit}", f"{name}_p95_{unit}")],
    *[f"{key}_{stat}" for key in ("actual_prompt_tokens", "completion_tokens")
      for stat in ("median", "min", "max")],
    "client_finish_reason_counts", "server_finish_reason_deltas", "gauge_sample_count",
    *[f"{key}_{stat}" for key in GAUGE_SELECTORS for stat in ("mean", "peak")],
    "model", "max_tokens", "temperature", "gauge_interval_s", "warmup_completions",
]


async def run_request(client, sequence):
    request_id = uuid.uuid4().hex
    prompt = f"request-{request_id}\n{FIXED_INSTRUCTION}\n{FIXED_BODY}"
    try:
        async with asyncio.timeout(180):
            result = await measure_streaming_request_async(client, MODEL, prompt, MAX_TOKENS)
        if result["prompt_tokens"] is None or result["completion_tokens"] is None:
            raise ValueError("Final usage is missing")
        if result["finish_reason"] not in {"stop", "length", "repetition"}:
            raise ValueError(f"Unexpected finish_reason={result['finish_reason']!r}")
    except Exception as exc:
        raise RuntimeError(f"Request sequence={sequence}, id={request_id} failed: {exc}") from exc
    return {"request_sequence": sequence, "request_id": request_id, **result}


async def wait_for_idle(http_client):
    deadline = time.perf_counter() + 60
    while True:
        snapshot = await get_metrics_snapshot(http_client, METRICS_URL, MODEL)
        if snapshot["gauges"]["running"] == 0 and snapshot["gauges"]["waiting"] == 0:
            return
        if time.perf_counter() >= deadline:
            raise RuntimeError("Engine did not reach Running=Waiting=0 within 60 s")
        await asyncio.sleep(0.2)


async def sample_gauges(http_client, interval, stop, snapshots):
    # Boundary snapshots supply the first/last samples. Do not catch scrape errors.
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            pass
        snapshots.append(await get_metrics_snapshot(http_client, METRICS_URL, MODEL))


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1] if ordered else None


def summarize(concurrency, requests, metrics, t0, t1, interval, warmup_completions):
    duration = t1["timestamp"] - t0["timestamp"]
    if duration <= 0 or t0["counters"].keys() != t1["counters"].keys():
        raise ValueError("Invalid snapshot duration or changed Counter series")
    deltas = {key: value - t0["counters"][key] for key, value in t1["counters"].items()}
    if any(value < 0 for value in deltas.values()):
        raise ValueError("Counter decreased: possible server reset; level is invalid")
    row = {
        "concurrency": concurrency,
        "measurement_duration_s": duration,
        "measured_request_count": len(requests),
        **{f"{key}_delta": deltas[key] for key in COUNTER_SELECTORS},
        **{f"{key}_per_s": deltas[key] / duration for key in (
            "generation_tokens", "prompt_tokens", "local_compute_prompt_tokens"
        )},
        "client_finish_reason_counts": json.dumps(
            dict(Counter(r["finish_reason"] for r in requests)), sort_keys=True
        ),
        "server_finish_reason_deltas": json.dumps({
            key.split(":", 1)[1]: value for key, value in deltas.items()
            if key.startswith("request_success:")
        }, sort_keys=True),
        "gauge_sample_count": len(metrics),
        "model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0,
        "gauge_interval_s": interval, "warmup_completions": warmup_completions,
    }
    for name, (key, unit) in LATENCIES.items():
        values = [r[key] for r in requests if r[key] is not None]
        row[f"{name}_valid_count"] = len(values)
        for percentile in (50, 95):
            row[f"{name}_p{percentile}_{unit}"] = nearest_rank(values, percentile / 100)
    for key in ("actual_prompt_tokens", "completion_tokens"):
        values = [r[key] for r in requests]
        row.update({f"{key}_median": statistics.median(values),
                    f"{key}_min": min(values), f"{key}_max": max(values)})
    for key in GAUGE_SELECTORS:
        values = [sample[key] for sample in metrics]
        row[f"{key}_mean"] = statistics.mean(values)
        row[f"{key}_peak"] = max(values)
    return row


async def run_level(client, http_client, concurrency, sample_count, interval):
    pending = set()
    sequence = 0
    warmup_completions = 0
    requests = []
    snapshots = []
    t0 = t1 = None
    t0_task = sampler = None
    stop_sampling = asyncio.Event()
    failure = None

    def replenish():
        nonlocal sequence
        sequence += 1
        pending.add(asyncio.create_task(run_request(client, sequence)))

    for _ in range(concurrency):
        replenish()
    try:
        while len(requests) < sample_count:
            watched = pending.copy()
            if t0_task is not None:
                watched.add(t0_task)
            if sampler is not None:
                watched.add(sampler)
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            if sampler in done:
                sampler.result()  # Propagate metrics failures immediately.
                raise RuntimeError("Gauge sampler stopped before t1")
            if t0_task in done:
                t0 = t0_task.result()
                t0_task = None
                snapshots.append(t0)
                sampler = asyncio.create_task(
                    sample_gauges(http_client, interval, stop_sampling, snapshots)
                )

            completed = done & pending
            # Inspect every ready result for errors, even if this batch hits the cutoff.
            ordered = sorted(completed, key=lambda task: (
                task.result()["stream_end"], task.result()["request_sequence"]
            ))
            for task in ordered:
                result = task.result()
                pending.remove(task)
                if t0 is None:
                    warmup_completions += 1
                elif result["request_start"] >= t0["timestamp"]:
                    requests.append({
                        "concurrency": concurrency,
                        "request_sequence": result["request_sequence"],
                        "request_id": result["request_id"],
                        "actual_prompt_tokens": result["prompt_tokens"],
                        **{key: result[key] for key in (
                            "completion_tokens", "finish_reason", "ttft_s", "e2e_s",
                            "tpot_s_per_token", "decode_tokens_per_s"
                        )},
                        "request_start_relative_s": result["request_start"] - t0["timestamp"],
                        "request_end_relative_s": result["stream_end"] - t0["timestamp"],
                    })
                if len(requests) == sample_count:
                    break  # No replacement for the cutoff completion or later completions.
                replenish()
                if t0 is None and t0_task is None and warmup_completions >= 2 * concurrency:
                    # The snapshot is watched alongside request tasks: no refill pause.
                    t0_task = asyncio.create_task(
                        get_metrics_snapshot(http_client, METRICS_URL, MODEL)
                    )

        t1 = await get_metrics_snapshot(http_client, METRICS_URL, MODEL)
        snapshots.append(t1)
    except Exception as exc:
        failure = exc
    finally:
        stop_sampling.set()
        # Only stop replenishing; existing inference requests finish naturally.
        auxiliary = [task for task in (t0_task, sampler) if task is not None]
        outcomes = await asyncio.gather(*pending, *auxiliary, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                print(f"C={concurrency} drain/metrics error: {outcome}", file=sys.stderr)
                if failure is None:
                    failure = outcome
    if failure is not None:
        raise RuntimeError(f"C={concurrency} failed; level not saved: {failure}") from failure

    metrics = [
        {"concurrency": concurrency, "sample_index": i,
         "relative_time_s": snapshot["timestamp"] - t0["timestamp"], **snapshot["gauges"]}
        for i, snapshot in enumerate(sorted(
            (s for s in snapshots if t0["timestamp"] <= s["timestamp"] <= t1["timestamp"]),
            key=lambda s: s["timestamp"],
        ), start=1)
    ]
    level = summarize(concurrency, requests, metrics, t0, t1, interval, warmup_completions)
    await wait_for_idle(http_client)
    return requests, level, metrics


def print_summary(rows):
    print(" C  gen tok/s  TTFT p50/p95(s)   E2E p50/p95(s)   TPOT p50(s/tok)  "
          "Running mean/peak  Waiting mean/peak  KV mean/peak(%)  preempt")
    def show(value):
        return "—" if value is None else f"{value:.4f}"

    for row in rows:
        print(
            f"{row['concurrency']:2d} {row['generation_tokens_per_s']:10.2f}  "
            f"{show(row['ttft_p50_s'])}/{show(row['ttft_p95_s']):<8}  "
            f"{show(row['e2e_p50_s'])}/{show(row['e2e_p95_s']):<8}  "
            f"{show(row['tpot_p50_s_per_token']):>15}  "
            f"{row['running_mean']:7.2f}/{row['running_peak']:<8.0f}  "
            f"{row['waiting_mean']:7.2f}/{row['waiting_peak']:<8.0f}  "
            f"{100 * row['kv_cache_usage_perc_mean']:6.2f}/"
            f"{100 * row['kv_cache_usage_perc_peak']:<7.2f}  {row['preemptions_delta']:.0f}"
        )


async def main(args):
    # Separate HTTP pools keep monitoring independent of inference connections.
    async with (
        AsyncOpenAI(base_url=BASE_URL, api_key="dummy", max_retries=0, timeout=120) as client,
        httpx.AsyncClient(trust_env=False) as http_client,
    ):
        await get_metrics_snapshot(http_client, METRICS_URL, MODEL)
        await wait_for_idle(http_client)
        print("Global warmup: 3 requests (excluded)", flush=True)
        for sequence in range(1, 4):
            await run_request(client, sequence)
        await wait_for_idle(http_client)

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        with ExitStack() as stack:
            writers = []
            files = []
            for suffix, columns in (
                ("requests", REQUEST_COLUMNS), ("levels", LEVEL_COLUMNS), ("metrics", METRIC_COLUMNS)
            ):
                path = RESULTS_DIR / f"concurrency-{run_id}-{suffix}.csv"
                output = stack.enter_context(path.open("x", newline="", encoding="utf-8"))
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                output.flush()
                files.append(output)
                writers.append(writer)
                print(f"CSV: {path}", flush=True)
            for concurrency in args.levels:
                print(f"C={concurrency}: warmup -> measurement -> drain", flush=True)
                requests, level, metrics = await run_level(
                    client, http_client, concurrency, args.samples_per_level, args.gauge_interval
                )
                writers[0].writerows(requests)
                writers[1].writerow(level)
                writers[2].writerows(metrics)
                for output in files:
                    output.flush()
                rows.append(level)
                print_summary([level])
                if level["actual_prompt_tokens_min"] < 550 or level["actual_prompt_tokens_max"] > 600:
                    print("Note: actual prompt tokens extend outside the approximate 550–600 target.")
        print("\nSweep complete (server-window throughput and client samples are distinct populations).")
        print_summary(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, choices=LEVELS, default=LEVELS)
    parser.add_argument("--samples-per-level", type=int, default=128)
    parser.add_argument("--gauge-interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.samples_per_level < 1:
        parser.error("--samples-per-level must be positive")
    if not math.isfinite(args.gauge_interval) or args.gauge_interval <= 0:
        parser.error("--gauge-interval must be finite and positive")
    if args.levels != sorted(set(args.levels)):
        parser.error("--levels must be unique and ascending")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        asyncio.run(main(arguments))
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Experiment stopped: {exc or 'interrupted'}. Completed levels only are saved.", file=sys.stderr)
        raise SystemExit(1)
