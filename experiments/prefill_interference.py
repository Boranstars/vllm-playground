"""Fixed-cohort mixed Prefill–Decode experiment; client-observed stream events."""

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import sys
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from openai import AsyncOpenAI

from common import get_metrics_snapshot
from concurrency import BASE_URL, METRICS_URL, MODEL, RESULTS_DIR, FIXED_INSTRUCTION, FIXED_PARAGRAPH

CONDITIONS = ("baseline", "short", "long")
CONCURRENCY = 8
TRIGGER_TOKENS = 64
BACKGROUND_MAX_TOKENS = 512
GAUGE_INTERVAL = 0.2
COUNTERS = {
    "prompt_tokens_total": "prompt_tokens",
    "prompt_tokens_local_compute": "local_compute_prompt_tokens",
    "prompt_tokens_local_cache_hit": "local_cache_hit_tokens",
    "external_kv_transfer": "external_kv_transfer_tokens",
    "generation_tokens_total": "generation_tokens",
    "preemptions_total": "preemptions",
}
IDENTITY_COLUMNS = ["run_id", "block_index", "trial_index", "condition"]
METRIC_COLUMNS = [
    *IDENTITY_COLUMNS, "sample_ns", "relative_to_alignment_ms", "num_requests_running",
    "num_requests_waiting", "waiting_capacity", "waiting_deferred", "kv_cache_usage_perc",
]
TRIAL_COLUMNS = [
    *IDENTITY_COLUMNS, "seed", "smoke", "model", "temperature", "gauge_interval_s",
    "trial_start_ns", "trigger_ns", "alignment_ns", "foreground_request_start_ns",
    "trial_end_ns", "duration_s", "trigger_count", "background_concurrency",
    "background_target_prompt_tokens", "background_actual_prompt_tokens_min",
    "background_actual_prompt_tokens_median", "background_actual_prompt_tokens_max",
    "background_max_tokens", "trigger_tokens", *[f"trigger_count_bg{i}" for i in range(8)],
    "foreground_target_prompt_tokens", "foreground_actual_prompt_tokens",
    "foreground_completion_tokens", "foreground_ttft_s", "foreground_finish_reason",
    "background_token_bearing_events", "background_singleton_event_ratio",
    "background_two_token_event_ratio", "background_ge3_token_event_ratio",
    "background_multi_token_event_ratio", "background_max_delta_token_count",
    "token_ids_usage_all_match", "request_summaries", *COUNTERS,
    "prefix_cache_warning", "idle_running", "idle_waiting", "idle_kv_cache_usage_perc",
    "status", "error",
]


def make_schedule(blocks, seed, conditions=CONDITIONS):
    if blocks < 1 or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Expected positive blocks and distinct conditions")
    if any(condition not in CONDITIONS for condition in conditions):
        raise ValueError("Unknown condition")
    rng = random.Random(seed)
    schedule = []
    for block in range(1, blocks + 1):
        order = list(conditions)
        rng.shuffle(order)
        schedule.extend((block, condition) for condition in order)
    return schedule


@dataclass
class Request:
    role: str
    background_id: int | None
    target_prompt_tokens: int
    max_tokens: int
    client_request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    cumulative: int = 0
    start_ns: int | None = None
    end_ns: int | None = None
    prompt_count: int | None = None
    completion_tokens: int | None = None
    actual_prompt_tokens: int | None = None
    finish_reason: str | None = None
    ttft_s: float | None = None
    valid: bool = False
    error: str = ""
    prompt: str = field(init=False)

    def __post_init__(self):
        # Same early UUID, instruction and paragraph as concurrency/output_length.
        repetitions = 35 if self.target_prompt_tokens == 3072 else 6
        body = "\n".join([FIXED_PARAGRAPH] * repetitions)
        self.prompt = f"request-{self.client_request_id}\n{FIXED_INSTRUCTION}\n{body}"


def record_event(request, chunk, received_ns):
    """One lightweight record per parsed SSE event, including empty/usage events."""
    choice = chunk.choices[0] if chunk.choices else None
    token_ids = (getattr(choice, "token_ids", None) or []) if choice else []
    request.cumulative += len(token_ids)
    usage = chunk.usage
    prompt_ids = getattr(chunk, "prompt_token_ids", None)
    delta = choice.delta if choice else None
    event = {
        "event_index": len(request.events), "received_ns": received_ns,
        "delta_token_count": len(token_ids), "cumulative_token_count": request.cumulative,
        "token_ids": token_ids, "finish_reason": choice.finish_reason if choice else None,
        "has_usage": usage is not None,
        "usage_completion_tokens": usage.completion_tokens if usage else None,
        "usage_prompt_tokens": usage.prompt_tokens if usage else None,
        "prompt_token_ids_count": len(prompt_ids) if prompt_ids is not None else None,
        "delta_role": getattr(delta, "role", None),
        "has_content": bool(getattr(delta, "content", None)),
    }
    request.events.append(event)
    if request.cumulative >= TRIGGER_TOKENS and not request.ready.is_set():
        request.ready.set()


def validate_request(request):
    # Deliberately outside the consumer hot path: concatenate in original event order.
    generated_ids = [token for event in request.events for token in event["token_ids"]]
    usage_events = [event for event in request.events if event["has_usage"]]
    if not usage_events or not request.events[-1]["has_usage"]:
        raise ValueError("Final usage missing")
    final = usage_events[-1]
    request.completion_tokens = final["usage_completion_tokens"]
    request.actual_prompt_tokens = final["usage_prompt_tokens"]
    if len(generated_ids) != request.completion_tokens:
        raise ValueError(f"Token IDs/usage mismatch: {len(generated_ids)} != {request.completion_tokens}")
    if any(type(token) is not int for token in generated_ids):
        raise ValueError("Generated token IDs must be integers")
    if not 0 < request.completion_tokens <= request.max_tokens:
        raise ValueError("Completion count outside requested token limit")
    counts = [e["prompt_token_ids_count"] for e in request.events
              if e["prompt_token_ids_count"] is not None]
    if counts:
        request.prompt_count = counts[0]
        if any(count != request.actual_prompt_tokens for count in counts):
            raise ValueError("prompt_token_ids count disagrees with final usage")
    prompt = request.actual_prompt_tokens
    if prompt is None:
        raise ValueError("Actual prompt token count missing")
    if request.target_prompt_tokens == 3072:
        if prompt <= 2048:
            raise ValueError(f"Invalid Long: actual prompt {prompt} <= 2048-token lower bound")
        if not 3000 <= prompt <= 3200:
            raise ValueError(f"Long actual prompt {prompt} outside target range 3000–3200")
    elif not 550 <= prompt <= 600:
        raise ValueError(f"Background/Short actual prompt {prompt} outside target range 550–600")
    finishes = [e["finish_reason"] for e in request.events if e["finish_reason"] is not None]
    request.finish_reason = finishes[-1] if finishes else None
    if request.finish_reason not in {"stop", "length", "repetition"}:
        raise ValueError(f"Unexpected finish_reason={request.finish_reason!r}")
    bearing = [e for e in request.events if e["delta_token_count"] > 0]
    if not bearing:
        raise ValueError("No token-bearing events")
    request.ttft_s = (bearing[0]["received_ns"] - request.start_ns) / 1e9
    if request.role == "background" and request.cumulative < TRIGGER_TOKENS:
        raise ValueError(f"Background ended before trigger: {request.cumulative} < 64")
    request.valid = True


async def stream_request(client, request, alignment=None):
    # Build all arguments before recording API-call start; no prompt work at injection.
    kwargs = dict(model=MODEL, messages=[{"role": "user", "content": request.prompt}],
                  temperature=0, max_tokens=request.max_tokens, stream=True,
                  stream_options={"include_usage": True}, extra_body={"return_token_ids": True})
    stream = None
    try:
        request.start_ns = time.perf_counter_ns()
        if alignment is not None:
            alignment["foreground_request_start_ns"] = request.start_ns
            alignment["alignment_ns"] = request.start_ns
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            received_ns = time.perf_counter_ns()
            record_event(request, chunk, received_ns)
        request.end_ns = time.perf_counter_ns()
        validate_request(request)
    except BaseException as exc:
        request.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if request.end_ns is None:
            request.end_ns = time.perf_counter_ns()
        if stream is not None:
            try:
                async with asyncio.timeout(5):
                    await stream.close()
            except BaseException as exc:
                request.valid = False
                request.error += f"; stream close: {type(exc).__name__}: {exc}"
                raise


async def wait_for_idle(http_client, snapshots=None, before=None, requests=()):
    """Two idle scrapes, KV <= 0.1%; wait for completed counters to be exported too."""
    stable = 0
    expected_generation = sum(r.completion_tokens or 0 for r in requests)
    expected_prompt = sum(r.actual_prompt_tokens or 0 for r in requests)
    async with asyncio.timeout(60):
        while True:
            snapshot = await get_metrics_snapshot(http_client, METRICS_URL, MODEL)
            if snapshots is not None:
                snapshots.append(snapshot)
            g = snapshot["gauges"]
            accounted = before is None or (
                snapshot["counters"]["generation_tokens"] - before["counters"]["generation_tokens"]
                >= expected_generation
                and snapshot["counters"]["prompt_tokens"] - before["counters"]["prompt_tokens"]
                >= expected_prompt
            )
            idle = g["running"] == g["waiting"] == 0 and g["kv_cache_usage_perc"] <= 0.001
            stable = stable + 1 if idle and accounted else 0
            if stable >= 2:
                return snapshot
            await asyncio.sleep(GAUGE_INTERVAL)


async def sample_gauges(http_client, stop, snapshots):
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=GAUGE_INTERVAL)
            return
        except TimeoutError:
            snapshots.append(await get_metrics_snapshot(http_client, METRICS_URL, MODEL))


async def coordinate(client, backgrounds, foreground, alignment):
    for request in backgrounds:
        await request.ready.wait()
    # No await between checking the live cohort, counting progress, and timestamping.
    if any(r.end_ns is not None or any(e["finish_reason"] for e in r.events[-1:])
           for r in backgrounds):
        raise ValueError("Background completed before injection; cohort is no longer C=8")
    alignment["trigger_ns"] = time.perf_counter_ns()
    alignment["trigger_counts"] = [r.cumulative for r in backgrounds]
    alignment["trigger_count"] = 1
    if foreground is None:
        alignment["alignment_ns"] = alignment["trigger_ns"]
    else:
        await stream_request(client, foreground, alignment)


async def run_cohort(client, backgrounds, foreground, alignment, http_client, snapshots):
    stop = asyncio.Event()
    # TaskGroup propagates any background/foreground/metrics failure and closes siblings.
    async with asyncio.TaskGroup() as group:
        group.create_task(sample_gauges(http_client, stop, snapshots))
        background_tasks = [group.create_task(stream_request(client, r)) for r in backgrounds]
        coordinator = group.create_task(coordinate(client, backgrounds, foreground, alignment))
        await coordinator
        for task in background_tasks:
            await task
        stop.set()


def error_text(exc):
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(error_text(child) for child in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def event_distribution(events):
    deltas = [e["delta_token_count"] for e in events if e["delta_token_count"] > 0]
    n = len(deltas)
    return {
        "token_bearing_events": n,
        "singleton_event_ratio": sum(d == 1 for d in deltas) / n if n else None,
        "two_token_event_ratio": sum(d == 2 for d in deltas) / n if n else None,
        "ge3_token_event_ratio": sum(d >= 3 for d in deltas) / n if n else None,
        "multi_token_event_ratio": sum(d > 1 for d in deltas) / n if n else None,
        "max_delta_token_count": max(deltas, default=0),
    }


async def run_trial(client, http_client, identity, timeout):
    start = time.perf_counter_ns()
    backgrounds = [Request("background", i, 577, BACKGROUND_MAX_TOKENS) for i in range(8)]
    condition = identity["condition"]
    foreground = None if condition == "baseline" else Request(
        "foreground", None, 3072 if condition == "long" else 577, 1
    )
    requests = backgrounds + ([foreground] if foreground else [])
    snapshots, alignment, errors = [], {}, []
    before = after = None
    try:
        async with asyncio.timeout(timeout):
            before = await wait_for_idle(http_client)
            snapshots.append(before)
            await run_cohort(client, backgrounds, foreground, alignment, http_client, snapshots)
    except (Exception, asyncio.CancelledError) as exc:
        errors.append(error_text(exc))
    finally:
        try:
            after = await wait_for_idle(http_client, snapshots, before, requests)
        except (Exception, asyncio.CancelledError) as exc:
            errors.append("Drain: " + error_text(exc))
    end = time.perf_counter_ns()
    row = {
        **identity, "model": MODEL, "temperature": 0, "gauge_interval_s": GAUGE_INTERVAL,
        "trial_start_ns": start, "trial_end_ns": end, "duration_s": (end - start) / 1e9,
        **{k: alignment.get(k) for k in (
            "trigger_ns", "alignment_ns", "foreground_request_start_ns")},
        "trigger_count": alignment.get("trigger_count", 0),
        "background_concurrency": 8, "background_target_prompt_tokens": 577,
        "background_max_tokens": BACKGROUND_MAX_TOKENS, "trigger_tokens": TRIGGER_TOKENS,
        **{f"trigger_count_bg{i}": count for i, count in enumerate(alignment.get("trigger_counts", []))},
        "foreground_target_prompt_tokens": foreground.target_prompt_tokens if foreground else None,
        **{f"foreground_{name}": getattr(foreground, name) if foreground else None
           for name in ("actual_prompt_tokens", "completion_tokens", "ttft_s", "finish_reason")},
        **{f"background_{key}": value for key, value in event_distribution(
            [e for r in backgrounds for e in r.events]).items()},
        "token_ids_usage_all_match": all(r.completion_tokens is not None and
                                         r.cumulative == r.completion_tokens for r in requests),
        "request_summaries": json.dumps([{
            "client_request_id": r.client_request_id, "role": r.role,
            "background_id": r.background_id, "request_start_ns": r.start_ns,
            "request_end_ns": r.end_ns, "generated_ids_count": r.cumulative,
            "completion_tokens": r.completion_tokens, "actual_prompt_tokens": r.actual_prompt_tokens,
            "prompt_token_ids_count": r.prompt_count, "finish_reason": r.finish_reason,
            "valid": r.valid, "error": r.error,
        } for r in requests]),
    }
    counts = [r.actual_prompt_tokens for r in backgrounds if r.actual_prompt_tokens is not None]
    for name, fn in (("min", min), ("median", statistics.median), ("max", max)):
        row[f"background_actual_prompt_tokens_{name}"] = fn(counts) if counts else None
    if before is not None and after is not None:
        if before["counters"].keys() != after["counters"].keys():
            errors.append("Counter series changed")
        elif any(after["counters"][k] < v for k, v in before["counters"].items()):
            errors.append("Counter decreased: possible server reset")
        else:
            row.update({name: after["counters"][key] - before["counters"][key]
                        for name, key in COUNTERS.items()})
    row["prefix_cache_warning"] = (
        "local_cache_hit > 0: Prefix Cache reuse was not excluded"
        if row.get("prompt_tokens_local_cache_hit", 0) > 0 else ""
    )
    if after:
        row.update(idle_running=after["gauges"]["running"], idle_waiting=after["gauges"]["waiting"],
                   idle_kv_cache_usage_perc=after["gauges"]["kv_cache_usage_perc"])
    if not all(r.valid for r in requests) and not errors:
        errors.append("One or more requests invalid")
    row.update(status="failed" if errors else "success", error="; ".join(errors))
    align = alignment.get("alignment_ns")
    events = [{**{k: identity[k] for k in IDENTITY_COLUMNS},
               "role": r.role, "background_id": r.background_id,
               "client_request_id": r.client_request_id, **event, "alignment_ns": align,
               "relative_to_alignment_ms": (event["received_ns"] - align) / 1e6 if align else None}
              for r in requests for event in r.events]
    metrics = []
    for snapshot in sorted(snapshots, key=lambda s: s["timestamp"]):
        ns = round(snapshot["timestamp"] * 1e9)
        g = snapshot["gauges"]
        metrics.append({**{k: identity[k] for k in IDENTITY_COLUMNS}, "sample_ns": ns,
                        "relative_to_alignment_ms": (ns - align) / 1e6 if align else None,
                        "num_requests_running": g["running"], "num_requests_waiting": g["waiting"],
                        **{k: g[k] for k in ("waiting_capacity", "waiting_deferred", "kv_cache_usage_perc")}})
    return row, events, metrics


async def warmup(client, http_client):
    before = await wait_for_idle(http_client)
    requests = [Request("background", i, 577, 64) for i in range(8)]
    try:
        async with asyncio.timeout(180), asyncio.TaskGroup() as group:
            for request in requests:
                group.create_task(stream_request(client, request))
    finally:
        after = await wait_for_idle(http_client, before=before, requests=requests)
    return {
        "completion_tokens": [r.completion_tokens for r in requests],
        "all_valid": all(r.valid for r in requests),
        "counter_deltas": {k: after["counters"][k] - v for k, v in before["counters"].items()},
        "event_distribution": event_distribution([e for r in requests for e in r.events]),
        "idle_gauges": after["gauges"],
    }


def write_trial(row, events, metrics, trial_writer, event_file, metric_writer):
    for event in events:
        event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
    metric_writer.writerows(metrics)
    trial_writer.writerow(row)


async def main(args):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = RESULTS_DIR / f"prefill-interference-{run_id}"
    schedule = make_schedule(1 if args.smoke else args.trials_per_condition, args.seed)
    print(f"Run: {prefix}\nSchedule: {schedule}", flush=True)
    with ExitStack() as stack:
        event_file = stack.enter_context(open(f"{prefix}-events.jsonl", "x", encoding="utf-8"))
        trial_file = stack.enter_context(open(f"{prefix}-trials.csv", "x", newline="", encoding="utf-8"))
        metric_file = stack.enter_context(open(f"{prefix}-metrics.csv", "x", newline="", encoding="utf-8"))
        trial_writer = csv.DictWriter(trial_file, fieldnames=TRIAL_COLUMNS)
        metric_writer = csv.DictWriter(metric_file, fieldnames=METRIC_COLUMNS)
        trial_writer.writeheader()
        metric_writer.writeheader()
        async with (
            AsyncOpenAI(base_url=BASE_URL, api_key="dummy", max_retries=0, timeout=120) as client,
            httpx.AsyncClient(trust_env=False) as http_client,
        ):
            try:
                print("Warmup: C=8, output=64 (excluded)", flush=True)
                await warmup(client, http_client)
            except (Exception, asyncio.CancelledError) as exc:
                # Even server-unreachable/preflight failures leave a reviewable failure row.
                trial_writer.writerow({"run_id": run_id, "trial_index": 0,
                                       "condition": "warmup", "status": "failed",
                                       "error": error_text(exc)})
                raise
            for trial_index, (block, condition) in enumerate(schedule, 1):
                identity = dict(run_id=run_id, block_index=block, trial_index=trial_index,
                                condition=condition, seed=args.seed, smoke=args.smoke)
                row, events, metrics = await run_trial(client, http_client, identity, args.trial_timeout)
                write_trial(row, events, metrics, trial_writer, event_file, metric_writer)
                for output in (event_file, trial_file, metric_file):
                    output.flush()
                print(json.dumps({k: v for k, v in row.items() if k != "request_summaries"}), flush=True)
                if row["prefix_cache_warning"]:
                    print(f"WARNING: {row['prefix_cache_warning']}", file=sys.stderr)
                if row["status"] != "success":
                    raise RuntimeError(row["error"])
    print(f"Completed. Plot: uv run --no-sync python experiments/plot_prefill_interference.py {prefix}-trials.csv")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One randomized block, never a formal conclusion")
    parser.add_argument("--trials-per-condition", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--trial-timeout", type=float, default=180)
    args = parser.parse_args()
    if args.trials_per_condition < 1:
        parser.error("--trials-per-condition must be positive")
    if not math.isfinite(args.trial_timeout) or args.trial_timeout <= 0:
        parser.error("--trial-timeout must be finite and positive")
    return args


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except (Exception, KeyboardInterrupt) as failure:
        print("Experiment stopped: " + error_text(failure), file=sys.stderr)
        raise SystemExit(1)
