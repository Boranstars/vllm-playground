"""Two fixed-pool mechanism probes; no vLLM modifications or cache resets."""
import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import AsyncOpenAI

import common
import prefill_interference as stream_base
import token_budget_sweep as lifecycle

ROOT = Path(__file__).resolve().parents[1]
MODEL = stream_base.MODEL
URL = "http://127.0.0.1:8000"
SEED = 20261013
ADMISSION = ("BLOCKED_THEN_RELEASE", "RELEASE_THEN_INJECT")
GROWTH = ("CONTROL_GROWTH", "PRESSURE_GROWTH")
now = time.perf_counter_ns


def required_blocks(tokens, block_size):
    if tokens < 0 or block_size <= 0:
        raise ValueError("Invalid token/block count")
    return math.ceil(tokens / block_size)


def free_blocks(usage, num_gpu_blocks):
    if not 0 <= usage <= 1 or num_gpu_blocks <= 1:
        raise ValueError("Invalid pool usage")
    return (1 - usage) * (num_gpu_blocks - 1)


def counter_delta(before, after):
    if after < before:
        raise ValueError("Counter reset")
    return after - before


def admission_window(free, fg_blocks, holder_blocks, margin):
    return free < fg_blocks - margin and free + holder_blocks > fg_blocks + margin


def make_schedule(blocks, conditions, seed=SEED):
    rng = random.Random(seed)
    result = []
    for block in range(1, blocks + 1):
        order = list(conditions)
        rng.shuffle(order)
        result.extend((block, condition) for condition in order)
    return result


def growth_targets(capacity, block_size, n, prompt_tokens, max_model_len=4096):
    safe_context = math.floor(.70 * capacity / n / block_size) * block_size
    pressure_context = min(math.ceil(1.10 * capacity / n / block_size) * block_size,
                           (max_model_len - 32) // block_size * block_size)
    if safe_context <= max(prompt_tokens) or n * pressure_context < 1.10 * capacity:
        raise ValueError("Cohort cannot construct clean growth targets")
    return dict(safe_context=safe_context, pressure_context=pressure_context,
                safe_max_tokens=[safe_context - p for p in prompt_tokens],
                pressure_max_tokens=[pressure_context - p for p in prompt_tokens],
                safe_aggregate=n * safe_context, pressure_aggregate=n * pressure_context)


@dataclass(frozen=True)
class Pool:
    num_gpu_blocks: int
    block_size: int
    kv_cache_size_tokens: int
    max_model_len: int = 4096

    @property
    def usable_tokens(self):
        return (self.num_gpu_blocks - 1) * self.block_size

    @property
    def margin(self):
        return max(16, min(32, math.ceil(.01 * (self.num_gpu_blocks - 1))))

    def free(self, metric):
        return free_blocks(metric["kv_cache_usage_perc"], self.num_gpu_blocks)


def validate_trial(row):
    """Protocol validity independent of runtime; fails closed on missing evidence."""
    if row.get("error"):
        raise ValueError(row["error"])
    for name in ("initial_admitted", "drained", "stream_integrity", "abort_verified"):
        if row.get(name) is not True:
            raise ValueError(f"Missing evidence: {name}")
    if row["external_kv_transfer_delta"] != 0:
        raise ValueError("External KV transfer")
    condition = row["condition"]
    if condition in ADMISSION:
        if row["preemption_delta"] != 0 or row["prefix_hits_delta"] != 0:
            raise ValueError("Admission preemption or shared prefix confound")
        if not admission_window(row["trigger_free_blocks"], row["fg_required_blocks"],
                                row["holder_blocks_estimate"], row["margin_blocks"]):
            raise ValueError("Invalid admission window")
        if not row["release_headroom_verified"] or row["fg_first_ns"] is None:
            raise ValueError("Release did not restore headroom/progress")
        if condition == ADMISSION[0]:
            if (row["blocked_samples"] < 3 or row["blocked_span_ms"] < 100
                    or not row["fg_send_ns"] < row["release_ns"] < row["fg_first_ns"]):
                raise ValueError("No client-observed blocked interval")
        elif (not row["release_ns"] < row["headroom_ns"] <= row["fg_send_ns"]
              or row["waiting_plateau_ms"] >= 100):
            raise ValueError("Control ordering or sustained waiting")
    elif condition == GROWTH[0]:
        if row["preemption_delta"] != 0 or not row["all_completed"] or row["prefix_hits_delta"] != 0:
            raise ValueError("Invalid growth control")
    elif condition == GROWTH[1]:
        if row["preemption_delta"] <= 0 or row["max_kv_usage"] < .9:
            raise ValueError("No natural capacity preemption observed")
        if row["prefix_hits_before_preemption"] != 0:
            raise ValueError("Cross-request prefix contamination before preemption")
    else:
        raise ValueError("Unknown condition")


class Request(stream_base.Request):
    def __init__(self, role, prompt, count, limit):
        super().__init__(role, None, count, limit)
        self.prompt = prompt
        self.actual_prompt_tokens = count
        self.first_ns = None
        self.cancel_ns = None
        self.closed_ns = None
        self.cancelled = False
        self.stream = None

    def summary(self):
        return {k: getattr(self, k) for k in (
            "client_request_id", "role", "prompt", "actual_prompt_tokens", "max_tokens",
            "start_ns", "first_ns", "end_ns", "cancel_ns", "closed_ns", "cancelled",
            "cumulative", "completion_tokens", "finish_reason", "valid", "error")}


async def stream_request(client, request):
    """Use the existing OpenAI SSE parser and project token-ID event recorder."""
    try:
        request.start_ns = now()
        request.stream = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": request.prompt}],
            temperature=0, max_tokens=request.max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body={"return_token_ids": True, "ignore_eos": True})
        async for chunk in request.stream:
            stream_base.record_event(request, chunk, now())
            if request.cumulative and request.first_ns is None:
                request.first_ns = request.events[-1]["received_ns"]
        usages = [e for e in request.events if e["has_usage"]]
        if not usages or not request.events[-1]["has_usage"]:
            raise ValueError("Missing final usage")
        final = usages[-1]
        request.completion_tokens = final["usage_completion_tokens"]
        finishes = [e["finish_reason"] for e in request.events if e["finish_reason"]]
        request.finish_reason = finishes[-1] if finishes else None
        if (request.cumulative != request.completion_tokens
                or request.completion_tokens != request.max_tokens
                or final["usage_prompt_tokens"] != request.actual_prompt_tokens
                or request.finish_reason != "length"):
            raise ValueError("Token integrity / unexpected EOS or finish")
        request.valid = True
    except asyncio.CancelledError:
        if request.cancel_ns is None:
            request.error = "Unplanned cancellation"
            raise
        request.cancelled = True
    except BaseException as exc:
        request.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        request.end_ns = now()
        if request.stream is not None:
            await asyncio.wait_for(request.stream.close(), 5)
        request.closed_ns = now()


async def cancel(request, task):
    if task.done():
        await task
        raise ValueError("Designated request finished before cancellation")
    request.cancel_ns = now()
    task.cancel()
    await asyncio.wait_for(task, 5)
    if not request.cancelled or request.closed_ns is None:
        raise ValueError("Client stream cancellation failed")


def extra_metrics(text):
    result = {}
    for name in ("prefix_cache_hits_total", "prefix_cache_queries_total"):
        matches = re.findall(r"^vllm:" + name + r'\{([^\n]+)\}\s+(\S+)', text, re.M)
        values = [float(v) for labels, v in matches if f'model_name="{MODEL}"' in labels
                  and 'engine="0"' in labels]
        if len(values) != 1:
            raise ValueError(f"Missing/ambiguous {name}")
        result[name] = values[0]
    return result


async def snapshot(http):
    start = now()
    response = await http.get(URL + "/metrics", timeout=5)
    response.raise_for_status()
    stamp = now()
    parsed = common.parse_vllm_metrics(response.text, MODEL)
    return dict(sample_ns=stamp, scrape_ms=(stamp-start)/1e6, **parsed["gauges"],
                **parsed["counters"], **extra_metrics(response.text))


class Sampler:
    def __init__(self, http, writer, file, identity, interval=.05):
        self.http, self.writer, self.file = http, writer, file
        self.identity, self.interval = identity, interval
        self.rows = []
        self.task = None

    async def run(self):
        while True:
            start = time.monotonic()
            row = dict(**self.identity, **await snapshot(self.http))
            self.rows.append(row)
            self.writer.writerow(row)
            self.file.flush()
            await asyncio.sleep(max(0, self.interval - (time.monotonic()-start)))

    async def next(self):
        count = len(self.rows)
        while len(self.rows) == count:
            if self.task.done():
                await self.task
            await asyncio.sleep(.002)
        return self.rows[-1]


async def idle(http):
    stable = 0
    async with asyncio.timeout(30):
        while True:
            row = await snapshot(http)
            stable = stable + 1 if row["running"] == row["waiting"] == 0 and row["kv_cache_usage_perc"] <= .001 else 0
            if stable >= 3:
                return row
            await asyncio.sleep(.1)


async def make_prompt(http, target):
    # Unique nonce before any repeated text; chat-template prefix is < one block.
    prefix = uuid.uuid4().hex + "\nContinue listing integers and their English names. "
    async def tokenize(text):
        response = await http.post(URL + "/tokenize", json={
            "model": MODEL, "messages": [{"role": "user", "content": text}]})
        response.raise_for_status()
        return response.json()["tokens"]
    base = await tokenize(prefix)
    prompt = prefix + " data" * max(0, target-len(base))
    tokens = await tokenize(prompt)
    if len(tokens) != target:
        raise ValueError(f"Prompt generation count mismatch: {len(tokens)} != {target}")
    return prompt, tokens


async def make_requests(http, n, role, target, limit):
    requests, tokens = [], []
    for _ in range(n):
        prompt, ids = await make_prompt(http, target)
        requests.append(Request(role, prompt, len(ids), limit))
        tokens.append(ids)
    return requests, tokens


def check_tasks(tasks):
    for task in tasks:
        if task.done() and not task.cancelled() and task.exception():
            raise task.exception()


async def initial_admission(sampler, requests, tasks, before):
    async with asyncio.timeout(15):
        while True:
            m = await sampler.next()
            check_tasks(tasks)
            if m["preemptions"] != before["preemptions"]:
                raise ValueError("Preemption before initial admission")
            if m["running"] == len(requests) and m["waiting"] == 0 and all(r.first_ns for r in requests):
                return m


async def admission_trial(client, http, sampler, pool, row, requests, tasks, before, n):
    holders, ids = await make_requests(http, n, "holder", 80, pool.max_model_len-32-80)
    fg_list, fg_ids = await make_requests(http, 1, "foreground", 3104, 1)
    fg = fg_list[0]
    requests.extend(holders + [fg])
    all_ids = ids + fg_ids
    if len({tuple(t[:pool.block_size]) for t in all_ids}) != len(all_ids):
        raise ValueError("Shared first block")
    holder_tasks = [asyncio.create_task(stream_request(client, r)) for r in holders]
    tasks.extend(holder_tasks)
    await initial_admission(sampler, holders, holder_tasks, before)
    row["initial_admitted"] = True
    required = required_blocks(fg.actual_prompt_tokens, pool.block_size)
    async with asyncio.timeout(240):
        while True:
            m = await sampler.next()
            check_tasks(tasks)
            if any(t.done() for t in holder_tasks) or m["preemptions"] != before["preemptions"]:
                raise ValueError("Holder completion/preemption before trigger")
            # Conservative lower estimate: last generated token may not yet have KV.
            holder_blocks = max(0, (holders[0].actual_prompt_tokens + holders[0].cumulative-2)//pool.block_size)
            if m["running"] == n and m["waiting"] == 0 and admission_window(pool.free(m), required, holder_blocks, pool.margin):
                break
    row.update(trigger_ns=m["sample_ns"], trigger_free_blocks=pool.free(m),
               trigger_kv_usage=m["kv_cache_usage_perc"], holder_blocks_estimate=holder_blocks,
               fg_required_blocks=required, margin_blocks=pool.margin, holder_count=n,
               trigger_holder_counts=[r.cumulative for r in holders])
    fg_task = None
    if row["condition"] == ADMISSION[0]:
        fg_task = asyncio.create_task(stream_request(client, fg))
        tasks.append(fg_task)
        waiting = []
        async with asyncio.timeout(.6):
            while True:
                m = await sampler.next()
                check_tasks(tasks)
                if fg.first_ns or m["preemptions"] != before["preemptions"]:
                    raise ValueError("Foreground progressed/preemption during blocked probe")
                if m["waiting"] >= 1 and m["running"] == n:
                    waiting.append(m)
                else:
                    waiting = []
                if len(waiting) >= 3 and (waiting[-1]["sample_ns"]-waiting[0]["sample_ns"])/1e6 >= 100:
                    break
        row.update(first_waiting_ns=waiting[0]["sample_ns"], blocked_samples=len(waiting),
                   blocked_span_ms=(waiting[-1]["sample_ns"]-waiting[0]["sample_ns"])/1e6)
    await cancel(holders[0], holder_tasks[0])
    row["release_ns"] = holders[0].cancel_ns
    row["release_closed_ns"] = holders[0].closed_ns
    # A1 may already have allocated FG blocks before the first post-release scrape.
    async with asyncio.timeout(5):
        while True:
            m = await sampler.next()
            check_tasks(tasks)
            if m["kv_cache_usage_perc"] < row["trigger_kv_usage"]-1/(pool.num_gpu_blocks-1):
                row.setdefault("first_observed_kv_drop_ns", m["sample_ns"])
            if m["waiting"] == 0:
                row.setdefault("first_observed_waiting_zero_after_release_ns", m["sample_ns"])
            fg_allowance = required if fg_task is not None else 0
            running_bound = n if fg_task is not None else n-1
            if (pool.free(m) + fg_allowance > required + pool.margin
                    and m["kv_cache_usage_perc"] < row["trigger_kv_usage"]-.025
                    and m["running"] <= running_bound):
                row.update(headroom_ns=m["sample_ns"], post_release_free_blocks=pool.free(m),
                           post_release_free_plus_fg_blocks=pool.free(m)+fg_allowance,
                           release_headroom_verified=True)
                break
    if fg_task is None:
        fg_task = asyncio.create_task(stream_request(client, fg))
        tasks.append(fg_task)
    await asyncio.wait_for(asyncio.shield(fg_task), 15)
    row.update(fg_send_ns=fg.start_ns, fg_first_ns=fg.first_ns,
               fg_ttft_ms=(fg.first_ns-fg.start_ns)/1e6,
               release_to_first_ms=(fg.first_ns-row["release_ns"])/1e6)
    if any(r.end_ns is not None for r in holders[1:]):
        raise ValueError("Undesignated holder ended during admission probe")
    if row["condition"] == ADMISSION[0]:
        row["client_blocked_interval_ms"] = (row["release_ns"]-fg.start_ns)/1e6
    waits = [m for m in sampler.rows if fg.start_ns <= m["sample_ns"] <= fg.first_ns and m["waiting"] >= 1]
    row["waiting_plateau_ms"] = (waits[-1]["sample_ns"]-waits[0]["sample_ns"])/1e6 if len(waits)>1 else 0


async def growth_trial(client, http, sampler, pool, row, requests, tasks, before, n):
    cohort, ids = await make_requests(http, n, "growth", 80, 1)
    if len({tuple(t[:pool.block_size]) for t in ids}) != n:
        raise ValueError("Shared first block")
    targets = growth_targets(pool.kv_cache_size_tokens, pool.block_size, n, [80]*n)
    limits = targets["safe_max_tokens"] if row["condition"] == GROWTH[0] else targets["pressure_max_tokens"]
    for r, limit in zip(cohort, limits):
        r.max_tokens = limit
    row.update(cohort_size=n, **targets)
    requests.extend(cohort)
    tasks.extend(asyncio.create_task(stream_request(client, r)) for r in cohort)
    m = await initial_admission(sampler, cohort, tasks, before)
    row.update(initial_admitted=True, initial_admitted_ns=m["sample_ns"],
               initial_kv_usage=m["kv_cache_usage_perc"], cohort_send_ns=min(r.start_ns for r in cohort))
    previous = m
    async with asyncio.timeout(300):
        while True:
            m = await sampler.next()
            check_tasks(tasks)
            if row["condition"] == GROWTH[1] and m["preemptions"] > before["preemptions"]:
                row.update(first_observed_preemption_ns=m["sample_ns"],
                           time_to_preemption_s=(m["sample_ns"]-row["cohort_send_ns"])/1e9,
                           prefix_hits_before_preemption=counter_delta(before["prefix_cache_hits_total"], previous["prefix_cache_hits_total"]))
                await asyncio.sleep(1)
                break
            if all(t.done() for t in tasks):
                if row["condition"] == GROWTH[1]:
                    raise ValueError("Pressure workload completed without observed preemption")
                row["all_completed"] = True
                break
            previous = m


async def run_trial(client, http, pool, identity, files, n_holders, n_growth):
    metric_writer, metric_file, event_file, trial_file = files
    before = await idle(http)
    row = dict(**identity, trial_start_ns=now(), idle_before=before,
               initial_admitted=False, drained=False, stream_integrity=False,
               abort_verified=False, all_completed=False, blocked_samples=0,
               blocked_span_ms=0, error="")
    sampler = Sampler(http, metric_writer, metric_file, identity)
    sampler.task = asyncio.create_task(sampler.run())
    requests, tasks = [], []
    failure = None
    try:
        if identity["condition"] in ADMISSION:
            await admission_trial(client, http, sampler, pool, row, requests, tasks, before, n_holders)
        else:
            await growth_trial(client, http, sampler, pool, row, requests, tasks, before, n_growth)
    except BaseException as exc:
        failure = exc
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Requests may include a foreground not yet started; tasks map by start order.
        for request in requests:
            if request.start_ns is not None and request.end_ns is None:
                request.cancel_ns = now()
        for task in tasks:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if any(isinstance(result, BaseException) for result in results) and failure is None:
            failure = ValueError(f"Request failures: {results}")
            row["error"] = str(failure)
        try:
            after = await idle(http)
            # Let engine statistics catch up; require counters stable across a short idle window.
            await asyncio.sleep(.2)
            after = await idle(http)
            row.update(drained=True, idle_after=after,
                       preemption_delta=counter_delta(before["preemptions"], after["preemptions"]),
                       prefix_hits_delta=counter_delta(before["prefix_cache_hits_total"], after["prefix_cache_hits_total"]),
                       external_kv_transfer_delta=counter_delta(before["external_kv_transfer_tokens"], after["external_kv_transfer_tokens"]),
                       abort_count_delta=counter_delta(before["request_success:abort"], after["request_success:abort"]))
            # Disconnect abort pops output state before finished-request stats;
            # request_success{abort} is not a required acknowledgement on this path.
            row["abort_verified"] = all(r.closed_ns and not r.error for r in requests if r.cancelled)
            row["abort_evidence"] = "planned stream close + server Running/Waiting/KV drained; designated release also requires headroom recovery"
            row["counter_deltas"] = {k: counter_delta(before[k], after[k]) for k in before
                                     if k in common.COUNTER_SELECTORS or k.startswith("request_success:")}
        except BaseException as exc:
            failure = failure or exc
            row["error"] += f"; drain: {exc}"
        sampler.task.cancel()
        try:
            await sampler.task
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            failure = failure or exc
            row["error"] += f"; metrics: {exc}"
        row["requests"] = [r.summary() for r in requests]
        row["stream_integrity"] = all((r.valid or (r.cancelled and r.closed_ns and not r.error)) for r in requests)
        row["max_kv_usage"] = max((m["kv_cache_usage_perc"] for m in sampler.rows), default=0)
        row["trial_end_ns"] = now()
        row["duration_s"] = (row["trial_end_ns"]-row["trial_start_ns"])/1e9
        if failure is None:
            try:
                validate_trial(row)
            except ValueError as exc:
                failure = exc
                row["error"] = str(exc)
        row["status"] = "failed" if failure else "validated"
        for request in requests:
            for event in request.events:
                event_file.write(json.dumps(dict(**identity, client_request_id=request.client_request_id,
                                                 role=request.role, **event))+"\n")
        event_file.flush()
        trial_file.write(json.dumps(row)+"\n")
        trial_file.flush()
        print(json.dumps({k: row[k] for k in ("trial_index", "condition", "status", "duration_s", "error")}
                         | {k: row.get(k) for k in ("preemption_delta", "fg_ttft_ms", "release_to_first_ms", "max_kv_usage")}), flush=True)
    if failure:
        raise failure
    return row


async def cancellation_smoke(client, http, pool, directory):
    before = await idle(http)
    cohort, _ = await make_requests(http, 1, "cancel_smoke", 80, 1024)
    r = cohort[0]
    task = asyncio.create_task(stream_request(client, r))
    samples = []
    try:
        async with asyncio.timeout(15):
            while r.cumulative < 160:
                check_tasks([task])
                samples.append(await snapshot(http))
                await asyncio.sleep(.05)
        pre = await snapshot(http)
        await cancel(r, task)
        after = await idle(http)
        lifecycle.write_json(directory / "cancellation-smoke.json", dict(
            before=before, pre_cancel=pre, after=after, request=r.summary(), samples=samples,
            cancel_to_idle_ms=(after["sample_ns"]-r.cancel_ns)/1e6))
        if (pre["running"] != 1 or after["running"] != 0
                or pool.free(after)-pool.free(pre) < 8
                or after["prefix_cache_hits_total"] != before["prefix_cache_hits_total"]):
            raise ValueError("Cancellation/prefix smoke failed")
    finally:
        if not task.done():
            await cancel(r, task)
        await idle(http)


async def graceful_shutdown(process, log_path):
    """Signal API owner first, allowing it to drain/close EngineCore itself."""
    found = re.search(r"APIServer pid=(\d+)", log_path.read_text())
    api_pid = int(found[1]) if found else None
    if api_pid is None or api_pid not in lifecycle.session_processes(process.pid):
        return await lifecycle.shutdown(process)
    os.kill(api_pid, signal.SIGINT)
    deadline = time.monotonic()+45
    while time.monotonic()<deadline:
        process.poll()
        if process.returncode is not None and not lifecycle.session_processes(process.pid):
            lifecycle.require_free_port()
            if not lifecycle.expected_shutdown_returncode(process.returncode,["SIGINT"]):
                raise RuntimeError(f"Abnormal shutdown code {process.returncode}")
            return dict(signals=["SIGINT"], signal_target=api_pid, returncode=process.returncode,
                        remaining_pids=[], port_8000_released=True, timestamp=lifecycle.utc_now())
        await asyncio.sleep(.2)
    cleanup=await lifecycle.shutdown(process)
    raise RuntimeError(f"Graceful API shutdown exceeded 45s; fallback cleanup: {cleanup}")


async def run(args):
    directory = ROOT / "results" / "kv-pressure" / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_id=args.run_id, phase=args.phase, seed=SEED,
                    blocks=args.blocks, interval_s=.05, status="starting", started=lifecycle.utc_now())
    lifecycle.write_json(directory / "manifest.json", manifest)
    source_dir = directory / "source"
    source_dir.mkdir()
    for name in ("experiments/kv_pressure_experiment.py", "experiments/plot_kv_pressure.py",
                 "tests/test_kv_pressure_experiment.py", "notes/kv-pressure-experiment.md"):
        (source_dir / Path(name).name).write_text((ROOT/name).read_text())
    lifecycle.require_free_port()
    lifecycle.capture_environment(directory / "environment-start.txt")
    command = [str(ROOT / "scripts/serve_qwen3_0_6b.sh"), "--max-num-batched-tokens", "2048",
               "--host", "127.0.0.1", "--shutdown-timeout", "30"]
    log_path = directory / "server.log"
    metadata = dict(command=command, status="starting")
    rows = []
    with log_path.open("x") as log:
        # Read-only /server_info needs the dev router; bind it to loopback only.
        # No dev mutation endpoints are called. Identical smoke/formal setup.
        env = os.environ | {"VLLM_SERVER_DEV_MODE": "1"}
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            print(f"Starting {args.phase}: pid={process.pid} directory={directory}", flush=True)
            await lifecycle.wait_ready(process)
            metadata.update(lifecycle.startup_metadata(args.run_id, 2048, process.pid, command,
                                                       log_path, log_path.read_text(), manifest["started"]))
            async with httpx.AsyncClient(trust_env=False, timeout=10) as http, AsyncOpenAI(
                base_url=URL+"/v1", api_key="EMPTY", max_retries=0,
                timeout=httpx.Timeout(320, connect=10),
                http_client=httpx.AsyncClient(trust_env=False)) as client:
                raw = (await http.get(URL+"/metrics")).text
                (directory / "metrics-start.txt").write_text(raw)
                config_line = next(line for line in raw.splitlines() if line.startswith("vllm:cache_config_info{"))
                labels = dict(re.findall(r'(\w+)="([^"]*)"', config_line))
                metadata["cache_config_info"] = labels
                pool = Pool(int(labels["num_gpu_blocks"]), int(labels["block_size"]), int(labels["kv_cache_size_tokens"]))
                info_response = await http.get(URL+"/server_info?config_format=json", timeout=60)
                metadata["server_info_status"] = info_response.status_code
                if info_response.status_code == 200:
                    lifecycle.write_json(directory / "server-info.json", info_response.json())
                info_response.raise_for_status()
                actual = info_response.json()["vllm_config"]
                scheduler = actual["scheduler_config"]
                metadata["actual_scheduler_config"] = scheduler
                metadata["shutdown_timeout_s"] = actual["shutdown_timeout"]
                metadata["max_num_seqs"] = scheduler["max_num_seqs"]
                metadata["async_scheduling"] = scheduler["async_scheduling"]
                metadata["long_prefill_token_threshold"] = scheduler["long_prefill_token_threshold"]
                metadata["dev_router_readonly_usage"] = "GET /server_info only; loopback bind"
                if (scheduler["policy"] != "fcfs" or not scheduler["scheduler_reserve_full_isl"]
                        or scheduler["watermark"] != 0 or scheduler["max_num_batched_tokens"] != 2048
                        or scheduler.get("max_num_active_seqs") not in (None, scheduler["max_num_seqs"])
                        or actual.get("kv_transfer_config") is not None
                        or actual.get("kv_events_config") is not None):
                    raise ValueError("Unexpected scheduler/transfer configuration")
                log_text = log_path.read_text()
                # Startup scheduler config is emitted by current installation.
                scheduler_line = next((line for line in log_text.splitlines() if "SchedulerConfig(" in line), "")
                metadata["scheduler_log"] = scheduler_line
                if (pool.num_gpu_blocks*pool.block_size != pool.kv_cache_size_tokens
                        or metadata["kv_cache_capacity_tokens"] != pool.kv_cache_size_tokens
                        or metadata["max_model_len"] != 4096
                        or metadata["gpu_memory_utilization"] != .75
                        or metadata["observed_max_num_batched_tokens"] != 2048
                        or not metadata["prefix_cache_enabled"] or not metadata["chunked_prefill_enabled"]):
                    raise ValueError("Unexpected server configuration/capacity")
                # Explicit defaults cross-checked in local installed source, and persisted.
                metadata.update(pool=vars(pool), usable_tokens=pool.usable_tokens,
                                margin_blocks=pool.margin, scheduler_reserve_full_isl=True,
                                scheduler_watermark=0, scheduling_policy="fcfs",
                                scheduler_config_evidence="fresh server /server_info JSON and startup log")
                n_holders = args.holders or math.ceil((pool.usable_tokens-(3104-pool.margin*pool.block_size))/(4096-32))
                n_growth = args.growth_n or 12
                while True:
                    try:
                        targets = growth_targets(pool.kv_cache_size_tokens, pool.block_size, n_growth, [80]*n_growth)
                        break
                    except ValueError:
                        if args.growth_n or n_growth >= 16:
                            raise
                        n_growth += 2
                if metadata["max_num_seqs"] is None or metadata["max_num_seqs"] <= max(n_holders+1, n_growth):
                    raise ValueError("Unverified max_num_seqs headroom")
                metadata.update(n_holders=n_holders, n_growth=n_growth, growth_targets=targets,
                                foreground_prompt_tokens=3104, holder_prompt_tokens=80)
                if args.calibration_run:
                    prior = ROOT/"results"/"kv-pressure"/args.calibration_run
                    calibration = json.loads((prior/"metadata.json").read_text())
                    if json.loads((prior/"manifest.json").read_text())["status"] != "validated":
                        raise ValueError("Calibration did not pass")
                    for key in ("actual_scheduler_config", "n_holders", "n_growth", "max_model_len",
                                "gpu_memory_utilization", "observed_max_num_batched_tokens"):
                        if metadata[key] != calibration[key]:
                            raise ValueError(f"Formal/calibration mismatch: {key}")
                    metadata["calibration_run"] = args.calibration_run
                    metadata["lifecycle_only_change"] = "30s shutdown grace and API-first SIGINT; inference/scheduler config unchanged"
                lifecycle.write_json(directory / "metadata.json", metadata)
                print(json.dumps({k: metadata[k] for k in ("pool", "n_holders", "n_growth", "growth_targets")}), flush=True)
                warm, _ = await make_requests(http, 1, "warmup", 80, 8)
                await stream_request(client, warm[0])
                await idle(http)
                if args.phase == "smoke":
                    await cancellation_smoke(client, http, pool, directory)
                metric_fields = ["run_id", "trial_index", "block_index", "condition", *(await snapshot(http)).keys()]
                with (directory/"metrics.csv").open("x", newline="") as mf, (directory/"events.jsonl").open("x") as ef, (directory/"trials.jsonl").open("x") as tf:
                    writer = csv.DictWriter(mf, fieldnames=metric_fields)
                    writer.writeheader()
                    schedule = make_schedule(args.blocks, ADMISSION) + make_schedule(args.blocks, GROWTH, SEED+1)
                    manifest["schedule"] = schedule
                    lifecycle.write_json(directory / "manifest.json", manifest)
                    for index, (block, condition) in enumerate(schedule, 1):
                        if process.poll() is not None:
                            raise RuntimeError("Server exited unexpectedly")
                        identity = dict(run_id=args.run_id, trial_index=index, block_index=block, condition=condition)
                        rows.append(await run_trial(client, http, pool, identity, (writer, mf, ef, tf), n_holders, n_growth))
                (directory/"metrics-end.txt").write_text((await http.get(URL+"/metrics")).text)
                metadata["status"] = manifest["status"] = "validated"
        except BaseException as exc:
            metadata["status"] = manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                metadata["shutdown"] = await graceful_shutdown(process, log_path)
                lifecycle.capture_environment(directory/"environment-end.txt")
                print("Server shutdown and port release verified", flush=True)
            finally:
                lifecycle.write_json(directory/"metadata.json", metadata)
                lifecycle.write_json(directory/"manifest.json", manifest)
                # Always preserve a CSV, including failed trials written before propagation.
                trial_path = directory/"trials.jsonl"
                saved = [json.loads(line) for line in trial_path.read_text().splitlines()] if trial_path.exists() else []
                if saved:
                    keys = list(dict.fromkeys(k for row in saved for k in row))
                    with (directory/"trials.csv").open("x", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k,v in r.items()} for r in saved)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--holders", type=int)
    parser.add_argument("--growth-n", type=int)
    parser.add_argument("--calibration-run")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) or args.blocks < 1:
        parser.error("Invalid run ID or block count")
    if args.phase == "formal" and not args.calibration_run:
        parser.error("Formal requires --calibration-run naming a validated smoke run")
    asyncio.run(run(args))
