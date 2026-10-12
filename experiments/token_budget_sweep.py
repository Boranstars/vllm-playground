"""Owned server lifecycle: one 2048 smoke, then formal 2048/4096/1024 blocks.

Reuses prefill_interference's prompts, SSE measurement, warmup and drain protocol.
No dependency synchronization, server config mutation from the client, or retries
of failed measurements. A failure leaves raw files and stops the entire sweep.
"""

import argparse
import asyncio
import csv
import errno
import importlib.metadata
import json
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import AsyncOpenAI

import prefill_interference as runner
import plot_prefill_interference as analysis

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (2048, 4096, 1024)
CONDITIONS = ("baseline", "long")
SEED = 20261012
# Predeclared conservative guard: stop before block-pool capacity pressure.
MAX_KV_FRACTION = 0.8
MEASUREMENT_HALF_WINDOW_NS = 500_000_000
BACKGROUND_INTEGRITY_RULE = "window-coverage-v2"


def budget_plan(smoke=False):
    return (2048,) if smoke else BUDGETS


def output_directory(root, run_id, smoke=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    return Path(root) / (f"smoke-{run_id}" if smoke else run_id)


def write_json(path, value):
    """Atomic metadata update; strict JSON and explicit null for unknown fields."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("x", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require_free_port():
    # IPv4 wildcard bind also detects listeners that a loopback connect can miss.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", 8000))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError("Port 8000 is occupied; refusing to touch an unowned process") from exc
            raise


def session_processes(session_id):
    """Read Linux /proc; only signal our own newly created server session."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == session_id and fields[0] != "Z":
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return found


async def shutdown(process):
    stages = []
    for sig, timeout in ((signal.SIGINT, 45), (signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        process.poll()  # reap the session leader as soon as possible
        if process.returncode is not None and not session_processes(process.pid):
            break
        stages.append(sig.name)
        # Usually one group; also cover descendants creating their own group.
        for pid in session_processes(process.pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            if process.returncode is not None and not session_processes(process.pid):
                break
            await asyncio.sleep(0.2)
    process.poll()
    remaining = session_processes(process.pid)
    if process.returncode is None or remaining:
        raise RuntimeError(f"Server shutdown failed: live owned session PIDs={remaining}")
    deadline = time.monotonic() + 10
    while True:
        try:
            require_free_port()
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Server exited but port 8000 was not released")
            await asyncio.sleep(0.2)
    result = dict(signals=stages, returncode=process.returncode, remaining_pids=remaining,
                  port_8000_released=True, timestamp=utc_now())
    if not expected_shutdown_returncode(process.returncode, stages):
        raise RuntimeError(f"Abnormal server shutdown: {result}")
    return result


def expected_shutdown_returncode(returncode, signals):
    """uv/shell may encode SIGINT as 128+2 instead of subprocess's -2.

    Only accept the encoding of a signal WE sent, after process/port cleanup.
    A forced SIGKILL is always a lifecycle failure, regardless of exit code.
    """
    if "SIGKILL" in signals:
        return False
    allowed = {0}
    for name in signals:
        value = int(getattr(signal, name))
        allowed.update((-value, 128 + value))
    return returncode in allowed


async def wait_ready(process, timeout=300):
    deadline = time.monotonic() + timeout
    last_error = None
    async with httpx.AsyncClient(trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited during startup: code {process.returncode}")
            try:
                response = await client.get(runner.BASE_URL + "/models", timeout=3)
                response.raise_for_status()
                data = response.json()
                if runner.MODEL not in [model["id"] for model in data["data"]]:
                    raise RuntimeError(f"Unexpected served models: {data}")
                return data
            except (httpx.HTTPError, ValueError) as exc:
                last_error = str(exc)
            await asyncio.sleep(1)
    raise TimeoutError(f"Server readiness exceeded {timeout}s: {last_error}")


def startup_metadata(run_id, budget, pid, command, log_path, log_text, started):
    def match(pattern, convert=str):
        found = re.search(pattern, log_text)
        return convert(found[1]) if found else None

    observed_budget = match(r"['\"]max_num_batched_tokens['\"]:\s*(\d+)", int)
    return dict(
        run_id=run_id, budget=budget, model=runner.MODEL,
        vllm_version=match(r"Initializing a V1 LLM engine \(v([^)]*)\)"),
        installed_vllm_version=importlib.metadata.version("vllm"),
        max_model_len=match(r"Using max model len (\d+)", int),
        gpu_memory_utilization=match(r"['\"]gpu_memory_utilization['\"]:\s*([\d.]+)", float),
        requested_max_num_batched_tokens=budget, observed_max_num_batched_tokens=observed_budget,
        max_num_seqs=match(r"max_num_seqs[=:]\s*(\d+)", int),
        prefix_cache_enabled=match(r"enable_prefix_caching=(True|False)", lambda s: s == "True"),
        chunked_prefill_enabled=match(r"enable_chunked_prefill=(True|False)", lambda s: s == "True"),
        async_scheduling=match(r"async_scheduling=(True|False)", lambda s: s == "True"),
        long_prefill_token_threshold=match(r"long_prefill_token_threshold=(\d+)", int),
        max_num_scheduled_tokens=None,
        server_pid=pid, startup_timestamp=started, ready_timestamp=utc_now(),
        command_line=command, server_log_path=str(log_path),
        kv_cache_capacity_tokens=match(r"GPU KV cache size: ([\d,]+) tokens", lambda s: int(s.replace(",", ""))),
        available_kv_cache_memory_gib=match(r"Available KV cache memory: ([\d.]+) GiB", float),
        maximum_concurrency_4096=match(r"Maximum concurrency for 4,096 tokens per request: ([\d.]+)x", float),
        metadata_source="startup log (unknown fields remain null); command is the script invocation",
    )


def validate_trial(row, events, metrics):
    """Strict sweep checks atop the existing stream-level validation, plus raw replay."""
    if row["status"] != "success":
        raise ValueError(row["error"])
    summaries = json.loads(row["request_summaries"])
    bgs = [s for s in summaries if s["role"] == "background"]
    fgs = [s for s in summaries if s["role"] == "foreground"]
    if len(bgs) != 8 or {s["background_id"] for s in bgs} != set(range(8)):
        raise ValueError("Expected exactly 8 completed backgrounds")
    if any(s["completion_tokens"] is None or
           not 0 < s["completion_tokens"] <= runner.BACKGROUND_MAX_TOKENS for s in bgs):
        raise ValueError("Background completion count outside max_tokens upper bound")
    if len(fgs) != (row["condition"] == "long") or any(s["completion_tokens"] != 1 for s in fgs):
        raise ValueError("Foreground count/output != expected")
    if row["trigger_count"] != 1 or any(row[f"trigger_count_bg{i}"] < 64 for i in range(8)):
        raise ValueError("Invalid alignment trigger")
    window_start = row["alignment_ns"] - MEASUREMENT_HALF_WINDOW_NS
    window_end = row["alignment_ns"] + MEASUREMENT_HALF_WINDOW_NS
    for summary in bgs:
        if (summary["request_start_ns"] is None or summary["request_end_ns"] is None
                or summary["request_start_ns"] > window_start
                or summary["request_end_ns"] < window_end):
            raise ValueError(f"Background {summary['background_id']} lifecycle does not cover [-0.5,+0.5]s")
    grouped = defaultdict(list)
    for event in events:
        if any(event[key] != row[key] for key in runner.IDENTITY_COLUMNS):
            raise ValueError("Event identity mismatch")
        grouped[event["client_request_id"]].append(event)
    if set(grouped) != {s["client_request_id"] for s in summaries}:
        raise ValueError("Raw request set mismatch")
    for summary in summaries:
        group = grouped[summary["client_request_id"]]
        cumulative, previous_ns = 0, 0
        for index, event in enumerate(group):
            cumulative += len(event["token_ids"])
            if (event["event_index"] != index or event["cumulative_token_count"] != cumulative
                    or event["delta_token_count"] != len(event["token_ids"])
                    or event["received_ns"] < previous_ns
                    or event["alignment_ns"] != row["alignment_ns"]):
                raise ValueError("Raw event sequence/count/timestamp mismatch")
            previous_ns = event["received_ns"]
        if (not summary["valid"] or not group[-1]["has_usage"]
                or cumulative != summary["completion_tokens"]
                or group[-1]["usage_completion_tokens"] != cumulative
                or group[-1]["usage_prompt_tokens"] != summary["actual_prompt_tokens"]):
            raise ValueError("Final usage/ID/summary mismatch")
        if summary["role"] == "background":
            progress_at_alignment = sum(e["delta_token_count"] for e in group
                                        if e["received_ns"] <= row["alignment_ns"])
            if progress_at_alignment < runner.TRIGGER_TOKENS:
                raise ValueError(f"Background {summary['background_id']} raw trigger progress < 64")
            finishes = [e for e in group if e["finish_reason"] is not None]
            if (summary.get("error") or not finishes
                    or any(e["finish_reason"] not in {"stop", "length", "repetition"} for e in finishes)):
                raise ValueError(f"Background {summary['background_id']} did not complete successfully")
            # Stream closure/final usage can lag a natural stop. That lag must
            # never make a request ending inside the post window appear active.
            if min(e["received_ns"] for e in finishes) < window_end:
                raise ValueError(f"Background {summary['background_id']} finishes before post window ends")
    for key in ("prompt_tokens_local_cache_hit", "external_kv_transfer", "preemptions_total"):
        if row.get(key) != 0:
            raise ValueError(f"Integrity counter {key}={row.get(key)} (expected 0)")
    if row["generation_tokens_total"] != sum(s["completion_tokens"] for s in summaries):
        raise ValueError("Generation counter does not equal measured cohort")
    if row["prompt_tokens_total"] != sum(s["actual_prompt_tokens"] for s in summaries):
        raise ValueError("Prompt counter does not equal measured cohort")
    if not metrics:
        raise ValueError("No metrics samples")
    for key in ("num_requests_waiting", "waiting_capacity", "waiting_deferred"):
        if any(m[key] != 0 for m in metrics):
            raise ValueError(f"Observed {key}>0; workload/capacity anomaly")
    if max(m["kv_cache_usage_perc"] for m in metrics) >= MAX_KV_FRACTION:
        raise ValueError("KV block-pool usage >= predeclared 80% guard")
    if row["idle_running"] != 0 or row["idle_waiting"] != 0 or row["idle_kv_cache_usage_perc"] > 0.001:
        raise ValueError("Trial did not drain")


def analyze_extended(row, events, budget):
    result, gaps = analysis.analyze_trial(row, events)
    if not result["analysis_valid"]:
        raise ValueError(result["invalid_reason"])
    abnormal = analysis.stall_gaps(gaps)
    details = []
    for bg, group in abnormal.items():
        # Preserve old endpoint-based window membership; duration is the FULL
        # observed interval, including any portion crossing alignment.
        selected = [g for g in group if 0 <= g["relative_ms"] <= 500]
        details.append(dict(background_id=bg, significant_stall_count=len(selected),
                            significant_stall_total_duration_ms=sum(g["gap_ms"] for g in selected)))
    result.update(
        budget=budget,
        max_stall_gap_ms=result["post_peak_gap_ms"],
        cohort_max_gap_ms=max(g["gap_ms"] for group in gaps.values() for g in group if 0 <= g["relative_ms"] <= 500),
        stall_ratio=result["trial_stall_ratio"],
        significant_stall_count=statistics.median(d["significant_stall_count"] for d in details),
        significant_stall_total_duration_ms=statistics.median(d["significant_stall_total_duration_ms"] for d in details),
        foreground_ttft_ms=row["foreground_ttft_s"] * 1000 if row["foreground_ttft_s"] is not None else None,
        request_stall_summaries=json.dumps(details),
    )
    return result


async def run_clients(directory, run_id, budget, smoke):
    analyses = []
    schedule = runner.make_schedule(1 if smoke else 10, SEED, CONDITIONS)
    with ExitStack() as stack:
        files = [stack.enter_context((directory / name).open("x", newline=""))
                 for name in ("trials.csv", "events.jsonl", "metrics.csv")]
        trial_file, event_file, metric_file = files
        tw = csv.DictWriter(trial_file, fieldnames=[*runner.TRIAL_COLUMNS, "budget_label"])
        mw = csv.DictWriter(metric_file, fieldnames=runner.METRIC_COLUMNS)
        tw.writeheader()
        mw.writeheader()
        async with (AsyncOpenAI(base_url=runner.BASE_URL, api_key="dummy", max_retries=0, timeout=120) as client,
                    httpx.AsyncClient(trust_env=False) as http_client):
            warm = dict(started=utc_now(), protocol="C8, output=64; excluded")
            write_json(directory / "warmup.json", warm)
            try:
                warm["measurement"] = await runner.warmup(client, http_client)
                measurement = warm["measurement"]
                if not measurement["all_valid"] or measurement["completion_tokens"] != [64] * 8:
                    raise ValueError("C8 warmup completion/usage failure")
                for key in ("local_cache_hit_tokens", "external_kv_transfer_tokens", "preemptions"):
                    if measurement["counter_deltas"][key] != 0:
                        raise ValueError(f"Warmup integrity counter {key} != 0")
                warm.update(status="passed", completed=utc_now())
            except BaseException as exc:
                warm.update(status="failed", error=runner.error_text(exc))
                raise
            finally:
                write_json(directory / "warmup.json", warm)
            print(f"budget={budget}: C8 warmup passed; {len(schedule)} trials", flush=True)
            for index, (block, condition) in enumerate(schedule, 1):
                identity = dict(run_id=run_id, block_index=block, trial_index=index,
                                condition=condition, seed=SEED, smoke=smoke)
                row, events, metrics = await runner.run_trial(client, http_client, identity, 180)
                row["budget_label"] = budget  # Metadata only; never a server mutation.
                failure = None
                try:
                    validate_trial(row, events, metrics)
                    analyses.append(analyze_extended(row, events, budget))
                except Exception as exc:
                    failure = exc
                    row.update(status="failed", error=runner.error_text(exc))
                runner.write_trial(row, events, metrics, tw, event_file, mw)
                for output in files:
                    output.flush()
                print(f"budget={budget} trial={index}/{len(schedule)} block={block} {condition}: {row['status']} {row['error']}", flush=True)
                if failure:
                    raise failure
    # Verify files can be read back and analyze only persisted raw records.
    with (directory / "trials.csv").open() as source:
        persisted_trials = list(csv.DictReader(source))
    with (directory / "events.jsonl").open() as source:
        persisted_events = [json.loads(line) for line in source]
    if len(persisted_trials) != len(schedule):
        raise ValueError("Persisted trial count mismatch")
    for row, (block, condition), expected in zip(persisted_trials, schedule, analyses):
        if int(row["block_index"]) != block or row["condition"] != condition:
            raise ValueError("Persisted schedule mismatch")
        replay, _ = analysis.analyze_trial(row, [e for e in persisted_events if e["trial_index"] == int(row["trial_index"])])
        if not replay["analysis_valid"] or replay["trial_stall_ratio"] != expected["trial_stall_ratio"]:
            raise ValueError("Persisted event replay mismatch")
    write_csv(directory / "analysis.csv", analyses)
    return analyses


async def run_server_block(directory, run_id, budget, smoke):
    directory.mkdir()
    require_free_port()
    capture_environment(directory / "environment-start.txt")
    command = [str(ROOT / "scripts/serve_qwen3_0_6b.sh"), "--max-num-batched-tokens", str(budget)]
    log_path = directory / "server.log"
    metadata = dict(run_id=run_id, budget=budget, startup_timestamp=utc_now(),
                    command_line=command, server_log_path=str(log_path), status="starting")
    write_json(directory / "metadata.json", metadata)
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        metadata["server_pid"] = process.pid
        write_json(directory / "metadata.json", metadata)
        try:
            print(f"Starting budget={budget}, pid={process.pid}, log={log_path}", flush=True)
            models = await wait_ready(process)
            write_json(directory / "models.json", models)
            metadata.update(startup_metadata(run_id, budget, process.pid, command, log_path,
                                             log_path.read_text(), metadata["startup_timestamp"]))
            metadata["status"] = "ready"
            write_json(directory / "metadata.json", metadata)
            if (metadata["observed_max_num_batched_tokens"] != budget
                    or metadata["max_model_len"] != 4096
                    or metadata["gpu_memory_utilization"] != 0.75
                    or metadata["prefix_cache_enabled"] is not True
                    or metadata["chunked_prefill_enabled"] is not True):
                raise ValueError(f"Startup configuration sanity failed: {metadata}")
            rows = await run_clients(directory, run_id, budget, smoke)
            if process.poll() is not None:
                raise RuntimeError("Server exited unexpectedly during measurement")
            metadata["status"] = "validated"
        except BaseException as exc:
            metadata.update(status="failed", error=runner.error_text(exc))
            raise
        finally:
            try:
                metadata["shutdown"] = await shutdown(process)
                capture_environment(directory / "environment-end.txt")
                print(f"budget={budget}: server session exited and port 8000 released", flush=True)
            except BaseException as exc:
                metadata.update(status="failed", shutdown_error=runner.error_text(exc))
                raise
            finally:
                write_json(directory / "metadata.json", metadata)
    return rows


def capture_environment(path):
    lines = [f"timestamp={utc_now()}", f"platform={platform.platform()}", f"python={sys.version}",
             f"executable={sys.executable}"]
    for command in (["nvidia-smi"], ["git", "status", "--short"], ["git", "diff", "--stat"]):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
        lines.extend([f"$ {' '.join(command)} [exit={result.returncode}]", result.stdout, result.stderr])
    path.write_text("\n".join(lines))


async def run_phase(root, run_id, smoke):
    directory = output_directory(root, run_id, smoke)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_id=directory.name, smoke=smoke, started=utc_now(), status="running",
                    budgets=budget_plan(smoke), blocks_per_budget=1 if smoke else 10,
                    seed=SEED, conditions=CONDITIONS, max_kv_fraction_guard=MAX_KV_FRACTION,
                    background_integrity_rule=BACKGROUND_INTEGRITY_RULE,
                    schedule=runner.make_schedule(1 if smoke else 10, SEED, CONDITIONS))
    write_json(directory / "manifest.json", manifest)
    try:
        require_free_port()
        capture_environment(directory / "environment.txt")
        rows = []
        for budget in budget_plan(smoke):
            rows.extend(await run_server_block(directory / f"budget-{budget}", directory.name, budget, smoke))
        if not smoke:
            from plot_token_budget import summarize_and_plot
            summarize_and_plot(directory, rows)
        manifest.update(status="passed", completed=utc_now(), valid_trials=len(rows))
    except BaseException as exc:
        manifest.update(status="failed", error=runner.error_text(exc), stopped=utc_now())
        raise
    finally:
        write_json(directory / "manifest.json", manifest)
    return directory


async def main(args):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    smoke_path = await run_phase(args.output_root, run_id, True)
    print(f"SMOKE PASSED: {smoke_path}", flush=True)
    if not args.smoke_only:
        formal_path = await run_phase(args.output_root, run_id, False)
        print(f"FORMAL PASSED: {formal_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/token-budget-sweep")
    args = parser.parse_args()
    try:
        asyncio.run(main(args))
    except (Exception, KeyboardInterrupt) as exc:
        print("SWEEP STOPPED: " + runner.error_text(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
