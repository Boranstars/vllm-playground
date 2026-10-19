import json
import math
import re
import time


def measure_streaming_request(client, model, prompt, max_tokens):
    """Measure one streaming chat completion using client-observed timestamps."""
    first_nonempty_content_time = None
    prompt_tokens = None
    completion_tokens = None
    finish_reason = None

    request_start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta.content and first_nonempty_content_time is None:
                first_nonempty_content_time = time.perf_counter()
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    stream_end = time.perf_counter()
    ttft_s = (
        None
        if first_nonempty_content_time is None
        else first_nonempty_content_time - request_start
    )
    generation_duration_s = (
        None
        if first_nonempty_content_time is None
        else stream_end - first_nonempty_content_time
    )

    tpot_s_per_token = None
    decode_tokens_per_s = None
    if completion_tokens is not None and completion_tokens > 1 and generation_duration_s is not None:
        tpot_s_per_token = generation_duration_s / (completion_tokens - 1)
        if generation_duration_s > 0:
            decode_tokens_per_s = (completion_tokens - 1) / generation_duration_s

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_s": ttft_s,
        "e2e_s": stream_end - request_start,
        "generation_duration_s": generation_duration_s,
        "tpot_s_per_token": tpot_s_per_token,
        "decode_tokens_per_s": decode_tokens_per_s,
    }


async def measure_streaming_request_async(client, model, prompt, max_tokens):
    """Async counterpart with the same timing semantics, plus absolute boundaries."""
    first_nonempty_content_time = None
    prompt_tokens = None
    completion_tokens = None
    finish_reason = None

    request_start = time.perf_counter()
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    try:
        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.content and first_nonempty_content_time is None:
                    first_nonempty_content_time = time.perf_counter()
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
        stream_end = time.perf_counter()
    finally:
        await stream.close()

    ttft_s = (
        None if first_nonempty_content_time is None
        else first_nonempty_content_time - request_start
    )
    generation_duration_s = (
        None if first_nonempty_content_time is None
        else stream_end - first_nonempty_content_time
    )
    tpot_s_per_token = None
    decode_tokens_per_s = None
    if completion_tokens is not None and completion_tokens > 1 and generation_duration_s is not None:
        tpot_s_per_token = generation_duration_s / (completion_tokens - 1)
        if generation_duration_s > 0:
            decode_tokens_per_s = (completion_tokens - 1) / generation_duration_s

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_s": ttft_s,
        "e2e_s": stream_end - request_start,
        "generation_duration_s": generation_duration_s,
        "tpot_s_per_token": tpot_s_per_token,
        "decode_tokens_per_s": decode_tokens_per_s,
        "request_start": request_start,
        "stream_end": stream_end,
    }


FINISHED_REASONS = ("stop", "length", "abort", "error", "repetition")
GAUGE_SELECTORS = {
    "running": ("vllm:num_requests_running", {}),
    "waiting": ("vllm:num_requests_waiting", {}),
    "waiting_capacity": ("vllm:num_requests_waiting_by_reason", {"reason": "capacity"}),
    "waiting_deferred": ("vllm:num_requests_waiting_by_reason", {"reason": "deferred"}),
    "kv_cache_usage_perc": ("vllm:kv_cache_usage_perc", {}),
}
COUNTER_SELECTORS = {
    "generation_tokens": ("vllm:generation_tokens_total", {}),
    "prompt_tokens": ("vllm:prompt_tokens_total", {}),
    "local_compute_prompt_tokens": (
        "vllm:prompt_tokens_by_source_total", {"source": "local_compute"}
    ),
    "local_cache_hit_tokens": (
        "vllm:prompt_tokens_by_source_total", {"source": "local_cache_hit"}
    ),
    "external_kv_transfer_tokens": (
        "vllm:prompt_tokens_by_source_total", {"source": "external_kv_transfer"}
    ),
    "preemptions": ("vllm:num_preemptions_total", {}),
}
_SAMPLE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)\{(.*)\}\s+(\S+)(?:\s+\S+)?')
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\[\\"n])*)"\s*(?:,|$)')


def parse_vllm_metrics(text, model, engine="0"):
    """Resolve only required vLLM text samples; never guess missing series as zero."""
    required_names = {
        name for name, _ in (*GAUGE_SELECTORS.values(), *COUNTER_SELECTORS.values())
    } | {"vllm:request_success_total"}
    samples = []
    for line in text.splitlines():
        line = line.strip()
        name = re.split(r"[\s{]", line, maxsplit=1)[0]
        if name not in required_names:
            continue
        match = _SAMPLE.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed required metrics sample: {line}")
        labels = {}
        position = 0
        label_text = match[2]
        while position < len(label_text):
            label_match = _LABEL.match(label_text, position)
            if label_match is None or label_match[1] in labels:
                raise ValueError(f"Malformed/duplicate metric labels: {line}")
            labels[label_match[1]] = json.loads('"' + label_match[2] + '"')
            position = label_match.end()
        if labels.get("model_name") != model or labels.get("engine") != engine:
            continue
        value = float(match[3])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid metrics value: {line}")
        samples.append((name, labels, value))

    def resolve(name, selector):
        matches = [
            value for sample_name, labels, value in samples
            if sample_name == name and all(labels.get(k) == v for k, v in selector.items())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {name} sample for model_name={model!r}, "
                f"engine={engine!r}, {selector}; found {len(matches)}"
            )
        return matches[0]

    gauges = {key: resolve(name, labels) for key, (name, labels) in GAUGE_SELECTORS.items()}
    counters = {key: resolve(name, labels) for key, (name, labels) in COUNTER_SELECTORS.items()}
    reasons = set(FINISHED_REASONS)
    for name, labels, _ in samples:
        if name == "vllm:request_success_total":
            if not labels.get("finished_reason"):
                raise ValueError("request_success_total lacks finished_reason")
            reasons.add(labels["finished_reason"])
    for reason in sorted(reasons):
        counters[f"request_success:{reason}"] = resolve(
            "vllm:request_success_total", {"finished_reason": reason}
        )
    return {"counters": counters, "gauges": gauges}


async def get_metrics_snapshot(http_client, url, model, engine="0"):
    """Timestamp a fully received successful scrape on the request timing clock."""
    try:
        response = await http_client.get(url, timeout=5.0)
        response.raise_for_status()
        timestamp = time.perf_counter()
        snapshot = parse_vllm_metrics(response.text, model, engine)
    except Exception as exc:
        raise RuntimeError(f"Metrics snapshot failed at {url}: {exc}") from exc
    return {"timestamp": timestamp, **snapshot}
