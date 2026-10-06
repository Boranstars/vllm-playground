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
