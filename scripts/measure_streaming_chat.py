import time

from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

# 每段填充文本约含 8 个英文 token；重复次数只用于近似目标长度。
# 每个案例在开头使用不同标记和填充内容，避免 prompt 互为简单前缀。
cases = [
    ("~32", "CASE-32-ORCHID", "orchid violet garden blooms softly beside quiet rivers", 2),
    ("~128", "CASE-128-MAPLE", "maple leaves drift gently across the silver morning meadow", 14),
    ("~512", "CASE-512-COPPER", "copper lanterns glow warmly along the winding harbor walkway", 62),
    ("~1024", "CASE-1024-CEDAR", "cedar branches sway slowly above the distant mountain footpath", 126),
    ("~2048", "CASE-2048-INDIGO", "indigo clouds gather quietly beyond the wide northern coastline", 254),
]

results = []

for label, marker, filler, repetitions in cases:
    prompt = (
        f"{marker}. Read the following words, then explain KV Cache in one sentence.\n"
        + (filler + " ") * repetitions
    )

    request_start = time.perf_counter()
    first_nonempty_content_time = None
    generated_text = ""
    prompt_tokens = None
    completion_tokens = None

    # request_start 到首段非空生成文本之间的时间就是客户端观测 TTFT。
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        # 初始 SSE 块可能只有 role 或空 content；它不代表首个生成 token。
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                if first_nonempty_content_time is None:
                    first_nonempty_content_time = time.perf_counter()
                generated_text += content

        # 读取 vLLM 最终 usage；SSE 块数不等于 tokenizer token 数。
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    # stream_end 是客户端读完整个响应流的时刻。
    stream_end = time.perf_counter()
    end_to_end = stream_end - request_start

    if first_nonempty_content_time is None:
        ttft = None
        generation_duration = None
    else:
        ttft = first_nonempty_content_time - request_start
        generation_duration = stream_end - first_nonempty_content_time

    # TPOT / 吞吐量都是客户端观测近似值，使用服务端返回的 token 数。
    tpot = None
    if completion_tokens is not None and completion_tokens > 1 and generation_duration is not None:
        tpot = generation_duration / (completion_tokens - 1)

    throughput = None
    if completion_tokens is not None and generation_duration is not None and generation_duration > 0:
        throughput = completion_tokens / generation_duration

    results.append(
        {
            "case": label,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft": ttft,
            "end_to_end": end_to_end,
            "tpot": tpot,
            "throughput": throughput,
            "response": generated_text[:50].replace("\n", " "),
        }
    )


def format_value(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


print("\n客户端观测结果（TPOT 和输出吞吐量为近似值）")
print(
    f"{'案例':<8} {'prompt_tokens':>13} {'completion_tokens':>16} "
    f"{'TTFT(s)':>10} {'端到端(s)':>12} {'TPOT(s/token)':>15} {'吞吐(tokens/s)':>16}"
)
for result in results:
    print(
        f"{result['case']:<8} {format_value(result['prompt_tokens'], 0):>13} "
        f"{format_value(result['completion_tokens'], 0):>16} "
        f"{format_value(result['ttft']):>10} {format_value(result['end_to_end']):>12} "
        f"{format_value(result['tpot']):>15} {format_value(result['throughput']):>16}"
    )

print("\n生成文本片段（前约 50 个字符）")
for result in results:
    print(f"{result['case']}: {result['response']}")
