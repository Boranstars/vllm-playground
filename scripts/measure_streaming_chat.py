import random
import statistics
import time
import uuid

from openai import OpenAI


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

model = "Qwen/Qwen3-0.6B"
case_sizes = [("~32", 32), ("~128", 128), ("~512", 512), ("~1024", 1024), ("~2048", 2048)]
repetitions = 5
run_id = uuid.uuid4().hex

filler_words = """amber ancient autumn blue bright calm cedar clear copper cool coral cozy crimson crystal dawn distant eager early east emerald empty evening faint fair far fast fertile final fine fresh gentle golden green hidden high indigo inner ivory jade kind large late lilac lively lonely long loose lovely low lucky maple mellow mild modern mossy narrow natural near neat new noble north oak ocean open orange outside pale peaceful pearl plain quiet rapid rare red remote rich ripe river rosy round royal rural safe scarlet secret serene sharp shining silver simple slow small smooth soft solar solid south spacious spring steady still stone strong summer sunny sweet swift tender thin tidy tiny tranquil true velvet vivid warm western white wide wild willow wise wooden yellow young""".split()

results = []

# 一次不计入统计的短请求先建立连接并经过首次客户端运行开销。
warmup_prompt = f"warmup-{uuid.uuid4().hex} 请简单问候。"
warmup_stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": warmup_prompt}],
    temperature=0,
    max_tokens=64,
    stream=True,
    stream_options={"include_usage": True},
)
for _ in warmup_stream:
    pass

# 每个 prompt 都在开头放入本次运行和本请求的 UUID；指令放在标识之后，
# 并为每个请求随机生成填充词，避免相同 prompt 的前缀 KV 块被复用。
requests = []
for size_label, target_tokens in case_sizes:
    filler_count = max(0, target_tokens - 32)
    for repetition in range(1, repetitions + 1):
        request_id = uuid.uuid4().hex
        filler = " ".join(random.choices(filler_words, k=filler_count))
        prompt = (
            f"request-{request_id} run-{run_id} "
            f"请用一句话解释什么是 KV Cache。\n{filler}"
        )
        requests.append((size_label, repetition, request_id, prompt))

# 随机执行顺序，减少长度与服务器逐渐变热/变忙之间的系统性关联。
random.shuffle(requests)
print("实际执行顺序（序号、长度、重复编号、请求 ID）")
for order, (size_label, repetition, request_id, _) in enumerate(requests, start=1):
    print(f"{order:02d}. {size_label} rep={repetition} id={request_id[:8]}")

for order, (size_label, repetition, request_id, prompt) in enumerate(requests, start=1):
    first_nonempty_content_time = None
    prompt_tokens = None
    completion_tokens = None

    request_start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            # 忽略只有 role 或空 content 的初始 SSE 块，以首段非空文本作为 TTFT 终点。
            if content and first_nonempty_content_time is None:
                first_nonempty_content_time = time.perf_counter()

        # token 数来自最终 usage，而不是 SSE 块数。
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    stream_end = time.perf_counter()
    end_to_end = stream_end - request_start
    generation_duration = (
        None
        if first_nonempty_content_time is None
        else stream_end - first_nonempty_content_time
    )
    ttft = (
        None
        if first_nonempty_content_time is None
        else first_nonempty_content_time - request_start
    )

    tpot = None
    throughput = None
    if completion_tokens is not None and generation_duration is not None:
        if completion_tokens > 1:
            tpot = generation_duration / (completion_tokens - 1)
            # 解码吞吐量与 TPOT 使用相同 token 区间，二者互为倒数。
            if generation_duration > 0:
                throughput = (completion_tokens - 1) / generation_duration

    results.append(
        {
            "order": order,
            "size": size_label,
            "repetition": repetition,
            "request_id": request_id[:8],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft": ttft,
            "end_to_end": end_to_end,
            "tpot": tpot,
            "throughput": throughput,
        }
    )


def show(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


print("\nA. 原始测量结果（TPOT / 解码吞吐量为客户端观测近似值）")
print(
    f"{'#':>2} {'目标':>6} {'rep':>3} {'请求ID':>8} {'prompt':>7} {'completion':>10} "
    f"{'TTFT(s)':>9} {'E2E(s)':>9} {'TPOT(s/token)':>14} {'解码(tokens/s)':>15}"
)
for result in results:
    print(
        f"{result['order']:>2} {result['size']:>6} {result['repetition']:>3} "
        f"{result['request_id']:>8} {show(result['prompt_tokens'], 0):>7} "
        f"{show(result['completion_tokens'], 0):>10} {show(result['ttft']):>9} "
        f"{show(result['end_to_end']):>9} {show(result['tpot']):>14} "
        f"{show(result['throughput']):>15}"
    )

# median 可降低单次调度、网络和运行时波动对结论的影响。
print("\nB. 按目标 prompt 长度汇总（中位数；TTFT 同时给出最小值和最大值）")
print(
    f"{'目标':>6} {'n':>2} {'prompt 中位数':>13} {'TTFT 中位数':>12} "
    f"{'TTFT 最小':>10} {'TTFT 最大':>10} {'E2E 中位数':>12} "
    f"{'TPOT 中位数':>13} {'解码吞吐中位数':>15}"
)
for size_label, _ in case_sizes:
    group = [result for result in results if result["size"] == size_label]

    def median_of(key):
        values = [result[key] for result in group if result[key] is not None]
        return statistics.median(values) if values else None

    ttfts = [result["ttft"] for result in group if result["ttft"] is not None]
    print(
        f"{size_label:>6} {len(group):>2} {show(median_of('prompt_tokens'), 0):>13} "
        f"{show(median_of('ttft')):>12} {show(min(ttfts) if ttfts else None):>10} "
        f"{show(max(ttfts) if ttfts else None):>10} {show(median_of('end_to_end')):>12} "
        f"{show(median_of('tpot')):>13} {show(median_of('throughput')):>15}"
    )
