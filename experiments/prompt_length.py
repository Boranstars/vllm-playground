import csv
import random
import statistics
import uuid
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from common import measure_streaming_request


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

model = "Qwen/Qwen3-0.6B"
max_tokens = 64
case_sizes = [("~32", 32), ("~128", 128), ("~512", 512), ("~1024", 1024), ("~2048", 2048)]
repetitions = 5
run_id = uuid.uuid4().hex
results_dir = Path(__file__).resolve().parents[1] / "results"

filler_words = """amber ancient autumn blue bright calm cedar clear copper cool coral cozy crimson crystal dawn distant eager early east emerald empty evening faint fair far fast fertile final fine fresh gentle golden green hidden high indigo inner ivory jade kind large late lilac lively lonely long loose lovely low lucky maple mellow mild modern mossy narrow natural near neat new noble north oak ocean open orange outside pale peaceful pearl plain quiet rapid rare red remote rich ripe river rosy round royal rural safe scarlet secret serene sharp shining silver simple slow small smooth soft solar solid south spacious spring steady still stone strong summer sunny sweet swift tender thin tidy tiny tranquil true velvet vivid warm western white wide wild willow wise wooden yellow young""".split()

# 一次不计入统计的短请求用于建立连接并消除首次客户端开销。
warmup_prompt = f"warmup-{uuid.uuid4().hex} 请简单问候。"
measure_streaming_request(client, model, warmup_prompt, max_tokens)

# 请求 ID 位于 prompt 开头，run ID 紧随其后；随机填充也降低前缀缓存复用。
requests = []
for target_size, target_tokens in case_sizes:
    filler_count = max(0, target_tokens - 32)
    for repetition in range(1, repetitions + 1):
        request_id = uuid.uuid4().hex
        filler = " ".join(random.choices(filler_words, k=filler_count))
        prompt = (
            f"request-{request_id} run-{run_id} "
            f"请用一句话解释什么是 KV Cache。\n{filler}"
        )
        requests.append((target_size, repetition, request_id, prompt))

# 随机顺序运行，减少 prompt 长度与运行时状态变化的相关性。
random.shuffle(requests)
print("实际执行顺序（序号、长度、重复编号、请求 ID）")
for execution_order, (target_size, repetition, request_id, _) in enumerate(requests, start=1):
    print(f"{execution_order:02d}. {target_size} rep={repetition} id={request_id[:8]}")

results = []
for execution_order, (target_size, repetition, request_id, prompt) in enumerate(requests, start=1):
    measurement = measure_streaming_request(client, model, prompt, max_tokens)
    results.append(
        {
            "target_size": target_size,
            "repetition": repetition,
            "request_id": request_id,
            "execution_order": execution_order,
            "actual_prompt_tokens": measurement["prompt_tokens"],
            "completion_tokens": measurement["completion_tokens"],
            "finish_reason": measurement["finish_reason"],
            "ttft_s": measurement["ttft_s"],
            "e2e_s": measurement["e2e_s"],
            "tpot_s_per_token": measurement["tpot_s_per_token"],
            "decode_tokens_per_s": measurement["decode_tokens_per_s"],
        }
    )


def show(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


print("\nA. 原始测量结果（TPOT / 解码吞吐量为客户端观测近似值）")
print(
    f"{'#':>2} {'目标':>6} {'rep':>3} {'请求ID':>8} {'prompt':>7} {'completion':>10} "
    f"{'finish':>8} {'TTFT(s)':>9} {'E2E(s)':>9} {'TPOT(s/token)':>14} {'解码(tokens/s)':>15}"
)
for result in results:
    print(
        f"{result['execution_order']:>2} {result['target_size']:>6} {result['repetition']:>3} "
        f"{result['request_id'][:8]:>8} {show(result['actual_prompt_tokens'], 0):>7} "
        f"{show(result['completion_tokens'], 0):>10} {str(result['finish_reason'] or '—'):>8} "
        f"{show(result['ttft_s']):>9} {show(result['e2e_s']):>9} "
        f"{show(result['tpot_s_per_token']):>14} {show(result['decode_tokens_per_s']):>15}"
    )

# 用中位数汇总，降低单次调度与网络波动的影响。
print("\nB. 按目标 prompt 长度汇总（中位数；TTFT 同时给出最小值和最大值）")
print(
    f"{'目标':>6} {'n':>2} {'prompt 中位数':>13} {'TTFT 中位数':>12} "
    f"{'TTFT 最小':>10} {'TTFT 最大':>10} {'E2E 中位数':>12} "
    f"{'TPOT 中位数':>13} {'解码吞吐中位数':>15}"
)
for target_size, _ in case_sizes:
    group = [result for result in results if result["target_size"] == target_size]

    def median_of(key):
        values = [result[key] for result in group if result[key] is not None]
        return statistics.median(values) if values else None

    ttfts = [result["ttft_s"] for result in group if result["ttft_s"] is not None]
    print(
        f"{target_size:>6} {len(group):>2} {show(median_of('actual_prompt_tokens'), 0):>13} "
        f"{show(median_of('ttft_s')):>12} {show(min(ttfts) if ttfts else None):>10} "
        f"{show(max(ttfts) if ttfts else None):>10} {show(median_of('e2e_s')):>12} "
        f"{show(median_of('tpot_s_per_token')):>13} "
        f"{show(median_of('decode_tokens_per_s')):>15}"
    )

columns = [
    "target_size",
    "repetition",
    "request_id",
    "execution_order",
    "actual_prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "ttft_s",
    "e2e_s",
    "tpot_s_per_token",
    "decode_tokens_per_s",
    "model",
    "max_tokens",
]
results_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = results_dir / f"prompt_length-{timestamp}.csv"
suffix = 1
while csv_path.exists():
    csv_path = results_dir / f"prompt_length-{timestamp}-{suffix:02d}.csv"
    suffix += 1

with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    for result in results:
        writer.writerow({**result, "model": model, "max_tokens": max_tokens})

print(f"\nCSV 已保存：{csv_path}")
