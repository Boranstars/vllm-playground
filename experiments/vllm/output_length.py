import csv
import random
import statistics
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from common import measure_streaming_request


client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

model = "Qwen/Qwen3-0.6B"
max_token_cases = [16, 64, 128, 256, 512]
repetitions = 5
results_dir = Path(__file__).resolve().parents[1] / "results"

fixed_instruction = "请基于以下材料，持续生成连贯、详细的说明和例子，尽可能达到允许的最大输出长度。"
fixed_paragraph = (
    "A language model processes a sequence one token at a time. For each token, "
    "it projects hidden states into keys and values for every attention layer. "
    "These tensors represent information from the preceding context. Keeping them "
    "in memory lets later decoding steps attend to earlier tokens without "
    "recomputing their projections. The cache grows as new tokens are generated, "
    "so memory use depends on sequence length, layer count, head dimensions, and "
    "the chosen data type."
)
# 固定重复同一段材料，得到约 512 tokens 的基准 prompt；所有测量请求共用此 body。
fixed_base_body = "\n".join([fixed_paragraph] * 6)

# 先构造全部请求；32 位十六进制 UUID 长度固定，并置于共同指令和 body 之前。
requests = []
for target_max_tokens in max_token_cases:
    for repetition in range(1, repetitions + 1):
        request_id = uuid.uuid4().hex
        prompt = f"request-{request_id}\n{fixed_instruction}\n{fixed_base_body}"
        requests.append((target_max_tokens, repetition, request_id, prompt))

# 随机顺序运行，避免输出长度组与运行时状态变化相关联。
random.shuffle(requests)
print("实际执行顺序（序号、max_tokens、重复编号、请求 ID）")
for execution_order, (target, repetition, request_id, _) in enumerate(requests, start=1):
    print(f"{execution_order:02d}. max_tokens={target} rep={repetition} id={request_id[:8]}")

# 短 warmup 建立连接并消除首次客户端开销，不加入测量数据。
warmup_prompt = f"warmup-{uuid.uuid4().hex} 请简单问候。"
measure_streaming_request(client, model, warmup_prompt, max_token_cases[0])

results = []
for execution_order, (target, repetition, request_id, prompt) in enumerate(requests, start=1):
    measurement = measure_streaming_request(client, model, prompt, target)
    results.append(
        {
            "target_max_tokens": target,
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
    f"{'#':>2} {'max_tokens':>10} {'rep':>3} {'请求ID':>8} {'prompt':>7} "
    f"{'completion':>10} {'finish':>8} {'TTFT(s)':>9} {'E2E(s)':>9} "
    f"{'TPOT(s/token)':>14} {'解码(tokens/s)':>15}"
)
for result in results:
    print(
        f"{result['execution_order']:>2} {result['target_max_tokens']:>10} "
        f"{result['repetition']:>3} {result['request_id'][:8]:>8} "
        f"{show(result['actual_prompt_tokens'], 0):>7} "
        f"{show(result['completion_tokens'], 0):>10} "
        f"{str(result['finish_reason'] or '—'):>8} "
        f"{show(result['ttft_s']):>9} {show(result['e2e_s']):>9} "
        f"{show(result['tpot_s_per_token']):>14} "
        f"{show(result['decode_tokens_per_s']):>15}"
    )

print("\nB. 按 target max_tokens 汇总")
print(
    f"{'max_tokens':>10} {'n':>2} {'prompt 中位数':>13} {'prompt 范围':>13} "
    f"{'completion 中位数':>16} {'finish_reason':>22} {'TTFT 中位数':>12} "
    f"{'TTFT 最小':>10} {'TTFT 最大':>10} {'E2E 中位数':>12} "
    f"{'TPOT 中位数':>13} {'解码吞吐中位数':>15}"
)
for target in max_token_cases:
    group = [result for result in results if result["target_max_tokens"] == target]

    def median_of(key):
        values = [result[key] for result in group if result[key] is not None]
        return statistics.median(values) if values else None

    prompt_values = [result["actual_prompt_tokens"] for result in group]
    ttfts = [result["ttft_s"] for result in group if result["ttft_s"] is not None]
    finish_counts = Counter(result["finish_reason"] or "unknown" for result in group)
    finish_summary = ", ".join(f"{reason}:{count}" for reason, count in sorted(finish_counts.items()))
    prompt_range = f"{min(prompt_values)}–{max(prompt_values)}" if prompt_values else "—"

    print(
        f"{target:>10} {len(group):>2} {show(median_of('actual_prompt_tokens'), 0):>13} "
        f"{prompt_range:>13} {show(median_of('completion_tokens'), 0):>16} "
        f"{finish_summary:>22} {show(median_of('ttft_s')):>12} "
        f"{show(min(ttfts) if ttfts else None):>10} "
        f"{show(max(ttfts) if ttfts else None):>10} "
        f"{show(median_of('e2e_s')):>12} "
        f"{show(median_of('tpot_s_per_token')):>13} "
        f"{show(median_of('decode_tokens_per_s')):>15}"
    )

columns = [
    "target_max_tokens",
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
]
results_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csv_path = results_dir / f"output_length-{timestamp}.csv"
suffix = 1
while csv_path.exists():
    csv_path = results_dir / f"output_length-{timestamp}-{suffix:02d}.csv"
    suffix += 1

with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()
    for result in results:
        writer.writerow({**result, "model": model})

print(f"\nCSV 已保存：{csv_path}")
