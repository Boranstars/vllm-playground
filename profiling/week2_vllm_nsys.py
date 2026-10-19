import torch
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,  # 避免 warmup 缓存缩短正式请求的 Prefill。
        profiler_config={"profiler": "cuda"},
    )
    prompt = (
        "Explain in simple terms how a language model first processes an input "
        "prompt and then generates its answer one token at a time, using a short example."
    )
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)

    # Warmup：用完全相同的请求预热，减少首次执行开销对正式区间的影响。
    llm.generate(prompt, sampling, use_tqdm=False)

    llm.start_profile()
    try:
        # NVTX：只标记第二次同步 generate 调用，便于在 timeline 中定位。
        torch.cuda.nvtx.range_push("week2_vllm_generate")
        try:
            outputs = llm.generate(prompt, sampling, use_tqdm=False)
        finally:
            torch.cuda.nvtx.range_pop()
    finally:
        llm.stop_profile()

    result = outputs[0]
    completion = result.outputs[0]
    assert len(completion.token_ids) == 16
    print(f"prompt_tokens={len(result.prompt_token_ids)}, output_tokens={len(completion.token_ids)}")
    print(repr(completion.text))


if __name__ == "__main__":
    main()
