"""Fixed-shape forward comparison against the repository's official Triton tutorial."""

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

os.environ.setdefault("MPLCONFIGDIR", "/tmp/attention-comparison-matplotlib")
import torch
import triton

OFFICIAL = Path(__file__).resolve().parents[1] / "triton_examples/06-fused-attention.py"
LENGTHS = (1024, 2048, 4096, 8192)
SCALE = 64**-0.5
MB = 1_000_000  # Decimal MB, not MiB.


def load_attention():
    # The tutorial already guards bench_flash_attention.run with __main__.
    # Keep the real filename/module available for Triton's source inspection.
    spec = importlib.util.spec_from_file_location("official_fused_attention", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.attention


def eager_attention(q, k, v, causal, sm_scale):
    """Match the FP16 reference path in the official tutorial's test_op."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        # Inclusive diagonal: position i can attend to j <= i.
        mask = torch.ones((q.shape[-2], q.shape[-2]), device=q.device,
                          dtype=torch.bool).triu_(1)
        scores.masked_fill_(mask, float("-inf"))
        del mask
    # Release the FP16 scores before allocating the FP32 softmax output.
    scores = scores.float()
    probabilities = torch.softmax(scores, dim=-1)
    del scores
    probabilities = probabilities.to(q.dtype)
    return torch.matmul(probabilities, v)


def latency_sample(fn):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    del output
    return elapsed


def memory_sample(fn):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    del output
    return baseline, peak, peak - baseline, reserved


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifacts(out, rows, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, metrics in (
        ("latency.png", [("latency_median_ms", "Forward latency (ms)")]),
        ("peak_memory.png", [("peak_allocated_mb", "Total peak allocated (MB)"),
                             ("extra_peak_mb", "Extra peak above baseline (MB)")]),
    ):
        fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 4 * len(metrics)), squeeze=False)
        for i, (metric, ylabel) in enumerate(metrics):
            for j, causal in enumerate((False, True)):
                ax = axes[i, j]
                for impl in ("eager", "triton"):
                    selected = [r for r in rows if r["causal"] == causal and r["implementation"] == impl]
                    ax.plot([r["n_ctx"] for r in selected], [r[metric] for r in selected], "o-", label=impl)
                if metric != "latency_median_ms":
                    selected = [r for r in rows if r["causal"] == causal and r["implementation"] == "eager"]
                    ax.plot([r["n_ctx"] for r in selected], [r["input_qkv_mb"] for r in selected],
                            "k--", label="Q/K/V bytes (context)")
                ax.set(title=f"Causal={causal}", xlabel="N_CTX", ylabel=ylabel, yscale="log")
                ax.set_xticks(LENGTHS, labels=[str(n) for n in LENGTHS])
                ax.grid(alpha=0.3)
                ax.legend()
        fig.suptitle(f"{metadata['gpu']} | B=1 H=8 D=64 FP16")
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)

    lines = ["# Attention Forward 对照实验", "",
             f"GPU: {metadata['gpu']}；PyTorch {torch.__version__}；Triton {triton.__version__}。",
             "固定 B=1、H=8、D=64、FP16；scale=0.125；输入为固定 seed 的 N(0, 0.5²)。",
             "两实现共享同一组连续 Q/K/V，causal 包含对角线；关闭梯度，仅测 forward。",
             "", "## 方法", "",
             "直接导入原始示例的 attention(..., warp_specialize=False)，保留官方 autotune；未修改 Kernel。",
             "Eager 显式执行 FP16 QKᵀ/scale、可选上三角 mask、FP32 Softmax、概率转 FP16、PV，输出 FP16。",
             "官方 Kernel 使用 FP32 累积和在线 exp2 Softmax，概率在 dot 前转低精度，输出写回 FP16；运算顺序不同，因此不要求逐位一致。",
             f"每实现每配置 warmup={metadata['warmup']}，首次编译/autotune 不计入测量；随后 {metadata['repeats']} 次 CUDA Event 单次 forward 计时，交替执行两实现，报告中位数和 P25/P75。",
             f"显存独立重复 {metadata['memory_repeats']} 次，每次同步、释放缓存、重置峰值计数，保留输出至读取峰值；计时阶段不清缓存。",
             "记录 Q/K/V 理论字节、执行前 memory_allocated、max_memory_allocated 和两者差值（包括输出/临时量/官方 M）；单位为十进制 MB。",
             "max_memory_reserved 另存 CSV。PyTorch 分配器统计不包括 CUDA context、库内部非分配器分配或其他进程，不能等同 nvidia-smi 总占用。",
             "正确性在测量外执行，检查 dtype、有限值、max/mean absolute error、RMSE；采用官方 FP16 阈值 atol=1e-2、rtol=0。",
             "", "## 结果", "",
             "|Causal|N_CTX|Eager ms|Triton ms|加速比|Eager 额外 MB|Triton 额外 MB|最大绝对误差|通过|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for causal in (False, True):
        for n in LENGTHS:
            a, b = [next(r for r in rows if r['causal'] == causal and r['n_ctx'] == n and r['implementation'] == impl)
                    for impl in ('eager', 'triton')]
            lines.append(f"|{causal}|{n}|{a['latency_median_ms']:.4f}|{b['latency_median_ms']:.4f}|"
                         f"{a['latency_median_ms']/b['latency_median_ms']:.2f}×|{a['extra_peak_mb']:.3f}|"
                         f"{b['extra_peak_mb']:.3f}|{b['max_abs_error']:.6g}|{b['correct']}|")
    lines += ["", "Eager 物化 N×N 分数与概率矩阵，额外显存随序列长度近似平方增长；Triton 分块在线计算，避免保存完整注意力矩阵。",
              "这是本 GPU、输入分布及固定配置的一次实验；桌面 GPU 的其他进程、功耗与温度可能影响延迟。未扩展到 backward、其他 dtype 或其他实现。",
              "", "原始逐次数据：samples.csv；汇总：results.csv；环境及源码 SHA256：metadata.json。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "results/flash_attention/eager_vs_triton")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.memory_repeats) < 1:
        parser.error("measurement counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; no benchmark results were fabricated")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    attention = load_attention()
    props = torch.cuda.get_device_properties(0)
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), gpu=props.name,
                    gpu_total_mb=props.total_memory / MB, compute_capability=torch.cuda.get_device_capability(),
                    pytorch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                    python=sys.version, batch=1, heads=8, head_dim=64, dtype="float16", n_ctx=LENGTHS,
                    causal=[False, True], sm_scale=SCALE, warp_specialize=False, seed=args.seed,
                    input_distribution="normal(mean=0,std=0.5)", warmup=args.warmup,
                    repeats=args.repeats, memory_repeats=args.memory_repeats, atol=1e-2, rtol=0,
                    allow_tf32=False, allow_fp16_reduced_precision_reduction=False,
                    official_source=str(OFFICIAL), official_sha256=hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, samples = [], []
    for n in LENGTHS:
        q, k, v = [torch.empty((1, 8, n, 64), device="cuda", dtype=torch.float16).normal_(0, 0.5)
                   for _ in range(3)]
        for causal in (False, True):
            fns = dict(eager=lambda: eager_attention(q, k, v, causal, SCALE),
                       triton=lambda: attention(q, k, v, causal, SCALE, False))
            for fn in fns.values():
                for _ in range(args.warmup):
                    result = fn()
                    del result
            torch.cuda.synchronize()
            # Move to CPU so neither correctness output contaminates GPU memory baseline.
            reference = fns['eager']().cpu()
            actual = fns['triton']().cpu()
            error = actual.float() - reference.float()
            correct = bool(actual.dtype == reference.dtype == torch.float16
                           and torch.isfinite(actual).all() and torch.isfinite(reference).all()
                           and torch.allclose(actual, reference, atol=1e-2, rtol=0))
            errors = dict(correct=correct, max_abs_error=error.abs().max().item(),
                          mean_abs_error=error.abs().mean().item(), rmse=error.square().mean().sqrt().item())
            times, memories = {key: [] for key in fns}, {key: [] for key in fns}
            common = dict(n_ctx=n, causal=causal, batch=1, heads=8, head_dim=64, dtype="float16",
                          sm_scale=SCALE, gpu=props.name, pytorch=torch.__version__, triton=triton.__version__)
            for i in range(args.repeats):
                for key in list(fns)[::1 if i % 2 == 0 else -1]:
                    value = latency_sample(fns[key])
                    times[key].append(value)
                    samples.append(dict(**common, implementation=key, phase="latency", repeat=i,
                                        latency_ms=value, baseline_mb="", peak_allocated_mb="", extra_peak_mb="", peak_reserved_mb=""))
            for i in range(args.memory_repeats):
                for key in list(fns)[::1 if i % 2 == 0 else -1]:
                    values = memory_sample(fns[key])
                    memories[key].append(values)
                    samples.append(dict(**common, implementation=key, phase="memory", repeat=i, latency_ms="",
                                        **dict(zip(("baseline_mb", "peak_allocated_mb", "extra_peak_mb", "peak_reserved_mb"),
                                                   (value / MB for value in values)))))
            for key in fns:
                quantiles = torch.tensor(times[key]).quantile(torch.tensor([0.25, 0.75])).tolist()
                baseline, peak, extra, reserved = max(memories[key], key=lambda item: item[1])
                rows.append(dict(**common, implementation=key, input_qkv_mb=sum(t.numel()*t.element_size() for t in (q,k,v))/MB,
                                 baseline_mb=baseline/MB, peak_allocated_mb=peak/MB, extra_peak_mb=extra/MB,
                                 peak_reserved_mb=reserved/MB, latency_median_ms=statistics.median(times[key]),
                                 latency_p25_ms=quantiles[0], latency_p75_ms=quantiles[1], **errors))
            write_csv(out / "results.csv", rows)
            write_csv(out / "samples.csv", samples)
            print(f"N={n} causal={causal}: correct={correct}, max_error={errors['max_abs_error']:.6g}, "
                  f"eager={statistics.median(times['eager']):.4f}ms triton={statistics.median(times['triton']):.4f}ms", flush=True)
            del fns
        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()
    artifacts(out, rows, metadata)
    if not all(r['correct'] for r in rows):
        raise AssertionError("Numerical check failed; see results.csv")


if __name__ == "__main__":
    main()
