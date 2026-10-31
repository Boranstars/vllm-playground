# PyTorch Eager vs 官方 Triton FlashAttention Forward

从项目根目录运行（需要可访问的 CUDA GPU，以及项目环境中的 PyTorch、Triton、pytest、matplotlib）：

```bash
.venv/bin/python experiments/triton/attention_comparison/compare.py
```

仅运行指定的 B=1、H=8、D=64、FP16、N_CTX=1024/2048/4096/8192、causal=False/True。
输入为 seed=20 的 N(0, 0.5²)，scale=1/sqrt(64)=0.125；每个长度的两种 mask 模式复用同一组输入。
可用 `--output-dir`、`--warmup`、`--repeats`、`--memory-repeats`、`--seed` 调整保存目录或测量设置。
默认 10 次预热、50 次计时、3 次显存采样。首次运行的官方 autotune 可能耗时较长。
GPU 型号动态读取，并用于 CSV、图表和报告；默认使用第 0 张可见 CUDA GPU，未绑定特定型号。
需要选择其他物理 GPU 时，可通过 `CUDA_VISIBLE_DEVICES` 指定。

代码结构：

- `load_attention()`：按文件路径导入旁边官方示例，复用 `attention` 接口。官方已有 main guard，无需修改文件。
- `eager_attention()`：显式 QKᵀ → scale → 可选 causal mask → FP32 Softmax → 转 FP16 → PV。没有 SDPA 或 torch.compile。
- `latency_sample()` / `memory_sample()`：两实现共用的测量入口；显存与延迟分开测量。
- `main()`：输入、预热、输出校验、交替采样、CSV 和环境记录。
- `artifacts()`：基于本次测量输出图表与中文简报。

输出默认保存在项目根目录的 `results/flash_attention/eager_vs_triton/`，与启动时的工作目录无关：

- `samples.csv`：每次 latency/memory 原始采样；不适用的字段为空。
- `results.csv`：16 行（8 组 × 2 实现），延迟中位数/P25/P75，显存峰值及数值误差。误差字段是该配置两实现的成对比较，在两行中重复记录。
- `metadata.json`：GPU、版本、参数、数值设置、官方文件与实验脚本 SHA256。
- `latency.png`：两个 causal 模式的延迟曲线。
- `peak_memory.png`：总分配峰值与额外峰值，并标出输入 Q/K/V 字节数供参照；十进制 MB。
- `report.md`：方法、数据表与结论。

现有结果由实验目录移至上述统一目录，未重新运行。`metadata.json` 的 `script_sha256`
保留实际测量时的脚本哈希；之后仅调整默认输出路径，因此它与当前脚本哈希不同。
项目 `.gitignore` 已忽略 `results/`，这些结果保留在本地。

显存汇总选择三次采样中总峰值最高的一次，baseline/extra/reserved 均来自该次采样。
额外峰值 = peak allocated − 执行前 allocated；基础输入字节单独计算，不假定 allocator baseline 恰好等于输入。
Eager 的 mask 在 forward 内创建，计入延迟与显存。图表中的分配器统计不代表整卡显存占用。
正确性阈值沿用官方 FP16 检查的 atol=0.01、rtol=0；失败会写出结果并以非零状态退出。
CUDA 不可用或 OOM/Kernel 异常会直接报错；已经完成的配置逐组写入 CSV，不会伪造缺失结果。
