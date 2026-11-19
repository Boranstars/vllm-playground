# Qwen2.5-VL 单图分辨率验证

## 环境与 Cold Start 证据

2026-11-19，AutoDL 单卡 RTX 5090，`nvidia-smi` 报告 32,607 MiB 显存。
现有服务的模型为 `Qwen/Qwen2.5-VL-7B-Instruct`，`/version` 返回 vLLM 0.31.0。
运行参数为 BF16、`127.0.0.1:8000`、`max_model_len=4096`、`max_num_seqs=1`、
`gpu_memory_utilization=0.85`、每个 Prompt 限一张图和零个视频。
服务进程使用 `HF_HOME=/root/autodl-tmp/huggingface`。

| 项目 | 可核对的证据 |
| --- | --- |
| 模型权重加载 | 服务列出目标模型并完成实际图文生成，说明加载成功。原始日志中的加载耗时及权重显存占用待补。 |
| 显存预算 | 进程参数和在线 `/metrics` 均报告 `gpu_memory_utilization=0.85`。启动时各部分的显存拆分待补。 |
| KV Cache | 在线 `/metrics` 的 `vllm:cache_config_info` 报告 9,961 个 GPU blocks，每块 16 Tokens，共 159,376 Tokens。 |
| Encoder Cache | 原始启动日志输出到了当时的终端，本次未找到持久化副本。Encoder Cache 的初始化预算、profiling 配置和请求命中情况均待核对。 |

此前的首次 448×448 图文请求据部署记录已正确识别图片，API 返回 `prompt_tokens=288`。
该次请求的完整 Prompt、输出和流式时间戳未提供，因此本次直接用新实验的 Prompt 和 Processor 序列做对照。

## 输入与实测

脚本在三个原生尺寸的白色画布上直接绘制同样的场景，左侧红色正方形、右侧蓝色圆形。
每个尺寸依次发送一次 PNG，请求文本固定为
`Name the color and shape on the left and on the right. Answer in one sentence.`，
`temperature=0`，`max_tokens=32`。测试使用已经运行的服务。

| 原图 / Processor 尺寸 | `image_grid_thw` | 理论 / Processor Patches | 理论 / Processor 视觉 Tokens | Processor / API Prompt Tokens | API Completion Tokens | TTFT (ms) | E2E (ms) | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 224×224 | (1, 16, 16) | 256 / 256 | 64 / 64 | 103 / 103 | 30 | 184.2 | 458.4 | completed / stop |
| 448×448 | (1, 32, 32) | 1,024 / 1,024 | 256 / 256 | 295 / 295 | 30 | 127.1 | 403.5 | completed / stop |
| 896×896 | (1, 64, 64) | 4,096 / 4,096 | 1,024 / 1,024 | 1,063 / 1,063 | 20 | 225.7 | 423.6 | completed / stop |

224 和 448 的输出均为 “The color on the left is red and the shape is a square, while the color on the right is blue and the shape is a circle.”
896 的输出为 “The color on the left is red, and the shape on the right is a blue circle.”，漏报左侧形状。
运行时间为 11:39:30–11:39:31 UTC，原始输出保存在本地忽略目录 `../results`，
文件分别为 `vlm-resolution-20261119T113930253892Z.json` 和 `vlm-processor-20261119T113930253892Z.json`。

## Token 对照

- 理论预测以单张图（`T=1`）、14 像素 Patch 和 `s=2` 为前提。三个输入均按 28 像素对齐。
  Transformers 5.17.0 从本地 snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5` 加载 Processor，
  三张图实际网格乘以 Patch Size 得到的预处理尺寸与原图一致。边长每翻倍，Patch 数和视觉 Token 数分别变为四倍。
  其他输入尺寸以实际网格为准，Processor 可根据对齐和像素预算调整尺寸。[Qwen2.5-VL 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Processor 报告 `spatial_merge_size=2`。Merger 前网格有 `T × H × W` 个 Patch，与 `pixel_values` 的行数相等。
  每组 2×2 空间 Patch 对应一个 LLM 视觉位置，数量为 `T × (H / 2) × (W / 2)`。
  Processor 序列中展开后的 `<|image_pad|>` ID 数逐项等于该值。[Qwen2.5-VL 技术报告](https://arxiv.org/abs/2502.13923)
- 从本地 `input_ids` 数得每个 Prompt 还有 39 个非图像位置，包括默认 system 文本、用户文本、
  消息角色与边界、assistant 生成前缀和 `<|vision_start|>` / `<|vision_end|>`。
  三个完整序列长度均与 API Usage 相等。39 由本次文本和 chat template 决定，换 Prompt 或模板需要重新计数。

## 后续 Infra 问题

TTFT 从客户端发起请求计到收到首段非空文本，E2E 计到流耗尽，包含最终 Usage。
单次观测包含客户端、传输、服务端预处理、排队与模型执行等开销，无法从中单独估计 Vision Encoder 延迟，
也无法把差异归因于 Window Attention、ViT 或缓存状态。当前证据覆盖 Processor 输出和 API Usage，
GPU 内部的实际算子时序有待独立观测。

后续可以核对 Encoder Cache 和 Processor Cache 的预算与命中，追踪视觉特征如何进入 LLM Prefill，
再用源码和 GPU Profiling 拆解 Encoder、Prefill 和 Decode 的时间与调度行为。
