# Mixed Prefill–Decode Interference Experiment

本实验研究：8 个固定 background 请求全部进入稳定 Decode 后，插入不同规模的
Prefill workload，是否使已有 stream 出现客户端可观察的短暂输出 stall。
Primary observable 是 **client-observed token-bearing stream event timing**。

当前 baseline 已默认启用 Chunked Prefill。这里没有关闭/开启 Chunked Prefill 的
对照，因此不能证明 Chunked Prefill 对 TPOT 的 causal improvement，也不能把
客户端时间戳解释成 GPU 上的 token generation timestamp。

## 服务器与源码依据

保持 `scripts/serve_qwen3_0_6b.sh` 原样：

```bash
vllm serve Qwen/Qwen3-0.6B --gpu-memory-utilization 0.75 --max-model-len 4096
```

2026-10-11 本地安装源码、原脚本启动日志及 `EngineArgs.create_engine_config`
（`UsageContext.OPENAI_API_SERVER`）核实以下配置；未发现与预设 baseline 的冲突：

| 参数 | 实际值 |
| --- | --- |
| vLLM / GPU | 0.31.0 / RTX 4060 Laptop 8188 MiB |
| enable_chunked_prefill | True |
| max_num_batched_tokens | 2048 |
| max_num_scheduled_tokens | None，Scheduler 回退到 2048 |
| max_num_seqs | 256 |
| max_num_active_seqs | None，Scheduler 回退到 256 |
| long_prefill_token_threshold | 0 |
| scheduling policy | fcfs |
| async_scheduling / scheduler class | True / AsyncScheduler |
| enable_prefix_caching | True |
| max_model_len | 4096 |

源码位置均相对于 `.venv/lib/python3.12/site-packages/vllm/`：

- `entrypoints/openai/chat_completion/protocol.py`：`return_token_ids` 描述，
  prompt IDs 仅在首个 chunk 返回，choice token IDs 为每个 event 的增量。
- `entrypoints/openai/chat_completion/serving.py`：首个 role event 的 prompt IDs、
  空文本时仍返回 token IDs、finish event 中的 token IDs，以及末尾 usage。
- `v1/engine/output_processor.py`：`RequestOutputCollector.put` 在 producer
  超前于 consumer 时合并 DELTA 输出。
- `config/scheduler.py`、`engine/arg_utils.py`：配置默认值与 AsyncScheduler 选择。
- `v1/core/sched/scheduler.py`：active sequence / token budget 回退以及每轮 budget。

Runner 固定使用本地端口 8000、model `Qwen/Qwen3-0.6B`、engine `0`，沿用现有实验。
它不通过 API 自动断言服务的启动参数；正式实验前须再次核对服务日志/配置。
Counters 包含该 engine 的全部流量，因此实验期间必须保持其他 inference 流量为空。

## Workload 与同步

每次 run 先运行一个不计入结果的 warmup cohort：C=8、相同短 prompt、max_tokens=64。
之后每个 trial 一次性启动 8 个 background tasks，全部结束前不启动下一 trial。
**没有 closed-loop replenishment**，避免 replacement 自己制造 Prefill。

| 设置 | 值及原因 |
| --- | --- |
| Background C | 8：形成持续的多 stream Decode workload，同时远低于 max_num_seqs=256；不是宣称 C=8 为通用最优值 |
| Background prompt | 目标约 577 actual tokens，复用已熟悉的 concurrency/output_length 材料 |
| Background output | max_tokens=512，为 trigger 前后窗口保留 Decode 运行时间；实际 completion 从 usage 校验，不强制 ignore_eos |
| Trigger | 所有 background 的累计真实 generated token IDs >=64，降低初始 Prefill/首 token 的影响；不用 sleep 替代 |
| Temperature | 全部为 0 |
| Foreground output | max_tokens=1，只制造短暂 Prefill 加首 token 的 workload，避免长期第 9 个 Decode 请求 contention |

Coordinator 只执行一次。在所有 ready events set 后，确认 cohort 尚未结束，记录
`trigger_ns=perf_counter_ns()`，并保存 8 个真实累计计数；多 token event 可跨越 64。
Prompt/UUID 事先构造。Short/Long 的 `foreground_request_start_ns` 在实际调用
`client.chat.completions.create` 前记录。Baseline 不创建 foreground 请求。

| Condition | Foreground | Alignment / control |
| --- | --- | --- |
| Baseline | 无 | trigger_ns；给出相同进度附近自然波动和客户端 jitter 的对照 |
| Short | 约 577 prompt tokens、output=1 | foreground_request_start_ns；普通小 Prefill 的干扰 |
| Long | 约 3072 prompt tokens、output=1 | foreground_request_start_ns；超过单轮 scheduler budget 的大 Prefill 干扰 |

Short 明显小于 2048，C=8 的 Decode 通常只需少量 token budget，理论上可在单轮
剩余 budget 内容纳；这不是对 GPU 实际调度轮次的直接观测。
Long actual prompt 必须 >2048；在没有缓存复用时，全部 Prefill 不可能放入一个
2048-token iteration。3072 左右与 4096 的上下文上限有充分余量。
`foreground_request_start_ns` 是客户端提交时间，不是 GPU 开始 Prefill 的时间。
请求构建、HTTP、队列、调度都可能产生延迟。保留 trigger 和 request-start 两个边界。

## Prompt 与 Prefix Cache

直接导入无顶层执行副作用的 `concurrency.py` 常量，短 prompt 逐字复用其构造：
`request-<32 hex UUID>\n` + 固定中文指令 + 6 次英文 paragraph。
Long 使用相同材料，重复 35 次；本地 tokenizer 仅用于估算，未成为运行时依赖。
所有请求包含新 UUID，位于共同指令/body 之前，尽量降低 Prefix Cache reuse。

最终 authoritative token count 来自 streaming `prompt_token_ids` 的 count 和 final
usage，两者都存在时必须一致。Background/Short 验证 550–600，Long 验证 3000–3200，
且对 Long <=2048 明确报 invalid/failure，绝不继续当作有效 Long。
UUID tokenization 本身会引起小幅长度变化，不承诺每个 prompt 恰好 577 或 3072。
`request_summaries` 保留 UUID，可结合固定材料、重复次数重构用户 prompt。

每 trial 记录 `local_cache_hit` counter delta，若 >0 输出警告并保存
`prefix_cache_warning`。唯一 prefix 不是 cache hit=0 的逻辑保证；若发生缓存复用，
必须按该警告解释数据，不能仅凭 logical prompt >2048 推断全部本地计算跨 budget。
服务器 Prefix Cache 始终保持开启。

## Streaming measurement

所有请求使用 AsyncOpenAI、`max_retries=0`、`stream=True`、
`stream_options={"include_usage": True}`、`extra_body={"return_token_ids": True}`。
不启用 logprobs 或 continuous usage。每个解析后的 SSE event 均保存，包括
role、空文本、token-bearing、finish、final usage；SDK 的 `[DONE]` 终止标记不是
解析后的 completion event，不伪造记录。

Hot path 在 SDK 产出 event 后立即取 `perf_counter_ns()`，只读取 token IDs、
增量/累计计数、轻量 metadata，append 内存 record，并在达到阈值时 set ready。
不 print、不写文件、不序列化 Pydantic 对象、不 tokenize、不计算分位数。
完成请求后按顺序拼接 generated IDs，要求长度等于 final usage.completion_tokens。
缺 final usage、计数不一致、prompt 不符、异常 finish 等均使 request/trial 失败。
不把空 `delta.content` 判为无 token；TTFT 取第一个 token-bearing event。

一个 SSE event 可以包含多个 token。`return_token_ids` 解决真实增量计数、trigger
进度、event multiplicity 和 usage 完整性；它无法提供 event 内每个 token 的生成时刻。
不把多 token event 拆成相同 timestamp，也不插值伪造 per-token ITL。

对每个 background 独立排序，仅取 `delta_token_count>0` 的 events：

```text
event_gap_j = received_ns(j) - received_ns(previous token-bearing event)
```

将 gap 归到后一个 event 的 relative time。只有相邻两次都是 singleton 的 gap
才属于 **client-observed singleton-token event interval** diagnostic。
即使本次所有 event 都是 singleton，仍受 SDK、HTTP、event loop 和 OS 调度影响，
不等于 GPU-side decode ITL。每 trial 保存 event 总数、1-token/2-token/>=3-token
比例、multi-token 比例和最大 delta，analysis 另保留 singleton pair 数量。

平均 TPOT 会把短暂 stall 摊入整段 512-token 输出，并掩盖相对 injection 的位置，
所以不作为 primary measurement。Prometheus 也是辅助数据，约 0.2 s 一次 scrape，
不能用于定位几十毫秒的瞬时 stall。Gauge 的 `sample_ns` 从现有 common helper 的
monotonic scrape timestamp 转换；KV 值是 block-pool fraction，不是总 VRAM 比例。

## 分析与图形

每个 request：pre window `[-500, 0)` ms；post window `[0, 500]` ms。
预声明最少各有 2 个 gap，且 pre median >0。任一 background 不满足就将整个 trial
标为 `analysis_valid=False`，保存原因；不补零、不删除该请求后假装仍有完整 cohort。

```text
pre_median_gap = median(pre gaps)
post_peak_gap = max(post gaps)
request_stall_ratio = post_peak_gap / pre_median_gap
trial_stall_ratio = median(8 request_stall_ratios)
```

`analysis.csv` 的 `pre_median_gap_ms` 是 8 个 request pre median 的中位数；
`post_peak_gap_ms` 是 8 个 request post peak 的中位数。Trial ratio 是 request ratios
的中位数，不是这两个汇总数的商，也不是全 cohort 的单个全局 max。
`request_window_summaries` 还保存每个请求的窗口 gap 数和数值供复查。

Plotter 只读 trials CSV 及同 prefix 的 events JSONL，绝不发 inference。
使用 matplotlib，输出 analysis CSV 及三个 PNG：

1. `-event-gap-timeline.png`：默认 `[-0.5,+1.0] s`，每个 gap 画为实际相邻接收
   时刻的区间 `[t_previous,t_current]`。每 condition 一个 panel，每 trial 一行，
   每行保留 8 条 background 轨道；颜色表示完整 gap 时长，刻线标出接收边界。
   不做 binning、平滑或跨 trial 聚合。原先的 endpoint-bin 汇总图已由此区间图替代。
2. `-stall-ratio.png`：每个 analysis-valid trial 一个点，附 condition median。
3. `-foreground-ttft.png`：成功 trial 的 Short/Long TTFT 点和 median；不含 Baseline。

另用 `--stall-only` 可生成 `-stall-only-timeline.png`：每 request 独立计算 pre
window `[-500,0)` ms 的 median gap，只保留严格 `gap > 2 × pre_median_gap` 的
区间，仍按真实接收边界画线。Pre window 按后一个 event 的时间归属，不随展示
窗口改变。该图是后续添加的诊断可视化，2× 不是统计显著性阈值，不修改主 stall ratio。
正常 gap 不绘制；每 trial 的 8 条轨道仍保留，不合并相邻异常 gap。

失败/analysis-invalid trials 保留在 CSV 并明确标记；不混入有效 timeline/stall ratio。
少量或空窗口数据也可以出图。TTFT 不依赖 background 窗口是否足够，成功请求仍可展示。

## Schedule、输出与失败处理

Formal 默认 10 blocks，每 block 恰好 baseline/short/long 各一次；固定 seed 20261011
随机排列 block 内顺序，共 30 trials。这样减少 laptop thermal/clock/time drift 与
condition 的系统关联；不能完全消除漂移。seed 决定顺序，UUID 保持每请求唯一。
Smoke 强制仅 1 block、3 trials，用于验证 protocol 和测量链路，不产生正式科学结论。

结果前缀 `results/prefill-interference-<timestamp>-<8 hex run-id>`，results 保持 gitignored：

- `-events.jsonl`：每 parsed SSE event 一行。trial/request identity、背景编号、event
  序号、接收/对齐时间、相对 ms、delta/cumulative token count、完整 generated IDs、
  finish/usage 字段、prompt IDs count，以及少量 role/content-presence metadata。
- `-trials.csv`：每 trial 一行，配置/seed、所有时间边界、8 个 trigger counts、prompt
  范围、foreground TTFT/usage、event 分布、request summaries、counter deltas、
  idle gauges、cache warning 和 success/failed/error。
- `-metrics.csv`：边界、约 0.2 s sampling 及清场的 Running/Waiting、capacity/deferred、KV。
  全部使用相同 alignment。Failure 未形成 alignment 时保留空值。
- `-analysis.csv`：每 trial 一行，窗口汇总、ratio、event 分布、TTFT、valid/reason。

Trial 结束后等待所有 streams 完成，随后连续两次观察 Running=Waiting=0、
KV fraction <=0.001，并等待累计 prompt/generation counters 至少覆盖已完成 requests，
避免指标导出刷新晚于 stream 完成时过早取差值。每 trial 不重启 server。
逐 trial 结束后才 JSON/CSV 落盘并 flush。warmup 不进入结果，只有 preflight/warmup
失败时写 trial_index=0、condition=warmup 的 failure record。

正常请求全部自然结束；background 提前退出、foreground/metrics 失败或 trial timeout
会由 TaskGroup 取消并等待 sibling tasks；每个已打开 stream 在 finally 内 close。
请求 HTTP timeout=120 s、trial 默认总 timeout=180 s、warmup timeout=180 s、scrape
5 s、close 5 s、idle wait 60 s；无无限等待。失败保存可用 partial events/metrics 与
failure row，然后停止 run，不静默替换 trial。Counter 下降或系列变化也明确失败。

## 命令与离线验证

沿用项目 `--no-sync`，避免 uv 同步时移除现有手动安装的 vLLM/OpenAI 包。不新增依赖。

```bash
./scripts/serve_qwen3_0_6b.sh
uv run --no-sync python experiments/prefill_interference.py --smoke
uv run --no-sync python experiments/prefill_interference.py --trials-per-condition 10 --seed 20261011
uv run --no-sync python experiments/plot_prefill_interference.py results/prefill-interference-<run-id>-trials.csv
uv run --no-sync python experiments/plot_prefill_interference.py results/prefill-interference-<run-id>-trials.csv --timeline-only
uv run --no-sync python experiments/plot_prefill_interference.py results/prefill-interference-<run-id>-trials.csv --stall-only
uv run --no-sync python -m unittest discover -s tests -v
```

可用 `--trial-timeout` 设置 trial deadline；plotter 的 `--window-ms LEFT RIGHT`
只改变 timeline 展示，不改变预声明的分析窗口。`--bin-ms` 仅保留兼容，区间图不使用它。
`--timeline-only` 仅更新 full-interval PNG，`--stall-only` 仅写 stall-only PNG，均不写
CSV 或其他 PNG；两者互斥。默认命令仍输出 analysis CSV 和原三张图。

`tests/test_prefill_interference.py` 使用 unittest/fake async streams，覆盖 role/empty/
single/multi/finish/usage，62+3=65，trigger 一次，Baseline 无 foreground，真实
request-start 对齐，usage 缺失/不一致、Long<=2048、prompt IDs 不一致、取消关闭、
background 提前退出、foreground failure、metrics failure、timeout 与无遗留 tasks；
另覆盖 gap/singleton pair/window/invalid、block schedule、common timing API 签名、
原 prompt 复用、JSONL/CSV schema、稀疏数据三 PNG 和 gitignored results。

## 2026-10-11 实际 smoke 记录（不属于正式结论）

仅运行一次 smoke：`20261011-211120-78a1b24d`，执行顺序 Baseline → Long → Short。
三 trial measurement success 且 analysis valid；warmup C=8 也正常完成。

| Condition | Background actual prompt | Foreground prompt | Foreground TTFT (s) | Pre median gap (ms) | Post peak 汇总 (ms) | Trial stall ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 573–580 | — | — | 9.423 | 11.433 | 1.213 |
| Short | 575–580 | 580 | 0.054864 | 9.556 | 34.417 | 3.605 |
| Long | 575–580 | 3071 (>2048) | 0.219071 | 9.306 | 115.050 | 12.363 |

Trigger counts（按 bg0…bg7）：

- Baseline：`[66,64,64,64,65,65,65,64]`
- Long：`[65,66,65,64,64,65,64,64]`
- Short：`[66,65,64,64,65,65,64,64]`

全部 24 个 background 都完成 512 tokens；两个 foreground 各完成 1 token，finish
均为 length。26 个测量请求的 generated IDs 总数均与 final usage 匹配，首 event
prompt IDs count 也均与 usage 一致。Warmup 8 个请求通过相同完整性验证。

每 condition 有 4096 background token-bearing events；合计 12288。
Singleton=100%，2-token=0%，>=3-token=0%，multi-token=0%，max delta=1。
计入两个 foreground 则为 12290 token-bearing events；所有类型 SSE events 共12342，
Gauge samples 共92。本次没有实际 multi-token event，不能据此假定以后不存在；
其处理逻辑由合成测试验证。

三 trial 的 local_cache_hit、external KV transfer、preemption deltas 均为 0。
Running 采样峰值 Baseline/Short/Long=8/8/9；Waiting/capacity/deferred 全部为0。
KV fraction 峰值分别约23.36%/23.19%/23.41%，无清场异常。Short 的额外 running
request 可能被0.2 s采样遗漏，不能据此否认其存在。每 trial 清场和关闭前最后一次
scrape 均为 Running=Waiting=KV=0。

三 PNG 已生成并目视检查。Raw JSONL 重放再次验证每 event 累计计数、alignment、
request event 序号、全部 usage、trigger、cohort 完成及 CSV/metrics 完整性。
13 项 unittest 通过；全部 experiment/test Python syntax 检查通过。

服务器按 SIGINT 正常退出（exit code 0），日志包含 engine/client/FastAPI shutdown
complete。退出时 Python resource_tracker 报告清理 1 个 leaked semaphore；这是
实际观察到的服务器退出警告，未伴随 measurement 或 idle 失败。保留原日志供复查：

- `results/prefill-interference-server-20261011.log`
- `results/prefill-interference-smoke-20261011.log`
- `results/prefill-interference-config-20261011.log`
- `results/prefill-interference-20261011-final-metrics.txt`

这些数字只表明本次客户端 stream 在注入附近可观察到不同规模的 gap；样本量每
condition=1，未控制更多运行波动，不能宣称稳定效应量、显著性或 Chunked Prefill 的
因果改善。此节仅记录当时执行的 smoke；后续正式 run
`20261011-213351-40137cab` 的结果与新版图形记录在 [Week 1 experiments](week1-experiments.md#mixed-prefilldecode-interference)。
