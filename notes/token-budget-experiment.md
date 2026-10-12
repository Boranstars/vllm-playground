# Token budget / Chunked Prefill A/B experiment

## Question and controls

固定 active Decode workload 与 Long Prefill workload，只改变 server-side
`max_num_batched_tokens = 1024 / 2048 / 4096`。研究 Prefill scheduling 切分、
client-observed token-bearing event gaps 与 foreground TTFT 的关系。
不评选最优 budget，不以 hypothesis 是否成立作为 acceptance criterion。

复用 `prefill_interference.py` / `common.py` 的请求、prompt、measurement、metrics、
C8 warmup 和 drain。模型 Qwen/Qwen3-0.6B，max_model_len=4096，GPU fraction=0.75；
原启动脚本仅增加参数透传，orchestrator 仅传入 `--max-num-batched-tokens`。
不改依赖，不修改其他服务环境。Budget label 仅用于 metadata，client 不修改 server。

固定 C=8 cohort，无 replacement；background prompt 约577 actual tokens，max_tokens=512（上限），
temperature=0，每请求开头 UUID unique prefix。全部 background 观测到 >=64 generated
IDs 且仍 active 后 alignment。Baseline 不注入；Long 用原35段 prompt（约3070 actual
prompt tokens，validation 3000–3200，明显小于4096），output=1，同样 early UUID。
UUID 导致少量 actual prompt 长度变化；prompt 构造跨 budget 相同，usage/prompt IDs 为准。

## Source-derived expectation and pre-experiment hypotheses

以下采用已完成的 vLLM 0.31.0 本地源码审计，不重新推断运行时配置：RUNNING requests
先分配，余量给 Prefill；没有独立固定 chunk_size；long_prefill_token_threshold=0。
在无 cache hit、KV充足、8个decode每轮约1 token条件下，约3070-token Prefill 的预测：

| Budget | Approximate scheduled tokens | Scheduling rounds |
| --- | --- | --- |
| 1024 | 1016, 1016, 1016, 22 | 4 |
| 2048 | 2040, 1030 | 2 |
| 4096 | 3070 | 1 |

**实验前 hypothesis**：小 budget 单次 decode stall 可能较短、异常区间更多、TTFT
可能较长；大 budget 单次 mixed iteration 可能更重、stall 更大，但轮数较少、TTFT
可能降低。GPU execution efficiency、batch shape、runtime overhead、async scheduling
可能令关系非单调。客户端 stall 数不要求等于 scheduler rounds；本实验没有 scheduler
iteration trace，不能从客户端 interval 个数直接声称实测 chunk 数。

## Protocol and failure rules

`experiments/token_budget_sweep.py` 默认自动完成 smoke → formal：

1. Smoke：fresh server budget=2048，poll `/v1/models`（300s timeout），C8 output64
   warmup 一次，再1个 block（Baseline+Long）。保存并验证后关闭，自有session进程退出、
   port8000释放后才通过。关键失败保留数据并停止，不进入formal、不自动重试失败测量。
2. Formal固定server block顺序 **2048 → 4096 → 1024**。每档一个全新server生命周期，
   warmup一次，10 blocks，每block各1 Baseline/Long，block内部用 seed=20261012 shuffle。
   同一可重现condition schedule用于三个budget，共60 trials，不运行512。
3. 每trial前及后复用drain：连续2次Running=Waiting=0、KV fraction<=0.001，
   结束时还等待counter导出覆盖全部完成请求，timeout60s。Trial timeout180s。
4. 端口已被其他进程占用则停止报告，不杀已有进程。subprocess新session，独立server log；
   仅关闭自己session中的进程，依次SIGINT(45s)、SIGTERM(20s)、SIGKILL(10s)。
   强制SIGKILL、异常exit、遗留进程或端口不释放均作为failure停止sweep。
5. 当前完整性规则为 `window-coverage-v2`（2026-10-12用户明确授权修订，替代原来的
   background恰好512要求）：8个background均须成功完成，0<completion_tokens<=512，
   final usage存在，IDs与usage匹配；alignment时每个已观测到>=64 IDs。
   每个request_start<=alignment−0.5s，request_end>=alignment+0.5s，且实际接收到的
   finish event不得早于alignment+0.5s，防止延迟的final usage/stream closure掩盖提前stop。
   post window结束前自然stop使trial无效；恰在+0.5s结束可接受。原窗口gap数量检查仍适用。
   `max_tokens=512`保持上限语义，不启用ignore_eos，不改prompt或请求参数；Long仍须output1。
   raw replay验证序号、累加计数、timestamps、request identities和trigger进度。
   新manifest记录规则版本；历史failed状态与postmortem不追溯覆盖。任何失败trial仍落盘。
6. local prefix cache hit、external KV、preemption必须0；sampled Waiting/capacity/deferred
   必须0。预声明KV block-pool fraction>=80%为停止guard（独立于stall hypothesis）。
   这不是额外measurement阈值。采样约200ms，无法排除采样之间短暂waiting。
7. multi-token event本身不作失败：保存singleton ratio、max delta，不拆分event造timestamp。
   只在全部formal blocks完整且integrity通过后生成formal plots。

## Measurement and analysis semantics

streaming `return_token_ids=True`，每parsed SSE用`perf_counter_ns()`；token-bearing event
保存delta IDs、timestamp、cumulative count。TTFT从API call开始到首token-bearing event。
这些是 **client-observed token-bearing event gaps**，不是GPU token timestamp或kernel ITL。

保持旧窗口：pre `[-500,0)`ms，post `[0,500]`ms，**按后一个event时间归窗**。
每请求至少各2 gaps，pre median>0；每请求ratio=post peak/pre median；trial ratio是8请求ratio
中位数。pre/post-peak也分别取8请求中位数，不能把这两个汇总值相除替代trial ratio。

新增字段：

- `max_stall_gap_ms`：原 `post_peak_gap_ms` 的明确别名（8个request峰值的中位数）。
- `cohort_max_gap_ms`：额外保留8请求全局最大post gap，避免混淆max含义。
- `significant_stall_count`：每请求post-window中严格`gap > 2 × request pre median`
  的原始interval数，再取8请求中位数；“significant”不是统计显著性检验。
- `significant_stall_total_duration_ms`：各请求这些interval的完整时长之和，再取8请求中位数。
  不减去normal gap，不将8请求重复相加为wall time。跨alignment的gap计完整长度；
  endpoint>500ms的gap按原窗口规则不纳入post统计，timeline仍按真实区间画出可见部分。
- `foreground_ttft_ms`；另保存每请求stall count/duration以便审计。

图形使用所有trial points及median、不平滑。Stall-only timeline分别画3 budget，
每budget全10个Long trials、每trial8轨道，区间为实际`[previous receipt,current receipt]`，
展示窗`[-0.1,+0.5]s`，仅显示2×异常interval，接收边界有刻线。Baseline另提供同类图。
Raw interval不合并、不命名为scheduler chunk。所有baseline/long比较按budget分别描述；
`paired-comparisons.csv`提供同block Long−Baseline及比值（baseline为0时ratio=null），
`paired-summary.csv`汇总这些配对差值。昨天数据只写independent replication CSV，不入formal图。

## Startup metadata and confounders

记录实际启动命令、PID、timestamps、版本、budget、可靠可解析的配置、KV capacity及
profiling memory。无法从runtime日志可靠确认的max_num_seqs/async scheduling等记null；
用户提供的已知默认值不冒充该次runtime观测。启动前后nvidia-smi与git状态单独保存。

每budget仅一个server block，因此budget与block时间仍有混杂；非单调顺序只避免与budget
大小完全共线，不能消除thermal/clock/桌面GPU workload漂移。UUID长度、client event loop、
HTTP/SSE batching、async scheduling、JIT/compilation与startup profiling均可能影响观察。
不同budget可能有不同KV capacity；若Waiting/preemption=0且KV远离capacity，作为记录但未
观察到激活的潜在confounder。Waiting=0不等于无资源竞争，preemption=0不等于完全无压力。

## Outputs and invocation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/token_budget_sweep.py
# optional smoke only:
.venv/bin/python experiments/token_budget_sweep.py --smoke-only
```

新路径：`results/token-budget-sweep/smoke-<run-id>/` 和
`results/token-budget-sweep/<run-id>/`。目录排他创建，不覆盖旧结果。
每档包含server.log、metadata.json、warmup.json、models.json、environment-start/end.txt、
events.jsonl、trials.csv、metrics.csv、analysis.csv。根目录manifest/environment，formal另有
summary、integrity、paired comparison、replication及plots。失败manifest保存exact reason。

执行记录和结果在运行完成后追加；以上hypotheses在实验前固定。

## 2026-10-12 执行记录：smoke 关闭检查失败，formal 未启动

Run ID：`smoke-20261012-203600-0fe18da7`。
路径：`results/token-budget-sweep/smoke-20261012-203600-0fe18da7/`。
实际仅启动budget2048；未产生formal run，1024/4096未运行。

Smoke readiness成功，C8 output64 warmup通过，Baseline与Long各1 trial测量及analysis有效。
但发送SIGINT后启动器返回130，初版checker仅接受0、-SIGINT、-SIGTERM，导致：

```text
RuntimeError: Abnormal server shutdown:
{'signals': ['SIGINT'], 'returncode': 130, 'remaining_pids': [],
 'port_8000_released': True, 'timestamp': '2026-10-12T12:38:58.017118+00:00'}
```

这是orchestrator对uv的128+signal退出码表示遗漏。已修正纯逻辑：只有自己发出的signal
所对应的负数/128+signal编码才可接受，且仍先确认进程退出、端口释放；SIGKILL始终失败。
新增regression与完整synthetic绘图测试，共20项unittest通过；`git diff --check`通过。
**按本轮smoke失败即停止规则，修复后没有重跑smoke、没有继续formal、没有重标manifest成功。**
`manifest.json`和`metadata.json`保留原failure，`postmortem.json`另存复核说明及数字。

独立复核8000端口无listener、启动器PID53214和EngineCore PID53784均已退出。
关闭日志含FastAPI Application shutdown complete；vLLM内部另记录force killing EngineCore，
以及resource_tracker清理1个leaked semaphore。Orchestrator本身只发出SIGINT，没有SIGTERM
或SIGKILL升级。不能只凭进程消失忽略这些实际日志。

Startup：vLLM0.31.0，budget2048，max_model_len4096，GPU fraction0.75，Prefix Cache和
Chunked Prefill日志确认开启；KV capacity=36,976 tokens，available KV memory=3.95GiB，
4096长度最大并发估计9.03x。max_num_seqs、async scheduling等未可靠确认，metadata为null。
日志出现Numba需要NumPy<=2.4但当前2.5的warning，之后server仍成功ready；未修改依赖。

完整性（不含warmup）：16个background各512，Long output1，prompt3073；background
prompt573–580；IDs/usage全部匹配。8193 token-bearing events，singleton100%，max delta1。
local cache hit/external KV/preemption各0；sampled waiting/capacity/deferred各0；KV峰值
23.5498%，最终idle Running/Waiting/KV均0。

| Smoke condition | Pre gap ms | Post-peak ms | Trial stall ratio | FG TTFT ms | Significant count | Total significant duration ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8.806 | 10.715 | 1.217 | — | 0 | 0 |
| Long | 9.686 | 114.502 | 11.819 | 217.054 | 2 | 197.533 |

以上每condition只有1 trial，background汇总定义与上文相同；Long cohort全局max为
114.611ms。只是smoke描述，不是formal结果，也不足以评估outlier分布或跨budget关系。
没有生成formal plots、summary或2048 independent replication comparison。
六种图形生成路径已用temporary synthetic数据测试；测试图不作为实验结果保留。

## 2026-10-12 新 smoke 通过；formal 因 background 提前结束停止

用户授权用已修复checker从全新smoke开始，本轮没有修改代码、实验协议、threshold或workload。
旧 `smoke-20261012-203600-0fe18da7` 保持failed，未复用、未重标记。

- 新smoke：`smoke-20261012-204435-6ce7f0d8`，manifest=passed；C8 warmup与两个trial通过。
  SIGINT shutdown returncode=0，自有session无剩余进程，8000端口释放；singleton100%。
- 新formal：`20261012-204435-6ce7f0d8`，计划2048→4096→1024；实际仅启动2048。
  前11 trials通过（5 Baseline、6 Long）；trial12/block6/Baseline失败并立即停止。
  4096和1024未启动，没有replacement trial，没有完成60-trial sweep。
- Exact error：`ValueError: Background completion length != 512`。
  background_id=5，client_request_id=`0d5b3990335b423198b9d1e0cc85d747`，
  prompt580，generated IDs=508，final usage completion_tokens=508，finish_reason=`stop`。
  该trial其余7个background各512、finish_reason=length。Stream计数有效不等于fixed512
  workload有效；没有缺usage或IDs不一致。没有保存更具体stop cause，不能仅凭stop断定原因。
- 全12个attempted trials共102个测量请求，raw replay全部IDs/final usage一致。
  49114个token-bearing events，singleton ratio=0.9995113409618439，max delta=4。
  本次存在multi-token SSE events，不能把event间隔当成逐token生成时间。
- 全部已执行trial的local cache hit/external KV/preemption均0，sampled
  Waiting/capacity/deferred均0；KV峰值23.5906%；最终Running/Waiting/KV均0。
- Formal2048 metadata：vLLM0.31.0，budget2048，max_model_len4096，GPU fraction0.75，
  Prefix Cache/Chunked Prefill开启；KV capacity36912 tokens、3.94GiB，最大并发估计9.01x。
  新smoke同budget的capacity为37984 tokens、4.06GiB，差异已保留，无观察到capacity pressure。
- Formal停止后只使用SIGINT，returncode=0，自有session进程完全退出、8000端口释放；
  独立查询再次确认。本次shutdown日志仍有vLLM内部force-kill EngineCore及semaphore warning，
  并在shutdown过程中记录AsyncLLM output_handler EngineDeadError；不是trial12长度失败的原因。

结果目录：`results/token-budget-sweep/20261012-204435-6ce7f0d8/`。
保留failed manifest、budget2048的server log/metadata/warmup/models/events/trials/metrics；
另写`postmortem.json`保存exact failure和离线复核。未生成formal plots、完整summary或
2048 independent replication comparison，不将partial run呈现为完成的formal experiment。

## Trial 12 离线复核与用户授权的 window-coverage-v2 修订

本次仅离线复核、修改validator和测试，未启动server或下一次formal run。
按用户明确授权，将background精确512要求改为上述窗口覆盖规则；保持请求
max_tokens=512、temperature、prompt/UUID、trigger和2×threshold不变，不启用ignore_eos。
旧run的failed manifest、trials CSV、metadata和postmortem保留原始历史状态。

`20261012-204435-6ce7f0d8` trial12/background5：

| 相对于alignment的时间 | ms |
| --- | ---: |
| Request start | -803.386898 |
| First token-bearing event | -674.922557 |
| Last token / finish event (`stop`) | +5071.428288 |
| Request end | +5071.712173 |

508 IDs与final usage一致，alignment前已观测65 IDs；finish比post window右边界晚
4571.428288ms。8个background的生命周期均完整覆盖[-500,+500]ms，全部通过新规则
完整性验证及原分析。**Trial12按新规则有效**；这是离线重新评价，不把历史中止的run
改写成已完成sweep，也没有补充任何trial。

该trial全部4092个background token-bearing events均为singleton，multi-token event=0。
每请求post peak interval两端的delta都为1；8请求post窗口均无gap超过各自2×pre median。
background5 pre median=9.2809445ms，threshold=18.561889ms，post peak=11.4622ms，
区间[+417.068512,+428.530712]ms。Trial post-peak中位数11.480244ms、stall ratio
1.2350046062，significant interval count和total duration均0。因此multi-token events
没有参与这个trial的peak；这里也不存在达到既定阈值的significant post-window stall。
之前整个partial run的99.951% singleton统计不能套用到trial12。

独立审计文件：
`results/token-budget-sweep/20261012-204435-6ce7f0d8/trial-12-window-coverage-review.json`。
包含每个background的起止/finish/trigger、峰值区间两端delta、event distribution、完整
analysis以及原文件SHA256。对历史fixed512 failure只在内存中清除该旧rule判定后执行
新validator；源文件前后hash一致。新运行manifest会记录`background_integrity_rule`。

26项unittest通过，新增覆盖：508 tokens完整覆盖有效、自然stop在post window内无效、
final usage延迟不能掩盖提前stop、pre window覆盖、+500ms边界、raw trigger和512上限。

## 正式 sweep 完成：20261012-210208-840f1f50

用户授权按已测试的window-coverage-v2规则重新执行。全新smoke
`smoke-20261012-210208-840f1f50`通过，随后全新formal run
`20261012-210208-840f1f50`按2048→4096→1024完成60/60有效trials（每档10个配对blocks）。
三个server均SIGINT关闭、returncode0、无遗留自有进程且port8000释放。
未改workload或threshold，未启用ignore_eos，未把历史partial run并入本次数据。

完整描述性报告及所有raw artifacts：
[正式run报告](../results/token-budget-sweep/20261012-210208-840f1f50/report.md)。
每档KV capacity2048/4096/1024为36912/35824/34608tokens，KV峰值均<26%；
cache hits、externalKV、preemption、sampled Waiting/capacity/deferred均0。
独立raw replay验证60trials、510requests均满足当前完整性规则并与保存analysis一致。

Long在1024/2048/4096的TTFT中位数217.224/209.377/199.408ms，
post-peak中位数72.287/111.453/181.705ms，stall ratio中位数8.326/12.281/20.769。
Significant原始interval count为3/2/1（每组所有trial相同），这不是scheduler round计数。

8个multi-token SSE events（delta3）全在2048 Baseline trial10、约+1.247s，
位于分析窗之外；所有480个significant post intervals及480个request post peaks的
两端都是singleton。2048整体singleton99.9902304%，另两档100%。
一个2048 Long background自然stop于485tokens但完整覆盖窗口，按当前规则有效并保留。
4096 Long trial11 pre median10.857ms偏高，ratio16.741偏低，未删除或改动。

已生成并目视检查六张plot，配对比较、replication及event-multiplicity审计文件均已保存。
昨天2048与本次只做独立sanity comparison；hypotheses仍为实验前陈述，不据本轮结果
评选最优budget。完整范围、时间结构、startup日志警告与thermal/KV capacity混杂见报告。
