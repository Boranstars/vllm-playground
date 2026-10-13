vllm 源码探究
---

# 需要搞懂的问题
1. 一个请求在哪里从“API request”变成 engine 内部的 Request？
2. WAITING 和 RUNNING 到底是谁维护的？
3. Scheduler.schedule() 一轮究竟在算什么？
4. num_new_tokens、token budget 在哪里限制本轮工作量？
5. KV block 的需求在哪里计算，allocate_slots() 返回失败意味着什么？
6. Scheduler 一轮结束以后到底产出什么对象？
7. ModelRunner 拿到这个结果后，如何把它变成真正的 model execution？

# checkpoint1

## 阅读指导

**Checkpoint 1：内部 Request 的创建与进入 waiting**

本次只跟踪一个**普通、新建、非 streaming 的文本生成请求**。下面链接打开 installed source；submodule 中对应 `third_party/vllm/vllm/` 下的同名路径，关键文件内容一致。

先打开 [v1/engine/core.py](/home/boran/PycharmProjects/vllm-playground/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py)，从 `EngineCore.preprocess_add_request()` 开始。围绕它做以下几个小跳转，都属于同一个 checkpoint：

| 顺序 | 文件 / 类 / 函数 | 本次阅读范围 |
|---|---|---|
| 1 | `core.py` → `EngineCore.preprocess_add_request()` | 看 `Request.from_engine_core_request(...)` 的对象转换与返回值 |
| 2 | [v1/request.py](/home/boran/PycharmProjects/vllm-playground/.venv/lib/python3.12/site-packages/vllm/v1/request.py) → `Request.from_engine_core_request()`、`Request.__init__()` | 看字段传递，以及 `status`、`num_computed_tokens` 的初始化 |
| 3 | `core.py` → `EngineCoreProc.process_input_sockets()`、`_handle_client_request()`、`EngineCore.add_request()` | 只看 ADD 接收、输入队列交接、ADD 分派和交给 scheduler 的调用 |
| 4 | [v1/core/sched/scheduler.py](/home/boran/PycharmProjects/vllm-playground/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py) → `Scheduler.add_request()` | 看是否已有同 ID 请求，再看新请求的 else 分支 |
| 5 | 同一 `scheduler.py` → `_enqueue_waiting_request()` 及两个判断辅助函数 | 看阻塞状态判断、是否持有 KV blocks 的判断；回看 `Scheduler.__init__()` 的容器定义 |

这次阅读只回答三个问题：

1. `EngineCoreRequest` 在哪里变成内部 `Request`？`WAITING` 初始状态由谁设置？
2. `EngineCore.add_request()` 把请求交给谁？新请求最终加入哪个队列，又登记在哪个字典？
3. 这条新请求入队路径有没有调用 `allocate_slots()`，或把请求加入 `self.running`？据此，**进入 waiting 与 admission 是否是同一步？**

与实验的对应：这里先把 **Running / Waiting** 从观测指标对应到实际容器，再辨认 **Continuous Batching** 的请求入口与后续调度的分界。你在 homogeneous concurrency 实验里同时发出的多个请求，可以先逐个套进这条入队路径；它们何时获得 KV 空间、进入 RUNNING，留到后面的 admission checkpoint。

## 探索中的问题与结论

### 1. EngineCoreRequest 是什么，从哪里来？

它是前端交给 EngineCore 的请求数据结构，定义在 `v1/engine/__init__.py`。常规路径由 `InputProcessor.process_inputs()` 构造，携带输入 token、采样参数、请求 ID 等；跨进程时经 EngineCoreClient 编码发送。

以下是 `EngineCoreRequest` 中部分原始字段定义：

```python
request_id: str
prompt_token_ids: list[int] | None
mm_features: list[MultiModalFeatureSpec] | None
sampling_params: SamplingParams | None
pooling_params: PoolingParams | None
arrival_time: float
lora_request: LoRARequest | None
cache_salt: str | None
data_parallel_rank: int | None
```

### 2. 为什么还要转换成内部 Request，是否舍弃字段？

`Request.from_engine_core_request()` 保留执行所需输入，并通过构造函数建立可变执行状态：`status`、输出 token、`num_computed_tokens`、抢占计数等。`data_parallel_rank` 用于 client 路由；`current_wave` 单独返回并传递；`external_req_id` 的映射由前端 OutputProcessor 保存。

以下摘录 `Request.__init__()` 中分散的初始化语句：

```python
self.status = RequestStatus.WAITING
self._output_token_ids: list[int] = []
self.num_computed_tokens = 0
self.num_preemptions = 0
```

生成请求的 `self.max_tokens` 来自 `sampling_params.max_tokens`，限制单个请求的输出长度；它与 scheduler 每轮的 `max_num_batched_tokens` 不同。

### 3. cache_salt 是否说明跨用户复用会串对话？

`cache_salt` 是可选的缓存隔离条件，不同 salt 可以隔离相同前缀，但不会自动识别用户。相同模型及相关计算条件下，共享相同前缀的 KV 本身不会串对话；隔离也可避免跨租户通过命中耗时探测前缀。TODO：后续看 `kv_cache_utils.py` 中缓存 hash 如何加入 salt。

### 4. 状态为什么不只有 WAITING / RUNNING？

`RequestStatus` 还区分等待 grammar、远端 KV、streaming 输入等条件，以及不同结束原因。`PREEMPTED` 单独记录被抢占但尚未结束的请求；`is_finished()` 通过 `status > PREEMPTED` 判断结束。状态与所属队列不是一回事，实验中的 Waiting / Running 是汇总指标。

以下摘录状态枚举成员与结束判断，中间省略其他方法：

```python
WAITING = enum.auto()
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
WAITING_FOR_REMOTE_KVS = enum.auto()
WAITING_FOR_STREAMING_REQ = enum.auto()
RUNNING = enum.auto()
PREEMPTED = enum.auto()
# Note: anything after PREEMPTED will be considered
# as a finished status.
FINISHED_STOPPED = enum.auto()
FINISHED_LENGTH_CAPPED = enum.auto()
FINISHED_ABORTED = enum.auto()
FINISHED_IGNORED = enum.auto()
FINISHED_ERROR = enum.auto()
FINISHED_REPETITION = enum.auto()
```

```python
@staticmethod
def is_finished(status: "RequestStatus") -> bool:
    return status > RequestStatus.PREEMPTED
```

### 5. process_input_sockets 做什么，如何接上之前的函数？

它是 EngineCore 侧后台输入线程入口。对 ADD：ZMQ 接收 → 解码为 `EngineCoreRequest` → `preprocess_add_request()` → 内部 `Request` → 写入 `input_queue`。ZMQ 是消息通信库，可使用 IPC 或 TCP；这里先把它当作进程间消息通道。

### 6. 谁读取 input_queue，怎么找到调用者？

`_process_input_queue()` 读取消息并调用 `_handle_client_request()`，后者按 ADD / ABORT / UTILITY 等类型分发。ADD 再经 `EngineCore.add_request()` 到 `Scheduler.add_request()`。查调用者可用 PyCharm 的 Find Usages；跨队列衔接时，要同时追写入和读取。

对应的写入语句与非阻塞读取片段分别是：

```python
self.input_queue.put_nowait((request_type, request))
```

```python
while not self.input_queue.empty():
    req = self.input_queue.get_nowait()
    self._handle_client_request(*req)
```

### 7. EngineCore 是多线程的吗，还有输出队列吗？

这里有后台输入线程、engine 主循环和后台输出线程。输入线程写 `input_queue`，主循环处理消息并推进执行、写 `output_queue`，`process_output_sockets()` 读取输出并发送。ABORT 同时写 `aborts_queue` 和 `input_queue`：前者允许尽早处理，后者保留消息顺序；取消处理是幂等的。

### 8. run_busy_loop 的骨架和两段输入处理是什么意思？

外层每轮处理输入消息、发布计数、推进 engine step、再发布计数。输入处理分两段：没有执行工作时阻塞等消息，直到有工作；随后非阻塞地处理已到达消息，队列空了就返回执行。`has_work()` 不等于消息队列非空；第一段也不是只 try 一次。合成一个循环也可实现，两段只是明确不同等待与退出条件。

`EngineCoreProc.run_busy_loop()` 的原始骨架：

```python
@fault_tolerant_wrapper
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # Publish request counts before and after GPU step to ensure freshness.
        self._maybe_publish_request_counts()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
        self._maybe_publish_request_counts()

    raise SystemExit
```

第一段的循环条件是 `while not self.has_work() and self.is_running()`；其读取片段如下。默认可阻塞等消息，配置为非阻塞时，空队列由 `queue.Empty` 处理；第二段使用上面的 `get_nowait()` 循环。

```python
block = self.process_input_queue_block
try:
    req = self.input_queue.get(block=block)
    self._handle_client_request(*req)
except queue.Empty:
    break
if not block:
    break
```

### 9. KVConnector / ECConnector 与传递参数是什么关系？

两者是可选传输组件：KVConnector 传递 KV cache，例如 prefill / decode 分离；ECConnector 传递 encoder cache，例如多模态编码与后续执行分离。请求中的 transfer params 是传输信息，connector 是处理它的组件；`EngineCore.add_request()` 此处只检查并打印缺少组件的 warning。TODO：具体传输机制暂不深入。

### 10. _enqueue_waiting_request 是不是只入队？

是。受条件阻塞的请求加入 `deferred_waiting` 集合，同时按是否持有 KV blocks 进入 `kv_holding_waiting` 或 `waiting`，两个 if 独立。普通新请求进入 `waiting`，外层还登记到 `self.requests`；此处没有分配 KV，也没有加入 RUNNING，入队与 admission 是不同步骤。

```python
def _enqueue_waiting_request(self, request: Request) -> None:
    if self._is_blocked_waiting_status(request.status):
        self.deferred_waiting.add(request)
    if self._holds_kv_blocks(request):
        self.kv_holding_waiting.add_request(request)
    else:
        self.waiting.add_request(request)
```

# checkpoint2

## 阅读指导

**Checkpoint 2：只看 `schedule()` 的顶层骨架。** 本次用函数名和关键语句定位，不列行数。仍以 installed source 为准；submodule 对应文件一致。

先打开 [v1/engine/core.py](/home/boran/PycharmProjects/vllm-playground/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py)。

| 函数 | 建议阅读的小范围 | 这一段观察什么 |
|---|---|---|
| `EngineCoreProc._process_engine_step()` | 从函数入口到 `outputs, model_executed = self.step_fn()` | 找到 engine step 实际调用的入口 |
| `EngineCore.__init__()` | 只看 `self.step_fn = (...)` 的赋值 | 确认它可能指向哪两个函数 |
| `EngineCore.step()` | 从函数入口读到 `self.scheduler.schedule(...)`，停止 | 找到普通路径的 scheduler 调用 |
| `EngineCore.step_with_batch_queue()` | 搜索其中的 `self.scheduler.schedule(...)`，只看外围 `if` | 标记另一条调用路径，不读 batch queue 机制 |

然后打开 [v1/core/sched/scheduler.py](/home/boran/PycharmProjects/vllm-playground/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py)，进入 **`Scheduler.schedule()`**。按文件顺序做下面几个小跳转，暂时跳过循环内部的大段逻辑。

| 阅读范围的起止锚点 | 这一段观察什么 |
|---|---|
| 从函数入口，到 `token_budget = self.max_num_scheduled_tokens` 及紧邻的 `input_budget` 初始化 | 看本轮建立了哪些结果容器；标记 budget 的首次赋值 |
| 从注释 `First, schedule the RUNNING requests.`，到循环内 `request = self.running[req_index]` | 看 RUNNING 块的位置、循环条件和读取的容器 |
| 从注释 `Next, schedule the WAITING requests.`，到 `request = request_queue.peek_request()` | 看 WAITING 块的位置、外围条件和读取的容器；与上一块比较顺序 |
| 从注释 `Check if the scheduling constraints are satisfied.`，到这组 `assert` 结束 | 看请求处理之后有哪些顶层检查，只看检查对象 |
| 搜索 `new_reqs_data` 和 `cached_reqs_data` 的赋值，只看赋值及调用参数 | 看输出构造之前准备了哪些数据；不进入辅助函数 |
| 从 `scheduler_output = SchedulerOutput(...)` 到右括号 | 看本轮结果如何封装，先扫字段名 |
| 从注释 `Build the connector meta for ECConnector` 往后，读到 `return scheduler_output` | 看构造后还有哪些收尾调用，以及最终返回的位置；不进入被调用函数 |

另外只放三个书签，**不读机制**：

- **Token budget**：函数开头的 `token_budget = self.max_num_scheduled_tokens`。
- **首次 KV allocation 调用**：RUNNING 块内，注释 `Schedule newly needed KV blocks for the request.` 下的 `self.kv_cache_manager.allocate_slots(...)`。
- **Preemption 分支**：同一块内，注释 `Preempt the lowest-priority request.`；再定位其后的 `self._preempt_request(...)`，不进入函数。

## 探索中的问题与结论

### 调度骨架与后续阅读地图

这里的“输入”主要是 Scheduler 持有或通过管理器访问的状态，不是 `schedule()` 的形参；本轮 token budget 在函数入口初始化。图中保留主骨架与已观察的关键分支，其他辅助分支省略。末尾的模型执行仅标出后续方向。

**详细骨架**（点击图片查看 SVG）

[![详细骨架](diagrams/vllm-scheduler-checkpoint2-detailed.png)](diagrams/vllm-scheduler-checkpoint2-detailed.svg)


**简化主线**（点击图片查看 SVG）

[![简化主线](diagrams/vllm-scheduler-checkpoint2-simplified.png)](diagrams/vllm-scheduler-checkpoint2-simplified.svg)

WAITING 的外围条件是 `not preempted_reqs and self._pause_state == PauseState.UNPAUSED`；因此 RUNNING 中发生实际抢占后，本轮会跳过 WAITING。图中的循环箭头表示扫描或重试入口，具体条件留给对应 checkpoint。

| 后续 checkpoint | 深入内容 |
|---|---|
| Checkpoint 3 | `num_new_tokens` 与本轮 token budget 的计算、裁剪 |
| Checkpoint 4 | WAITING admission 与 KV allocation |
| Checkpoint 5 | RUNNING growth、allocation 失败与 preemption |
| Checkpoint 6 | `SchedulerOutput` 的具体字段与交接接口 |

### 1. step 与 step_with_batch_queue 是否一个等结果，一个直接返回？

`EngineCoreProc._process_engine_step()` 调用 `step_fn()`；具体路径在 `EngineCore.__init__()` 中选择：

```python
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
```

两条路径都会调用 scheduler。`step()` 提交本轮 batch 后通过 `future.result()` 等结果；batch queue 路径满足条件时先返回主循环，之后仍会取出较早提交的 batch 并等结果。这里的提交单位是一轮 batch，返回也不是向 API 用户返回整个请求的完成结果。TODO：batch queue 机制。

### 2. _should_throttle_prefills 是否影响本轮 prefill？

它向 `schedule()` 传入一个可选的 prefill 节奏控制开关。普通 `EngineCore` 的实现如下，DP 版本则重写判断；返回 True 也不代表必然禁止所有 prefill，scheduler 还会结合其他条件。

```python
def _should_throttle_prefills(self) -> bool:
    """Whether to defer new prefills this step (DP prefill balancing).
    Overridden by the DP engine core; never throttles otherwise."""
    return False
```

TODO：DP prefill 节奏控制的具体条件。

### 3. RUNNING 和 WAITING 的处理顺序，以及预算是否共享？

RUNNING 先处理，WAITING 后处理；WAITING 使用同一轮剩余的 `token_budget`，没有重新获得一份预算。以下分别摘录两块的循环入口，以及 RUNNING 中扣减预算的语句：

```python
while req_index < len(self.running) and token_budget > 0:
```

```python
token_budget -= num_new_tokens
```

```python
while token_budget > 0:
```

这对应 Mixed Prefill–Decode interference / token budget 实验中的工作量竞争，但不能把顺序直接理解为 decode 先、prefill 后：RUNNING 也可能包含未完成的 chunked prefill。

### 4. 为什么 num_new_tokens 随限制不断减少？

已经观察到它会受到多种条件裁剪，然后用于申请本轮 KV 空间。它表示本轮计划为请求计算的 token 数，可包含 prefill 工作，并不专指新生成的输出 token。

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

TODO：初始工作量如何计算、各项预算如何裁剪；本次没有深入这些规则。

### 5. allocation 失败后，是否立即抢占某个请求？

已定位 RUNNING 中的失败分支：成功则退出分配循环；失败但 connector 有 pending block frees 时也先退出；否则继续选择 victim。以下代码只是选择候选，真正执行抢占在后面的 `_preempt_request()`：

```python
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),
    )
else:
    preempted_req = self.running[-1]
```

TODO：抢占如何改变状态、释放空间和重新入队。

### 6. 已在 RUNNING，是否意味着资源已经分配完成？

不意味着整个生命周期的 KV 空间已经提前分配。请求通常持有已有 blocks，后续逐轮、逐请求按需增长；容量够时不必新增 block，也不是每轮重新分配全部空间。申请空间的请求与 victim 可以不同，某个请求本轮尚未被遍历也可能成为 victim。

因此，RUNNING 不保证本轮一定执行，已有 KV blocks 也不保证后续增长有空间。这个结论限定于 KV 空间；其他资源各有管理方式。TODO：RUNNING growth 与 preemption 的具体机制。

### 7. 调度后的 assert 在检查什么？

以下是 `Scheduler.schedule()` 中的一组原始检查：

```python
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

assert token_budget >= 0
assert input_budget >= 0
assert len(self.running) <= self.max_num_running_reqs
```

它们检查本轮总 token 数不超上限、两份剩余预算不为负，以及 RUNNING 请求数量不超上限。这是验证前面的调度结果，不是在这里才裁剪工作量；RUNNING 数量也不等于本轮安排计算的请求数量。TODO：`input_budget` 与 `token_budget` 的具体区别。

### 8. 新请求 / cached 请求数据是什么，构造 SchedulerOutput 后还有什么？

`new_reqs_data` 提供建立请求执行状态所需的信息，`cached_reqs_data` 为执行侧已有请求传递更新，也包含恢复请求；这里的 cached 不代表命中 prefix KV cache。构造输出就是将这些数据与本轮计划一起传参封装，以下摘录构造参数：

```python
scheduled_new_reqs=new_reqs_data,
scheduled_cached_reqs=cached_reqs_data,
num_scheduled_tokens=num_scheduled_tokens,
total_num_scheduled_tokens=total_num_scheduled_tokens,
```

附加 metadata 等处理之后，末尾更新调度状态并返回；返回值由 engine 的 `scheduler_output` 接住。

```python
with record_function_or_nullcontext("schedule: update_after_schedule"):
    self._update_after_schedule(scheduler_output)
return scheduler_output
```

TODO：输出字段的具体用途、状态更新，以及如何交给 ModelRunner。

### 9. 函数太长，如何梳理才有收获？

本次只沿“初始化 → RUNNING → WAITING → 检查与数据准备 → 构造输出 → 收尾返回”识别边界。WAITING 入口缩小为“外围条件 → 循环入口 → 选队列 → 取请求”；比较两块共享预算的关系，比只认函数位置更有意义。

当前最重要的三个认识：RUNNING 不等于本轮一定执行；WAITING 使用剩余预算；`schedule()` 产出本轮执行计划。具体的 token 计算、admission、preemption 和 KVCacheManager 内部留到后续 checkpoints。
