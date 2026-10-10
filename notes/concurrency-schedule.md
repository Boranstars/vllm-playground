1. schedule 要管理什么，流程是什么？

```text
                     Incoming requests
                            │
                            ▼
                        Scheduler
                  ┌─────────┴─────────┐
                  │                   │
          Compute constraint     Memory constraint
           token budget           KV capacity
                  │                   │
                  └─────────┬─────────┘
                            │
                per-request token allocation
                            │
                    Continuous Batching
                            │
                    packed GPU workload
                            │
                           GPU
                            │
                     request state update
                            │
                       next iteration
```

2. 设计实验的大致流程
```text
Server 已启动
     │
     ▼
Global warmup
     │
     ▼
选择 concurrency = C
     │
     ▼
启动 C 个 in-flight requests
     │
     ▼
Closed-loop warmup
     │
     ▼
──────── t0 ────────
开始 measurement
始终保持 C in-flight
     │
     │
     │  收集 128 个有效 request
     │
     │
──────── t1 ────────
结束 measurement
     │
     ▼
停止补充新请求
     │
     ▼
drain 剩余 requests
     │
     ▼
等待 server idle
     │
     ▼
下一个 concurrency
```

```text
time ───────────────────────────────────────→


启动 C 个 workers
│
├──── warmup / settle ────┤
│                         │
│    concurrency ≈ C      │
│                         │
                          t0
                           │
                           │ measurement
                           │
                           │ concurrency ≈ C
                           │
                           │ collect 128 requests
                           │
                           │
                          t1
                           │
                           ├──── drain ────┤
                           │
                                        idle
```

4. 如何理解实验？
```text
Load
我加了多少压力？
→ concurrency / Running

Capacity
系统多做了多少活？
→ aggregate throughput

Cost
用户体验付出多少代价？
→ TTFT / TPOT / E2E

Bottleneck
为什么开始不划算？
→ throughput flattening
   Waiting
   KV
   preemption
   GPU profiling
```