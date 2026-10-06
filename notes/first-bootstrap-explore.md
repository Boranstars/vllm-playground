1. APIServer 和 EngineCore 各自干什么？
```text
                    用户世界
                       │
                       ▼
                 ┌───────────┐
                 │ APIServer │
                 └─────┬─────┘
                       │
              “要生成什么？”
                       │
                       ▼
                 ┌───────────┐
                 │EngineCore │
                 └─────┬─────┘
                       │
              “这一轮怎么算？”
                       │
                       ▼
             Scheduler / KV Cache
                       │
                       ▼
                 Model Runner
                       │
                       ▼
                      GPU
```

2. 模型加载进 GPU 之后，占了多少显存？
```text
8 GB 显卡显存空间分布（--gpu-memory-utilization 0.75）
┌─────────────────────────────────────────────────────────────┬──────────────┐
│                    vLLM 预算控制区域 (5.76 GiB)               │ 安全保留预留 │
├──────────────┬──────────────┬─────────────┬─────────────────┼──────────────┤
│ 静态模型权重 │ 峰值激活内存 │ CUDA Graph  │ PagedAttention  │ 未分配空间   │
│   1.40 GiB   │   0.60 GiB   │  0.18 GiB   │ KV Cache 动态池 │ (防止系统/   │
│ (Weights)    │ (Peak Act)   │ (Graph Pool)│    3.76 GiB     │  桌面崩溃)   │
└──────────────┴──────────────┴─────────────┴─────────────────┴──────────────┘
```
3. torch.compile 和 CUDA Graph 各自在优化什么？

torch.compile / Inductor 有点类似我以前 TensorRT build engine 的过程：它会根据模型 graph、shape、dtype 和硬件，对计算进行 lowering、fusion、kernel selection/codegen 等优化；而 CUDA Graph 并不改变这些计算实现，它主要优化这些已经确定的 GPU 操作如何被低开销地反复提交。

```text
PyTorch model
     │
     ▼
TorchDynamo
“把计算图抓出来”
     │
     ▼
Computation Graph
     │
     ▼
TorchInductor
“优化、融合、lowering、选择/生成 kernel”
     │
     ├── cuBLAS
     ├── FlashAttention
     ├── native kernel
     └── Triton kernel
     │
     ▼
一组实际 GPU operations
     │
     ▼
CUDA Graph
“把这一串 GPU 操作 capture / replay”
     │
     ▼
GPU
```

4. KV Cache 为什么需要单独占大量显存？

5. 为什么 vLLM 必须在真正接请求之前做
   profile / warmup / graph capture / KV cache allocation？
```text
我想最大化利用 GPU 显存
        ↓
就必须知道其它东西实际占多少
        ↓
所以先 profile / warmup
        ↓
CUDA Graph 自己也吃显存
        ↓
也要先纳入测量
        ↓
剩下的才能安全给 KV Cache
        ↓
最终 runtime 状态确定
        ↓
再 capture / warmup 最终路径
        ↓
开始 serving
```
---

## 整体流程
```text
1. 解析配置
      ↓
2. 加载模型
   知道 weights 多大
      ↓
3. compile / load compiled artifacts
   确定模型怎么算
      ↓
4. profiling + warmup
   测 activation / runtime 开销
      ↓
5. CUDA Graph memory profiling
   确定 graph 还需要多少显存
      ↓
6. KV Cache sizing / allocation
   把剩余显存变成 token capacity
      ↓
7. JIT / kernel warmup
   把最终执行路径准备好
      ↓
8. Final CUDA Graph capture
   建立 serving runtime graph
      ↓
9. Engine ready
      ↓
10. API Server 接受请求
```

## 术语结构
```text
Serving
│
├── API / request handling
│
├── Engine / scheduling
│   ├── requests
│   ├── KV Cache
│   └── batching
│
├── Model Runtime
│
├── Compiler
│   ├── graph capture
│   ├── fusion
│   ├── lowering
│   └── codegen
│
├── execution optimization
│   └── CUDA Graph
│
└── Kernels
    ├── FlashAttention
    ├── FlashInfer
    ├── native
    └── Triton
```

## 实验
1.同一套 workload，只因为 cache state 不同，TTFT 就能从几百毫秒变到十几毫秒，而 Decode 性能却不会因此完全恢复？
