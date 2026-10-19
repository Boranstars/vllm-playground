import torch

device = "cuda"

a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

# Warmup：避免第一次执行的初始化/JIT等因素污染主要观察区间
for _ in range(5):
    c = a @ b

torch.cuda.synchronize()

# 给真正想观察的区域加一个人类可读标签
torch.cuda.nvtx.range_push("week2_large_gemm")

for _ in range(10):
    c = a @ b

torch.cuda.synchronize()

torch.cuda.nvtx.range_pop()