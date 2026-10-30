import random
import warnings

import torch
import triton
import triton.language as tl
import triton.testing


@triton.jit
def layernorm_kernel(input_ptr, output_ptr, weight_ptr, bias_ptr,
                     # mean_ptr, var_ptr, # 现在暂时不使用mean和var
                     feature_size, row_stride,
                     BLOCKSIZE: tl.constexpr, eps):
    pid = tl.program_id(0)

    input_ptr +=  pid * row_stride # 提前先计算好每一行的起始地址
    output_ptr += pid * row_stride # 提前先计算好每一行的起始地址

    means_partial = tl.zeros([BLOCKSIZE], dtype=tl.float32)
    vars_partial = tl.zeros([BLOCKSIZE], dtype=tl.float32)

    # 分块计算均值
    for off in range(0, feature_size, BLOCKSIZE):
        cols = off  + tl.arange(0, BLOCKSIZE)
        mask = cols < feature_size
        means_partial += tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32) # 到这里实际上读取的地址矩阵是input_ptr + pid * row_stride + cols

    means = tl.sum(means_partial, axis=0) / feature_size # 按列方向求和，得到每一行的均值

    # 分块计算方差
    for off in range(0, feature_size, BLOCKSIZE):
        cols = off + tl.arange(0, BLOCKSIZE)
        mask = cols < feature_size
        vals = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vars_partial += tl.where(mask, (vals - means) * (vals - means), 0.0)

    var = tl.sum(vars_partial, axis=0) / feature_size # 按列方向求和，得到每一行的方差

    rstd = 1.0 / tl.sqrt(var + eps)

    # 分块写入
    for off in range(0, feature_size, BLOCKSIZE):
        cols = off + tl.arange(0, BLOCKSIZE)
        mask = cols < feature_size

        # 计算读取指针和掩码
        input_ptrs = input_ptr + cols
        weight_ptrs = weight_ptr + cols
        bias_ptrs = bias_ptr + cols
        output_ptrs = output_ptr + cols


        x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

        x_norm = (x - means) * rstd
        w = tl.load(weight_ptrs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = x_norm * w + b

        tl.store(output_ptrs, y, mask=mask)








class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, normalized_shape: list[int],
                weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Applies Layer Normalization over a mini-batch of inputs as described in the paper
        根据官方，我们可选重载Usage 1 (Combined forward and ctx)
        """


        # input的shape通常为[batch_size, seq_len, feature_size]，我们可以把第一维展开
        if not input.is_contiguous():
            input = input.contiguous()
            warnings.warn("input tensor is not contiguous, making it contiguous for layernorm forward pass.", UserWarning)



        output: torch.Tensor = torch.empty_like(input)
        input = input.reshape(-1, input.shape[-1])  # 展开第一维
        M, N = input.shape  # M是batch_size * seq_len，N是feature_size
        BLOCKSIZE = triton.next_power_of_2(N)  # 这里使用特征维度的下一个2的幂作为block size，暂不限制输入的大小
        MAX_FUSED_SIZE: int = 64 * 1024 // input.element_size()  # 64KB的限制，防止block过大导致性能下降
        BLOCKSIZE = min(BLOCKSIZE, MAX_FUSED_SIZE)  # 限制block size不超过最大限制
        if N > BLOCKSIZE:
            raise RuntimeError(f"Feature size {N} is too large for layernorm kernel, max fused size is {MAX_FUSED_SIZE}.")
        grid = lambda meta: (M,)

        output_2d = output.view(-1, output.shape[-1])
        layernorm_kernel[grid](input, output_2d, weight, bias,
                               # mean, var, # 现在暂时不使用mean和var
                               N, input.stride(0), BLOCKSIZE=BLOCKSIZE, eps=eps)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass is not implemented for LayerNorm.")



# 指定layernorm的apply方法为LayerNorm.apply，这样我们就可以直接使用layernorm来调用前向传播
layernorm = LayerNorm.apply


def benchmark_layernorm(device):
    cases = [
        ("Original", (8, 16, 1000), torch.float16),
        ("Case A: many programs", (4096, 1024), torch.float16),
        ("Case B: large feature", (64, 4096), torch.float16),
        ("Case C: FP32", (128, 1000), torch.float32),
    ]
    for name, shape, dtype in cases:
        input = torch.randn(shape, device=device, dtype=dtype)
        weight = torch.randn((shape[-1],), device=device, dtype=dtype)
        bias = torch.randn((shape[-1],), device=device, dtype=dtype)
        normalized_shape = [shape[-1]]
        eps = 1e-5

        print(f"\n{name} | shape={shape} | dtype={dtype}")
        benchmarks = [
            ("Triton", lambda: layernorm(input, normalized_shape, weight, bias, eps)),
            ("PyTorch", lambda: torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)),
        ]
        random.shuffle(benchmarks)
        print("Benchmark order: " + " -> ".join(label for label, _ in benchmarks))
        for label, fn in benchmarks:
            ms, p20, p90 = triton.testing.do_bench(
                fn, quantiles=[0.5, 0.2, 0.9], warmup=10, rep=1000,
            )
            print(f"{label} LayerNorm Benchmark: {ms:.4f} ms (p20: {p20:.4f} ms, p90: {p90:.4f} ms)")


if __name__ == "__main__":
    torch.manual_seed(42)

    device = triton.runtime.driver.active.get_active_torch_device()
    dtype = torch.float16

    # 创建测试数据
    M, N = 128, 1000
    input = torch.randn((8, 16, N), device=device, dtype=dtype)
    weight = torch.randn((N,), device=device, dtype=dtype)
    bias = torch.randn((N,), device=device, dtype=dtype)

    normalized_shape = [N]
    eps = 1e-5

    # Triton 实现
    output_tri = layernorm(
        input, normalized_shape, weight, bias, eps
    )


    # PyTorch 参考实现
    output_ref = torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, eps
    )

    torch.cuda.synchronize()  # 确保所有CUDA操作完成
    # 正确性比较
    torch.testing.assert_close(
        output_tri,
        output_ref,
        atol=1e-2,
        rtol=0
    )

    max_error = (output_tri - output_ref).abs().max().item()

    print("LayerNorm Forward: PASS")
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output_tri.shape}")
    print(f"Max absolute error: {max_error}")
    print(f"Input dtype:  {input.dtype}")
    print(f"Output dtype: {output_tri.dtype}")

    benchmark_layernorm(device)
