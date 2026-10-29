import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_v0(a_ptr, b_ptr, c_ptr,
                     M: int, N: int, K: int,
                     stride_am: int, stride_ak: int,
                     stride_bk: int, stride_bn: int,
                     stride_cm: int, stride_cn: int,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
                     ):
    # 计算pid, 这里使用朴素二维网格, pid_m表示行的索引, pid_n表示列的索引
    pid_m: int = tl.program_id(0)
    pid_n: int = tl.program_id(1)

    # 计算每个线程块的偏移量, 这里使用了triton提供的tl.arange函数来生成一个范围内的整数序列, 然后加上pid_m和pid_n的偏移量, 得到每个线程块对应的矩阵块的起始位置
    offset_m: tl.tensor = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n: tl.tensor = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k: tl.tensor = tl.arange(0, BLOCK_K)

    # 计算矩阵指针，这里使用了广播机制，将offset_m和offset_k进行广播，得到每个线程块对应的矩阵块的指针

    # [BLOCK_M, BLOCK_K] = [BLOCK_M, 1] + [1, BLOCK_K]
    # a[i, j] = a_ptr + i * stride_am + j * stride_ak
    a_ptrs: tl.tensor = (a_ptr
                         + offset_m[:, None] * stride_am
                         + offset_k[None, :] * stride_ak)
    # [BLOCK_K, BLOCK_N] = [BLOCK_K, 1] + [1, BLOCK_N]
    b_ptrs: tl.tensor = (b_ptr +
                         offset_k[:, None] * stride_bk +
                         offset_n[None, :] * stride_bn)

    # [BLOCK_M, BLOCK_N] = [BLOCK_M, 1] + [1, BLOCK_N]
    c_ptrs: tl.tensor = (c_ptr +
                         offset_m[:, None] * stride_cm +
                         offset_n[None, :] * stride_cn)

    # 累加器存储中间结果，采用更高的fp32精度进行计算，避免精度损失
    acc: tl.tensor = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        current_k = k * BLOCK_K + offset_k  # 每个块的K索引(起始位置 + 偏移量)

        mask_a = (offset_m[:, None] < M) & (current_k[None, :] < K)  # A[M,K],需要考虑两个方向的越界，利用广播机制生成两边
        mask_b = (current_k[:, None] < K) & (offset_n[None, :] < N)  # B[K,N],需要考虑两个方向的越界，利用广播机制生成两边
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # 每次累加
        acc = tl.dot(a_tile, b_tile, acc=acc)

        # tile前进

        # 这里需要让各自指针沿着K方向移动一个BLOCK,反映在指针上就是沿着K方向移动BLOCK_K个元素, 也就是stride_ak和stride_bk的步长
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回C矩阵

    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    c = acc.to(tl.float16)

    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def matmul_kernel_v1(a_ptr, b_ptr, c_ptr,
                     M: int, N: int, K: int,
                     stride_am: int, stride_ak: int,
                     stride_bk: int, stride_bn: int,
                     stride_cm: int, stride_cn: int,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP_M: tl.constexpr
                     ):
    # 计算pid, 这里使用一维网格

    pid: int = tl.program_id(0)

    num_pid_m: int = tl.cdiv(M, BLOCK_M)
    num_pid_n: int = tl.cdiv(N, BLOCK_N)

    # 按M方向为GROUP,增加GPU的L2 cache复用可能性

    group_size = GROUP_M * num_pid_n
    pid_group_offset: int = pid % group_size
    num_group = pid // group_size # 该pid在第几个group中

    # 我们需要看看剩下的是否是完整的，之前已经可以通过num_group确定这是第几个，再看看剩下的是否完整即可
    first_pid_m = num_group * GROUP_M
    remaining_m = num_pid_m - first_pid_m
    group_size_m:int = tl.minimum(remaining_m, GROUP_M)

    # 接下来计算二维索引，其中pid_m跨group就增加对应行数，pid_n不需要
    pid_m: int = pid_group_offset % group_size_m + num_group * GROUP_M
    pid_n: int = pid_group_offset // group_size_m


    # 计算每个线程块的偏移量, 这里使用了triton提供的tl.arange函数来生成一个范围内的整数序列, 然后加上pid_m和pid_n的偏移量, 得到每个线程块对应的矩阵块的起始位置
    offset_m: tl.tensor = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n: tl.tensor = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k: tl.tensor = tl.arange(0, BLOCK_K)

    # 计算矩阵指针，这里使用了广播机制，将offset_m和offset_k进行广播，得到每个线程块对应的矩阵块的指针

    # [BLOCK_M, BLOCK_K] = [BLOCK_M, 1] + [1, BLOCK_K]
    # a[i, j] = a_ptr + i * stride_am + j * stride_ak
    a_ptrs: tl.tensor = (a_ptr
                         + offset_m[:, None] * stride_am
                         + offset_k[None, :] * stride_ak)
    # [BLOCK_K, BLOCK_N] = [BLOCK_K, 1] + [1, BLOCK_N]
    b_ptrs: tl.tensor = (b_ptr +
                         offset_k[:, None] * stride_bk +
                         offset_n[None, :] * stride_bn)

    # [BLOCK_M, BLOCK_N] = [BLOCK_M, 1] + [1, BLOCK_N]
    c_ptrs: tl.tensor = (c_ptr +
                         offset_m[:, None] * stride_cm +
                         offset_n[None, :] * stride_cn)

    # 累加器存储中间结果，采用更高的fp32精度进行计算，避免精度损失
    acc: tl.tensor = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        current_k = k * BLOCK_K + offset_k  # 每个块的K索引(起始位置 + 偏移量)

        mask_a = (offset_m[:, None] < M) & (current_k[None, :] < K)  # A[M,K],需要考虑两个方向的越界，利用广播机制生成两边
        mask_b = (current_k[:, None] < K) & (offset_n[None, :] < N)  # B[K,N],需要考虑两个方向的越界，利用广播机制生成两边
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # 每次累加
        acc = tl.dot(a_tile, b_tile, acc=acc)

        # tile前进

        # 这里需要让各自指针沿着K方向移动一个BLOCK,反映在指针上就是沿着K方向移动BLOCK_K个元素, 也就是stride_ak和stride_bk的步长
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回C矩阵

    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    c = acc.to(tl.float16)

    tl.store(c_ptrs, c, mask=mask_c)


@triton.autotune(
    configs=[
        # 1. 保留现有的小 Tile 与基线配置
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),

        # 2. 扩大 M/N 方向的 Tile
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),

        # 3. 单独增加 K 方向的 Tile
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64,
             "BLOCK_K": 64, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128,
             "BLOCK_K": 64, "GROUP_M": 4},
            num_warps=4, num_stages=3,
        ),

        # 保留第一轮找到的最优 Tile，只改变 num_warps
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=8,
            num_stages=3,
        ),
        # 保留第一轮找到的最优 Tile，只改变 num_stages
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128,
             "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4,
            num_stages=5,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_v2(a_ptr, b_ptr, c_ptr,
                     M: int, N: int, K: int,
                     stride_am: int, stride_ak: int,
                     stride_bk: int, stride_bn: int,
                     stride_cm: int, stride_cn: int,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP_M: tl.constexpr
                     ):
    # 计算pid, 这里使用一维网格

    pid: int = tl.program_id(0)

    num_pid_m: int = tl.cdiv(M, BLOCK_M)
    num_pid_n: int = tl.cdiv(N, BLOCK_N)

    # 按M方向为GROUP,增加GPU的L2 cache复用可能性

    group_size = GROUP_M * num_pid_n
    pid_group_offset: int = pid % group_size
    num_group = pid // group_size # 该pid在第几个group中

    # 我们需要看看剩下的是否是完整的，之前已经可以通过num_group确定这是第几个，再看看剩下的是否完整即可
    first_pid_m = num_group * GROUP_M
    remaining_m = num_pid_m - first_pid_m
    group_size_m:int = tl.minimum(remaining_m, GROUP_M)

    # 接下来计算二维索引，其中pid_m跨group就增加对应行数，pid_n不需要
    pid_m: int = pid_group_offset % group_size_m + num_group * GROUP_M
    pid_n: int = pid_group_offset // group_size_m


    # 计算每个线程块的偏移量, 这里使用了triton提供的tl.arange函数来生成一个范围内的整数序列, 然后加上pid_m和pid_n的偏移量, 得到每个线程块对应的矩阵块的起始位置
    offset_m: tl.tensor = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n: tl.tensor = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k: tl.tensor = tl.arange(0, BLOCK_K)

    # 计算矩阵指针，这里使用了广播机制，将offset_m和offset_k进行广播，得到每个线程块对应的矩阵块的指针

    # [BLOCK_M, BLOCK_K] = [BLOCK_M, 1] + [1, BLOCK_K]
    # a[i, j] = a_ptr + i * stride_am + j * stride_ak
    a_ptrs: tl.tensor = (a_ptr
                         + offset_m[:, None] * stride_am
                         + offset_k[None, :] * stride_ak)
    # [BLOCK_K, BLOCK_N] = [BLOCK_K, 1] + [1, BLOCK_N]
    b_ptrs: tl.tensor = (b_ptr +
                         offset_k[:, None] * stride_bk +
                         offset_n[None, :] * stride_bn)

    # [BLOCK_M, BLOCK_N] = [BLOCK_M, 1] + [1, BLOCK_N]
    c_ptrs: tl.tensor = (c_ptr +
                         offset_m[:, None] * stride_cm +
                         offset_n[None, :] * stride_cn)

    # 累加器存储中间结果，采用更高的fp32精度进行计算，避免精度损失
    acc: tl.tensor = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        current_k = k * BLOCK_K + offset_k  # 每个块的K索引(起始位置 + 偏移量)

        mask_a = (offset_m[:, None] < M) & (current_k[None, :] < K)  # A[M,K],需要考虑两个方向的越界，利用广播机制生成两边
        mask_b = (current_k[:, None] < K) & (offset_n[None, :] < N)  # B[K,N],需要考虑两个方向的越界，利用广播机制生成两边
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # 每次累加
        acc = tl.dot(a_tile, b_tile, acc=acc)

        # tile前进

        # 这里需要让各自指针沿着K方向移动一个BLOCK,反映在指针上就是沿着K方向移动BLOCK_K个元素, 也就是stride_ak和stride_bk的步长
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回C矩阵

    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    c = acc.to(tl.float16)

    tl.store(c_ptrs, c, mask=mask_c)

def matmul_v0(a: torch.Tensor, b: torch.Tensor, blocks: tuple[int, int, int]) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Inner dimensions must match for matrix multiplication"
    M, K = a.shape
    _, N = b.shape

    c = torch.empty_like(torch.empty((M, N), device=a.device, dtype=a.dtype))
    BLOCK_M, BLOCK_N, BLOCK_K = blocks

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']),)  # 使用二维网格

    matmul_kernel_v0[grid](a, b, c,
                           M, N, K,
                           a.stride(0), a.stride(1),
                           b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    return c


def matmul_v1(a: torch.Tensor, b: torch.Tensor, blocks: tuple[int, ...]) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Inner dimensions must match for matrix multiplication"
    M, K = a.shape
    _, N = b.shape

    c = torch.empty_like(torch.empty((M, N), device=a.device, dtype=a.dtype))
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, *_ = blocks

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)  # 使用一维网格

    matmul_kernel_v1[grid](a, b, c,
                           M, N, K,
                           a.stride(0), a.stride(1),
                           b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)

    return c

def matmul_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Inner dimensions must match for matrix multiplication"
    M, K = a.shape
    _, N = b.shape

    c = torch.empty_like(torch.empty((M, N), device=a.device, dtype=a.dtype))


    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)  # 使用一维网格

    matmul_kernel_v2[grid](a, b, c,
                           M, N, K,
                           a.stride(0), a.stride(1),
                           b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
             )

    return c

def do_benchmark():

    for size in [1024, 2048, 4096]:
        M = N = K = size

        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        bench_fns = {
            "v0": lambda: matmul_v0(a, b, blocks=(64, 64, 32)),
            "v1": lambda: matmul_v1(a, b, blocks=(64, 64, 32, 4)),
            "v2": lambda: matmul_v2(a, b), # v2使用autotune,不需要手动指定blocks
        }

        p50_samples = {"v0": [], "v1": [], "v2": []}

        # 提前完成编译和当前尺寸的 autotune，正式测量稳态性能。
        for fn in bench_fns.values():
            fn()
        torch.cuda.synchronize()

        # 遍历全部排列，每个版本在每个位置各出现两次。
        orders = [
            ("v0", "v1", "v2"),
            ("v1", "v2", "v0"),
            ("v2", "v0", "v1"),
            ("v2", "v1", "v0"),
            ("v1", "v0", "v2"),
            ("v0", "v2", "v1"),
        ]
        for round_idx, order in enumerate(orders):

            results = {}

            for name in order:
                q = triton.testing.do_bench(
                    bench_fns[name],
                    warmup=100,
                    rep=500,
                    quantiles=[0.2, 0.5, 0.8],
                )

                results[name] = q
                p50_samples[name].append(q[1])

            v0_p50 = results["v0"][1]
            v1_p50 = results["v1"][1]
            v2_p50 = results["v2"][1]
            improvement_v1 = (v0_p50 - v1_p50) / v0_p50 * 100
            improvement_v2 = (v1_p50 - v2_p50) / v1_p50 * 100

            print(
                f"Size={size}³, Round={round_idx + 1}, "
                f"Order={order}, "
                f"v0 P50={v0_p50:.3f} ms, "
                f"v1 P50={v1_p50:.3f} ms, "
                f"v2 P50={v2_p50:.3f} ms, "
                f"v1 vs v0={improvement_v1:+.2f}%, "
                f"v2 vs v1={improvement_v2:+.2f}%"
            )

        # 汇总六轮各自的 P50，再取中位数
        v0_median = statistics.median(p50_samples["v0"])
        v1_median = statistics.median(p50_samples["v1"])
        v2_median = statistics.median(p50_samples["v2"])
        improvement_v1 = (v0_median - v1_median) / v0_median * 100
        improvement_v2 = (v1_median - v2_median) / v1_median * 100

        # Torch 作为参考基线
        torch_q = triton.testing.do_bench(
            lambda: a @ b,
            warmup=100,
            rep=500,
            quantiles=[0.2, 0.5, 0.8],
        )

        print(
            f"SUMMARY Size={size}³ | "
            f"v0 median-of-P50={v0_median:.3f} ms | "
            f"v1 median-of-P50={v1_median:.3f} ms | "
            f"v2 median-of-P50={v2_median:.3f} ms | "
            f"v1 vs v0={improvement_v1:+.2f}% | "
            f"v2 vs v1={improvement_v2:+.2f}% | "
            f"Torch P50={torch_q[1]:.3f} ms"
        )

def benchmark_num_stages():
    sizes = [1024, 2048, 4096]

    for size in sizes:
        M = N = K = size
        a = torch.randn(
            (size, size),
            device="cuda",
            dtype=torch.float16,
        )
        b = torch.randn(
            (size, size),
            device="cuda",
            dtype=torch.float16,
        )

        # 输出和 launch 参数均提前准备，计时函数内只发射固定配置的 kernel。
        c3 = torch.empty((M, N), device=a.device, dtype=a.dtype)
        c5 = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(M, 64) * triton.cdiv(N, 128),)
        launch = matmul_kernel_v1[grid]
        args3 = (
            a, b, c3, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c3.stride(0), c3.stride(1),
        )
        args5 = (
            a, b, c5, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c5.stride(0), c5.stride(1),
        )

        def stage3():
            launch(
                *args3,
                BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=4,
                num_warps=4, num_stages=3,
            )

        def stage5():
            launch(
                *args5,
                BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=4,
                num_warps=4, num_stages=5,
            )

        # 先触发编译，避免把 JIT 编译时间算进 benchmark
        stage3()
        stage5()
        torch.cuda.synchronize()

        # 检查两种配置彼此一致，并分别与 PyTorch 对比。
        expected = a @ b
        torch.testing.assert_close(c3, expected, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(c5, expected, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(
            c3,
            c5,
            atol=1e-2,
            rtol=1e-3,
        )

        bench_fns = {3: stage3, 5: stage5}
        p50_samples = {3: [], 5: []}
        for round_idx in range(4):
            order = (3, 5) if round_idx % 2 == 0 else (5, 3)
            results = {}
            for stages in order:
                q = triton.testing.do_bench(
                    bench_fns[stages],
                    warmup=100,
                    rep=500,
                    quantiles=[0.2, 0.5, 0.8],
                )
                results[stages] = q[1]
                p50_samples[stages].append(q[1])

            change = (results[5] - results[3]) / results[3] * 100
            print(
                f"Size={size}^3, Round={round_idx + 1}, Order={order} | "
                f"stages=3 P50={results[3]:.6f} ms | "
                f"stages=5 P50={results[5]:.6f} ms | "
                f"stage5 latency change vs stage3={change:+.4f}%"
            )

        median3 = statistics.median(p50_samples[3])
        median5 = statistics.median(p50_samples[5])
        change = (median5 - median3) / median3 * 100
        print(
            f"SUMMARY Size={size}^3 | "
            f"stages=3 median P50={median3:.6f} ms | "
            f"stages=5 median P50={median5:.6f} ms | "
            f"stage5 latency change vs stage3={change:+.4f}% "
            f"(negative=faster)\n"
        )
if __name__ == '__main__':

    M: int = 256
    K: int = 130
    N: int = 256
    blocks: tuple[int, int, int] = (16, 16, 16)
    group = 4

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)


    c_0 = matmul_v0(a, b, blocks=blocks)
    c_1 = matmul_v1(a, b, blocks=blocks + (group,))
    c_2 = matmul_v2(a, b)
    c_torch: torch.Tensor = a @ b


    try:
        torch.testing.assert_close(c_torch, c_0, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(c_torch, c_1, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(c_torch, c_2, atol=1e-2, rtol=1e-3)
        print("Results match with PyTorch implementation.")
    except AssertionError as e:
        print("Results do not match with PyTorch implementation.")
        print(e)


    # benchmark
    # do_benchmark()
    benchmark_num_stages()
