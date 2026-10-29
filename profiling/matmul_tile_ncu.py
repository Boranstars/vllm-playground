"""Run with python -m profiling.matmul_tile_ncu --case v1 (or tile128)."""

import argparse

import torch
import triton

from experiments.triton.kernel.matmul_kernel import matmul_kernel_v1


def main():
    parser = argparse.ArgumentParser(description="NCU comparison of fixed matmul tiles")
    parser.add_argument("--case", choices=["v1", "tile128"], required=True)
    args = parser.parse_args()

    M = N = K = 2048
    BLOCK_M, BLOCK_K, GROUP_M = 64, 32, 4
    BLOCK_N = 64 if args.case == "v1" else 128
    num_warps, num_stages = 4, 3
    print(
        f"case={args.case} M={M} N={N} K={K} dtype=torch.float16 device=cuda "
        f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} "
        f"GROUP_M={GROUP_M} num_warps={num_warps} num_stages={num_stages}",
        flush=True,
    )

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    def launch():
        matmul_kernel_v1[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M, num_warps=num_warps, num_stages=num_stages,
        )

    launch()  # Warmup/JIT; skip this matching launch in NCU.
    torch.cuda.synchronize()
    launch()  # Profile this matching launch.
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
