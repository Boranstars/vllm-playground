import argparse

import torch
import triton

from experiments.triton.kernel.matmul_kernel import matmul_v0, matmul_v1


def parse_args():


    parser = argparse.ArgumentParser()
    # TODO: add more arguments for M, K, N, blocks, etc.Do they need to be added?
    parser.add_argument("--size", type=int, default=4096, help="Size of the square matrices (M=N=K=size)")
    parser.add_argument("--impl", type=str, choices=["v0", "v1"], default="v0", help="Which matmul implementation to use")

    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    M = N = K = args.size
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    blocks = (64, 64, 32)  # Default block sizes
    if args.impl == "v1":
        blocks = blocks + (4,)  # Add GROUP_M for v1

    for _ in range(5):  # Warm-up runs
        if args.impl == "v0":
            _ = matmul_v0(a, b, blocks=blocks)
        else:
            _ = matmul_v1(a, b, blocks=blocks)

    torch.cuda.synchronize()  # Ensure all operations are complete before timing

    if args.impl == "v0":
        result = matmul_v0(a, b, blocks=blocks)
    else:
        result = matmul_v1(a, b, blocks=blocks)

    torch.cuda.synchronize()  # Ensure all operations are complete before checking results



    expected = a @ b

    torch.testing.assert_close(
        result,
        expected,
        atol=1e-2,
        rtol=1e-3,
    )
    print(f"Matrix multiplication using {args.impl} passed for size {args.size}.")



if __name__ == "__main__":
    main()