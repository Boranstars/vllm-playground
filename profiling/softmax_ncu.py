"""Run five warmups and one target Softmax launch for Nsight Compute."""

import argparse
from pathlib import Path
import sys

import torch

# Allow direct execution with .venv/bin/python profiling/softmax_ncu.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("simple", "official"), required=True)
    parser.add_argument("--M", type=int, default=4096, help="Number of rows")
    parser.add_argument("--N", type=int, default=4096, help="Number of columns")
    args = parser.parse_args()
    if args.M <= 0 or args.N <= 0:
        parser.error("M and N must be positive")
    return args


def main() -> torch.Tensor:
    args = parse_args()

    if args.impl == "simple":
        from experiments.triton.kernel.softmax_kernel import softmax
    else:
        from experiments.triton.triton_examples.fused_softmax import softmax

    torch.manual_seed(0)
    x = torch.randn((args.M, args.N), device="cuda", dtype=torch.float32)

    # Warm up CUDA, Triton JIT, and execution; each call launches one Softmax kernel.
    for _ in range(5):
        output = softmax(x)
    torch.cuda.synchronize()  # Finish all setup/warmup work before the target launch.

    # Exactly one target launch: NCU skips five matching warmups and captures this one.
    output = softmax(x)
    torch.cuda.synchronize()  # Wait for the asynchronous target launch to finish.
    return output


if __name__ == "__main__":
    output = main()  # Keep the target tensor alive until the process exits.
