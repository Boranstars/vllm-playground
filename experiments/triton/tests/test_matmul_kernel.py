import pytest
import torch

from experiments.triton.kernel.matmul_kernel import matmul


def test_matmul_basic():

    M = 256
    K = 128
    N = 256

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    actual = matmul(a, b, blocks=(16, 16, 16))
    expected = a @ b

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-2,
        rtol=0,
    )

@pytest.mark.parametrize(
    "M,K,N,blocks",
    [
        (256, 128, 256, (16, 16, 16)),
        (250, 130, 270, (16, 16, 16)),
        (256, 128, 256, (16, 32, 16)),
        (256, 128, 256, (32, 16, 16)),

        (256, 130, 256, (16, 16, 16)),
        (256, 128, 270, (16, 16, 16)),
        (250, 128, 256, (16, 16, 16)),

        (4096,2048,4096,(32,32,32)),
        (4096,2048,4096,(64,64,64)),


    ],
)
def test_matmul(M, K, N, blocks):

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    actual = matmul(a, b, blocks=blocks)
    expected = a @ b

    ref32 = a.float() @ b.float()

    err_triton = (actual.float() - ref32).abs()
    err_torch = (expected.float() - ref32).abs()

    print("Triton vs FP32 ref:")
    print("max:", err_triton.max().item())
    print("mean:", err_triton.mean().item())

    print("Torch vs FP32 ref:")
    print("max:", err_torch.max().item())
    print("mean:", err_torch.mean().item())

    diff = (actual.float() - expected.float()).abs()

    idx = torch.nonzero(diff > 1e-2)

    print(idx)

    for i, j in idx:
        print(
            "index:",
            (i.item(), j.item()),
            "triton:",
            actual[i, j].item(),
            "torch:",
            expected[i, j].item(),
            "fp32:",
            ref32[i, j].item(),
            "triton_err:",
            (actual[i, j].float() - ref32[i, j]).abs().item(),
            "torch_err:",
            (expected[i, j].float() - ref32[i, j]).abs().item(),
        )

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-2,
        rtol=1e-3,
    )# fp16下相邻数间隔为2^-10,也就是 9.77e-4,所以1e-3相对误差是一个比较宽松的误差范围，允许1相对ULP
