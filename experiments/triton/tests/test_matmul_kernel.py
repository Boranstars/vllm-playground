import pytest
import torch

from experiments.triton.kernel.matmul_kernel import matmul_v0, matmul_v1


@pytest.mark.parametrize("matmul_impl", [matmul_v0, matmul_v1], ids=["v0", "v1"])
def test_matmul_basic(matmul_impl):

    M = 256
    K = 128
    N = 256

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    blocks = (16, 16, 16)
    if matmul_impl is matmul_v1:
        blocks = blocks + (4,)
    actual = matmul_impl(a, b, blocks=blocks)
    expected = a @ b

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-2,
        rtol=0,
    )

@pytest.mark.parametrize("matmul_impl", [matmul_v0, matmul_v1], ids=["v0", "v1"])
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
def test_matmul(M, K, N, blocks, matmul_impl):

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    if matmul_impl is matmul_v1:
        blocks = blocks + (4,)
    actual = matmul_impl(a, b, blocks=blocks)
    expected = a @ b



    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-2,
        rtol=1e-3,
    )# fp16下相邻数间隔为2^-10,也就是 9.77e-4,所以1e-3相对误差是一个比较宽松的误差范围，允许1相对ULP


def test_matmul_v1_incomplete_group():
    M, K, N = 130, 128, 256
    blocks = (16, 16, 16)
    GROUP_M = 4
    # ceil(M / BLOCK_M) = 9，M 方向按 4 + 4 + 1 分组。
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    actual = matmul_v1(a, b, blocks=blocks + (GROUP_M,))
    expected = a @ b

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-3)
