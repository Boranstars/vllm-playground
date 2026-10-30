import pytest
import torch

from experiments.triton.kernel.rmsnorm_kernel import rmsnorm


FORWARD_CASES = [
    ("fp16_tiny_1x1", (1, 1), torch.float16, "random"),
    ("fp16_non_power_of_two_7x31", (7, 31), torch.float16, "random"),
    ("fp16_multidimensional_8x16x1000", (8, 16, 1000), torch.float16, "random"),
    ("fp16_many_programs_4096x1024", (4096, 1024), torch.float16, "random"),
    ("fp16_large_feature_64x4096", (64, 4096), torch.float16, "random"),
    ("fp32_128x1000", (128, 1000), torch.float32, "random"),
    ("fp32_large_feature_64x4096", (64, 4096), torch.float32, "random"),
    ("fp16_noncontiguous_8x16x1000", (8, 16, 1000), torch.float16, "sliced"),
    ("fp16_constant_7x31", (7, 31), torch.float16, "constant"),
    ("fp32_constant_7x31", (7, 31), torch.float32, "constant"),
    ("fp16_zero_7x31", (7, 31), torch.float16, "zero"),
    ("fp32_zero_7x31", (7, 31), torch.float32, "zero"),
]


@pytest.mark.parametrize(
    "name,shape,dtype,kind", FORWARD_CASES, ids=[case[0] for case in FORWARD_CASES],
)
@torch.no_grad()
def test_rmsnorm_forward(name, shape, dtype, kind):
    max_error = None
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(42)
        if kind == "sliced":
            base = torch.randn(
                (*shape[:-1], shape[-1] * 2),
                device=device, dtype=dtype, generator=generator,
            )
            x = base[..., ::2]
            assert not x.is_contiguous(), "Sliced input must exercise contiguous handling"
        elif kind in ("constant", "zero"):
            x = torch.full(shape, 2.0 if kind == "constant" else 0.0, device=device, dtype=dtype)
        else:
            x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        weight = torch.randn(shape[-1], device=device, dtype=dtype, generator=generator)
        eps = 1e-5
        atol = rtol = 2e-3 if dtype == torch.float16 else 1e-5

        expected = torch.nn.functional.rms_norm(x, [shape[-1]], weight, eps)
        if kind == "sliced":
            # Pass the original strided view so the wrapper handles conversion.
            with pytest.warns(UserWarning, match="not contiguous"):
                actual = rmsnorm(x, [shape[-1]], weight, eps)
        else:
            actual = rmsnorm(x, [shape[-1]], weight, eps)

        # FP32 subtraction avoids rounding the reported FP16 error.
        if actual.shape == expected.shape:
            max_error = (actual.float() - expected.float()).abs().max().item()
        assert actual.shape == x.shape, f"Shape: {actual.shape} != {x.shape}"
        assert actual.dtype == x.dtype, f"Dtype: {actual.dtype} != {x.dtype}"
        assert actual.device == x.device, f"Device: {actual.device} != {x.device}"
        assert torch.isfinite(actual).all().item(), "Output contains NaN or Inf"
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        if kind == "constant":
            # RMSNorm does not subtract the mean: c / sqrt(c*c + eps) * weight.
            constant_expected = (2.0 / (4.0 + eps) ** 0.5 * weight.float()).to(dtype)
            torch.testing.assert_close(
                actual, constant_expected.expand_as(actual), atol=atol, rtol=rtol,
            )
        elif kind == "zero":
            torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
    except Exception as exc:
        error_text = "N/A (no comparable output)" if max_error is None else f"{max_error:.8e}"
        print(f"{name}: FAIL | max_abs_error={error_text} | {type(exc).__name__}: {exc}")
        raise
    print(f"{name}: PASS | max_abs_error={max_error:.8e}")
