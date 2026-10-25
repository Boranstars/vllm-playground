import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = input_ptr + pid * input_row_stride

    col_offset = tl.arange(0, BLOCKSIZE)
    mask = col_offset < n_cols
    # load the row of the input matrix, applying a mask to handle cases where n_cols is not a multiple of BLOCKSIZE
    row = tl.load(row_start + col_offset, mask=mask, other=-float('inf'))
    row_max = tl.max(row, axis=0) # scalar, the maximum value in the row
    row_exp = tl.exp(row - row_max) # [BLOCKSIZE,] subtracting row_max for numerical stability
    row_sum = tl.sum(row_exp, axis=0) # scalar, the sum of exponentials in the row
    row_softmax = row_exp / row_sum # [BLOCKSIZE,] the softmax values for the row

    # store the values back to the output matrix, applying the same mask
    output_row_start = output_ptr + pid * output_row_stride
    output_ptrs = output_row_start + col_offset
    tl.store(output_ptrs, row_softmax, mask=mask)





def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCKSIZE = triton.next_power_of_2(n_cols)
    y: torch.Tensor = torch.empty_like(x)
    grid = lambda meta: (n_rows,)

    # Launch the kernel
    softmax_kernel[grid](x, y, x.stride(0), y.stride(0), n_cols, BLOCKSIZE=BLOCKSIZE)

    return y


if __name__ == '__main__':
    # Test the softmax function
    x = torch.randn(1823, 781, device='cuda')
    y = softmax(x)
    y_torch = torch.softmax(x, dim=1)

    # Check if the results are close
    assert torch.allclose(y, y_torch, atol=1e-6), "Softmax results do not match!"
    print("Softmax results match with PyTorch implementation.")