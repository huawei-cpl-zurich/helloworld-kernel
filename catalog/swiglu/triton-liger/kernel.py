# Copyright 2024 LinkedIn Corporation; BSD-2-Clause (see LICENSE).
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


# NOTE on `gate_multiplier.to(tl.float32)` below: scalar kernel params are
# specialized to fp32 by eager Triton but to fp64 by Inductor when these kernels
# are launched from inside torch.compile. An fp64 scalar silently promotes the
# whole silu/sigmoid chain to float64, which halves throughput (108us vs 56us at
# 8192x3072 on H100). The explicit cast is a no-op in eager and keeps the
# compiled path identical.


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, gate_multiplier, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32) * gate_multiplier.to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)
