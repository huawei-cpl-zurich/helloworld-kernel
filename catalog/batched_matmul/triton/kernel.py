@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DOT_PAD_ONLY_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)
    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for i in range(num_iters):
        if DIVISIBLE_K:
            mask_a = None if DIVISIBLE_M else mask_m[:, None]
            mask_b = None if DIVISIBLE_N else mask_n[None, :]
        else:
            mask_k = offs_k < K - i * TILE_K
            mask_a = (
                mask_k[None, :] if DIVISIBLE_M else mask_m[:, None] & mask_k[None, :]
            )
            mask_b = (
                mask_k[:, None] if DIVISIBLE_N else mask_k[:, None] & mask_n[None, :]
            )
        if DIVISIBLE_K and DIVISIBLE_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        if DIVISIBLE_K and DIVISIBLE_N:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        if DOT_PAD_ONLY_K:
            extension.compile_hint(a, "dot_pad_only_k")
            extension.compile_hint(b, "dot_pad_only_k")
        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M:
        mask_c = mask_n[None, :]
    elif DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    if DIVISIBLE_M and DIVISIBLE_N:
        tl.store(o_ptrs, o)
    else:
        tl.store(o_ptrs, o, mask_c)
