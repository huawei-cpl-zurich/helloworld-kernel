@triton.jit
def _unpermute_kernel(
    Input,              # (num_tokens * top_k, hidden_dim) - expert outputs in sorted order
    Output,             # (num_tokens, hidden_dim) - combined output
    SortedIdx,          # (num_tokens * top_k,) - maps sorted position -> flat (token, k) index
    TopK_Weights,       # (num_tokens, top_k) - gating weights
    num_tokens,
    hidden_dim,
    top_k,
    stride_in_t,
    stride_out_t,
    stride_w_t,         # weights stride between tokens
    BLOCK_D: tl.constexpr,
):
    """
    Unpermute expert outputs back to original token order with weighted combination.

    Each program handles one (token, hidden_dim_block) tile. For each token,
    it gathers the top_k expert outputs from their sorted positions, multiplies
    by gating weights, and accumulates.
    """
    pid_t = tl.program_id(0)   # token index
    pid_d = tl.program_id(1)   # hidden dim block

    if pid_t >= num_tokens:
        return

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_dim

    # subhadipmitra, 2026-03-14: accumulate in FP32 even though inputs are FP16.
    # The weighted sum across top_k experts can lose significant precision in FP16,
    # especially when one expert dominates (weight ~0.99) and others are small (~0.01).
    # Saw ~2% max error without this on Mixtral configs
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k in range(top_k):
        # Load gating weight for this (token, k)
        weight = tl.load(TopK_Weights + pid_t * stride_w_t + k).to(tl.float32)

        # subhadipmitra, 2026-03-14: restore_idx maps flat_idx -> sorted_position,
        # so we can look up where each (token, k) pair ended up after the sort.
        # This avoids needing an inverse sort at unpermute time
        flat_idx = pid_t * top_k + k
        sorted_pos = tl.load(SortedIdx + flat_idx)

        # Load expert output from sorted position
        in_ptrs = Input + sorted_pos * stride_in_t + d_offs
        expert_out = tl.load(in_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        acc += weight * expert_out

    # Write combined output
    out_ptrs = Output + pid_t * stride_out_t + d_offs
    tl.store(out_ptrs, acc.to(tl.float16), mask=d_mask)
