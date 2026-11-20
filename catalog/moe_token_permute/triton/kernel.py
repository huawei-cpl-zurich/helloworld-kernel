@triton.jit
def _permute_kernel(
    Input,              # (num_tokens, hidden_dim)
    Output,             # (num_tokens * top_k, hidden_dim)
    TopK_Indices,       # (num_tokens, top_k) - expert assignments
    SortedIdx,          # (num_tokens * top_k,) - sort order by expert
    num_tokens,
    hidden_dim,
    top_k,
    stride_in_t,       # input stride between tokens
    stride_out_t,      # output stride between tokens
    stride_idx_t,      # top_k_indices stride between tokens
    BLOCK_D: tl.constexpr,   # tile size over hidden dimension
):
    """
    Permute tokens to expert-contiguous layout.

    Each program handles one (token_slot, hidden_dim_block) tile, where
    token_slot indexes into the sorted output and hidden_dim_block tiles
    over the hidden dimension for coalesced access.

    The kernel reads from the original token (determined by SortedIdx)
    and writes to the contiguous output position.
    """
    # pid_t: which slot in the sorted output
    # pid_d: which block of hidden_dim
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_t >= num_tokens * top_k:
        return

    # Which position in the flattened (num_tokens * top_k) array does this
    # sorted slot correspond to?
    src_flat_idx = tl.load(SortedIdx + pid_t)

    # subhadipmitra, 2026-03-14: the flat index encodes both token_id and which
    # of its top_k experts this copy belongs to. We only need token_id for the
    # permute (to know which row of Input to read from). The k_idx matters for
    # the unpermute when we need to look up the corresponding gating weight.
    src_token = src_flat_idx // top_k

    # Compute hidden dim offsets for this block
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < hidden_dim

    # Read from original token position
    in_ptrs = Input + src_token * stride_in_t + d_offs
    vals = tl.load(in_ptrs, mask=d_mask, other=0.0)

    # Write to sorted output position - this write is contiguous across programs
    # which is the whole point of the permutation
    out_ptrs = Output + pid_t * stride_out_t + d_offs
    tl.store(out_ptrs, vals, mask=d_mask)
