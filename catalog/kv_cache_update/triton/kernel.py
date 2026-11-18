# SPDX-License-Identifier: Apache-2.0
"""KDA state-copy kernel and metadata validation for worker-owned plans.
@triton.jit
def _kda_state_copy_kernel(
    cache_ptr,
    packed_ptr,
    indices_ptr,
    flags_ptr,
    cache_rows,
    cache_stride_elements,
    payload_elements,
    index_stride,
    flag_stride,
    TO_CACHE: tl.constexpr,
    HAS_FLAGS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one payload tile for one selected state.

    Grid axis 0 selects an entry in indices; axis 1 selects a payload tile.
    Promote before address multiplication to avoid 32-bit intermediate overflow.
    Invalid cache rows use a safe row address AND a false memory-access mask.
    """
    selected = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    element = tile * BLOCK_SIZE + lane
    in_payload = element < payload_elements

    cache_row = tl.load(indices_ptr + selected * index_stride).to(tl.int64)
    valid_row = (cache_row >= 0) & (cache_row < cache_rows)
    safe_row = tl.where(valid_row, cache_row, 0).to(tl.int64)

    # These offsets are in ELEMENTS, not bytes. Typed pointers handle item size.
    cache_offset = safe_row * cache_stride_elements + element
    packed_offset = selected * payload_elements + element

    if TO_CACHE:
        # Flags describe INITIAL state only; never suppress a valid final write.
        copy_mask = in_payload & valid_row
        value = tl.load(packed_ptr + packed_offset, mask=copy_mask, other=0)
        tl.store(cache_ptr + cache_offset, value, mask=copy_mask)
    else:
        should_read = valid_row
        if HAS_FLAGS:
            flag = tl.load(flags_ptr + selected * flag_stride)
            should_read = should_read & (flag != 0)
        # Masked reads become zero. Always write the full valid packed payload,
        # so invalid indices / false flags do not leave uninitialized output.
        value = tl.load(
            cache_ptr + cache_offset,
            mask=in_payload & should_read,
            other=0,
        )
        tl.store(packed_ptr + packed_offset, value, mask=in_payload)
