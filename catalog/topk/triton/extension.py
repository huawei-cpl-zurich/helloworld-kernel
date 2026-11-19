@triton.jit
def program_id(
    axis: int,
) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)
