@triton.jit
def static_gate_one(a_ptr,b_ptr,c_ptr,stride:tl.constexpr,n_cols:tl.constexpr,BLOCK_SIZE:tl.constexpr):
    original._swiglu_forward_kernel(a_ptr,b_ptr,c_ptr,stride,tl.full((),1.0,tl.float32),n_cols,BLOCK_SIZE)
