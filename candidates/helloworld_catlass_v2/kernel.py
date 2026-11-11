import catlass.tla as tla
@tla.kernel
def forward(x: tla.Tensor, y: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=32):
            tid, _, _ = tla.arch.thread_idx()
            threads, _, _ = tla.arch.thread_block_dim()
            for i in tla.range(tid, x.origin_shape[0], threads):
                y[i] = x[i] + tla.Float32(1.0)
        tla.pipe_barrier(tla.pipes.ALL)
