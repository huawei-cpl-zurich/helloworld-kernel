import torch
def reference(updates,slots,cache):
    indices=slots.to(torch.int64)
    valid=(indices>=0)&(indices<cache.shape[0])
    cache.index_copy_(0,indices[valid],updates[valid])
    return {'updated':cache}
