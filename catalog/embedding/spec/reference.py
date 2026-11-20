import torch
def reference(weight,indices):
    return {'out':weight.index_select(0,indices.to(torch.int64))}
