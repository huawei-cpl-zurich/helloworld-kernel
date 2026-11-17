import torch
def reference(a,b):
    return {'out':torch.bmm(a.float(),b.float()).to(a.dtype)}
