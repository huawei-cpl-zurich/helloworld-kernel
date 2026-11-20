import torch
def reference(a,b):
    return {'out':(torch.nn.functional.silu(a.float())*b.float()).to(a.dtype)}
