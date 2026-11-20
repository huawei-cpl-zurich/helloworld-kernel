import torch
def reference(a,b,scale_a,scale_b):
    value=(a.float()@b.float())*scale_a[:,None]*scale_b[None,:]
    return {'out':value.to(torch.float16)}
