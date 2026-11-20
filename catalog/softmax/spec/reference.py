import torch
def reference(x):
    return {'out':torch.softmax(x.float(),dim=-1).to(x.dtype)}
