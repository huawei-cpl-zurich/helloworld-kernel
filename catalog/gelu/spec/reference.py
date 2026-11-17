import torch
def reference(x,approximate_tanh):
    value=torch.nn.functional.gelu(x.float(),approximate='tanh' if approximate_tanh else 'none')
    x.copy_(value.to(x.dtype))
    return {'out':x}
