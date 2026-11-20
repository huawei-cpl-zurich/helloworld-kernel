import torch
def reference(x):
    value=x.float()
    scale=value.abs().amax(dim=-1)*(1.0/127.0)
    divisor=torch.where(scale>0,scale,torch.ones_like(scale))
    quantized=torch.round(value/divisor[:,None]).clamp(-127,127).to(torch.int8)
    return {'quantized':quantized,'scale':scale}
