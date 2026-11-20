import torch
def reference(x,residual,weight,epsilon):
    value=x.float()+residual.float()
    reciprocal=torch.rsqrt((value*value).mean(dim=-1,keepdim=True)+epsilon)
    out=(value*reciprocal).to(x.dtype)*weight
    x.copy_(out)
    residual.copy_(value.to(residual.dtype))
    return {'out':x,'residual_out':residual}
