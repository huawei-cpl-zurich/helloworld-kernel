import torch
def reference(x,weight,bias,epsilon):
    value=x.float()
    mean=value.mean(dim=-1)
    centered=value-mean[:,None]
    rstd=torch.rsqrt((centered*centered).mean(dim=-1)+epsilon)
    out=(centered*rstd[:,None])*weight.float()+bias.float()
    return {'out':out.to(x.dtype),'mean':mean,'rstd':rstd}
