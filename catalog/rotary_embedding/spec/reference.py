import torch
def reference(state,cos,sin,interleaved):
    x=state.float()
    cosine=cos.float()[:,None,:];sine=sin.float()[:,None,:]
    if interleaved:
        left=x[...,0::2];right=x[...,1::2]
        out=torch.stack((left*cosine-right*sine,left*sine+right*cosine),dim=-1).flatten(-2)
    else:
        left,right=x.chunk(2,dim=-1)
        out=torch.cat((left*cosine-right*sine,left*sine+right*cosine),dim=-1)
    return {'out':out.to(state.dtype)}
