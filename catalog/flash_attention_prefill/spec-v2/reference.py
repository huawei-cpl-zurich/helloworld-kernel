import torch
def reference(q,k,v,sm_scale):
    n=q.shape[-2]
    scores=torch.matmul(q.float(),k.float().transpose(-1,-2))*sm_scale
    scores=scores.masked_fill(torch.ones((n,n),dtype=torch.bool,device=q.device).triu(1),float('-inf'))
    outputs={'out':torch.matmul(scores.softmax(dim=-1),v.float()).to(q.dtype),'lse':torch.logsumexp(scores,dim=-1)}
    if q.device.type=='npu':
        import torch_npu
        outputs={name:torch_npu.npu_format_cast(value,2) if torch_npu.get_npu_format(value)!=2 else value for name,value in outputs.items()}
    return outputs
