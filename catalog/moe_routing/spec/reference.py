import torch
def reference(logits,k,renormalize):
    probability=torch.softmax(logits.float(),dim=-1)
    weights,experts=torch.topk(probability,int(k),dim=-1,largest=True,sorted=True)
    if renormalize:weights=weights/(weights.sum(dim=-1,keepdim=True)+1e-8)
    rows=logits.shape[0]
    source_rows=torch.arange(rows,dtype=torch.int32,device=logits.device)[:,None]+torch.arange(int(k),dtype=torch.int32,device=logits.device)[None,:]*rows
    return {'weights':weights,'experts':experts,'source_rows':source_rows}
