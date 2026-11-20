import torch
def reference(x,k,largest):
    values,indices=torch.topk(x,int(k),dim=-1,largest=largest,sorted=True)
    return {'values':values,'indices':indices}
