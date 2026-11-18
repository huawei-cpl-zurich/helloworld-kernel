import torch
def reference(a,b,group_list):
    m,k=a.shape;n=b.shape[1];output=torch.empty((m,n),dtype=a.dtype,device=a.device)
    start=0
    for g,end in enumerate(group_list.tolist()):
        output[start:end]=(a[start:end].float()@b[g*k:(g+1)*k].float()).to(a.dtype)
        start=end
    return {'out':output}
