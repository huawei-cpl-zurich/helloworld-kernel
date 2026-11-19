import torch
def reference(tokens,indices,weights):
    n,k=weights.shape
    gathered=tokens[indices.long()].reshape(n,k,tokens.shape[1]).float()
    return {'output':(gathered*weights[:,:,None]).sum(dim=1).to(tokens.dtype)}
