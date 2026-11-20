import torch
def reference(q,k_cache,v_cache,block_table,q_start,q_lengths,kv_lengths,sm_scale):
    batch,heads,dim=q.shape
    pages,block_size,kv_heads,_=k_cache.shape
    table=block_table.reshape(batch,pages//batch)
    results=[]
    assert q_lengths.tolist()==[1]*batch and q_start.tolist()==list(range(batch))
    for b in range(batch):
        length=int(kv_lengths[b])
        indices=table[b].to(torch.int64)
        keys=k_cache.index_select(0,indices).reshape(-1,kv_heads,dim)[:length].float()
        values=v_cache.index_select(0,indices).reshape(-1,kv_heads,dim)[:length].float()
        rows=[]
        for h in range(heads):
            kh=h//(heads//kv_heads)
            logits=(q[b,h].float()@keys[:,kh,:].T)*sm_scale
            rows.append(torch.softmax(logits,dim=-1)@values[:,kh,:])
        results.append(torch.stack(rows))
    return {'out':torch.stack(results).to(q.dtype)}
