import torch


def reference(q, kv_cache, block_table, kv_lengths, sm_scale):
    pages = kv_cache.index_select(0, block_table.to(torch.int64))
    kv = pages.flatten(0, 1).float()
    scores = (q.float() @ kv.T) * sm_scale
    positions = torch.arange(kv.shape[0], device=kv.device)
    scores = scores.masked_fill(positions >= kv_lengths[:, None, None], float("-inf"))
    return {"out": (scores.softmax(dim=-1) @ kv[:, :512]).to(q.dtype)}
