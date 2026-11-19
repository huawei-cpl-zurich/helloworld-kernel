import torch
def reference(tokens,positions):
    return {'output':tokens[(positions.long()//2)].clone()}
