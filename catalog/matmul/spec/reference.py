def reference(a,b):
    return {'out':(a.float()@b.float()).to(a.dtype)}
