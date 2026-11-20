def reference(x,gamma,*,epsilon):
    rstd=(x.float().square().mean(dim=-1)+epsilon).rsqrt()
    return {'y':(x.float()*rstd[:,None]*gamma.float()).to(x.dtype),'rstd':rstd}
