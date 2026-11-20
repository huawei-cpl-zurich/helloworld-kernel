import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
from triton.language.extra.cann import extension
path=Path(__file__).parent/'kernel.py'
names={'_attn_fwd_inner','_attn_fwd'}
functions=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in functions}==names
module=types.ModuleType('upstream_flash_attention');module.__file__=str(path)
module.triton=triton;module.tl=tl;module.extension=extension;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
def build(signature):
    q,k,v=signature['inputs'];z,h,n,d=q['shape'];assert d==64 and n in (128,256)
    scale,=signature['scalars'];assert scale['name']=='sm_scale'
    constants={'acc':None,'sm_scale':scale['value'],'Z':z,'H':h,'N_CTX':n,'HEAD_DIM':d,'BLOCK_M':64,'BLOCK_N':64,'STAGE':3}
    for tensor,last in [('q','k'),('k','k'),('v','k'),('o','n')]:
        constants.update({f'stride_{tensor}z':h*n*d,f'stride_{tensor}h':n*d,f'stride_{tensor}'+('m' if tensor in ('q','o') else 'n'):d,f'stride_{tensor}'+last:1})
    return {'kernel':module._attn_fwd,'bindings':[['Q','input','q'],['K','input','k'],['V','input','v'],['M','output','lse'],['Out','output','out']],'constants':constants,'grid':[20,1,1]}
