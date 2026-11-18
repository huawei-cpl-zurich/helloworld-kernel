import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
path=Path(__file__).parent/'kernel.py';names={'_block_indexer','_grouped_matmul_nd_kernel'}
functions=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in functions}==names
module=types.ModuleType('upstream_grouped_matmul');module.__file__=str(path)
module.triton=triton;module.tl=tl;module.al=al;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
def build(signature):
    a,b,groups=signature['inputs'];m,k=a['shape'];packed,n=b['shape'];g=groups['shape'][0]
    assert packed==g*k and not signature['scalars']
    return {'kernel':module._grouped_matmul_nd_kernel,
      'bindings':[['A','input','a'],['B','input','b'],['C','output','out'],['group_list','input','group_list']],
      'constants':{'GROUPS':g,'N':n,'K':k,'GROUP_LIST_TYPE':0,'BLOCK_M':64,'BLOCK_N':64,'BLOCK_K':64,'BLOCK_THRESHOLD':8},'grid':[32,1,1]}
