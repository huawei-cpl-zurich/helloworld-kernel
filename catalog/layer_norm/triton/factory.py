import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
path=Path(__file__).parent/'kernel.py'
functions=[node for node in ast.parse(path.read_bytes()).body
           if isinstance(node,ast.FunctionDef) and node.name=='_layer_norm_fwd_fused']
assert len(functions)==1
module=types.ModuleType('upstream_layer_norm');module.__file__=str(path)
module.triton=triton;module.tl=tl;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)

def build(signature):
    x,weight,bias=signature['inputs']
    rows,cols=x['shape']
    epsilon,=signature['scalars'];assert epsilon['name']=='epsilon'
    return {'kernel':module._layer_norm_fwd_fused,
      'bindings':[['X','input','x'],['Y','output','out'],['W','input','weight'],
                  ['B','input','bias'],['Mean','output','mean'],['Rstd','output','rstd']],
      'constants':{'stride':cols,'N':cols,'eps':epsilon['value'],
                   'BLOCK_SIZE':min(32768,triton.next_power_of_2(cols))},
      'grid':[rows,1,1]}
