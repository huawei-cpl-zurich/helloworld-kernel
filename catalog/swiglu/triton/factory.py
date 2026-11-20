import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
path=Path(__file__).parent/'kernel.py'
functions=[node for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.FunctionDef)
           and node.name in {'silu','_swiglu_forward_kernel'}]
assert {node.name for node in functions}=={'silu','_swiglu_forward_kernel'}
module=types.ModuleType('upstream_swiglu');module.__file__=str(path)
module.triton=triton;module.tl=tl;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)

def build(signature):
    a,b=signature['inputs'];out,=signature['outputs']
    rows,cols=a['shape'];assert not signature['scalars'] and b['shape']==out['shape']==[rows,cols]
    return {'kernel':module._swiglu_forward_kernel,
      'bindings':[['a_ptr','input','a'],['b_ptr','input','b'],['c_ptr','output','out']],
      'constants':{'stride':cols,'n_cols':cols,'BLOCK_SIZE':min(5120,triton.next_power_of_2(cols))},
      'grid':[rows,1,1]}
