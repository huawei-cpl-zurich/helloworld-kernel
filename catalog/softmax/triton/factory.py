import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
path=Path(__file__).parent/'kernel.py'
functions=[node for node in ast.parse(path.read_bytes()).body
           if isinstance(node,ast.FunctionDef) and node.name=='softmax_kernel']
assert len(functions)==1
module=types.ModuleType('upstream_softmax');module.__file__=str(path)
module.triton=triton;module.tl=tl;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)

def build(signature):
    x,=signature['inputs'];out,=signature['outputs']
    rows,cols=x['shape'];assert not signature['scalars'] and out['shape']==[rows,cols]
    return {'kernel':module.softmax_kernel,
      'bindings':[['output_ptr','output','out'],['input_ptr','input','x']],
      'constants':{'input_row_stride':cols,'output_row_stride':cols,'n_rows':rows,
                   'n_cols':cols,'BLOCK_SIZE':triton.next_power_of_2(cols)},
      'grid':[min(32,rows),1,1]}
