import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
from triton.language.math import rsqrt
path=Path(__file__).parent/'kernel.py'
tree=ast.parse(path.read_bytes())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_rms_norm_forward_kernel']
constants=[n for n in tree.body if isinstance(n,ast.AnnAssign)]
assert len(functions)==1 and len(constants)==3
module=types.ModuleType('upstream_rms_norm');module.__file__=str(path)
module.triton=triton;module.tl=tl;module.rsqrt=rsqrt;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=constants+functions,type_ignores=[]),str(path),'exec'),module.__dict__)
def build(signature):
    x,gamma=signature['inputs'];rows,cols=x['shape'];epsilon,=signature['scalars']
    assert gamma['shape']==[cols] and epsilon['name']=='epsilon'
    return {'kernel':module._rms_norm_forward_kernel,
      'bindings':[['Y_ptr','output','y'],['X_ptr','input','x'],['W_ptr','input','gamma'],['RSTD_ptr','output','rstd']],
      'constants':{'Y_row_stride':cols,'X_row_stride':cols,'W_row_stride':0,'RSTD_row_stride':1,
        'n_cols':cols,'eps':epsilon['value'],'offset':0.0,'casting_mode':1,'BLOCK_SIZE':triton.next_power_of_2(cols)},
      'grid':[rows,1,1]}
