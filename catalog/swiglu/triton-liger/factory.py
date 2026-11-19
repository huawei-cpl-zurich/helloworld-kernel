import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
root=Path(__file__).parent
def load(filename,names,module_name,**globals):
    path=root/filename
    functions=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions}==set(names)
    module=types.ModuleType(module_name);module.__file__=str(path)
    module.triton=triton;module.tl=tl;module.__dict__.update(globals);sys.modules[module_name]=module
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
original=load('kernel.py',{'silu','_swiglu_forward_kernel'},'original_liger_swiglu')
entry=load('entrypoint.py',{'static_gate_one'},'liger_static_swiglu',original=original)
def build(signature):
    a,b=signature['inputs'];out,=signature['outputs'];rows,cols=a['shape']
    if cols > 4096:
        raise ValueError('Liger SwiGLU supports at most 4096 columns per row')
    assert b['shape']==out['shape']==a['shape'] and not signature['scalars']
    return {'kernel':entry.static_gate_one,
      'bindings':[['a_ptr','input','a'],['b_ptr','input','b'],['c_ptr','output','out']],
      'constants':{'stride':cols,'n_cols':cols,'BLOCK_SIZE':4096},'grid':[rows,1,1]}
