import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
root=Path(__file__).parent

def definitions(filename,names,module_name):
    path=root/filename
    functions=[node for node in ast.parse(path.read_bytes()).body
               if isinstance(node,ast.FunctionDef) and node.name in names and any('triton.jit' in ast.unparse(d) for d in node.decorator_list)]
    assert {node.name for node in functions}==set(names)
    for node in functions:
        # Unwrap host autotuning/cache decorators; preserve original JIT and body.
        node.decorator_list=[d for d in node.decorator_list if
            (isinstance(d,ast.Attribute) and ast.unparse(d)=='triton.jit') or
            (isinstance(d,ast.Call) and ast.unparse(d.func)=='triton.jit')]
    module=types.ModuleType(module_name);module.__file__=str(path)
    module.triton=triton;module.tl=tl
    sys.modules[module_name]=module
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
module=definitions('kernel.py',['rms_norm_kernel'],'original_flaggems_rms')

def build(signature):
    x,gamma=signature['inputs'];rows,cols=x['shape']
    if cols > 4096:
        raise ValueError('FlagGems RMSNorm supports at most 4096 columns per row')
    assert gamma['shape']==[cols]
    epsilon,=signature['scalars'];assert epsilon['name']=='epsilon'
    parameters=set(module.rms_norm_kernel.arg_names)
    old='Y' in parameters
    constants={'y_stride_r':cols,'y_stride_c':1,'x_stride_r':cols,'x_stride_c':1,
               'N':cols,'eps':epsilon['value'],'BLOCK_SIZE':4096}
    bindings=[['Y' if old else 'out_ptr','output','y'],['INV_RMS','output','rstd'],
              ['X' if old else 'in_ptr','input','x'],['W' if old else 'w_ptr','input','gamma']]
    assert parameters==set(constants)|{b[0] for b in bindings}
    return {'kernel':module.rms_norm_kernel,'bindings':bindings,'constants':constants,'grid':[rows,1,1]}
