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
module=definitions('kernel.py',['fused_add_rms_norm_kernel'],'upstream_flaggems_add_rms_norm')

def build(signature):
    x,residual,weight=signature['inputs'];rows,cols=x['shape'];epsilon,=signature['scalars']
    assert epsilon['name']=='epsilon'
    return {'kernel':module.fused_add_rms_norm_kernel,
      'bindings':[['X','input','x'],['R','input','residual'],['W','input','weight']],
      'constants':{'x_stride_r':cols,'x_stride_c':1,'r_stride_r':cols,'r_stride_c':1,
                   'N':cols,'eps':epsilon['value'],'BLOCK_SIZE':min(8192,triton.next_power_of_2(cols))},
      'grid':[rows,1,1]}
