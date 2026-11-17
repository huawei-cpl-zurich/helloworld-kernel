import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
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
    module.triton=triton;module.tl=tl;module.extension=extension
    module.ext=globals().get('ext');sys.modules[module_name]=module
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
module=definitions('kernel.py',['gelu_'],'upstream_flaggems_gelu')

def build(signature):
    x,=signature['inputs'];mode,=signature['scalars'];assert mode['name']=='approximate_tanh'
    count=x['shape'][0]
    return {'kernel':module.gelu_,'bindings':[['x_ptr','input','x']],
      'constants':{'n_elements':count,'USE_TANH':mode['value'],'BLOCK_SIZE':4096},
      'grid':[(count+4095)//4096,1,1]}
