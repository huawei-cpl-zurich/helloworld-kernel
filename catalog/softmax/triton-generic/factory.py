import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
root=Path(__file__).parent

def definitions(filename,names,module_name,**bindings):
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
    module.triton=triton;module.tl=tl;module.__dict__.update(bindings);sys.modules[module_name]=module
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
ext=definitions('extension.py',['program_id'],'upstream_flaggems_ext')
module=definitions('kernel.py',['prev_multiple_of','softmax_kernel_inner'],'original_flaggems_softmax',ext=ext)

def build(signature):
    x,=signature['inputs'];rows,cols=x['shape']
    assert cols<=4096 and not signature['scalars']
    return {'kernel':module.softmax_kernel_inner,
      'bindings':[['output_ptr','output','out'],['input_ptr','input','x']],
      'constants':{'M':rows,'N':cols,'TILE_N':4096,'ONE_TILE_PER_CTA':True},'grid':[rows,1,1]}
