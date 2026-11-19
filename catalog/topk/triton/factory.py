import ast,sys,types
from pathlib import Path
import triton
import torch
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
    module.triton=triton;module.tl=tl;module.extension=extension;module.torch=torch
    module.ext=globals().get('ext');module.tle=globals().get('ext');sys.modules[module_name]=module
    constants=[node for node in ast.parse(path.read_bytes()).body if isinstance(node,(ast.Assign, ast.AnnAssign))]
    exec(compile(ast.Module(body=constants,type_ignores=[]),str(path),'exec'),module.__dict__)
    # Triton 3.6 requires constexpr values, not only constexpr annotations.
    for node in constants:
        if isinstance(node,ast.AnnAssign):
            name=node.target.id
            module.__dict__[name]=tl.constexpr(module.__dict__[name])
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
ext=definitions('extension.py',['program_id'],'upstream_flaggems_topk_extension')
module=definitions('kernel.py',['_get_finfo_val', 'topk_stage1_kernel'],'upstream_flaggems_topk')

def build(signature):
    x,=signature['inputs'];rows,cols=x['shape'];scalars={v['name']:v['value'] for v in signature['scalars']}
    return {'kernel':module.topk_stage1_kernel,
      'bindings':[['y_ptr','output','values'],['index_ptr','output','indices'],['x_ptr','input','x']],
      'constants':{'k':scalars['k'],'N':cols,'CHUNK_SIZE':triton.next_power_of_2(cols),
                   'DESCENDING':scalars['largest']},'grid':[rows,1,1]}
