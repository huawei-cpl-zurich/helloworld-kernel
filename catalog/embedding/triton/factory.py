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
    module.ext=globals().get('ext');module.tle=globals().get('ext');sys.modules[module_name]=module
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
ext=definitions('extension.py',['program_id','num_programs'],'upstream_flaggems_ext')
module=definitions('kernel.py',['embedding_kernel'],'upstream_flaggems_embedding')

def build(signature):
    weight,indices=signature['inputs'];tokens,=indices['shape'];_,cols=weight['shape']
    return {'kernel':module.embedding_kernel,'bindings':[['out_ptr','output','out'],
      ['in_ptr','input','indices'],['weight_ptr','input','weight']],
      'constants':{'N':cols,'BLOCK_SIZE':min(8192,triton.next_power_of_2(cols))},
      'grid':[tokens,1,1]}
