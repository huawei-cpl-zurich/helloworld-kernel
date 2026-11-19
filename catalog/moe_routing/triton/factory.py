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
module=definitions('kernel.py',['topk_gating_softmax_kernel'],'upstream_flaggems_moe_routing')

def build(signature):
    logits,=signature['inputs'];rows,experts=logits['shape'];scalars={v['name']:v['value'] for v in signature['scalars']}
    return {'kernel':module.topk_gating_softmax_kernel,
      'bindings':[['input_ptr','input','logits'],['output_ptr','output','weights'],
                  ['indices_ptr','output','experts'],['source_rows_ptr','output','source_rows']],
      'constants':{'finished_ptr':None,'num_rows':rows,'k':scalars['k'],'num_experts':experts,
                   'start_expert':0,'end_expert':experts,'renormalize':scalars['renormalize'],
                   'INDEX_TY':tl.int64,'BLOCK_SIZE_ROWS':32,'BLOCK_SIZE_EXPERTS':32},
      'grid':[(rows+31)//32,1,1]}
