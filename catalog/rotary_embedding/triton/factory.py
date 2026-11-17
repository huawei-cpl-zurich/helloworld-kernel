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
module=definitions('kernel.py',['rotary_embedding_rw_kernel', 'rotary_embedding_siso_kernel'],'upstream_flaggems_rotary_embedding')

def build(signature):
    state,cos,sin=signature['inputs'];tokens,heads,dim=state['shape']
    interleaved,=signature['scalars'];assert interleaved['name']=='interleaved'
    return {'kernel':module.rotary_embedding_siso_kernel,
      'bindings':[['state_out','output','out'],['state','input','state'],
                  ['cos','input','cos'],['sin','input','sin']],
      'constants':{'stride_state_n':heads*dim,'stride_state_h':dim,'stride_state_d':1,
                   'stride_cos_n':dim//2,'stride_cos_d':1,'num_tokens':tokens,
                   'num_heads':heads,'BLOCK_N':1,'BLOCK_H':triton.next_power_of_2(heads),
                   'BLOCK_D':dim,'rotary_interleaved':interleaved['value']},
      'grid':[tokens,1,1]}
