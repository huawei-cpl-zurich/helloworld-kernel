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
module=definitions('kernel.py',['bmm_kernel'],'upstream_flaggems_batched_matmul')

def build(signature):
    a,b=signature['inputs'];batch,m,k=a['shape'];batch2,k2,n=b['shape']
    assert batch==batch2 and k==k2 and m%64==0 and n%64==0
    constants={'M':m,'N':n,'K':k,'BATCH':batch,'TILE_M':64,'TILE_N':64,'TILE_K':64,
        'GROUP_M':1,'DOT_PAD_ONLY_K':False,'DIVISIBLE_M':m%64==0,
        'DIVISIBLE_N':n%64==0,'DIVISIBLE_K':k%64==0}
    # Upstream revisions introduced BATCH and DOT_PAD_ONLY_K at different times.
    parameters=set(module.bmm_kernel.arg_names)
    constants={key:value for key,value in constants.items() if key in parameters}
    assert parameters=={'A','B','O'}|set(constants),'Unsupported original BMM signature'
    return {'kernel':module.bmm_kernel,'bindings':[['A','input','a'],['B','input','b'],['O','output','out']],
      'constants':constants,'grid':[(m+63)//64,(n+63)//64,batch]}
