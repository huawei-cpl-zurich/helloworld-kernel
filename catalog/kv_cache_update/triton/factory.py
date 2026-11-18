import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
path=Path(__file__).parent/'kernel.py'
functions=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='_kda_state_copy_kernel']
assert len(functions)==1
module=types.ModuleType('original_vllm_cache_copy');module.__file__=str(path)
module.triton=triton;module.tl=tl;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
def build(signature):
    updates,slots,cache=signature['inputs'];tokens,kv,heads,dim=updates['shape']
    assert kv==2 and slots['shape']==[tokens] and cache['shape'][1:]==updates['shape'][1:]
    assert not signature['scalars']
    payload=kv*heads*dim
    return {'kernel':module._kda_state_copy_kernel,
      'bindings':[['cache_ptr','output','updated'],['packed_ptr','input','updates'],['indices_ptr','input','slots']],
      'constants':{'flags_ptr':None,'cache_rows':cache['shape'][0],
                   'cache_stride_elements':payload,'payload_elements':payload,
                   'index_stride':1,'flag_stride':1,'TO_CACHE':True,'HAS_FLAGS':False,'BLOCK_SIZE':4096},
      'grid':[tokens,(payload+4095)//4096,1]}
