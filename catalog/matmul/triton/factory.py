import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
path=Path(__file__).parent/'kernel.py'
parsed=ast.parse(path.read_bytes(),filename=str(path))
functions=[node for node in parsed.body if isinstance(node,ast.FunctionDef)
           and node.name in ('get_autotune_config','matmul_kernel','leaky_relu_custom')]
assert {node.name for node in functions}=={'get_autotune_config','matmul_kernel','leaky_relu_custom'}
module=types.ModuleType('upstream_matmul');module.__file__=str(path)
module.triton=triton;module.tl=tl;module.extension=extension
sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
fn=module.matmul_kernel
if hasattr(fn,'fn') and hasattr(fn.fn,'arg_names'):fn=fn.fn

def build(signature):
    assert not signature['scalars']
    a,b=signature['inputs'];out=signature['outputs'][0]
    m,k=a['shape'];k2,n=b['shape'];assert k==k2 and out['shape']==[m,n]
    return {'kernel':fn,'bindings':[['a_ptr','input','a'],['b_ptr','input','b'],['c_ptr','output','out']],
      'constants':{'M':m,'N':n,'K':k,'stride_am':k,'stride_ak':1,'stride_bk':n,'stride_bn':1,
        'stride_cm':n,'stride_cn':1,'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':64,'ACTIVATION':''},
      'grid':[((m+63)//64)*((n+63)//64),1,1]}
