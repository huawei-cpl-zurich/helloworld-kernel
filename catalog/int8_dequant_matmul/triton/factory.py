import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
path=Path(__file__).parent/'kernel.py'
functions=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='channel_kernel']
assert len(functions)==1
functions[0].decorator_list=[d for d in functions[0].decorator_list if ast.unparse(d)=='tr.jit']
module=types.ModuleType('upstream_int8_scaled_channel');module.__file__=str(path)
module.tr=triton;module.tl=tl;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
def build(signature):
    a,b,sa,sb=signature['inputs'];m,k=a['shape'];k2,n=b['shape'];assert k==k2 and not signature['scalars']
    # Original batched operator specialized to one batch: no tensor transformation.
    return {'kernel':module.channel_kernel,
      'bindings':[['A','input','a'],['B','input','b'],['SA','input','scale_a'],['SB','input','scale_b'],['C','output','out']],
      'constants':{'M':m,'N':n,'K':k,'ab':0,'am':k,'ak':1,'bb':0,'bk':n,'bn':1,
                   'sab':0,'sam':1,'sbb':0,'sbn':1,'cb':0,'cm':n,'cn':1,'BM':64,'BN':64,'BK':64,'G':4},
      'grid':[((m+63)//64)*((n+63)//64),1,1]}
