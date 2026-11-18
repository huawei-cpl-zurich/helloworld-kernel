import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
path=Path(__file__).parent/'kernel.py'
NAME='_unpermute_kernel'
functions=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name==NAME]
assert len(functions)==1
module=types.ModuleType('upstream_moe_'+NAME);module.__file__=str(path);module.triton=triton;module.tl=tl;sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)
def build(signature):
    tokens,indices,weights=signature['inputs'];n,k=weights['shape'];total,h=tokens['shape'];assert k==2 and total==n*k and indices['shape']==[n*k] and not signature['scalars']
    return {'kernel':getattr(module,NAME),'bindings':[['Input','input','tokens'],['Output','output','output'],['SortedIdx','input','indices'],['TopK_Weights','input','weights']],
      'constants':{'num_tokens':n,'hidden_dim':h,'top_k':2,'stride_in_t':h,'stride_out_t':h,'stride_w_t':2,'BLOCK_D':4096},'grid':[n,(h+4095)//4096,1]}
