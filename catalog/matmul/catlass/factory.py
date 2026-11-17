import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,importlib.util,sys,types
from pathlib import Path
import torch
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack
root=Path(__file__).parent
params_spec=importlib.util.spec_from_file_location('upstream_params',root/'helpers/params.py')
params=importlib.util.module_from_spec(params_spec)
sys.modules[params_spec.name]=params;params_spec.loader.exec_module(params)
TilingParams,SwizzleParams=params.TilingParams,params.SwizzleParams
path=root/'kernel.py'
parsed=ast.parse(path.read_bytes(),filename=str(path))
functions=[node for node in parsed.body if isinstance(node,ast.FunctionDef)
           and node.name=='basic_mmad_kernel']
assert len(functions)==1
module=types.ModuleType('upstream_kernel');module.__file__=str(path)
module.tla=tla;module.TilingParams=TilingParams;module.SwizzleParams=SwizzleParams
sys.modules[module.__name__]=module
futures=[n for n in parsed.body if isinstance(n,ast.ImportFrom) and n.module=='__future__']
exec(compile(ast.Module(body=futures+functions,type_ignores=[]),str(path),'exec'),module.__dict__)
basic_mmad_kernel=module.basic_mmad_kernel

def build(signature):
    assert not signature['scalars']
    descs=signature['inputs']+signature['outputs']
    tensors=[from_dlpack(torch.empty(d['shape'],dtype=getattr(torch,d['dtype'])),
                        layout_tag=tla.arch.RowMajor) for d in descs]
    assert all(t.data_ptr==0 for t in tensors)
    artifact=tla.compile(basic_mmad_kernel,*tensors,TilingParams(),
        SwizzleParams(SWIZZLE_DIRECTION=0,SWIZZLE_OFFSET=3),
        tla.params.HF32Mode.HF32_NEAREST_EVEN,False,False,options='--npu-arch 3510')
    m,n=signature['outputs'][0]['shape']
    return {'artifact':artifact,'bindings':[['input','a'],['input','b'],['output','out']],
            'block_num':min(32,((m+255)//256)*((n+255)//256))}
