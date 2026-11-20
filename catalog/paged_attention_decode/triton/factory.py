import ast,sys,types
from pathlib import Path
import triton
import triton.language as tl
path=Path(__file__).parent/'kernel.py'
names={'_fwd_kernel','_convert_pv','tanh'}
functions=[n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in functions}==names
module=types.ModuleType('upstream_paged_attention');module.__file__=str(path)
module.triton=triton;module.tl=tl;module.fast_expf=tl.math.exp;module.fast_dividef=tl.math.fdiv
sys.modules[module.__name__]=module
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),module.__dict__)

def build(signature):
    q,k,v,blocks,qstart,qlen,kvlen=signature['inputs']
    batch,heads,dim=q['shape'];pages,block_size,kv_heads,_=k['shape']
    scale,=signature['scalars'];assert scale['name']=='sm_scale'
    # Original generic paged forward kernel, specialized to one query/sequence.
    return {'kernel':module._fwd_kernel,'bindings':[['Q','input','q'],['K','input','k_cache'],
      ['V','input','v_cache'],['Q_start_loc','input','q_start'],['Q_seqlens','input','q_lengths'],
      ['KV_seqlens','input','kv_lengths'],['Block_offsets','input','block_table'],['Out','output','out']],
      'constants':{'sm_scale':scale['value'],'stride_qbs':heads*dim,'stride_qh':dim,'stride_qd':1,
        'stride_kp':block_size*kv_heads*dim,'stride_kbs':kv_heads*dim,'stride_kh':dim,'stride_kd':1,
        'stride_vp':block_size*kv_heads*dim,'stride_vbs':kv_heads*dim,'stride_vh':dim,'stride_vd':1,
        'stride_obs':heads*dim,'stride_oh':dim,'stride_od':1,'stride_boffb':pages//batch,
        'kv_group_num':heads//kv_heads,'window_size':-1,'head_size':dim,'head_size_v':dim,
        'logit_softcapping':0.0,'BLOCK_M':16,'BLOCK_DMODEL':dim,'BLOCK_DV':dim,
        'BLOCK_N':block_size,'BLOCK_DMODEL1':0},'grid':[1,heads,batch]}
