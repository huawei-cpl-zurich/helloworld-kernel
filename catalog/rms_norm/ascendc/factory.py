def build(signature):
    x,gamma=signature['inputs']; rows,cols=x['shape']
    assert gamma['shape']==[cols] and cols in (128,512) and rows==32
    epsilon=signature['scalars'][0]['value']
    # Static full-load schedule follows rms_norm_tiling_arch35.cpp for this domain.
    code=f'''
#include "kernel_operator.h"
struct RMSNormArch35TilingData {{
    uint64_t num_row,num_col,num_col_align,block_factor,col_flod_factor,ub_factor;
    float epsilon,avg_factor;
    uint64_t last_block_factor;
}};
#include "norm/rms_norm/op_kernel/arch35/rms_norm_regbase_perf.h"
extern "C" __global__ __aicore__ void cannery_rms_norm(GM_ADDR x,GM_ADDR gamma,GM_ADDR y,GM_ADDR rstd) {{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RMSNormArch35TilingData tiling={{{rows},{cols},{cols},1,{cols//2},1,{epsilon}f,{1.0/cols}f,1}};
    RmsNorm::KernelRmsNormRegBasePerf<half,half,false> kernel;
    kernel.Init(x,gamma,y,rstd,&tiling);
    kernel.Process();
}}
'''
    return {'source':code,'entrypoint':'cannery_rms_norm','block_num':32,
        'include_dirs':['norm/rms_norm/op_kernel'],
        'bindings':[['input','x'],['input','gamma'],['output','y'],['output','rstd']]}
