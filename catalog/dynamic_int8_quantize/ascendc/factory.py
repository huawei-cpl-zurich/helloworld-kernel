def build(signature):
    x,=signature['inputs'];rows,cols=x['shape']
    assert rows in (32,64) and cols in (128,512) and not signature['scalars']
    code=f"""
#include "kernel_operator.h"
#define ORIG_DTYPE_Y DT_INT8
#include "quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_full_load.h"
extern "C" __global__ __aicore__ void cannery_dynamic_int8(GM_ADDR x,GM_ADDR quantized,GM_ADDR scale) {{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DynamicQuantTilingDataArch35 tiling{{}};
    tiling.coreNum=1;tiling.rowLen={cols};tiling.headCoreNum=1;
    tiling.rowPerHeadCore={rows};tiling.rowPerTailCore={rows};
    tiling.multiRowNumHeadCore=8;tiling.multiRowNumTailCore=8;
    tiling.dstTypeMax=127.0f;
    AscendC::TPipe pipe;
    DynamicQuantNDOpt::DynamicQuantRegbaseFullLoad<half,int8_t,false,2,true> kernel(&pipe);
    kernel.Init(x,nullptr,quantized,scale,nullptr,nullptr,&tiling);
    kernel.Process();
}}
"""
    return {'source':code,'entrypoint':'cannery_dynamic_int8','block_num':1,
      'bindings':[['input','x'],['output','quantized'],['output','scale']]}
