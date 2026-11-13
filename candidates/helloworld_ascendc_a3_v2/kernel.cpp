#include "kernel_operator.h"

// Minimal admission fixture: executed by an A3 AI Core, without a performance claim.
extern "C" __global__ __vector__ __aicore__ void forward(GM_ADDR input, GM_ADDR output, uint64_t count) {
    AscendC::GlobalTensor<float> x;
    AscendC::GlobalTensor<float> y;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input), count);
    y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), count);
    for (uint64_t i = 0; i < count; ++i) {
        y.SetValue(i, x.GetValue(i) + 1.0f);
    }
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                     AscendC::DcciDst::CACHELINE_OUT>(y);
}
