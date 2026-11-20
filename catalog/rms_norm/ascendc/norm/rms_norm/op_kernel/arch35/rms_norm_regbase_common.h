/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm_regbase_common.h
 * \brief RmsNorm regbase common
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_COMMON_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_COMMON_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "kernel_utils.h"
#else
#include "kernel_operator_utils_intf.h"
#endif
#include "../rms_norm_base.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace RmsNorm {
using namespace AscendC;
using namespace AscendC::Reg;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
using AscendC::Reg::RegTensor;
#endif
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

constexpr AscendC::Reg::CastTrait castTraitFp322Fp8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait castTraitFp322Hifp8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND,
};

template <typename T_Y>
__aicore__ inline uint64_t GetOverflowMode()
{
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value ||
                  IsSameType<T_Y, hifloat8_t>::value) {
        return AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    }
#endif
    return 0;
}

template <typename T_Y>
__aicore__ inline void SetOverflowMode(uint64_t mode)
{
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value ||
                  IsSameType<T_Y, hifloat8_t>::value) {
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(mode);
    }
#endif
}

template <typename T, typename U, typename R>
__aicore__ inline void YCopyOutImpl(const U& dstTensor, const R& srcTensor, uint32_t blockCount, uint32_t blockLen,
                                    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams extParams{
        static_cast<uint16_t>(blockCount),           // blockCount
        static_cast<uint32_t>(blockLen * sizeof(T)), // blockLen
        srcStride,                                   // srcStride
        dstStride,                                   // dstStride
        0                                            // rsv
    };
    DataCopyPad(dstTensor, srcTensor, extParams);
}

/*!
 * DataCopy custom implement
 * @tparam T DataCopy data type
 * @tparam U Destination Operand type
 * @tparam R Source Operand type
 * @param dstTensor Destination Tensor
 * @param srcTensor Source Tensor
 * @param blockCount burst
 * @param blockLen burst length
 * @param padParams pad params
 * @return void
 */
template <typename T, typename U, typename R>
__aicore__ inline void DataCopyImpl(const U& dstTensor, const R& srcTensor, uint32_t blockCount, uint32_t blockLen,
                                    uint32_t srcStride = 0, uint32_t dstStride = 0,
                                    const DataCopyPadExtParams<T> padParams = {false, 0, 0, 0})
{
    DataCopyExtParams extParams{
        static_cast<uint16_t>(blockCount),           // blockCount
        static_cast<uint32_t>(blockLen * sizeof(T)), // blockLen
        srcStride,                                   // srcStride
        dstStride,                                   // dstStride
        0                                            // rsv
    };
    if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
        DataCopyPad(dstTensor, srcTensor, extParams, padParams);
    } else {
        DataCopyPad(dstTensor, srcTensor, extParams);
    }
}

template <typename T_X>
__aicore__ inline void CopyInX(TQue<QuePosition::VECIN, 1>& inQueueX, GlobalTensor<T_X>& srcGm, uint64_t gmOffset,
                               uint32_t blockLen, uint32_t left = 0, uint32_t right = 0)
{
    LocalTensor<T_X> xLocal = inQueueX.AllocTensor<T_X>();
    DataCopyPadExtParams<T_X> padParams{
        true,                        // isPad
        static_cast<uint8_t>(left),  // leftPadding
        static_cast<uint8_t>(right), // rightPadding
        static_cast<T_X>(0.0)        // paddingValue
    };
    DataCopyImpl<T_X>(xLocal, srcGm[gmOffset], 1, blockLen, 0, 0, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T_X>
__aicore__ inline void CopyOutX(GlobalTensor<T_X>& xGm, TQue<QuePosition::VECOUT, 1>& outQueueX, uint64_t gmOffset,
                                uint32_t blockLen)
{
    LocalTensor<T_X> xLocal = outQueueX.DeQue<T_X>();
    DataCopyImpl<T_X>(xGm[gmOffset], xLocal, 1, blockLen);
    outQueueX.FreeTensor(xLocal);
}

template <typename T_Y>
__aicore__ inline void CopyOutY(GlobalTensor<T_Y>& yGm, TQue<QuePosition::VECOUT, 1>& outQueueY, uint64_t gmOffset,
                                uint32_t blockLen)
{
    LocalTensor<T_Y> yLocal = outQueueY.DeQue<T_Y>();
    YCopyOutImpl<T_Y>(yGm[gmOffset], yLocal, 1, blockLen);
    outQueueY.FreeTensor(yLocal);
}

/*!
 * rstd = 1 / sqrt(mean / n + epsilon)
 * @param rstdLocal store mean value Tensor
 * @param epsilon RmsNorm's attr
 * @param count The num of mean
 * @return void
 */
__aicore__ inline void ComputeRstd(LocalTensor<float>& rstdLocal, float epsilon, float avgFactor, uint32_t count)
{
    __ubuf__ float* rstdLocalAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();

    uint32_t calCount = count;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);
    __VEC_SCOPE__
    {
        RegTensor<float> vReg, srcReg, dstReg;
        MaskReg maskReg;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            maskReg = UpdateMask<float>(sreg);
            LoadAlign(srcReg, rstdLocalAddr + i * V_LENGTH);
            Muls(srcReg, srcReg, avgFactor, maskReg);
            Adds(dstReg, srcReg, epsilon, maskReg);
            Sqrt(vReg, dstReg, maskReg);
            Duplicate(srcReg, float(1.0), maskReg);
            Div(dstReg, srcReg, vReg, maskReg);
            StoreAlign(rstdLocalAddr + i * V_LENGTH, dstReg, maskReg);
        }
    }
}

template <typename DG, bool IS_GEMMA>
__aicore__ inline void GemmaWithOutFloat(RegTensor<DG> gammaReg1, RegTensor<DG> gammaReg2,
                                         RegTensor<float>& gammaFp32Reg1, RegTensor<float>& gammaFp32Reg2,
                                         MaskReg pregMask)
{
    if constexpr (IS_GEMMA) {
        RegTensor<float> gammaTmp1, gammaTmp2;
        Cast<float, DG, castTraitB162B32>(gammaTmp1, gammaReg1, pregMask);
        Cast<float, DG, castTraitB162B32>(gammaTmp2, gammaReg2, pregMask);
        Adds(gammaFp32Reg1, gammaTmp1, 1.0f, pregMask);
        Adds(gammaFp32Reg2, gammaTmp2, 1.0f, pregMask);
    } else {
        Cast<float, DG, castTraitB162B32>(gammaFp32Reg1, gammaReg1, pregMask);
        Cast<float, DG, castTraitB162B32>(gammaFp32Reg2, gammaReg2, pregMask);
    }
}

template <typename DG, bool IS_GEMMA>
__aicore__ inline void GemmaWithFloat(__ubuf__ DG* gammaAddr1, __ubuf__ DG* gammaAddr2, RegTensor<DG>& gammaReg1,
                                      RegTensor<DG>& gammaReg2, MaskReg maskReg, uint16_t i)
{
    if constexpr (IS_GEMMA) {
        RegTensor<float> gammaTmp1, gammaTmp2;
        LoadAlign(gammaTmp1, gammaAddr1 + i * V_LENGTH);
        LoadAlign(gammaTmp2, gammaAddr2 + i * V_LENGTH);
        Adds(gammaReg1, gammaTmp1, 1.0f, maskReg);
        Adds(gammaReg2, gammaTmp2, 1.0f, maskReg);
    } else {
        LoadAlign(gammaReg1, gammaAddr1 + i * V_LENGTH);
        LoadAlign(gammaReg2, gammaAddr2 + i * V_LENGTH);
    }
}

/*!
 * compute multi N yLocal = xLocal * rstd * gammaLocal
 *
 * @param xLocal input xLocal
 * @param gammaLocal input gammaLocal
 * @param yLocal output yLocal
 * @param rstd input rstd
 * @param count vector commpute length
 * @return void
 */
template <typename DX, typename DG, bool IS_GEMMA>
__aicore__ inline void ComputeYMultiN(LocalTensor<float>& xLocal, LocalTensor<DG>& gammaLocal, LocalTensor<DX>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t offset, uint32_t count, uint32_t curRows)
{
    uint32_t calCount = count / 2;
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);

    __ubuf__ float* xAddr1 = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* xAddr2 = (__ubuf__ float*)xLocal.GetPhyAddr() + calCount;
    __ubuf__ DG* gammaAddr1 = (__ubuf__ DG*)gammaLocal.GetPhyAddr();
    __ubuf__ DG* gammaAddr2 = (__ubuf__ DG*)gammaLocal.GetPhyAddr() + calCount;
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __ubuf__ DX* yAddr1 = (__ubuf__ DX*)yLocal.GetPhyAddr();
    __ubuf__ DX* yAddr2 = (__ubuf__ DX*)yLocal.GetPhyAddr() + calCount;

    if constexpr (!IsSameType<DX, float>::value && !IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(curRows); row++) {
                uint32_t sreg = (uint32_t)calCount;
                RegTensor<DX> yB16Reg1, yB16Reg2;
                RegTensor<DG> gammaReg1, gammaReg2;
                RegTensor<float> rstdReg;
                RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
                RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
                MaskReg pregMask;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + offset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    pregMask = UpdateMask<float>(sreg);
                    LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                    LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                    LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg2, gammaAddr2 + i * V_LENGTH);
                    GemmaWithOutFloat<DG, IS_GEMMA>(gammaReg1, gammaReg2, gammaFp32Reg1, gammaFp32Reg2, pregMask);
                    Mul(dst1Reg, xReg1, rstdReg, pregMask);
                    Mul(dst2Reg, xReg2, rstdReg, pregMask);
                    Mul(yReg1, dst1Reg, gammaFp32Reg1, pregMask);
                    Mul(yReg2, dst2Reg, gammaFp32Reg2, pregMask);
                    Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, pregMask);
                    Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, pregMask);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, pregMask);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, pregMask);
                }
                offset++;
                xAddr1 += count;
                xAddr2 += count;
                yAddr1 += count;
                yAddr2 += count;
            }
        }
    } else if constexpr (!IsSameType<DX, float>::value and IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(curRows); row++) {
                uint32_t sreg = (uint32_t)calCount;
                RegTensor<DX> yB16Reg1, yB16Reg2;
                RegTensor<DG> gammaReg1, gammaReg2;
                RegTensor<float> rstdReg;
                RegTensor<float> xReg1, dst1Reg, yReg1;
                RegTensor<float> xReg2, dst2Reg, yReg2;
                MaskReg maskReg;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + offset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                    LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                    GemmaWithFloat<DG, IS_GEMMA>(gammaAddr1, gammaAddr2, gammaReg1, gammaReg2, maskReg, i);
                    Mul(dst1Reg, xReg1, rstdReg, maskReg);
                    Mul(dst2Reg, xReg2, rstdReg, maskReg);
                    Mul(yReg1, dst1Reg, gammaReg1, maskReg);
                    Mul(yReg2, dst2Reg, gammaReg2, maskReg);
                    Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                    Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, maskReg);
                    StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, maskReg);
                }
                offset++;
                xAddr1 += count;
                xAddr2 += count;
                yAddr1 += count;
                yAddr2 += count;
            }
        }
    } else {
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < static_cast<uint16_t>(curRows); row++) {
                uint32_t sreg = (uint32_t)calCount;
                RegTensor<float> rstdReg;
                RegTensor<float> xReg1, gammaReg1, yReg1, vRegTmp1;
                RegTensor<float> xReg2, gammaReg2, yReg2, vRegTmp2;
                MaskReg maskReg;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + offset);
                for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                    maskReg = UpdateMask<float>(sreg);
                    LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                    LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                    GemmaWithFloat<DG, IS_GEMMA>(gammaAddr1, gammaAddr2, gammaReg1, gammaReg2, maskReg, i);
                    Mul(vRegTmp1, xReg1, rstdReg, maskReg);
                    Mul(vRegTmp2, xReg2, rstdReg, maskReg);
                    Mul(yReg1, vRegTmp1, gammaReg1, maskReg);
                    Mul(yReg2, vRegTmp2, gammaReg2, maskReg);
                    StoreAlign(yAddr1 + i * V_LENGTH, yReg1, maskReg);
                    StoreAlign(yAddr2 + i * V_LENGTH, yReg2, maskReg);
                }
                offset++;
                xAddr1 += count;
                xAddr2 += count;
                yAddr1 += count;
                yAddr2 += count;
            }
        }
    }
}

/*!
 * compute yLocal = xLocal * rstd * gammaLocal
 *
 * @param xLocal input xLocal
 * @param gammaLocal input gammaLocal
 * @param yLocal output yLocal
 * @param rstd input rstd
 * @param count vector commpute length
 * @return void
 */
template <typename DX, typename DG, bool IS_GEMMA>
__aicore__ inline void ComputeLatterY(LocalTensor<DX>& xLocal, LocalTensor<DG>& gammaLocal, LocalTensor<DX>& yLocal,
                                      LocalTensor<float>& rstdLocal, uint32_t offset, uint32_t count)
{
    uint32_t calCount = count / 2;
    uint32_t sreg = (uint32_t)calCount;
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);

    __ubuf__ DX* xAddr1 = (__ubuf__ DX*)xLocal.GetPhyAddr();
    __ubuf__ DX* xAddr2 = (__ubuf__ DX*)xLocal.GetPhyAddr() + calCount;
    __ubuf__ DG* gammaAddr1 = (__ubuf__ DG*)gammaLocal.GetPhyAddr();
    __ubuf__ DG* gammaAddr2 = (__ubuf__ DG*)gammaLocal.GetPhyAddr() + calCount;
    __ubuf__ float* srcAddr2 = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __ubuf__ DX* yAddr1 = (__ubuf__ DX*)yLocal.GetPhyAddr();
    __ubuf__ DX* yAddr2 = (__ubuf__ DX*)yLocal.GetPhyAddr() + calCount;

    if constexpr (!IsSameType<DX, float>::value and !IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            RegTensor<DX> xB16Reg1, yB16Reg1, xB16Reg2, yB16Reg2;
            RegTensor<DG> gammaReg1, gammaReg2;
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
            RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg1, xAddr1 + i * V_LENGTH);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg2, xAddr2 + i * V_LENGTH);
                LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg1, gammaAddr1 + i * V_LENGTH);
                LoadAlign<DG, LoadDist::DIST_UNPACK_B16>(gammaReg2, gammaAddr2 + i * V_LENGTH);
                if constexpr (IS_GEMMA) {
                    RegTensor<float> gammaTmp1, gammaTmp2;
                    Cast<float, DG, castTraitB162B32>(gammaTmp1, gammaReg1, maskReg);
                    Cast<float, DG, castTraitB162B32>(gammaTmp2, gammaReg2, maskReg);
                    Adds(gammaFp32Reg1, gammaTmp1, 1.0f, maskReg);
                    Adds(gammaFp32Reg2, gammaTmp2, 1.0f, maskReg);
                } else {
                    Cast<float, DG, castTraitB162B32>(gammaFp32Reg1, gammaReg1, maskReg);
                    Cast<float, DG, castTraitB162B32>(gammaFp32Reg2, gammaReg2, maskReg);
                }
                Cast<float, DX, castTraitB162B32>(xReg1, xB16Reg1, maskReg);
                Cast<float, DX, castTraitB162B32>(xReg2, xB16Reg2, maskReg);
                Mul(dst1Reg, xReg1, rstdReg, maskReg);
                Mul(dst2Reg, xReg2, rstdReg, maskReg);
                Mul(yReg1, dst1Reg, gammaFp32Reg1, maskReg);
                Mul(yReg2, dst2Reg, gammaFp32Reg2, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, maskReg);
            }
        }
    } else if constexpr (!IsSameType<DX, float>::value and IsSameType<DG, float>::value) {
        __VEC_SCOPE__
        {
            RegTensor<DX> xB16Reg1, yB16Reg1, xB16Reg2, yB16Reg2;
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
            RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg1, xAddr1 + i * V_LENGTH);
                LoadAlign<DX, LoadDist::DIST_UNPACK_B16>(xB16Reg2, xAddr2 + i * V_LENGTH);
                if constexpr (IS_GEMMA) {
                    RegTensor<float> gammaTmp1, gammaTmp2;
                    LoadAlign(gammaTmp1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaTmp2, gammaAddr2 + i * V_LENGTH);
                    Adds(gammaFp32Reg1, gammaTmp1, 1.0f, maskReg);
                    Adds(gammaFp32Reg2, gammaTmp2, 1.0f, maskReg);
                } else {
                    LoadAlign(gammaFp32Reg1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaFp32Reg2, gammaAddr2 + i * V_LENGTH);
                }
                Cast<float, DX, castTraitB162B32>(xReg1, xB16Reg1, maskReg);
                Cast<float, DX, castTraitB162B32>(xReg2, xB16Reg2, maskReg);
                Mul(dst1Reg, xReg1, rstdReg, maskReg);
                Mul(dst2Reg, xReg2, rstdReg, maskReg);
                Mul(yReg1, dst1Reg, gammaFp32Reg1, maskReg);
                Mul(yReg2, dst2Reg, gammaFp32Reg2, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                Cast<DX, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr1 + i * V_LENGTH, yB16Reg1, maskReg);
                StoreAlign<DX, StoreDist::DIST_PACK_B32>(yAddr2 + i * V_LENGTH, yB16Reg2, maskReg);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, gammaReg1, yReg1, vRegTmp1;
            RegTensor<float> xReg2, gammaReg2, yReg2, vRegTmp2;
            MaskReg maskReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, srcAddr2 + offset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                maskReg = UpdateMask<float>(sreg);
                LoadAlign(xReg1, xAddr1 + i * V_LENGTH);
                LoadAlign(xReg2, xAddr2 + i * V_LENGTH);
                if constexpr (IS_GEMMA) {
                    RegTensor<float> gammaTmp1, gammaTmp2;
                    LoadAlign(gammaTmp1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaTmp2, gammaAddr2 + i * V_LENGTH);
                    Adds(gammaReg1, gammaTmp1, 1.0f, maskReg);
                    Adds(gammaReg2, gammaTmp2, 1.0f, maskReg);
                } else {
                    LoadAlign(gammaReg1, gammaAddr1 + i * V_LENGTH);
                    LoadAlign(gammaReg2, gammaAddr2 + i * V_LENGTH);
                }
                Mul(vRegTmp1, xReg1, rstdReg, maskReg);
                Mul(vRegTmp2, xReg2, rstdReg, maskReg);
                Mul(yReg1, vRegTmp1, gammaReg1, maskReg);
                Mul(yReg2, vRegTmp2, gammaReg2, maskReg);
                StoreAlign(yAddr1 + i * V_LENGTH, yReg1, maskReg);
                StoreAlign(yAddr2 + i * V_LENGTH, yReg2, maskReg);
            }
        }
    }
}

} // namespace RmsNorm
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_COMMON_H
