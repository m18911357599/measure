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
 * \file transpose_n_last.h
 * \brief transpose_n_last
 */
#ifndef TRANSPOSE_N_LAST_H
#define TRANSPOSE_N_LAST_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

template <typename T>
class TransposeNLast : public TransposeBase<T> {
public:
    __aicore__ inline TransposeNLast(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void GetBlockLoopNum();
    __aicore__ inline void GetLoopParams(int64_t n);
    __aicore__ inline int64_t GetDstAddressOffset(int64_t loopIdx);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void CopyIn(int64_t loopIdx, int64_t inputBlockLen, int64_t inCutLoopSize);
    __aicore__ inline void CopyOut(
        int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[], int64_t loopDstStride[]);
    __aicore__ inline void DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[]);

private:
    int64_t blockIdx_;

    // buffer
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> vecQue_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    // tiling params
    const TransposeOpTilingData* tiling_;
    int64_t reducedInputShape_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t reducedOutputShape_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t inUbInputShapeMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t inUbOutputShapeMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t inUbInputShapeTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t inUbOutputShapeTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t outputBlockLenMain_ = 0;
    int64_t outputBlockLenTail_ = 0;

    // core params
    int64_t blkProcessNum_ = 0;
    int64_t blkProcessIdxStart_ = 0;
    int64_t blkProcessIdxEnd_ = 0;

    // addressOffset params
    int64_t dstAddressOffsetMixedBase_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t dstLoopNumArray_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

    // dataCopy params
    int64_t loopSizeMain_[TRANSPOSE_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t loopSrcStrideMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t loopDstStrideMain_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t loopSizeTail_[TRANSPOSE_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t loopSrcStrideTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t loopDstStrideTail_[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
};

template <typename T>
__aicore__ inline void TransposeNLast<T>::Init(
    GM_ADDR x, GM_ADDR y, const TransposeOpTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    ParseTilingData();
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(vecQue_, BUFFER_NUM, tiling_->ubSize / BUFFER_NUM);
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    blkProcessNum_ = tiling_->blkFactor;
    blkProcessIdxStart_ = blockIdx_ * tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blkProcessNum_ += 1;
        blkProcessIdxStart_ += blockIdx_;
    } else {
        blkProcessIdxStart_ += tiling_->blkTailFactor;
    }
    blkProcessIdxEnd_ = blkProcessIdxStart_ + blkProcessNum_;
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[])
{
    if (num == 0) {
        return;
    }
    for (int64_t i = tiling_->permSize - 1; i >= 0; i--) {
        mixedBase[i] = num % bases[i];
        num /= bases[i];
        if (num == 0) {
            break;
        }
    }
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::ParseTilingData()
{
    // input in ub shape main and tail
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        reducedInputShape_[i] = tiling_->inputShape[i];
        if (i > tiling_->inCutIndex) {
            inUbInputShapeMain_[i] = reducedInputShape_[i];
            inUbInputShapeTail_[i] = reducedInputShape_[i];
        } else {
            inUbInputShapeMain_[i] = 1;
            inUbInputShapeTail_[i] = 1;
        }
    }
    inUbInputShapeMain_[tiling_->inCutIndex] = tiling_->inUbFactor;
    inUbInputShapeTail_[tiling_->inCutIndex] = tiling_->inTailFactor;

    // output in ub shape main and tail
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        reducedOutputShape_[i] = reducedInputShape_[tiling_->perm[i]];
        inUbOutputShapeMain_[i] = inUbInputShapeMain_[tiling_->perm[i]];
        inUbOutputShapeTail_[i] = inUbInputShapeTail_[tiling_->perm[i]];
    }

    outputBlockLenMain_ = inUbOutputShapeMain_[tiling_->permSize - 1];
    outputBlockLenTail_ = inUbOutputShapeTail_[tiling_->permSize - 1];
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::GetBlockLoopNum()
{
    const int64_t permSize = tiling_->permSize;
    int64_t dstLoopNumArrayTmp[TRANSPOSE_MAX_AXIS_NUM];
    int64_t rightProducts[TRANSPOSE_MAX_AXIS_NUM + 1];
    rightProducts[permSize] = 1;

    for (int64_t i = permSize - 1; i >= 0; i--) {
        rightProducts[i] = rightProducts[i + 1] * reducedOutputShape_[i];
    }
    for (int64_t i = 0; i < permSize; i++) {
        dstAddressOffsetMixedBase_[i] = Ops::Base::CeilDiv(reducedInputShape_[i], inUbInputShapeMain_[i]);
        dstLoopNumArrayTmp[i] = inUbOutputShapeMain_[i] * rightProducts[i + 1];
    }
    for (int64_t i = 0; i < permSize; i++) {
        dstLoopNumArray_[tiling_->perm[i]] = dstLoopNumArrayTmp[i];
    }
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::GetLoopParams(int64_t n)
{
    const int64_t permSize = tiling_->permSize;
    const int64_t reverseIdx = permSize - 1 - n;
    // main and tail loopSize
    loopSizeMain_[n] = inUbInputShapeMain_[reverseIdx];
    loopSizeTail_[n] = inUbInputShapeTail_[reverseIdx];
    // main and tail last axis aligned to ub block size
    int64_t alignedStrideMain =
        Ops::Base::CeilAlign(static_cast<int64_t>(loopSizeMain_[0] * sizeof(T)), BLOCK_SIZE_BYTE) / sizeof(T);
    int64_t alignedStrideTail =
        Ops::Base::CeilAlign(static_cast<int64_t>(loopSizeTail_[0] * sizeof(T)), BLOCK_SIZE_BYTE) / sizeof(T);
    // main loopSrcStride and loopDstStride
    if (loopSizeMain_[n] != 1) {
        loopSrcStrideMain_[n] = alignedStrideMain;
        if (n == 0) {
            loopSrcStrideMain_[n] = 1;
        }
        for (int64_t i = 1; i < n; i++) {
            loopSrcStrideMain_[n] *= inUbInputShapeMain_[permSize - 1 - i];
        }
        loopDstStrideMain_[n] = 1;
        for (int64_t i = permSize - 1; i >= 0; i--) {
            if (tiling_->perm[i] != reverseIdx) {
                loopDstStrideMain_[n] *= reducedOutputShape_[i];
            } else {
                break;
            }
        }
    }
    // tail loopSrcStride and loopDstStride
    if (tiling_->inTailFactor != 0 && loopSizeTail_[n] != 1) {
        loopSrcStrideTail_[n] = alignedStrideTail;
        for (int64_t i = 1; i < n; i++) {
            loopSrcStrideTail_[n] *= inUbInputShapeTail_[permSize - 1 - i];
        }
        loopDstStrideTail_[n] = 1;
        for (int64_t i = permSize - 1; i >= 0; i--) {
            if (tiling_->perm[i] != reverseIdx) {
                loopDstStrideTail_[n] *= reducedOutputShape_[i];
            } else {
                break;
            }
        }
    }
}

template <typename T>
__aicore__ inline int64_t TransposeNLast<T>::GetDstAddressOffset(int64_t loopIdx)
{
    int64_t dstAddressOffsetMixedBaseRes[TRANSPOSE_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    DecimalToMixed(loopIdx, dstAddressOffsetMixedBase_, dstAddressOffsetMixedBaseRes);

    int64_t dstAddressOffset = 0;
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        dstAddressOffset += dstAddressOffsetMixedBaseRes[i] * dstLoopNumArray_[i];
    }
    return dstAddressOffset;
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::CopyIn(int64_t loopIdx, int64_t inputBlockLen, int64_t inCutLoopSize)
{
    int64_t inputBlockLenMain = inputBlockLen * tiling_->inUbFactor;
    int64_t inputBlockLenTail = inputBlockLen * tiling_->inTailFactor;
    int64_t srcAddressOffset = loopIdx * inputBlockLenMain;
    if (tiling_->inTailFactor != 0) {
        srcAddressOffset =
            (loopIdx - loopIdx / inCutLoopSize) * inputBlockLenMain + loopIdx / inCutLoopSize * inputBlockLenTail;
    }
    int64_t blockLen = inputBlockLenMain;
    int64_t lastAxisLen = inUbInputShapeMain_[tiling_->permSize - 1];
    if (tiling_->inTailFactor != 0 && (loopIdx + 1) % inCutLoopSize == 0) {
        blockLen = inputBlockLenTail;
        lastAxisLen = inUbInputShapeTail_[tiling_->permSize - 1];
    }
    LocalTensor<T> bindLocalIn = vecQue_.AllocTensor<T>();
    DataCopyExtParams copyInParams{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    copyInParams.blockLen = blockLen * sizeof(T);
    copyInParams.blockCount = 1;
    if ((lastAxisLen * sizeof(T)) % BLOCK_SIZE_BYTE != 0) {
        copyInParams.blockLen = lastAxisLen * sizeof(T);
        copyInParams.blockCount = blockLen / lastAxisLen;
    }
    DataCopyPad(bindLocalIn, inputGM_[srcAddressOffset], copyInParams, padParams);
    vecQue_.EnQue(bindLocalIn);
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::CopyOut(
    int64_t loopIdx, int64_t loopSize[], int64_t loopSrcStride[], int64_t loopDstStride[])
{
    int64_t dstAddressOffset = GetDstAddressOffset(loopIdx);
    DataCopyExtParams copyOutParams{1, 0, 0, 0, 0};
    LoopModeParams loopParams;

    copyOutParams.blockLen = loopSize[0] * sizeof(T);
    copyOutParams.blockCount = loopSize[1];
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = copyOutParams.blockCount == 1 ? 0 : (loopDstStride[1] - loopSize[0]) * sizeof(T);

    loopParams.loop1Size = loopSize[2];
    loopParams.loop1SrcStride = loopSrcStride[2] * sizeof(T);
    loopParams.loop1DstStride = loopDstStride[2] * sizeof(T);
    loopParams.loop2Size = loopSize[3];
    loopParams.loop2SrcStride = loopSrcStride[3] * sizeof(T);
    loopParams.loop2DstStride = loopDstStride[3] * sizeof(T);

    LocalTensor<T> bindLocalOut = vecQue_.DeQue<T>();
    for (int64_t loop7Idx = 0; loop7Idx < loopSize[7]; loop7Idx++) {
        for (int64_t loop6Idx = 0; loop6Idx < loopSize[6]; loop6Idx++) {
            for (int64_t loop5Idx = 0; loop5Idx < loopSize[5]; loop5Idx++) {
                for (int64_t loop4Idx = 0; loop4Idx < loopSize[4]; loop4Idx++) {
                    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
                    DataCopyPad(
                        outputGM_
                            [dstAddressOffset + loop7Idx * loopDstStride[7] + loop6Idx * loopDstStride[6] +
                             loop5Idx * loopDstStride[5] + loop4Idx * loopDstStride[4]],
                        bindLocalOut
                            [loop7Idx * loopSrcStride[7] + loop6Idx * loopSrcStride[6] + loop5Idx * loopSrcStride[5] +
                             loop4Idx * loopSrcStride[4]],
                        copyOutParams);
                    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
                }
            }
        }
    }
    vecQue_.FreeTensor(bindLocalOut);
}

template <typename T>
__aicore__ inline void TransposeNLast<T>::ProcessPerCore()
{
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        GetLoopParams(i);
    }
    GetBlockLoopNum();
    int64_t inputBlockLen = 1;
    for (int64_t i = tiling_->permSize - 1; i > tiling_->inCutIndex; i--) {
        inputBlockLen *= reducedInputShape_[i];
    }
    int64_t inCutLoopSize = Ops::Base::CeilDiv(reducedInputShape_[tiling_->inCutIndex], tiling_->inUbFactor);
    for (int64_t loopIdx = blkProcessIdxStart_; loopIdx < blkProcessIdxEnd_; loopIdx++) {
        if (tiling_->inTailFactor != 0 && (loopIdx + 1) % inCutLoopSize == 0) { // tail
            CopyIn(loopIdx, inputBlockLen, inCutLoopSize);
            CopyOut(loopIdx, loopSizeTail_, loopSrcStrideTail_, loopDstStrideTail_);
        } else {
            CopyIn(loopIdx, inputBlockLen, inCutLoopSize);
            CopyOut(loopIdx, loopSizeMain_, loopSrcStrideMain_, loopDstStrideMain_);
        }
    }
}
} // namespace Transpose

#endif // TRANSPOSE_N_LAST_H