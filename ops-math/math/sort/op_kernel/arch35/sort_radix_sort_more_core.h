/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sort_radix_sort_more_core.h
 * \brief
 */

#ifndef RADIX_SORT_MORE_CORE_H
#define RADIX_SORT_MORE_CORE_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "util_type_simd.h"
#include "sort_tiling_data.h" // 不引入的话无法识别 SortRegBaseTilingData 结构体

namespace Sort {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

const int32_t VF_LEN_B32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
const int32_t VF_LEN_B16 = Ops::Base::GetVRegSize() / sizeof(int16_t);
const int32_t VF_LEN_B8 = Ops::Base::GetVRegSize() / sizeof(int8_t);
const int32_t VF_LEN_B64 = Ops::Base::GetVRegSize() / sizeof(int64_t);

// 基数排序正负0优化
const uint16_t TWIDDLED_MINUS_ZERO_BITS_FP16 = 0x7fff;
const uint16_t TWIDDLED_ZERO_BITS_FP16 = 0x8000;
const uint32_t TWIDDLED_MINUS_ZERO_BITS_FP32 = 0x7fffffff;
const uint32_t TWIDDLED_ZERO_BITS_FP32 = 0x80000000;

const uint16_t LOWEST_KEY_VALUE_B16 = uint16_t(-1);
const uint32_t LOWEST_KEY_VALUE_B32 = uint32_t(-1);
const uint16_t XOR_OP_VALUE_B16 = (uint16_t(1) << 15);
const uint32_t ZERO_VALUE_FLAG_B32 = 0;
const uint16_t ZERO_VALUE_FLAG_B16 = 0;
const uint32_t SHIFT_BIT_NUM = 8;
const uint32_t HIST_MASK_OUT_LEN = 8;
const uint64_t XOR_OP_VALUE_B64 = 0x8000000000000000;
const uint32_t RADIX_SORT_NUM = 256;
const uint8_t XOR_OP_VALUE_B8 = (uint8_t(1) << 7);
const uint32_t XOR_OP_VALUE = 0x80000000;
const int16_t STATE_BIT_SHF_VALUE = 30;
const int16_t STATE_BIT_SHF_VALUE_B64 = 62;
const int32_t AGGREGATE_READY_FLAG = 1;
const int32_t PREFIX_READY_FLAG = 2;
const uint32_t NOT_INIT_MODE = 0;
const uint32_t AGG_READY_MODE = 1;
const uint32_t PREFIX_READY_MODE = 2;
const uint32_t NOT_INIT_COUNT_INDEX = 0;
const uint32_t AGG_READY_COUNT_INDEX = 8;
const uint32_t PREFIX_READY_COUNT_INDEX = 16;
const uint32_t THREAD_DIM_NUM = 1024; // simt 开1024线程
const uint32_t AGGREGATE_READY_MASK = 0x40000000;
const uint64_t AGGREGATE_READY_MASK_B64 = 0x4000000000000000;
const uint32_t PREFIX_READY_MASK = 0x80000000;
const uint64_t PREFIX_READY_MASK_B64 = 0x8000000000000000;
const uint32_t VALUE_MASK = 0x3fffffff;
const uint64_t VALUE_MASK_B64 = 0x3fffffffffffffff;
const uint32_t CONST_2 = 2;

// T1输入x dtype T2输出Idx dtype UT无符号的数据类型
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
class SortRadixMoreCore {
public:
    __aicore__ inline SortRadixMoreCore(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
        const SortRegBaseTilingData *__restrict tilingData, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ClearWorkSapce();
    __aicore__ inline LocalTensor<UT> PreProcess(LocalTensor<T1> inputX, uint32_t numTileData);
    __aicore__ inline void PreGlobalExcusiveSum(LocalTensor<UT> &inputXCopy, LocalTensor<T3> &blockExcusiveUb,
        LocalTensor<uint16_t> &histUb, LocalTensor<uint16_t> &histCumsumUb, LocalTensor<uint8_t> &inputB8Ub,
        uint32_t currTileSize, uint32_t round, uint32_t tileId);
    __aicore__ inline void GetGlobalExcusiveSum(uint32_t round, uint32_t sortLoopRound, GlobalTensor<T1> inputX);
    __aicore__ inline void ScatterBlockHist2Global(LocalTensor<uint16_t> blockHist,
        LocalTensor<T3> blockHistWithFlag, GlobalTensor<T3> allblockHistToGm, uint32_t tileId,
        uint32_t round);
    __aicore__ inline void ProcessRadixSort(GlobalTensor<T1> inputXGm, int64_t gmOffset, uint32_t sortLoopRound);
    __aicore__ inline void ParserTilingData();
    __aicore__ inline void ComputeOnePass(uint32_t round, uint32_t sortLoopRound, GlobalTensor<T1> inputXGm);
    __aicore__ inline void DataCopyWorkSpaceToUb(LocalTensor<uint8_t> inputX8Ub, LocalTensor<uint16_t> blockExcusiveUb,
        LocalTensor<uint16_t> blockHistUb, uint32_t unSortId, uint32_t tileId, uint32_t curSize);
    __aicore__ inline void LookbackGlobal(LocalTensor<T3> nowTileHistBuffer,
        GlobalTensor<T3> allTileHistBuffer, LocalTensor<uint32_t> ubFlagTensor, uint32_t tileId, uint32_t round);
    __aicore__ inline void AddPrevfixMask(LocalTensor<T3> &blockHistWithFlag,
        GlobalTensor<T3> blockHistToGm, uint32_t tileId, uint32_t round);
    __aicore__ inline void ScatterKeysGlobal(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
        LocalTensor<uint32_t> xInputIndexLocal, LocalTensor<uint8_t> inputX8BitValue,
        LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
        LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
        uint32_t cureTileSize);
    __aicore__ inline void ScatterOutInt32(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
        LocalTensor<uint32_t> xInputIndexLocal, LocalTensor<uint8_t> inputX8BitValue,
        LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
        LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
        uint32_t cureTileSize);
    __aicore__ inline void ScatterOutInt32ToInt64(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
        LocalTensor<uint32_t> xInputIndexLocal, LocalTensor<uint8_t> inputX8BitValue,
        LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<uint32_t> blockDataInGlobalPos,
        LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
        uint32_t cureTileSize);
    __aicore__ inline void ScatterOutInt64(LocalTensor<T1> xInputValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
        LocalTensor<uint32_t> xInputIndexLocal, LocalTensor<uint8_t> inputX8BitValue,
        LocalTensor<uint16_t> blockExcusiveSum, LocalTensor<T3> blockDataInGlobalPos,
        LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist, uint32_t round, T3 tileDataStart,
        uint32_t cureTileSize);
    __aicore__ inline uint64_t CeilDivMul(uint64_t a, uint64_t b);

protected:
    GlobalTensor<T1> inputXGm_;
    GlobalTensor<T1> outValueGm_;
    GlobalTensor<uint32_t> outIdxGm_;

    GlobalTensor<T1> outValueDbWK_;
    GlobalTensor<uint32_t> outIdxDbWK_;
    GlobalTensor<uint8_t> xB8GmWk_;
    GlobalTensor<uint16_t> histTileGmWk_;
    GlobalTensor<uint16_t> histCumsumTileGmWk_;
    GlobalTensor<uint32_t> globalHistGmWk_;
    GlobalTensor<T3> globalHistGmWkTmp_;
    GlobalTensor<uint32_t> excusiveBinsGmWk_;
    GlobalTensor<T3> excusiveBinsGmWkTmp_;

    TPipe *pipe_;
    const SortRegBaseTilingData *tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueIndex_;
    TQue<QuePosition::VECIN, 1> inQueueGlobalHist_;
    TQue<QuePosition::VECIN, 1> blockExcusiveInQue_;
    TQue<QuePosition::VECIN, 1> blockHistInQue_;
    TBuf<TPosition::VECCALC> tmpUb_;
    TQue<QuePosition::VECOUT, 1> blockUbFlagQue_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inputB8Que_;
    TQue<QuePosition::VECOUT, 1> outIdxQueue_;
    TQue<QuePosition::VECOUT, 1> outValueQueue_;
    TQue<QuePosition::VECOUT, 1> blockHistFlagUbQue_;

    DoubleBufferSimd<T1> inputXDbGm_;
    DoubleBufferSimd<uint32_t> idxDbGm_;

    static constexpr SortConfig sortConfigMuti{ SortType::RADIX_SORT, false };
    int64_t totalDataNum_ = 0;
    uint32_t numTileData_ = 0;
    int64_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t lastDimTileNum_ = 0;
    uint32_t sortLoopTimes_ = 0;
    uint32_t lastDimRealCore_ = 0;
    uint32_t tmpUbSize_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t realCoreNum_ = 0;
    uint32_t clearCore1_ = 0;
    uint32_t clearCore0_ = 0;
    uint32_t clearSize_ = 0;
    uint32_t clearCout_ = 0;
    uint32_t clearCoreSize0_ = 0;
    uint32_t clearCoreSize1_ = 0;
    uint32_t factor_ = 1;
    DataCopyExtParams copyParams{ 1, 1, 0, 0, 0 };
    uint64_t oneBlock_ = Ops::Base::GetUbBlockSize();
};

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex,
    GM_ADDR workspace, const SortRegBaseTilingData *__restrict tilingData, TPipe *pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tilingData_ = tilingData;
    ParserTilingData();
    realCoreNum_ = GetBlockNum();
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        factor_ = CONST_2;
    }

    inputXGm_.SetGlobalBuffer((__gm__ T1 *)x);
    outValueGm_.SetGlobalBuffer((__gm__ T1 *)value);
    outIdxGm_.SetGlobalBuffer((__gm__ uint32_t *)sortIndex);
    uint64_t wkOffset = clearCoreSize0_ * clearCore0_;
    uint64_t oneBlockNumB32 = oneBlock_ / sizeof(int32_t);
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        wkOffset = wkOffset * CONST_2;
    }
    wkOffset = CeilDivMul(wkOffset, oneBlockNumB32);
    excusiveBinsGmWk_.SetGlobalBuffer((__gm__ uint32_t *)workspace, wkOffset);
    wkOffset = wkOffset * sizeof(uint32_t);

    uint64_t histOffset = clearCout_ * clearSize_ * clearCore1_;
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        histOffset = histOffset * CONST_2;
    }
    histOffset = CeilDivMul(histOffset, oneBlockNumB32);
    globalHistGmWk_.SetGlobalBuffer((__gm__ uint32_t *)(workspace + wkOffset), histOffset);
    wkOffset = wkOffset + histOffset * sizeof(uint32_t);

    uint64_t dbOffset = totalDataNum_ * unsortedDimParallel_;
    if constexpr (sizeof(T3) == sizeof(int64_t)) {
        dbOffset = dbOffset * CONST_2;
    }
    dbOffset = CeilDivMul(dbOffset, oneBlockNumB32);
    outIdxDbWK_.SetGlobalBuffer((__gm__ uint32_t *)(workspace + wkOffset), dbOffset);
    wkOffset = wkOffset + dbOffset * sizeof(uint32_t);

    uint64_t histTileOffset = lastDimTileNum_ * RADIX_SORT_NUM * unsortedDimParallel_;
    histTileGmWk_.SetGlobalBuffer((__gm__ uint16_t *)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);
    histCumsumTileGmWk_.SetGlobalBuffer((__gm__ uint16_t *)(workspace + wkOffset), histTileOffset);
    wkOffset = wkOffset + histTileOffset * sizeof(uint16_t);

    uint64_t xB8Offset = lastDimTileNum_ * numTileData_ * unsortedDimParallel_;
    xB8Offset = CeilDivMul(xB8Offset, oneBlock_);
    xB8GmWk_.SetGlobalBuffer((__gm__ uint8_t *)(workspace + wkOffset), xB8Offset);
    wkOffset = wkOffset + xB8Offset * sizeof(uint8_t);

    dbOffset = totalDataNum_ * unsortedDimParallel_;
    dbOffset = CeilDivMul(dbOffset * sizeof(T1), oneBlock_) / sizeof(T1);
    outValueDbWK_.SetGlobalBuffer((__gm__ T1 *)(workspace + wkOffset), dbOffset);

    pipe_->InitBuffer(inQueueX_, 1, numTileData_ * sizeof(T1));
    pipe_->InitBuffer(inQueueIndex_, 1, numTileData_ * sizeof(T3));
    pipe_->InitBuffer(inQueueGlobalHist_, 1, RADIX_SORT_NUM * sizeof(T3));
    pipe_->InitBuffer(outValueQueue_, 1, numTileData_);
    pipe_->InitBuffer(blockExcusiveInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    pipe_->InitBuffer(blockHistInQue_, 1, RADIX_SORT_NUM * sizeof(uint16_t));
    pipe_->InitBuffer(blockUbFlagQue_, 1, RADIX_SORT_NUM * sizeof(T3));
    pipe_->InitBuffer(inputB8Que_, 1, numTileData_);
    pipe_->InitBuffer(outIdxQueue_, 1, numTileData_ * sizeof(uint32_t));
    pipe_->InitBuffer(tmpUb_, tmpUbSize_);
    pipe_->InitBuffer(blockHistFlagUbQue_, 1, RADIX_SORT_NUM * sizeof(T3));

    globalHistGmWkTmp_ = globalHistGmWk_.template ReinterpretCast<T3>();
    excusiveBinsGmWkTmp_ = excusiveBinsGmWk_.template ReinterpretCast<T3>();
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline uint64_t SortRadixMoreCore<T1, T2, UT, T3, isDescend>::CeilDivMul(uint64_t a, uint64_t b)
{
    return ((a + b - 1) / b) * b;
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ParserTilingData()
{
    totalDataNum_ = tilingData_->lastAxisNum;                // h轴大小
    numTileData_ = tilingData_->numTileDataSize;             // ub循环块大小
    unsortedDimNum_ = tilingData_->unsortedDimNum;           // b轴大小
    unsortedDimParallel_ = tilingData_->unsortedDimParallel; // b轴使用的核数
    lastDimTileNum_ = tilingData_->lastDimTileNum;           // h轴循环次数
    sortLoopTimes_ = tilingData_->sortLoopTimes;             // b轴循环次数
    lastDimRealCore_ = tilingData_->lastDimNeedCore;         // h轴需要的核数
    tmpUbSize_ = tilingData_->tmpUbSize;                     // 高级api需要用的ub大小

    clearCore1_ = tilingData_->keyParams0;     // 用于清零的globalHistGmWk_的核
    clearCore0_ = tilingData_->keyParams1;     // 用于清零excusiveBinsGmWk_的核
    clearSize_ = tilingData_->keyParams2;      // 每次清零的ub大小，按照大的globalHistGmWk_所需ub算
    clearCout_ = tilingData_->keyParams3;      // 清零globalHistGmWk_ ub循环次数
    clearCoreSize0_ = tilingData_->keyParams4; // 清零excusiveBinsGmWk_,每个核处理多少个数
    clearCoreSize1_ = tilingData_->keyParams5; // 清零globalHistGmWk_，每个核处理多少
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ClearWorkSapce()
{
    if (blockIdx_ < clearCore1_) {
        AscendC::LocalTensor<uint32_t> tmpUb = tmpUb_.Get<uint32_t>();
        if constexpr (sizeof(T3) == sizeof(uint32_t)) {
            Duplicate(tmpUb, static_cast<uint32_t>(0), clearSize_);
        } else {
            Duplicate(tmpUb, static_cast<uint32_t>(0), clearSize_ * factor_);
        }
        event_t eventIdMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdMte3);
        if (blockIdx_ < clearCore0_) {
            copyParams.blockLen = clearCoreSize0_ * sizeof(T3);
            uint32_t offset = blockIdx_ * clearCoreSize0_  * factor_;
            DataCopyPad(excusiveBinsGmWk_[offset], tmpUb, copyParams);
            for (uint32_t i = 0; i < clearCout_; i++) {
                uint32_t histGmOffset = (blockIdx_ * clearCoreSize1_ + i * clearSize_) * factor_;
                copyParams.blockLen = clearSize_ * sizeof(T3);
                DataCopyPad(globalHistGmWk_[histGmOffset], tmpUb, copyParams);
            }
        } else {
            for (uint32_t i = 0; i < clearCout_; i++) {
                uint32_t histGmOffset = (blockIdx_ * clearCoreSize1_ + i * clearSize_) * factor_;
                copyParams.blockLen = clearSize_ * sizeof(T3);
                DataCopyPad(globalHistGmWk_[histGmOffset], tmpUb, copyParams);
            }
        }
    }
}

template <typename T>
__aicore__ inline T SortGetMin(int64_t left, uint32_t right)
{
    return (left > static_cast<int64_t>(right) ? static_cast<T>(right) : static_cast<T>(left));
}
template <typename UT, uint64_t isDescend>
__aicore__ inline void Twiddle(uint16_t repeatTime, uint32_t vfLen, uint32_t inputNum, RegTensor<UT> &xorReg,
    __local_mem__ UT *xValuePtr, __local_mem__ UT *uXValuePtr)
{
    MicroAPI::MaskReg xorMask;
    MicroAPI::RegTensor<UT> inputReg;
    MicroAPI::RegTensor<UT> vnotReg;
    MicroAPI::RegTensor<UT> xorResult;
    for (uint16_t i = 0; i < repeatTime; i++) {
        xorMask = MicroAPI::UpdateMask<UT>(inputNum);
        // load input
        MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, vfLen);
        // vxor
        MicroAPI::Xor(xorResult, inputReg, xorReg, xorMask);
        if constexpr (isDescend == 0) {
            // sts
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, xorResult, vfLen, xorMask);
        } else {
            MicroAPI::Not(vnotReg, xorResult, xorMask);
            // sts
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, vfLen, xorMask);
        }
    }
}
template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB32(LocalTensor<T1> inputX, LocalTensor<UT> uInputX, uint32_t numTileData)
{
    __local_mem__ UT *xValuePtr = (__ubuf__ UT *)inputX.GetPhyAddr();
    __local_mem__ UT *uXValuePtr = (__ubuf__ UT *)uInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B32);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorValue;
        MicroAPI::MaskReg xorMask;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorValue, XOR_OP_VALUE, maskB32);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B32, inputNum, xorValue, xValuePtr, uXValuePtr);
    }
}
template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB16(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT *xValuePtr = (__ubuf__ UT *)inputX.GetPhyAddr();
    __local_mem__ UT *uXValuePtr = (__ubuf__ UT *)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B16);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorReg;
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorReg, XOR_OP_VALUE_B16, maskB16);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B16, inputNum, xorReg, xValuePtr, uXValuePtr);
    }
}
template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB8(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT *xValuePtr = (__ubuf__ UT *)inputX.GetPhyAddr();
    __local_mem__ UT *uXValuePtr = (__ubuf__ UT *)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B8);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorReg;
        MicroAPI::MaskReg maskB8 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorReg, XOR_OP_VALUE_B8, maskB8);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B8, inputNum, xorReg, xValuePtr, uXValuePtr);
    }
}
template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInB64(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT *xValuePtr = (__ubuf__ UT *)inputX.GetPhyAddr();
    __local_mem__ UT *uXValuePtr = (__ubuf__ UT *)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B64);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> xorReg;
        MicroAPI::MaskReg predicateDefaultB64 = MicroAPI::CreateMask<UT>();
        MicroAPI::Duplicate(xorReg, XOR_OP_VALUE_B64, predicateDefaultB64);
        Twiddle<UT, isDescend>(repeatTime, VF_LEN_B64, inputNum, xorReg, xValuePtr, uXValuePtr);
    }
}
template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInFp16(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ uint16_t *xValuePtr = (__ubuf__ uint16_t *)inputX.GetPhyAddr();
    __local_mem__ uint16_t *uXValuePtr = (__ubuf__ uint16_t *)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B16);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> inputReg, vnotReg;
        MicroAPI::RegTensor<uint16_t> xorMaskReg, vandMask;
        MicroAPI::RegTensor<uint16_t> twiddledZeroReg;
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg xorMask;
        MicroAPI::Duplicate(xorMaskReg, LOWEST_KEY_VALUE_B16, maskB16);
        MicroAPI::Duplicate(vandMask, XOR_OP_VALUE_B16, maskB16);
        MicroAPI::Duplicate(twiddledZeroReg, TWIDDLED_ZERO_BITS_FP16, maskB16);

        for (uint16_t i = 0; i < repeatTime; i++) {
            xorMask = MicroAPI::UpdateMask<uint16_t>(inputNum);
            // load input
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, VF_LEN_B16);
            // vand
            MicroAPI::RegTensor<uint16_t> andValueOne;
            MicroAPI::And(andValueOne, inputReg, vandMask, maskB16);
            // not equal
            MicroAPI::MaskReg cmpValueOne;
            MicroAPI::CompareScalar<uint16_t, CMPMODE::NE>(cmpValueOne, andValueOne, ZERO_VALUE_FLAG_B16, maskB16);
            // vsel
            MicroAPI::RegTensor<uint16_t> finalMaskOne;
            MicroAPI::Select(finalMaskOne, xorMaskReg, vandMask, cmpValueOne);
            // vxor
            MicroAPI::RegTensor<uint16_t> xorVectorOne;
            MicroAPI::Xor(xorVectorOne, inputReg, finalMaskOne, maskB16);
            
            // 目前每个round都会做twiddleIn，因此直接在twiddleIn方法将负0转换为正0
            // 如果后期需要改为单次twiddleIn，则需要在提取位数时将负0转换为正0
            // get -0.0 mask
            MicroAPI::MaskReg minusZeroMask;
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(minusZeroMask, xorVectorOne, TWIDDLED_MINUS_ZERO_BITS_FP16, maskB16);
            // change -0.0 to +0.0
            MicroAPI::RegTensor<uint16_t> resultReg;
            MicroAPI::Select(resultReg, twiddledZeroReg, xorVectorOne, minusZeroMask);

            // sts
            if constexpr (isDescend == 0) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, resultReg,
                    VF_LEN_B16, xorMask);
            } else {
                // reverse op
                MicroAPI::Not(vnotReg, resultReg, xorMask);
                // sts
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, VF_LEN_B16,
                    xorMask);
            }
        }
    }
}
template <typename T1, typename UT, uint64_t isDescend>
__aicore__ inline void TwiddleInFp32(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ uint32_t *xValuePtr = (__ubuf__ uint32_t *)inputX.GetPhyAddr();
    __local_mem__ uint32_t *uXValuePtr = (__ubuf__ uint32_t *)uintInputX.GetPhyAddr();
    uint16_t repeatTime = CeilDivision(numTileData, VF_LEN_B32);
    uint32_t inputNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> inputReg, vnotReg;
        MicroAPI::RegTensor<uint32_t> xorMaskReg, vandMask;
        MicroAPI::RegTensor<uint32_t> twiddledZeroReg;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg xorMask;
        MicroAPI::Duplicate(xorMaskReg, LOWEST_KEY_VALUE_B32, maskB32);
        MicroAPI::Duplicate(vandMask, XOR_OP_VALUE, maskB32);
        MicroAPI::Duplicate(twiddledZeroReg, TWIDDLED_ZERO_BITS_FP32, maskB32);
        for (uint16_t i = 0; i < repeatTime; i++) {
            xorMask = MicroAPI::UpdateMask<uint32_t>(inputNum);
            // load input
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputReg, xValuePtr, VF_LEN_B32);
            // vand
            MicroAPI::RegTensor<uint32_t> andValueOne;
            MicroAPI::And(andValueOne, inputReg, vandMask, maskB32);
            // not equal
            MicroAPI::MaskReg cmpValueOne;
            MicroAPI::CompareScalar<uint32_t, CMPMODE::NE>(cmpValueOne, andValueOne, ZERO_VALUE_FLAG_B32, xorMask);
            // vsel
            MicroAPI::RegTensor<uint32_t> finalMaskOne;
            MicroAPI::Select(finalMaskOne, xorMaskReg, vandMask, cmpValueOne);
            // vxor
            MicroAPI::RegTensor<uint32_t> xorVectorZero;
            MicroAPI::Xor(xorVectorZero, inputReg, finalMaskOne, maskB32);

            // 目前每个round都会做twiddleIn，因此直接在twiddleIn方法将负0转换为正0
            // 如果后期需要改为单次twiddleIn，则需要在提取位数时将负0转换为正0
            // get -0.0 mask
            MicroAPI::MaskReg minusZeroMask;
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(minusZeroMask, xorVectorZero, TWIDDLED_MINUS_ZERO_BITS_FP32, maskB32);
            // change -0.0 to +0.0
            MicroAPI::RegTensor<uint32_t> resultReg;
            MicroAPI::Select(resultReg, twiddledZeroReg, xorVectorZero, minusZeroMask);

            if constexpr (isDescend == 0) {
                // sts
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, resultReg,
                    VF_LEN_B32, xorMask);
            } else {
                // ~
                MicroAPI::Not(vnotReg, resultReg, maskB32);
                // sts
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(uXValuePtr, vnotReg, VF_LEN_B32,
                    xorMask);
            }
        }
    }
}
template <typename T1, typename UT>
__aicore__ inline void ReverseInputData(LocalTensor<T1> inputX, LocalTensor<UT> uintInputX, uint32_t numTileData)
{
    __local_mem__ UT *inputXValuePtr = (__ubuf__ UT *)inputX.GetPhyAddr();
    __local_mem__ UT *reverseInputXPtr = (__ubuf__ UT *)uintInputX.GetPhyAddr();
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(UT);
    uint16_t repeatTime = CeilDivision(numTileData, vfLen);
    uint32_t inputElementNum = numTileData;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<UT> inputVectorOne;
        MicroAPI::RegTensor<UT> vnotVectorZero;
        MicroAPI::MaskReg predicateDefaultB8 = MicroAPI::CreateMask<UT>();
        for (uint16_t i = 0; i < repeatTime; i++) {
            MicroAPI::MaskReg vnotMask = MicroAPI::UpdateMask<UT>(inputElementNum);
            // load input
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr, vfLen);
            // ~
            MicroAPI::Not(vnotVectorZero, inputVectorOne, predicateDefaultB8);
            // sts
            MicroAPI::DataCopy<UT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(reverseInputXPtr, vnotVectorZero, vfLen,
                vnotMask);
        }
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline LocalTensor<UT> SortRadixMoreCore<T1, T2, UT, T3, isDescend>::PreProcess(LocalTensor<T1> inputX,
    uint32_t numTileData)
{
    LocalTensor<UT> inputXCopy = inputX.template ReinterpretCast<UT>();
    if constexpr (IsSameType<int32_t, T1>::value) {
        TwiddleInB32<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
    } else if constexpr (IsSameType<half, T1>::value || IsSameType<bfloat16_t, T1>::value) {
        TwiddleInFp16<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
    } else if constexpr (IsSameType<float, T1>::value) {
        TwiddleInFp32<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
    } else if constexpr (IsSameType<int16_t, T1>::value) {
        TwiddleInB16<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
    } else if constexpr (IsSameType<int8_t, T1>::value) {
        TwiddleInB8<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
    } else if constexpr (IsSameType<int64_t, T1>::value) {
        TwiddleInB64<T1, UT, isDescend>(inputX, inputXCopy, numTileData);
    } else {
        if (isDescend == 1) {
            ReverseInputData<T1, UT>(inputX, inputXCopy, numTileData);
        }
    }
    return inputXCopy;
}
template <typename T1>
__aicore__ inline void CopyDataIn(GlobalTensor<T1> inputX, LocalTensor<T1> xLocal, uint64_t tileOffset,
    uint32_t currTileSize)
{
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T1)) / sizeof(T1);
    DataCopyPadExtParams<T1> padParams{ false, 0, 0, 0 };
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T1);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
}
template <typename T3>
__aicore__ inline void ComputeSumChist(RegTensor<uint16_t> &chist0, RegTensor<uint16_t> &chist1,
    RegTensor<uint16_t> &hist0, RegTensor<uint16_t> &hist1, MaskReg &maskB16, MaskReg &maskB32,
    __local_mem__ T3 *blockExcusiveUbRPtr, __local_mem__ T3 *blockExcusiveUbWPtr,
    __local_mem__ uint16_t *histUbPtr, __local_mem__ uint16_t *histCumsumUbPtr)
{
    // get excusive sum
    MicroAPI::RegTensor<uint16_t> excusiveSumZero, excusiveSumOne, zeroReg;
    MicroAPI::Sub(excusiveSumZero, chist0, hist0, maskB16);
    MicroAPI::Sub(excusiveSumOne, chist1, hist1, maskB16);

    // 保存每个tile cumsumhist结果
    // store hist to ub
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist0, VF_LEN_B16, maskB16);
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histUbPtr, hist1, VF_LEN_B16, maskB16);
    // store excusive sum to ub
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histCumsumUbPtr, excusiveSumZero,
        VF_LEN_B16, maskB16);
    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histCumsumUbPtr, excusiveSumOne,
        VF_LEN_B16, maskB16);

    // cast B16 to B32
    MicroAPI::Duplicate(zeroReg, 0, maskB16);
    MicroAPI::RegTensor<uint32_t> sum0, sum1, sum2, sum3;
    MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)sum0, (MicroAPI::RegTensor<uint16_t> &)sum1, excusiveSumZero,
        zeroReg);
    MicroAPI::Interleave((MicroAPI::RegTensor<uint16_t> &)sum2, (MicroAPI::RegTensor<uint16_t> &)sum3, excusiveSumOne,
        zeroReg);
    if constexpr (sizeof(T3) == sizeof(uint32_t)) {
        // load global excusive
        MicroAPI::RegTensor<uint32_t> sumIn0, sumIn1, sumIn2, sumIn3;
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn0, blockExcusiveUbRPtr, VF_LEN_B32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn1, blockExcusiveUbRPtr, VF_LEN_B32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn2, blockExcusiveUbRPtr, VF_LEN_B32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(sumIn3, blockExcusiveUbRPtr, VF_LEN_B32);
        // add block ans to global excusive
        MicroAPI::Add(sumIn0, sumIn0, sum0, maskB32);
        MicroAPI::Add(sumIn1, sumIn1, sum1, maskB32);
        MicroAPI::Add(sumIn2, sumIn2, sum2, maskB32);
        MicroAPI::Add(sumIn3, sumIn3, sum3, maskB32);
        // vsts to global
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn0, VF_LEN_B32,
            maskB32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn1, VF_LEN_B32,
            maskB32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn2, VF_LEN_B32,
            maskB32);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, sumIn3, VF_LEN_B32,
            maskB32);
    } else {
        // cast B32 to B64
        MicroAPI::MaskReg maskB64 = MicroAPI::CreateMask<int64_t>();
        MicroAPI::RegTensor<int64_t> sum0Int64, sum1Int64, sum2Int64, sum3Int64;
        MicroAPI::RegTensor<int64_t> sum4Int64, sum5Int64, sum6Int64, sum7Int64;
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t> &)sum0Int64, (MicroAPI::RegTensor<uint32_t> &)sum1Int64,
            sum0, (MicroAPI::RegTensor<uint32_t> &)zeroReg);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t> &)sum2Int64, (MicroAPI::RegTensor<uint32_t> &)sum3Int64,
            sum1, (MicroAPI::RegTensor<uint32_t> &)zeroReg);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t> &)sum4Int64, (MicroAPI::RegTensor<uint32_t> &)sum5Int64,
            sum2, (MicroAPI::RegTensor<uint32_t> &)zeroReg);
        MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t> &)sum6Int64, (MicroAPI::RegTensor<uint32_t> &)sum7Int64,
            sum3, (MicroAPI::RegTensor<uint32_t> &)zeroReg);
        // load global excusive
        MicroAPI::RegTensor<int64_t> int64In0, int64In1, int64In2, int64In3;
        MicroAPI::RegTensor<int64_t> int64In4, int64In5, int64In6, int64In7;
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In0, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In1, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In2, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In3, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In4, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In5, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In6, blockExcusiveUbRPtr, VF_LEN_B64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(int64In7, blockExcusiveUbRPtr, VF_LEN_B64);
        // add block ans to global excusive
        MicroAPI::Add(int64In0, int64In0, sum0Int64, maskB64);
        MicroAPI::Add(int64In1, int64In1, sum1Int64, maskB64);
        MicroAPI::Add(int64In2, int64In2, sum2Int64, maskB64);
        MicroAPI::Add(int64In3, int64In3, sum3Int64, maskB64);
        MicroAPI::Add(int64In4, int64In4, sum4Int64, maskB64);
        MicroAPI::Add(int64In5, int64In5, sum5Int64, maskB64);
        MicroAPI::Add(int64In6, int64In6, sum6Int64, maskB64);
        MicroAPI::Add(int64In7, int64In7, sum7Int64, maskB64);
        // vsts to global
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In0, VF_LEN_B64,
            maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In1, VF_LEN_B64,
            maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In2, VF_LEN_B64,
            maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In3, VF_LEN_B64,
            maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In4, VF_LEN_B64,
            maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In5, VF_LEN_B64,
            maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In6, VF_LEN_B64,
            maskB64);
        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockExcusiveUbWPtr, int64In7, VF_LEN_B64,
            maskB64);
    }
}

template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB32(LocalTensor<UT> &inputX, LocalTensor<T3> &blockExcusiveUb,
    LocalTensor<uint16_t> &histUb, LocalTensor<uint16_t> &histCumsumUb, LocalTensor<uint8_t> &inputB8Ub,
    uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint32_t *inXPtr = (__ubuf__ uint32_t *)inputX.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbWPtr = (__ubuf__ T3 *)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t *histUbPtr = (__ubuf__ uint16_t *)histUb.GetPhyAddr();
    __local_mem__ uint16_t *histCumsumUbPtr = (__ubuf__ uint16_t *)histCumsumUb.GetPhyAddr();
    __local_mem__ uint8_t *inputB8UbPtr = (__ubuf__ uint8_t *)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint32_t> in0, in1, in2, in3;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::MaskReg histMask;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B32);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in1, inXPtr, VF_LEN_B32);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in2, inXPtr, VF_LEN_B32);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in3, inXPtr, VF_LEN_B32);
            // vshr
            MicroAPI::RegTensor<uint32_t> shift0, shift1, shift2, shift3;
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift0, in0, bitOffset, maskB32);
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift1, in1, bitOffset, maskB32);
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift2, in2, bitOffset, maskB32);
            MicroAPI::ShiftRights<uint32_t, int16_t>(shift3, in3, bitOffset, maskB32);
            // get and value
            MicroAPI::RegTensor<uint16_t> deInter0, deInter1, deInter2, deInter3;
            MicroAPI::DeInterleave(deInter0, deInter1, (MicroAPI::RegTensor<uint16_t> &)shift0,
                (MicroAPI::RegTensor<uint16_t> &)shift1);
            MicroAPI::DeInterleave(deInter2, deInter3, (MicroAPI::RegTensor<uint16_t> &)shift2,
                (MicroAPI::RegTensor<uint16_t> &)shift3);

            MicroAPI::RegTensor<uint8_t> deInter0B8, deInter1B8;
            MicroAPI::DeInterleave(deInter0B8, deInter1B8, (MicroAPI::RegTensor<uint8_t> &)deInter0,
                (MicroAPI::RegTensor<uint8_t> &)deInter2);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8,
                histMask);
            // get hist
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
            // get cusum
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
            histUbPtr, histCumsumUbPtr);
    }
}
template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB64(LocalTensor<UT> &inputX, LocalTensor<T3> &blockExcusiveUb,
    LocalTensor<uint16_t> &histUb, LocalTensor<uint16_t> &histCumsumUb, LocalTensor<uint8_t> &inputB8Ub,
    uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint64_t *inXPtr = (__ubuf__ uint64_t *)inputX.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbWPtr = (__ubuf__ T3 *)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t *histUbPtr = (__ubuf__ uint16_t *)histUb.GetPhyAddr();
    __local_mem__ uint8_t *inputB8UbPtr = (__ubuf__ uint8_t *)inputB8Ub.GetPhyAddr();
    __local_mem__ uint16_t *histCumsumUbPtr = (__ubuf__ uint16_t *)histCumsumUb.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint64_t> in0, in1, in2, in3, in4, in5, in6, in7;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::MaskReg histMask;
        MicroAPI::MaskReg maskB64 = MicroAPI::CreateMask<uint64_t>();
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in1, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in2, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in3, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in4, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in5, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in6, inXPtr, VF_LEN_B64);
            MicroAPI::DataCopy<uint64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in7, inXPtr, VF_LEN_B64);
            // vshr
            MicroAPI::RegTensor<uint64_t> shift0, shift1, shift2, shift3, shift4, shift5, shift6, shift7;
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift0, in0, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift1, in1, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift2, in2, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift3, in3, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift4, in4, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift5, in5, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift6, in6, bitOffset, maskB64);
            MicroAPI::ShiftRights<uint64_t, int16_t>(shift7, in7, bitOffset, maskB64);
            // get and value
            MicroAPI::RegTensor<uint32_t> deInter0, deInter1, deInter2, deInter3, deInter4, deInter5, deInter6,
                deInter7;
            MicroAPI::DeInterleave(deInter0, deInter1, (MicroAPI::RegTensor<uint32_t> &)shift0,
                (MicroAPI::RegTensor<uint32_t> &)shift1);
            MicroAPI::DeInterleave(deInter2, deInter3, (MicroAPI::RegTensor<uint32_t> &)shift2,
                (MicroAPI::RegTensor<uint32_t> &)shift3);
            MicroAPI::DeInterleave(deInter4, deInter5, (MicroAPI::RegTensor<uint32_t> &)shift4,
                (MicroAPI::RegTensor<uint32_t> &)shift5);
            MicroAPI::DeInterleave(deInter6, deInter7, (MicroAPI::RegTensor<uint32_t> &)shift6,
                (MicroAPI::RegTensor<uint32_t> &)shift7);
            MicroAPI::RegTensor<uint16_t> deInter0B16, deInter1B16, deInter2B16, deInter3B16;
            MicroAPI::DeInterleave(deInter0B16, deInter1B16, (MicroAPI::RegTensor<uint16_t> &)deInter0,
                (MicroAPI::RegTensor<uint16_t> &)deInter2);
            MicroAPI::DeInterleave(deInter2B16, deInter3B16, (MicroAPI::RegTensor<uint16_t> &)deInter4,
                (MicroAPI::RegTensor<uint16_t> &)deInter6);
            MicroAPI::RegTensor<uint8_t> deInter0B8, deInter1B8;
            MicroAPI::DeInterleave(deInter0B8, deInter1B8, (MicroAPI::RegTensor<uint8_t> &)deInter0B16,
                (MicroAPI::RegTensor<uint8_t> &)deInter2B16);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8,
                histMask);
            // get hist
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
            // get cusum
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
            histUbPtr, histCumsumUbPtr);
    }
}
template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB16(LocalTensor<UT> &inputX, LocalTensor<T3> &blockExcusiveUb,
    LocalTensor<uint16_t> &histUb, LocalTensor<uint16_t> &histCumsumUb, LocalTensor<uint8_t> &inputB8Ub,
    uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint16_t *inputXValuePtr = (__ubuf__ uint16_t *)inputX.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbWPtr = (__ubuf__ T3 *)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t *histUbPtr = (__ubuf__ uint16_t *)histUb.GetPhyAddr();
    __local_mem__ uint16_t *histCumsumUbPtr = (__ubuf__ uint16_t *)histCumsumUb.GetPhyAddr();
    __local_mem__ uint8_t *inputB8UbPtr = (__ubuf__ uint8_t *)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg histMask;
        MicroAPI::RegTensor<uint16_t> in0, in1;
        MicroAPI::RegTensor<uint16_t> shift0, shift1;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inputXValuePtr, VF_LEN_B16);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in1, inputXValuePtr, VF_LEN_B16);
            // vshr
            MicroAPI::ShiftRights<uint16_t, int16_t>(shift0, in0, bitOffset, maskB16);
            MicroAPI::ShiftRights<uint16_t, int16_t>(shift1, in1, bitOffset, maskB16);
            // get and value
            MicroAPI::RegTensor<uint8_t> deInter0B8, deInter1B8;
            MicroAPI::DeInterleave(deInter0B8, deInter1B8, (MicroAPI::RegTensor<uint8_t> &)shift0,
                (MicroAPI::RegTensor<uint8_t> &)shift1);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, deInter0B8, VF_LEN_B8,
                histMask);
            // get hist
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::FREQUENCY>(hist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::FREQUENCY>(hist1, deInter0B8, histMask);
            // get cusum
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::ACCUMULATE>(chist0, deInter0B8, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::ACCUMULATE>(chist1, deInter0B8, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
            histUbPtr, histCumsumUbPtr);
    }
}
template <typename UT, typename T3>
__aicore__ inline void GetGlobalExcusiveSumB8(LocalTensor<UT> &inputX, LocalTensor<T3> &blockExcusiveUb,
    LocalTensor<uint16_t> &histUb, LocalTensor<uint16_t> &histCumsumUb, LocalTensor<uint8_t> &inputB8Ub,
    uint32_t currTileSize, uint32_t round)
{
    uint32_t bitOffset = round * SHIFT_BIT_NUM;
    uint16_t repeatTime = CeilDivision(currTileSize, VF_LEN_B8);
    __local_mem__ uint8_t *inXPtr = (__ubuf__ uint8_t *)inputX.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbWPtr = (__ubuf__ T3 *)blockExcusiveUb.GetPhyAddr();
    __local_mem__ T3 *blockExcusiveUbRPtr = blockExcusiveUbWPtr;
    __local_mem__ uint16_t *histUbPtr = (__ubuf__ uint16_t *)histUb.GetPhyAddr();
    __local_mem__ uint16_t *histCumsumUbPtr = (__ubuf__ uint16_t *)histCumsumUb.GetPhyAddr();
    __local_mem__ uint8_t *inputB8UbPtr = (__ubuf__ uint8_t *)inputB8Ub.GetPhyAddr();
    uint32_t inputElementNum = currTileSize;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg histMask;
        MicroAPI::RegTensor<uint8_t> in0;
        MicroAPI::RegTensor<uint16_t> hist0, hist1, chist0, chist1;
        MicroAPI::RegTensor<uint16_t> chistVectorZero, chistVectorOne, zeroReg;
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::MaskReg maskB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::Duplicate(hist0, 0, maskB16);
        MicroAPI::Duplicate(hist1, 0, maskB16);
        MicroAPI::Duplicate(chist0, 0, maskB16);
        MicroAPI::Duplicate(chist1, 0, maskB16);
        for (uint16_t i = 0; i < repeatTime; i++) {
            histMask = MicroAPI::UpdateMask<uint8_t>(inputElementNum);
            // load input
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(in0, inXPtr, VF_LEN_B8);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(inputB8UbPtr, in0, VF_LEN_B8,
                histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::FREQUENCY>(hist0, in0, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::FREQUENCY>(hist1, in0, histMask);
            // get cusum
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN0,
                MicroAPI::HistogramsType::ACCUMULATE>(chist0, in0, histMask);
            MicroAPI::Histograms<uint8_t, uint16_t, MicroAPI::HistogramsBinType::BIN1,
                MicroAPI::HistogramsType::ACCUMULATE>(chist1, in0, histMask);
        }
        ComputeSumChist<T3>(chist0, chist1, hist0, hist1, maskB16, maskB32, blockExcusiveUbRPtr, blockExcusiveUbWPtr,
            histUbPtr, histCumsumUbPtr);
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::PreGlobalExcusiveSum(LocalTensor<UT> &inputXCopy,
    LocalTensor<T3> &blockExcusiveUb, LocalTensor<uint16_t> &histUb, LocalTensor<uint16_t> &histCumsumUb,
    LocalTensor<uint8_t> &inputB8Ub, uint32_t currTileSize, uint32_t round, uint32_t tileId)
{
    if constexpr (sizeof(T1) == sizeof(int32_t)) {
        GetGlobalExcusiveSumB32<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
            round);
    } else if constexpr (sizeof(T1) == sizeof(int16_t)) {
        GetGlobalExcusiveSumB16<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
            round);
    } else if constexpr (sizeof(T1) == sizeof(int8_t)) {
        GetGlobalExcusiveSumB8<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
            round);
    } else if constexpr (sizeof(T1) == sizeof(int64_t)) {
        GetGlobalExcusiveSumB64<UT, T3>(inputXCopy, blockExcusiveUb, histUb, histCumsumUb, inputB8Ub, currTileSize,
            round);
    }
}

template <typename T3>
__simt_vf__ LAUNCH_BOUND(RADIX_SORT_NUM)__aicore__
    void SimtGlobalOffset(uint32_t excusiveBinOffset, __gm__ T3 *excusiveBinsGm, __ubuf__ T3 *blockExcusiveBuffer, __gm__ T3* outGM_)
{
    for (int32_t i = Simt::GetThreadIdx(); i < RADIX_SORT_NUM; i += RADIX_SORT_NUM) {
        int32_t offset = i;
        T3 srcData = blockExcusiveBuffer[offset];
        Simt::AtomicAdd<T3>(excusiveBinsGm + excusiveBinOffset + offset, srcData);
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::GetGlobalExcusiveSum(uint32_t round,
    uint32_t sortLoopRound, GlobalTensor<T1> inputX)
{
    uint32_t startTileId = blockIdx_ % lastDimRealCore_;
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    uint32_t unsortedDimIndex = unSortId + sortLoopRound * unsortedDimParallel_;
    int64_t xUnsortOffset = unSortId * totalDataNum_;
    if (unsortedDimIndex < unsortedDimNum_) {
        LocalTensor<uint32_t> blockExcusiveUb = blockUbFlagQue_.AllocTensor<uint32_t>();

        uint32_t excusiveBinOffset = unSortId * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
        if constexpr (sizeof(T3) == sizeof(uint32_t)) {
            Duplicate(blockExcusiveUb, static_cast<uint32_t>(0), RADIX_SORT_NUM);
        } else {
            Duplicate(blockExcusiveUb, static_cast<uint32_t>(0), RADIX_SORT_NUM * CONST_2);
        }
        for (uint32_t tileId = startTileId; tileId < lastDimTileNum_; tileId += lastDimRealCore_) {
            uint64_t tileOffset = tileId * numTileData_;    // 可能超过INT64最大值，用uint64
            uint64_t remainTileDataNum = totalDataNum_ - tileOffset;
            if (totalDataNum_ < tileOffset) {
                break;
            }
            uint32_t currTileSize = SortGetMin<uint32_t>(remainTileDataNum, numTileData_);
            // copy gm to ub
            LocalTensor<T1> xLocal = inQueueX_.AllocTensor<T1>();

            if (round == 0) {
                CopyDataIn<T1>(inputX[xUnsortOffset], xLocal, tileOffset, currTileSize);
            } else {
                CopyDataIn<T1>(inputXDbGm_.Current()[xUnsortOffset], xLocal, tileOffset, currTileSize);
            }
            inQueueX_.EnQue(xLocal);
            xLocal = inQueueX_.DeQue<T1>();
            // convert singed data to unsinged
            LocalTensor<UT> xUbCopy = PreProcess(xLocal, currTileSize);
            LocalTensor<uint8_t> inputB8Ub = inputB8Que_.AllocTensor<uint8_t>();
            LocalTensor<uint16_t> histUb = outIdxQueue_.AllocTensor<uint16_t>();
            LocalTensor<uint16_t> histCumsumUb = outValueQueue_.AllocTensor<uint16_t>();
            LocalTensor<T3> blockExcusiveUbTmp = blockExcusiveUb.template ReinterpretCast<T3>();
            PreGlobalExcusiveSum(xUbCopy, blockExcusiveUbTmp, histUb, histCumsumUb, inputB8Ub, currTileSize, round,
                tileId);
            inQueueX_.FreeTensor(xLocal);
            inputB8Que_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(inputB8Ub);
            inputB8Ub = inputB8Que_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, uint8_t>();
            copyParams.blockCount = 1;
            copyParams.blockLen = currTileSize * sizeof(uint8_t);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(xB8GmWk_[xUnsortOffset + tileOffset], inputB8Ub, copyParams);
            inputB8Que_.FreeTensor(inputB8Ub);

            outIdxQueue_.EnQue(histUb);
            histUb = outIdxQueue_.DeQue<uint16_t>();
            copyParams.blockCount = 1;
            copyParams.blockLen = RADIX_SORT_NUM * sizeof(uint16_t);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(histTileGmWk_[unSortId * RADIX_SORT_NUM * lastDimTileNum_ + tileId * RADIX_SORT_NUM], histUb,
                copyParams);
            outIdxQueue_.FreeTensor(histUb);

            outValueQueue_.EnQue(histCumsumUb);
            histCumsumUb = outValueQueue_.DeQue<uint16_t>();
            DataCopyPad(histCumsumTileGmWk_[unSortId * RADIX_SORT_NUM * lastDimTileNum_ + tileId * RADIX_SORT_NUM],
                histCumsumUb, copyParams);
            outValueQueue_.FreeTensor(histCumsumUb);
        }
        blockUbFlagQue_.EnQue(blockExcusiveUb);
        blockExcusiveUb = blockUbFlagQue_.DeQue<uint32_t>();
        if constexpr (sizeof(T3) == sizeof(uint32_t)) {
            copyParams.blockCount = 1;
            copyParams.blockLen = RADIX_SORT_NUM * sizeof(uint32_t);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;

            SetAtomicAdd<int32_t>();
            DataCopyPad(excusiveBinsGmWk_[excusiveBinOffset], blockExcusiveUb, copyParams);
            SetAtomicNone();
        } else {
            // 超过int32表示范围，计算globaloffset时int32已经不够了，所以用simt的int64的atomicadd
            Simt::VF_CALL<SimtGlobalOffset<T3>>(Simt::Dim3(RADIX_SORT_NUM), excusiveBinOffset,
                (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockExcusiveUb.GetPhyAddr()), (__gm__ T3 *)(outIdxGm_.GetPhyAddr()));
        }
        blockUbFlagQue_.FreeTensor(blockExcusiveUb);
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ScatterBlockHist2Global(
    LocalTensor<uint16_t> blockHist, LocalTensor<T3> blockHistWithFlag, GlobalTensor<T3> allblockHistToGm,
    uint32_t tileId, uint32_t round)
{
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    int64_t unSortIdOffset =
        unSortId * lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM * lastDimTileNum_;
    __local_mem__ uint16_t *blockHistPtr = (__ubuf__ uint16_t *)blockHist.GetPhyAddr();
    __local_mem__ T3 *blockHistWithFlagPtr = (__ubuf__ T3 *)blockHistWithFlag.GetPhyAddr();
    // 软同步，写入状态位2bit
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg predicateDefaultB16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::RegTensor<uint16_t> blockHistZero, blockHistOne;
        MicroAPI::RegTensor<uint16_t> lookaheadOutZero, lookaheadOutOne, lookaheadOutTwo, lookaheadOutThree;
        MicroAPI::RegTensor<uint16_t> zeroVector;
        MicroAPI::Duplicate(zeroVector, 0, predicateDefaultB16);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistZero, blockHistPtr, VF_LEN_B16);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistOne, blockHistPtr, VF_LEN_B16);
        // cast b16 to B32
        MicroAPI::Interleave(lookaheadOutZero, lookaheadOutOne, blockHistZero, zeroVector);
        MicroAPI::Interleave(lookaheadOutTwo, lookaheadOutThree, blockHistOne, zeroVector);
        if constexpr (IsSameType<T3, uint32_t>::value) {
            MicroAPI::RegTensor<uint32_t> aggregateReadyMask;
            MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint32_t>();
            MicroAPI::Duplicate(aggregateReadyMask, AGGREGATE_READY_MASK, predicateDefault);
            // add mask
            MicroAPI::RegTensor<uint32_t> lookaheadOutZeroMask, lookaheadOutOneMask, lookaheadOutTwoMask,
                lookaheadOutThreeMask;
            MicroAPI::Or(lookaheadOutZeroMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutZero, aggregateReadyMask,
                predicateDefault);
            MicroAPI::Or(lookaheadOutOneMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutOne, aggregateReadyMask,
                predicateDefault);
            MicroAPI::Or(lookaheadOutTwoMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutTwo, aggregateReadyMask,
                predicateDefault);
            MicroAPI::Or(lookaheadOutThreeMask, (MicroAPI::RegTensor<uint32_t> &)lookaheadOutThree, aggregateReadyMask,
                predicateDefault);
            // vst
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutZeroMask, VF_LEN_B32, predicateDefault);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, lookaheadOutOneMask,
                VF_LEN_B32, predicateDefault);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr, lookaheadOutTwoMask,
                VF_LEN_B32, predicateDefault);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutThreeMask, VF_LEN_B32, predicateDefault);
        } else {
            MicroAPI::RegTensor<int64_t> aggregateReadyMask;
            MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<int64_t>();
            MicroAPI::Duplicate(aggregateReadyMask, AGGREGATE_READY_MASK_B64, predicateDefault);
            MicroAPI::RegTensor<uint16_t> lookaheadOutZeroB64A, lookaheadOutZeroB64B;
            MicroAPI::RegTensor<uint16_t> lookaheadOutOneB64A, lookaheadOutOneB64B;
            MicroAPI::RegTensor<uint16_t> lookaheadOutTwoB64A, lookaheadOutTwoB64B;
            MicroAPI::RegTensor<uint16_t>lookaheadOutThreeB64A, lookaheadOutThreeB64B;
            MicroAPI::Interleave(lookaheadOutZeroB64A, lookaheadOutZeroB64B, lookaheadOutZero, zeroVector);
            MicroAPI::Interleave(lookaheadOutOneB64A, lookaheadOutOneB64B, lookaheadOutOne, zeroVector);
            MicroAPI::Interleave(lookaheadOutTwoB64A, lookaheadOutTwoB64B, lookaheadOutTwo, zeroVector);
            MicroAPI::Interleave(lookaheadOutThreeB64A, lookaheadOutThreeB64B, lookaheadOutThree, zeroVector);
            MicroAPI::RegTensor<int64_t> lookaheadOutZeroMaskB64A, lookaheadOutZeroMaskB64B;
            MicroAPI::RegTensor<int64_t> lookaheadOutOneMaskB64A, lookaheadOutOneMaskB64B;
            MicroAPI::RegTensor<int64_t> lookaheadOutTwoMaskB64A, lookaheadOutTwoMaskB64B;
            MicroAPI::RegTensor<int64_t> lookaheadOutThreeMaskB64A, lookaheadOutThreeMaskB64B;
            MicroAPI::Or(lookaheadOutZeroMaskB64A, (MicroAPI::RegTensor<int64_t> &)lookaheadOutZeroB64A,
                aggregateReadyMask, predicateDefault);
            MicroAPI::Or(lookaheadOutZeroMaskB64B, (MicroAPI::RegTensor<int64_t> &)lookaheadOutZeroB64B,
                aggregateReadyMask, predicateDefault);
            MicroAPI::Or(lookaheadOutOneMaskB64A, (MicroAPI::RegTensor<int64_t> &)lookaheadOutOneB64A,
                aggregateReadyMask, predicateDefault);
            MicroAPI::Or(lookaheadOutOneMaskB64B, (MicroAPI::RegTensor<int64_t> &)lookaheadOutOneB64B,
                aggregateReadyMask, predicateDefault);
            MicroAPI::Or(lookaheadOutTwoMaskB64A, (MicroAPI::RegTensor<int64_t> &)lookaheadOutTwoB64A,
                aggregateReadyMask, predicateDefault);
            MicroAPI::Or(lookaheadOutTwoMaskB64B, (MicroAPI::RegTensor<int64_t> &)lookaheadOutTwoB64B,
                aggregateReadyMask, predicateDefault);
            MicroAPI::Or(lookaheadOutThreeMaskB64A, (MicroAPI::RegTensor<int64_t> &)lookaheadOutThreeB64A,
                aggregateReadyMask, predicateDefault);
            MicroAPI::Or(lookaheadOutThreeMaskB64B, (MicroAPI::RegTensor<int64_t> &)lookaheadOutThreeB64B,
                aggregateReadyMask, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutZeroMaskB64A, VF_LEN_B64, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutZeroMaskB64B, VF_LEN_B64, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutOneMaskB64A, VF_LEN_B64, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutOneMaskB64B, VF_LEN_B64, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutTwoMaskB64A, VF_LEN_B64, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutTwoMaskB64B, VF_LEN_B64, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutThreeMaskB64A, VF_LEN_B64, predicateDefault);
            MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(blockHistWithFlagPtr,
                lookaheadOutThreeMaskB64B, VF_LEN_B64, predicateDefault);
        }
    }
    blockHistFlagUbQue_.EnQue(blockHistWithFlag);
    blockHistWithFlag = blockHistFlagUbQue_.DeQue<T3>();
    if (tileId < (lastDimTileNum_ - 1)) {
        DataCopyExtParams dataCopyParam;
        dataCopyParam.blockCount = 1;
        dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(T3);
        dataCopyParam.srcStride = 0;
        dataCopyParam.dstStride = 0;
        DataCopyPad(allblockHistToGm[unSortIdOffset + RADIX_SORT_NUM * tileId], blockHistWithFlag, dataCopyParam);
    }
    blockHistFlagUbQue_.FreeTensor(blockHistWithFlag);
}
template <typename T>
__aicore__ inline void CopyGlobalDataIn(GlobalTensor<T> inputX, LocalTensor<T> &xLocal,
    uint32_t tileOffset, uint32_t currTileSize)
{
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T)) / sizeof(T);
    DataCopyPadExtParams<T> padParams{ false, 0, 0, 0 };
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::LookbackGlobal(
    LocalTensor<T3> nowTileHistBuffer, GlobalTensor<T3> allTileHistBuffer,
    LocalTensor<uint32_t> ubFlagTensor, uint32_t tileId, uint32_t round)
{
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    int64_t unSortIdOffset =
        unSortId * lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM * lastDimTileNum_;
    __local_mem__ T3 *nowTileHistBufferPtr = (__ubuf__ T3 *)nowTileHistBuffer.GetPhyAddr();

    uint16_t repeatTime;
    if constexpr (IsSameType<T3, uint32_t>::value) {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B32;
    } else {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B64;
    }

    for (int i = tileId - 1; i >= 0; --i) {
        int mode = -1;
        uint32_t histTileOffset = RADIX_SORT_NUM * i;
        __local_mem__ uint32_t *ubFlagTensorPtr = (__ubuf__ uint32_t *)ubFlagTensor.GetPhyAddr();
        __local_mem__ T3 *tilePrevHistValuePtrCopy = nullptr;
        // 软同步，等待，读flag
        while (true) {
            LocalTensor<T3> xLocal = inQueueGlobalHist_.AllocTensor<T3>();
            CopyGlobalDataIn<T3>(allTileHistBuffer[unSortIdOffset], xLocal, histTileOffset, RADIX_SORT_NUM);
            inQueueGlobalHist_.EnQue(xLocal);
            LocalTensor<T3> tilePrevHistValue = inQueueGlobalHist_.DeQue<T3>();
            __local_mem__ T3 *tilePrevHistValuePtr = (__ubuf__ T3 *)tilePrevHistValue.GetPhyAddr();
            tilePrevHistValuePtrCopy = tilePrevHistValuePtr;
            // highest 2bit is state bits
            // 0 not inited, should retry
            // 1 aggragate ready accumlate and look i-1 block
            // 2 prefix ready accumlaye and stop
            // 3 undefined
            __VEC_SCOPE__
            {
                MicroAPI::MaskReg pRegSelect;
                MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<uint32_t>();
                // notInit/aggReady/prefixReady flag
                MicroAPI::RegTensor<uint32_t> notInitCount, aggReadyCount, prefixReadyCount;
                MicroAPI::Duplicate(notInitCount, 0, maskB32);
                MicroAPI::Duplicate(aggReadyCount, 0, maskB32);
                MicroAPI::Duplicate(prefixReadyCount, 0, maskB32);
                // flag tensor
                MicroAPI::RegTensor<uint32_t> onesVector, zerosVector;
                MicroAPI::Duplicate(onesVector, 1, maskB32);
                MicroAPI::Duplicate(zerosVector, 0, maskB32);
                // count notInit/aggReady/prefixReady number
                for (uint16_t i = 0; i < repeatTime; i++) {
                    MicroAPI::RegTensor<T3> prevTileHistValue;
                    MicroAPI::RegTensor<uint32_t> stateBitValue;
                    // load pre tile hist
                    if constexpr (IsSameType<T3, uint32_t>::value) {
                        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(prevTileHistValue,
                            tilePrevHistValuePtr, VF_LEN_B32);
                        // get highest 2bit
                        MicroAPI::ShiftRights<uint32_t, int16_t>(stateBitValue, prevTileHistValue, STATE_BIT_SHF_VALUE,
                            maskB32);
                        pRegSelect = maskB32;
                    } else {
                        // 此处通过逻辑位移，得到状态位
                        MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(prevTileHistValue,
                            tilePrevHistValuePtr, VF_LEN_B64);
                        MicroAPI::MaskReg maskB64 = MicroAPI::CreateMask<int64_t>();
                        MicroAPI::RegTensor<uint64_t> stateTmp;
                        MicroAPI::ShiftRights<uint64_t, int16_t>(stateTmp, (MicroAPI::RegTensor<uint64_t> &)prevTileHistValue,
                                                                STATE_BIT_SHF_VALUE_B64, maskB64);
                        MicroAPI::Pack(stateBitValue, stateTmp);
                        pRegSelect = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::H>();
                    }
                    // get not init cnt
                    MicroAPI::MaskReg maskNotInit;
                    MicroAPI::RegTensor<uint32_t> maskNotInitCount;
                    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskNotInit, stateBitValue, NOT_INIT_MODE, pRegSelect);
                    MicroAPI::Select(maskNotInitCount, onesVector, zerosVector, maskNotInit);
                    MicroAPI::Add(notInitCount, notInitCount, maskNotInitCount, maskNotInit);
                    // get aggrefate ready cnt
                    MicroAPI::MaskReg maskAggReady;
                    MicroAPI::RegTensor<uint32_t> maskAggReadyCount;
                    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskAggReady, stateBitValue, AGG_READY_MODE,
                        pRegSelect);
                    MicroAPI::Select(maskAggReadyCount, onesVector, zerosVector, maskAggReady);
                    MicroAPI::Add(aggReadyCount, aggReadyCount, maskAggReadyCount, maskAggReady);
                    // get prefix ready cnt
                    MicroAPI::MaskReg maskPrefixReady;
                    MicroAPI::RegTensor<uint32_t> maskPrefixReadyCount;
                    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(maskPrefixReady, stateBitValue, PREFIX_READY_MODE,
                        pRegSelect);
                    MicroAPI::Select(maskPrefixReadyCount, onesVector, zerosVector, maskPrefixReady);
                    MicroAPI::Add(prefixReadyCount, prefixReadyCount, maskPrefixReadyCount, maskPrefixReady);
                }
                // reduce sum noInitCount
                MicroAPI::ReduceSum(notInitCount, notInitCount, maskB32);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, notInitCount, HIST_MASK_OUT_LEN,
                    maskB32);
                // reduce sum aggReadyCount
                MicroAPI::ReduceSum(aggReadyCount, aggReadyCount, maskB32);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, aggReadyCount, HIST_MASK_OUT_LEN,
                    maskB32);
                // reduce sum prefixReadyCount
                MicroAPI::ReduceSum(prefixReadyCount, prefixReadyCount, maskB32);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                    MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ubFlagTensorPtr, prefixReadyCount, HIST_MASK_OUT_LEN,
                    maskB32);
            }
            // get (not init)/(aggregate ready)/(preifx ready) cnt
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
            uint32_t notInitCountScalar = ubFlagTensorPtr[NOT_INIT_COUNT_INDEX];
            uint32_t aggReadyCountScalar = ubFlagTensorPtr[AGG_READY_COUNT_INDEX];
            uint32_t PrefixReadyCountScalar = ubFlagTensorPtr[PREFIX_READY_COUNT_INDEX];
            if (aggReadyCountScalar == RADIX_SORT_NUM) {
                mode = AGGREGATE_READY_FLAG;
                inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                break;
            }
            if (PrefixReadyCountScalar == RADIX_SORT_NUM) {
                mode = PREFIX_READY_FLAG;
                inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
                break;
            }
            inQueueGlobalHist_.FreeTensor(tilePrevHistValue);
        }
        __local_mem__ T3 *nowTileHistBufferPtrCopy = nowTileHistBufferPtr;
        event_t eventIdWaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdWaitV);
        WaitFlag<HardEvent::S_V>(eventIdWaitV);
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<uint16_t>();
            MicroAPI::RegTensor<T3> histMask;
            if constexpr (IsSameType<T3, uint32_t>::value) {
                MicroAPI::Duplicate(histMask, VALUE_MASK, predicateDefault);
                for (uint16_t i = 0; i < repeatTime; i++) {
                    MicroAPI::RegTensor<uint32_t> nowTileHistVal, prevTileHistVal;
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(nowTileHistVal,
                        nowTileHistBufferPtr, VF_LEN_B32);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(prevTileHistVal,
                        tilePrevHistValuePtrCopy, VF_LEN_B32);
                    // remove mask and get value
                    MicroAPI::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
                    MicroAPI::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
                    // add prev hist value
                    // get value key same which tile id less and equal now
                    MicroAPI::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(nowTileHistBufferPtrCopy,
                        nowTileHistVal, VF_LEN_B32, predicateDefault);
                }
            } else {
                MicroAPI::Duplicate(histMask, VALUE_MASK_B64, predicateDefault);
                for (uint16_t i = 0; i < repeatTime; i++) {
                    MicroAPI::RegTensor<int64_t> nowTileHistVal, prevTileHistVal;
                    MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(nowTileHistVal,
                                                                                        nowTileHistBufferPtr,
                                                                                        VF_LEN_B64);
                    MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(prevTileHistVal,
                                                                                        tilePrevHistValuePtrCopy,
                                                                                        VF_LEN_B64);
                    // remove mask and get value
                    MicroAPI::And(nowTileHistVal, nowTileHistVal, histMask, predicateDefault);
                    MicroAPI::And(prevTileHistVal, prevTileHistVal, histMask, predicateDefault);
                    // add prev hist value
                    // get value key same which tile id less and equal now
                    MicroAPI::Add(nowTileHistVal, nowTileHistVal, prevTileHistVal, predicateDefault);
                    MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(nowTileHistBufferPtrCopy,
                                                                                        nowTileHistVal,
                                                                                        VF_LEN_B64,
                                                                                        predicateDefault);
                }
            }
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        if (mode == PREFIX_READY_FLAG) {
            break;
        }
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::AddPrevfixMask(
    LocalTensor<T3> &blockHistWithFlag, GlobalTensor<T3> blockHistToGm, uint32_t tileId, uint32_t round)
{
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    int64_t unSortIdOffset =
        unSortId * lastDimTileNum_ * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM * lastDimTileNum_;
    __local_mem__ T3 *histCumsumPtr = (__ubuf__ T3 *)blockHistWithFlag.GetPhyAddr();
    __local_mem__ T3 *histCumsumPtrCopy = histCumsumPtr;

    uint16_t repeatTime;
    if constexpr (IsSameType<T3, uint32_t>::value) {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B32;
    } else {
        repeatTime = RADIX_SORT_NUM / VF_LEN_B64;
    }
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T3> prefixReadyMask, prefixRemainMask;
        MicroAPI::MaskReg predicateDefault = MicroAPI::CreateMask<T3>();
        if constexpr (IsSameType<T3, uint32_t>::value) {
            MicroAPI::Duplicate(prefixReadyMask, PREFIX_READY_MASK, predicateDefault);
            MicroAPI::Duplicate(prefixRemainMask, VALUE_MASK, predicateDefault);
            for (uint16_t repate = 0; repate < repeatTime; repate++) {
                MicroAPI::RegTensor<uint32_t> keyCumsumValue;
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue, histCumsumPtr,
                    VF_LEN_B32);
                // remove mask and get value
                // 清除状态位
                MicroAPI::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
                // add new mask
                // 添加OK状态位
                MicroAPI::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histCumsumPtrCopy, keyCumsumValue,
                    VF_LEN_B32, predicateDefault);
            }
        } else {
            MicroAPI::Duplicate(prefixReadyMask, PREFIX_READY_MASK_B64, predicateDefault);
            MicroAPI::Duplicate(prefixRemainMask, VALUE_MASK_B64, predicateDefault);
            for (uint16_t repate = 0; repate < repeatTime; repate++) {
                MicroAPI::RegTensor<int64_t> keyCumsumValue;
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(keyCumsumValue,
                                                                                    histCumsumPtr,
                                                                                    VF_LEN_B64);
                // remove mask and get value
                MicroAPI::And(keyCumsumValue, keyCumsumValue, prefixRemainMask, predicateDefault);
                // add new mask
                MicroAPI::Or(keyCumsumValue, keyCumsumValue, prefixReadyMask, predicateDefault);
                MicroAPI::DataCopy<int64_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(histCumsumPtrCopy,
                                                                                    keyCumsumValue,
                                                                                    VF_LEN_B64,
                                                                                    predicateDefault);
            }
        }
    }
    // copy value to gm
    blockHistFlagUbQue_.EnQue(blockHistWithFlag);
    blockHistWithFlag = blockHistFlagUbQue_.DeQue<T3>();
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(T3);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(blockHistToGm[unSortIdOffset + RADIX_SORT_NUM * tileId], blockHistWithFlag, dataCopyParam);
}
template <typename T1, typename T2, typename T3, int32_t round>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM_NUM)__aicore__ void CopyOutGm(T3 tileDataStart, uint32_t cureTileSize,
    uint32_t outputXUnsortedAxisOffset, uint32_t unSortIdOffset,
    __ubuf__ uint16_t *blockExcusiveSumAddr,     // 当前tile块直方图局部前缀和
    __gm__ volatile T3 *excusiveBinsGmAddr,      // 全局前缀和（适配int64）
    __ubuf__ T3 *blockDataInGlobalPosAddr,       // 位置信息
    __ubuf__ uint32_t *sortedIndexLocalAddr,     // 8bit排序后的idx
    __ubuf__ T3 *xInputIndexLocalAddr,           // 8bit在workspace的idx
    __ubuf__ uint8_t *inputX8BitValueAddr,       // 8bit在workspace的value
    __ubuf__ T1 *xInputValueLocalAddr,           // tile块的X值
    __ubuf__ T3 *blockHistFlagAddr,              // blockHistFlag_(适配int64_t)
    __ubuf__ uint16_t *blockHistAddr,            // blockHist_, 当前tile块直方图统计
    __gm__ volatile T2 *indexDoubleBufferGmAddr, // 输出workspcae的idx
    __gm__ volatile T1 *inputXDoubleBufferAddr)  // 输出workspace的value
{
    for (int i = Simt::GetThreadIdx(); i < RADIX_SORT_NUM; i += THREAD_DIM_NUM) {
        // how many data key = i and block id le to now block id
        T3 blockHistCumsumVal = blockHistFlagAddr[i]; // lookahead_output
        // 高2比特为状态位
        if constexpr (IsSameType<T3, uint32_t>::value) {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK; // lookahead_output
        } else {
            blockHistCumsumVal = blockHistCumsumVal & VALUE_MASK_B64;
        }
        
        // block key=i excusive sum
        uint32_t blockExcusiveSumVal = blockExcusiveSumAddr[i]; // blk_exclusive_hist_val
        // block key=i num
        uint32_t blockHistVal = blockHistAddr[i]; // block_hist_val
        // global key<i num
        T3 globalKeyOffsetVal = excusiveBinsGmAddr[unSortIdOffset + i];
        // now block key=i pos
        // real stand for block data in global pos which not have in block pos
        T3 finalpos = globalKeyOffsetVal + blockHistCumsumVal - blockHistVal - blockExcusiveSumVal;
        blockDataInGlobalPosAddr[i] = finalpos;
    }
    Simt::ThreadBarrier();
    for (int i = Simt::GetThreadIdx(); i < cureTileSize; i += THREAD_DIM_NUM) {
        // i stand for pos
        // sorted lcoal index content  stand for data index
        // 本地排序后的数据索引
        T3 localDataIndex = static_cast<T3>(sortedIndexLocalAddr[i]);
        T3 dataInitIndex = 0;
        if constexpr (round != 0) {
            dataInitIndex = xInputIndexLocalAddr[localDataIndex];
        } else {
            dataInitIndex = tileDataStart + localDataIndex;
        }

        // blockDataInGlobalPos stand for one data in globa pos
        // i stand for data in now block pos
        T3 dataFinalGlobalPos = blockDataInGlobalPosAddr[inputX8BitValueAddr[localDataIndex]] + i;
        // store to gm
        inputXDoubleBufferAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = xInputValueLocalAddr[localDataIndex];
        indexDoubleBufferGmAddr[dataFinalGlobalPos + outputXUnsortedAxisOffset] = static_cast<T2>(dataInitIndex); // T2为int32，int64
    }
}

template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ScatterOutInt32(LocalTensor<T1> xInputValueLocal,
    LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> inputX8BitValue, LocalTensor<uint16_t> blockExcusiveSum,
    LocalTensor<T3> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
    uint32_t round, T3 tileDataStart, uint32_t cureTileSize)
{
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    uint32_t outputXUnsortedAxisOffset = unSortId * totalDataNum_;
    uint32_t unSortIdOffset = unSortId * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    if (round == 0) {
        Simt::VF_CALL<CopyOutGm<T1, uint32_t, T3, 0>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ T3 *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ T3 *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ uint32_t *)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else {
        Simt::VF_CALL<CopyOutGm<T1, uint32_t, T3, 1>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ T3 *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ T3 *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ uint32_t *)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ScatterOutInt32ToInt64(LocalTensor<T1> xInputValueLocal,
    LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> inputX8BitValue, LocalTensor<uint16_t> blockExcusiveSum,
    LocalTensor<uint32_t> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
    uint32_t round, T3 tileDataStart, uint32_t cureTileSize)
{
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    uint32_t outputXUnsortedAxisOffset = unSortId * totalDataNum_;
    uint32_t unSortIdOffset = unSortId * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
    if (round == 0) {
        Simt::VF_CALL<CopyOutGm<T1, uint32_t, T3, 0>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ uint32_t *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ uint32_t *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ uint32_t *)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else if (round < static_cast<uint32_t>(sizeof(T1) - 1)) {
        Simt::VF_CALL<CopyOutGm<T1, uint32_t, T3, 1>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ uint32_t *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ uint32_t *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ uint32_t *)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else {
        GlobalTensor<T2> outIdxT2 = (idxDbGm_.Alternate()).template ReinterpretCast<T2>();
        Simt::VF_CALL<CopyOutGm<T1, T2, T3, 1>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ uint32_t *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ uint32_t *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ T2 *)(outIdxT2.GetPhyAddr()), (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ScatterOutInt64(LocalTensor<T1> xInputValueLocal,
    LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> inputX8BitValue, LocalTensor<uint16_t> blockExcusiveSum,
    LocalTensor<T3> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
    uint32_t round, T3 tileDataStart, uint32_t cureTileSize)
{
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    uint32_t outputXUnsortedAxisOffset = unSortId * totalDataNum_;
    uint32_t unSortIdOffset = unSortId * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;

    if (round == 0) {
        Simt::VF_CALL<CopyOutGm<T1, T2, T3, 0>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ T3 *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ T3 *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ T2 *)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else {
        Simt::VF_CALL<CopyOutGm<T1, T2, T3, 1>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ T3 *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ T3 *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ T2 *)(idxDbGm_.Alternate().GetPhyAddr()),
            (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ScatterKeysGlobal(LocalTensor<T1> xInputValueLocal,
    LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<uint32_t> xInputIndexLocal,
    LocalTensor<uint8_t> inputX8BitValue, LocalTensor<uint16_t> blockExcusiveSum,
    LocalTensor<T3> blockDataInGlobalPos, LocalTensor<uint32_t> blockHistFlag, LocalTensor<uint16_t> blockHist,
    uint32_t round, T3 tileDataStart, uint32_t cureTileSize)
{
    if constexpr (sizeof(T1) == sizeof(int8_t)) {
        // int8时只循环一次,所以scatter时肯定要按照输出数据类型
        uint32_t unSortId = blockIdx_ / lastDimRealCore_;
        uint32_t outputXUnsortedAxisOffset = unSortId * totalDataNum_;
        uint32_t unSortIdOffset = unSortId * RADIX_SORT_NUM * sizeof(T1) + round * RADIX_SORT_NUM;
        GlobalTensor<T2> outIdxT2 = (idxDbGm_.Alternate()).template ReinterpretCast<T2>();
        Simt::VF_CALL<CopyOutGm<T1, T2, T3, 0>>(Simt::Dim3(THREAD_DIM_NUM), tileDataStart, cureTileSize,
            outputXUnsortedAxisOffset, unSortIdOffset, (__ubuf__ uint16_t *)(blockExcusiveSum.GetPhyAddr()),
            (__gm__ T3 *)(excusiveBinsGmWk_.GetPhyAddr()), (__ubuf__ T3 *)(blockDataInGlobalPos.GetPhyAddr()),
            (__ubuf__ uint32_t *)(sortedIndexLocal.GetPhyAddr()), (__ubuf__ T3 *)(xInputIndexLocal.GetPhyAddr()),
            (__ubuf__ uint8_t *)(inputX8BitValue.GetPhyAddr()), (__ubuf__ T1 *)(xInputValueLocal.GetPhyAddr()),
            (__ubuf__ T3 *)(blockHistFlag.GetPhyAddr()), (__ubuf__ uint16_t *)(blockHist.GetPhyAddr()),
            (__gm__ T2 *)(outIdxT2.GetPhyAddr()), (__gm__ T1 *)(inputXDbGm_.Alternate().GetPhyAddr()));
    } else if constexpr (sizeof(T2) == sizeof(int32_t)) {
        // 输出idx本省就是int32，无需cast
        ScatterOutInt32(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, inputX8BitValue, blockExcusiveSum,
            blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize);
    } else if constexpr (IsSameType<T3, uint32_t>::value){
        // 输出idx是int64，需要在最后一次scatter时cast为int64
        ScatterOutInt32ToInt64(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, inputX8BitValue, blockExcusiveSum,
            blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize);
    } else {
        // 计算过程中idx使用int64
        ScatterOutInt64(xInputValueLocal, sortedIndexLocal, xInputIndexLocal, inputX8BitValue, blockExcusiveSum,
            blockDataInGlobalPos, blockHistFlag, blockHist, round, tileDataStart, cureTileSize);
    }
}
template <typename T>
__aicore__ inline void CopyIndexDataIn(GlobalTensor<uint32_t> inputIndex, LocalTensor<uint32_t> &xLocal,
    uint64_t tileOffset, uint32_t currTileSize)
{
    DataCopyPadExtParams<uint32_t> padParams{ false, 0, 0, 0 };
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputIndex[tileOffset], dataCopyParam, padParams);
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::DataCopyWorkSpaceToUb(
    LocalTensor<uint8_t> inputX8Ub, LocalTensor<uint16_t> blockExcusiveUb, LocalTensor<uint16_t> blockHistUb,
    uint32_t unSortId, uint32_t tileId, uint32_t curSize)
{
    DataCopyPadExtParams<uint16_t> padParams{ false, 0, 0, 0 };
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = RADIX_SORT_NUM * sizeof(uint16_t);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(blockExcusiveUb,
        histCumsumTileGmWk_[unSortId * RADIX_SORT_NUM * lastDimTileNum_ + tileId * RADIX_SORT_NUM], dataCopyParam,
        padParams);
    DataCopyPad(blockHistUb, histTileGmWk_[unSortId * RADIX_SORT_NUM * lastDimTileNum_ + tileId * RADIX_SORT_NUM],
        dataCopyParam, padParams);

    DataCopyPadExtParams<uint8_t> padParamsB8{ false, 0, 0, 0 };
    DataCopyExtParams dataCopyParam1;
    dataCopyParam1.blockCount = 1;
    dataCopyParam1.blockLen = curSize * sizeof(uint8_t);
    dataCopyParam1.srcStride = 0;
    dataCopyParam1.dstStride = 0;
    DataCopyPad(inputX8Ub, xB8GmWk_[unSortId * totalDataNum_ + tileId * numTileData_], dataCopyParam1, padParamsB8);
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ComputeOnePass(uint32_t round,
    uint32_t sortLoopRound, GlobalTensor<T1> inputXGm)
{
    uint32_t startId = blockIdx_ % lastDimRealCore_;
    uint32_t unSortId = blockIdx_ / lastDimRealCore_;
    uint32_t unsortedDimIndex = unSortId + sortLoopRound * unsortedDimParallel_;
    int64_t xUnsortOffset = unSortId * totalDataNum_;
    if (unsortedDimIndex < unsortedDimNum_) {
        for (uint32_t tileId = startId; tileId < lastDimTileNum_; tileId += lastDimRealCore_) {
            uint64_t tileOffset = tileId * numTileData_;
            uint64_t tileDataStart = tileId * numTileData_; // 可能超INT64最大值，用uint64
            uint64_t remainTileDataNum = totalDataNum_ - tileDataStart;
            if (totalDataNum_ < tileDataStart) {
                break;
            }
            uint32_t currTileSize = SortGetMin<uint32_t>(remainTileDataNum, numTileData_);
            LocalTensor<T1> xLocal = inQueueX_.AllocTensor<T1>();
            if (round == 0) {
                CopyDataIn<T1>(inputXGm[xUnsortOffset], xLocal, tileOffset, currTileSize);
            } else {
                CopyDataIn<T1>(inputXDbGm_.Current()[xUnsortOffset], xLocal, tileOffset, currTileSize);
            }
            inQueueX_.EnQue(xLocal);
            xLocal = inQueueX_.DeQue<T1>();
            // get block hist/excusive
            LocalTensor<uint8_t> inputX8Ub = inputB8Que_.AllocTensor<uint8_t>();
            LocalTensor<uint16_t> blockExcusiveUb = blockExcusiveInQue_.AllocTensor<uint16_t>();
            LocalTensor<uint16_t> blockHistUb = blockHistInQue_.AllocTensor<uint16_t>();
            DataCopyWorkSpaceToUb(inputX8Ub, blockExcusiveUb, blockHistUb, unSortId, tileId, currTileSize);
            blockHistInQue_.EnQue(blockHistUb);
            blockExcusiveInQue_.EnQue(blockExcusiveUb);
            inputB8Que_.EnQue<QuePosition::VECIN, QuePosition::VECCALC>(inputX8Ub);
            inputX8Ub = inputB8Que_.DeQue<QuePosition::VECIN, QuePosition::VECCALC, uint8_t>();
            blockHistUb = blockHistInQue_.DeQue<uint16_t>();

            LocalTensor<T3> blockHistFlagUb = blockHistFlagUbQue_.AllocTensor<T3>();
            ScatterBlockHist2Global(blockHistUb, blockHistFlagUb, globalHistGmWkTmp_, tileId, round);

            // need add static
            // get sort need buffer
            LocalTensor<uint8_t> shareTmpBuffer = tmpUb_.Get<uint8_t>();
            AscendC::LocalTensor<uint32_t> sortedValueIndexLocal = outIdxQueue_.AllocTensor<uint32_t>();
            AscendC::LocalTensor<uint8_t> sortedValueLocal = outValueQueue_.AllocTensor<uint8_t>();
            AscendC::Sort<uint8_t, false, sortConfigMuti>(sortedValueLocal, sortedValueIndexLocal, inputX8Ub,
                shareTmpBuffer, static_cast<uint32_t>(currTileSize));
            outValueQueue_.FreeTensor(sortedValueLocal);
            outIdxQueue_.EnQue<uint32_t>(sortedValueIndexLocal);
            AscendC::LocalTensor<uint32_t> ubFlagTensor = blockUbFlagQue_.AllocTensor<uint32_t>();
            // not first tile
            LocalTensor<T3> blockHistFlagUb1 = blockHistFlagUbQue_.AllocTensor<T3>();
            LocalTensor<uint32_t> blockHistFlagUb1Tmp = blockHistFlagUb1.template ReinterpretCast<uint32_t>();
            if (tileId > 0) {
                // get key=xxx which block id less and equal to now
                LookbackGlobal(blockHistFlagUb1, globalHistGmWkTmp_, ubFlagTensor, tileId, round);
            }
            blockUbFlagQue_.FreeTensor(ubFlagTensor);
            // not last tile
            if (tileId < (lastDimTileNum_ - 1)) {
                // add prefix mask to block hist. 打上OK状态位，高2bit
                AddPrevfixMask(blockHistFlagUb1, globalHistGmWkTmp_, tileId, round);
            }
            blockHistFlagUbQue_.FreeTensor(blockHistFlagUb1);
            LocalTensor<uint32_t> xIndexLocal;
            if (round != 0) {
                xIndexLocal = inQueueIndex_.AllocTensor<uint32_t>();
                CopyIndexDataIn<T3>(idxDbGm_.Current()[xUnsortOffset * factor_], xIndexLocal, tileOffset * factor_, currTileSize);
                inQueueIndex_.EnQue(xIndexLocal);
                xIndexLocal = inQueueIndex_.DeQue<uint32_t>();
            }
            sortedValueIndexLocal = outIdxQueue_.DeQue<uint32_t>();
            AscendC::LocalTensor<T3> blockDataInGlobalPos = blockUbFlagQue_.AllocTensor<T3>();
            blockExcusiveUb = blockExcusiveInQue_.DeQue<uint16_t>();
            LocalTensor<uint32_t> blockHistFlagUb2 = blockHistFlagUbQue_.AllocTensor<uint32_t>();
            ScatterKeysGlobal(xLocal, sortedValueIndexLocal, xIndexLocal, inputX8Ub, blockExcusiveUb, blockDataInGlobalPos,
                blockHistFlagUb2, blockHistUb, round, tileDataStart, currTileSize);
            if (round != 0) {
                inQueueIndex_.FreeTensor(xIndexLocal);
            }
            blockHistFlagUbQue_.FreeTensor(blockHistFlagUb2);
            inQueueX_.FreeTensor(xLocal);
            inputB8Que_.FreeTensor(inputX8Ub);
            blockHistInQue_.FreeTensor(blockHistUb);
            blockUbFlagQue_.FreeTensor(blockDataInGlobalPos);
            blockExcusiveInQue_.FreeTensor(blockExcusiveUb);
            outIdxQueue_.FreeTensor(sortedValueIndexLocal);
        }
        idxDbGm_.selector_ ^= 1;
        inputXDbGm_.selector_ ^= 1;
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::ProcessRadixSort(GlobalTensor<T1> inputXGm,
    int64_t gmOffset, uint32_t sortLoopRound)
{
    ClearWorkSapce();
    SyncAll();

    if constexpr (sizeof(T2) == sizeof(uint32_t)) {
        if constexpr (sizeof(T1) == sizeof(int8_t)) {
            inputXDbGm_.SetDoubleBuffer(outValueDbWK_, outValueGm_[gmOffset]);
            idxDbGm_.SetDoubleBuffer(outIdxDbWK_, outIdxGm_[gmOffset]);
        } else {
            inputXDbGm_.SetDoubleBuffer(outValueGm_[gmOffset], outValueDbWK_);
            idxDbGm_.SetDoubleBuffer(outIdxGm_[gmOffset], outIdxDbWK_);
        }
    } else {
        if constexpr (sizeof(T1) == sizeof(int8_t)) {
            inputXDbGm_.SetDoubleBuffer(outValueDbWK_, outValueGm_[gmOffset]);
            idxDbGm_.SetDoubleBuffer(outIdxDbWK_, outIdxGm_[gmOffset * CONST_2]);
        } else {
            inputXDbGm_.SetDoubleBuffer(outValueGm_[gmOffset], outValueDbWK_);
            idxDbGm_.SetDoubleBuffer(outIdxGm_[gmOffset * CONST_2], outIdxDbWK_);
        }
    }
    for (uint32_t round = 0; round < static_cast<uint32_t>(sizeof(T1)); round++) {
        GetGlobalExcusiveSum(round, sortLoopRound, inputXGm);
        SyncAll();
        ComputeOnePass(round, sortLoopRound, inputXGm);
        SyncAll();
    }
}
template <typename T1, typename T2, typename UT, typename T3, uint64_t isDescend>
__aicore__ inline void SortRadixMoreCore<T1, T2, UT, T3, isDescend>::Process()
{
    if (blockIdx_ > realCoreNum_) {
        return;
    }
    for (uint32_t i = 0; i < sortLoopTimes_; i++) {
        int64_t loopOffset = i * unsortedDimParallel_ * totalDataNum_;
        ProcessRadixSort(inputXGm_[loopOffset], loopOffset, i);
    }
}
} // namespace Sort
#endif // namespace SortRadixMoreCore
