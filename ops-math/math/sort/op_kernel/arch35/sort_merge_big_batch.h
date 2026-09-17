/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sort_merge_big_batch.h
 * \brief Sort Merge for big batch scenario
 * \details This template handles the case where:
 *          1. Data type is fp32
 *          2. Batch dimension (B) >= maxCoreNum
 *          3. Sort axis (N) > 4096
 *          Uses AscendC::Sort for UB sorting and MrgSort for merging.
 */

#ifndef SORT_MERGE_BIG_BATCH_H
#define SORT_MERGE_BIG_BATCH_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace Sort {
using namespace AscendC;

constexpr uint32_t DEALING_CONCAT_NUM_ONCE = 16;
constexpr uint32_t DEALING_SORT_NUM_ONCE = 32;
constexpr uint32_t DEALING_EXTRACT_NUM_ONCE = 32;
constexpr uint32_t MERGE_LIST_MAX_NUM = 4;  // Four-way merge
constexpr uint32_t BLOCK_UB = Ops::Base::GetUbBlockSize();
constexpr uint32_t DOUBLE_BUFFER = 2;


/**
 * @brief SortMergeBigBatch class for handling large batch sort with merge
 * @tparam ValueType Input data type (float)
 * @tparam IndexType Index data type (int32_t or int64_t)
 * @tparam IsDescend Sort order: true for descending, false for ascending
 */
template <typename ValueType, typename IndexType, bool IsDescend>
class SortMergeBigBatch {
public:
    __aicore__ inline SortMergeBigBatch() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR indices, GM_ADDR workspace,
        const SortRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    struct MergeListContext {
        uint32_t remains[MERGE_LIST_MAX_NUM] = {0};
        uint32_t gmOffsets[MERGE_LIST_MAX_NUM] = {0};
        uint32_t srcOffsets[MERGE_LIST_MAX_NUM] = {0};
        uint32_t elemCounts[MERGE_LIST_MAX_NUM] = {0};
        uint32_t listCount = 0;
    };

    // Main processing functions
    __aicore__ inline void SortSingleBatchInUb(GlobalTensor<ValueType> inputX, int64_t batchOffset);
    __aicore__ inline uint32_t MergeSingleBatch();
    __aicore__ inline void ExtractAndCopyOut(int64_t batchIdx, uint32_t resultRegion);
    
    // Helper functions
    __aicore__ inline void CopyInBlock(GlobalTensor<ValueType> inputX, LocalTensor<ValueType> xLocal, 
        uint32_t elemOffset, uint32_t actualElem);
    __aicore__ inline void SortBlockToStruct(LocalTensor<ValueType> xLocal, LocalTensor<ValueType> sortedLocal,
        uint32_t actualElem, uint32_t baseOffset);
    __aicore__ inline void CopyOutToCache(LocalTensor<ValueType> sortedLocal, uint32_t blockId, uint32_t actualElem);
    
    // Merge helper functions
    __aicore__ inline void DoIncrementalMerge(int64_t dstOffsetBase, MergeListContext& ctx);
    __aicore__ inline void MergeOneGroup(uint32_t groupStart, uint32_t groupBlockCount,
        uint32_t fullBlockElemCount, uint32_t fullBlockSortLen, uint32_t lastBlockElemCount,
        uint32_t numBlocks, uint32_t pingPongFlag, uint32_t& cumulativeOffset,
        uint32_t& mergedGroupElemCount);
    __aicore__ inline void CopyBlockChunk(int64_t srcAddr, int64_t dstAddr, uint32_t elemCount);
    
    // DoIncrementalMerge sub-functions
    __aicore__ inline uint32_t LoadListsToUb(LocalTensor<ValueType> ubMainInput,
        uint16_t elementCountList[MERGE_LIST_MAX_NUM], const MergeListContext& ctx);
    __aicore__ inline uint32_t ExecuteMrgSort(LocalTensor<ValueType> dstLocal,
        LocalTensor<ValueType> ubMainInput, uint16_t elementCountList[MERGE_LIST_MAX_NUM],
        uint32_t listSortedNums[MERGE_LIST_MAX_NUM], uint32_t listCount);
    __aicore__ inline void CopyRemainingList(MergeListContext& ctx,
        int64_t dstOffsetBase, uint32_t& dstCumulativeOffset);

    // ExtractAndCopyOut sub-functions
    __aicore__ inline void ExtractAndCopyChunk(int64_t cacheBatchOffset, uint32_t cacheOffset,
        int64_t outputOffset, uint32_t elemProcessed, uint32_t elemCount);
    
    // Data members
    TPipe* pipe_;
    const SortRegBaseTilingData* tilingData_;
    uint32_t blockIdx_ = 0;
    
    // GM buffers
    GlobalTensor<ValueType> inputXGm_;
    GlobalTensor<ValueType> outValueGm_;
    GlobalTensor<IndexType> outIdxGm_;
    GlobalTensor<ValueType> cacheGm_;  // TMP_CACHE for intermediate sorted data (sort struct = 8 bytes)
    
    // Tiling parameters
    int64_t batchNum_ = 0;
    int64_t sortAxisNum_ = 0;
    uint32_t batchPerCore_ = 0;
    uint32_t blockSortSize_ = 0;     // Elements per UB sort block
    uint32_t extractChunkSize_ = 0;  // Elements per Phase3 extract iteration
    uint32_t blocksPerRow_ = 0;
    uint32_t maxCoreNum_ = 0;
    uint32_t alignNum_ = 0;          // blocksPerRow_ * blockSortSize_
    
    // Precomputed constants
    uint32_t blockSortLen_ = 0;      // GetSortLen(blockSortSize_): sort struct length per block
    uint32_t batchSortLen_ = 0;      // GetSortLen(alignNum_): sort struct length per batch
    uint32_t lastBlockSize_ = 0;     // Actual element count of the last block
    uint32_t sortBufferSize_ = 0;    // blockSortLen_ * sizeof(ValueType)
    uint32_t sortRepeatTimes_ = 0;   // blockSortSize_ / DEALING_SORT_NUM_ONCE
    uint32_t concatRepeatTimes_ = 0; // blockSortSize_ / DEALING_CONCAT_NUM_ONCE
    uint32_t maxMergeIterations_ = 0;  // INT32_MAX / blockSortSize_
    
    // Queues and buffers for UB sorting
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outValueQueue_, outIdxQueue_, outIdxInt64Queue_;
    // Temp buffers for Sort API
    TBuf<TPosition::VECCALC> concatTmpBuf_, sortTmpBuf_, indexTmpBuf_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> sortedOutQueue_;

    // Merge queues - single input queue for four-way merge, buffer holds 4 blocks
    TQue<QuePosition::VECIN, 1> mergeInQueue_;
    TQue<QuePosition::VECOUT, 1> mergeOutQueue_;
    
    // Phase 3 queue - independent from Phase 2, can use DOUBLE_BUFFER for pipelining
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> extractInQueue_;
};

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::Init(GM_ADDR x, GM_ADDR value, GM_ADDR indices,
    GM_ADDR workspace, const SortRegBaseTilingData* tilingData, TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tilingData_ = tilingData;
    
    // Parse tiling data
    batchNum_ = tilingData_->unsortedDimNum;
    sortAxisNum_ = tilingData_->lastAxisNum;
    batchPerCore_ = tilingData_->keyParams0;
    blockSortSize_ = tilingData_->numTileDataSize;
    extractChunkSize_ = tilingData_->keyParams4;
    blocksPerRow_ = tilingData_->lastDimTileNum;
    maxCoreNum_ = tilingData_->lastDimNeedCore;
    alignNum_ = tilingData_->keyParams3;
    maxMergeIterations_ = tilingData_->keyParams5;

    if (blockSortSize_ == 0 || extractChunkSize_ == 0 || blocksPerRow_ == 0 || sortAxisNum_ <= 0) {
        return;
    }

    // Precompute constants
    blockSortLen_ = AscendC::GetSortLen<ValueType>(blockSortSize_);
    batchSortLen_ = AscendC::GetSortLen<ValueType>(alignNum_);
    sortBufferSize_ = blockSortLen_ * sizeof(ValueType);
    sortRepeatTimes_ = blockSortSize_ / DEALING_SORT_NUM_ONCE;
    concatRepeatTimes_ = blockSortSize_ / DEALING_CONCAT_NUM_ONCE;
    lastBlockSize_ = static_cast<uint32_t>(
        sortAxisNum_ - static_cast<int64_t>(blocksPerRow_ - 1) * blockSortSize_);

    // Set GM buffers
    inputXGm_.SetGlobalBuffer((__gm__ ValueType*)x);
    outValueGm_.SetGlobalBuffer((__gm__ ValueType*)value);
    outIdxGm_.SetGlobalBuffer((__gm__ IndexType*)indices);
    
    // Cache stores sort struct data (8 bytes per element: index + value)
    // Each core has its own cache region, reused across batches
    // perBatchCacheLen: sort struct length for one batch (with ping-pong, in ValueType units)
    int64_t perCoreCacheLen = static_cast<int64_t>(batchSortLen_) * 2;  // ping-pong, reused per batch

    cacheGm_.SetGlobalBuffer((__gm__ ValueType*)workspace + static_cast<int64_t>(blockIdx_) * perCoreCacheLen);

    // Note: Queue/Buffer initialization is deferred to Process() per phase
    // to optimize UB usage. Each phase initializes only what it needs.
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::Process()
{
    // Calculate batch range for this core
    int64_t startBatch = static_cast<int64_t>(blockIdx_) * batchPerCore_;
    int64_t endBatch = (startBatch + batchPerCore_ < batchNum_) ?
                       (startBatch + batchPerCore_) : batchNum_;

    if (startBatch >= batchNum_) {
        return;  // This core has no work
    }

    // Process each batch through all 3 phases before moving to the next batch.
    // This allows cache (workspace) to be reused across batches within the same core,
    
    // Precompute buffer sizes (loop-invariant)
    uint32_t mergeBufferSize = MERGE_LIST_MAX_NUM * sortBufferSize_;
    uint32_t extractInSize = AscendC::GetSortLen<ValueType>(extractChunkSize_) * sizeof(ValueType);

    for (int64_t batchIdx = startBatch; batchIdx < endBatch; batchIdx++) {
        // ========== Phase 1: Sort blocks in UB ==========
        pipe_->InitBuffer(inQueueX_, DOUBLE_BUFFER, blockSortSize_ * sizeof(ValueType));
        pipe_->InitBuffer(concatTmpBuf_, sortBufferSize_);
        pipe_->InitBuffer(sortTmpBuf_, sortBufferSize_);
        pipe_->InitBuffer(sortedOutQueue_, DOUBLE_BUFFER, sortBufferSize_);
        pipe_->InitBuffer(indexTmpBuf_, blockSortSize_ * sizeof(uint32_t));

        SortSingleBatchInUb(inputXGm_, batchIdx * sortAxisNum_);

        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        pipe_->Reset();

        // ========== Phase 2: Merge sorted blocks (4-way) ==========
        pipe_->InitBuffer(mergeInQueue_, 1, mergeBufferSize);
        pipe_->InitBuffer(mergeOutQueue_, 1, mergeBufferSize);

        uint32_t resultRegion = MergeSingleBatch();

        eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        pipe_->Reset();

        // ========== Phase 3: Extract and copy output ==========
        pipe_->InitBuffer(extractInQueue_, DOUBLE_BUFFER, extractInSize);
        pipe_->InitBuffer(outValueQueue_, DOUBLE_BUFFER, extractChunkSize_ * sizeof(ValueType));
        pipe_->InitBuffer(outIdxQueue_, DOUBLE_BUFFER, extractChunkSize_ * sizeof(uint32_t));
        if constexpr (IsSameType<int64_t, IndexType>::value) {
            pipe_->InitBuffer(outIdxInt64Queue_, DOUBLE_BUFFER, extractChunkSize_ * sizeof(int64_t));
        }

        ExtractAndCopyOut(batchIdx, resultRegion);

        pipe_->Reset();
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::SortSingleBatchInUb(
    GlobalTensor<ValueType> inputX, int64_t batchOffset)
{
    for (uint32_t blockId = 0; blockId < blocksPerRow_; blockId++) {
        uint32_t elemOffset = blockId * blockSortSize_;
        uint32_t actualElem = (blockId < blocksPerRow_ - 1) ? blockSortSize_ : lastBlockSize_;

        LocalTensor<ValueType> xLocal = inQueueX_.AllocTensor<ValueType>();
        CopyInBlock(inputX[batchOffset], xLocal, elemOffset, actualElem);
        inQueueX_.EnQue(xLocal);

        xLocal = inQueueX_.DeQue<ValueType>();
        LocalTensor<ValueType> sortedLocal = sortedOutQueue_.AllocTensor<ValueType>();
        SortBlockToStruct(xLocal, sortedLocal, actualElem, elemOffset);

        inQueueX_.FreeTensor(xLocal);
        sortedOutQueue_.EnQue(sortedLocal);
        sortedLocal = sortedOutQueue_.DeQue<ValueType>();

        CopyOutToCache(sortedLocal, blockId, actualElem);
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::CopyInBlock(
    GlobalTensor<ValueType> inputX, LocalTensor<ValueType> xLocal, uint32_t elemOffset, uint32_t actualElem)
{
    ValueType defaultValue = IsDescend ? static_cast<ValueType>(-INFINITY) : static_cast<ValueType>(NAN);

    // For the last block (actualElem < blockSortSize_), Sort/Concat APIs process up to
    // CeilAlign(actualElem, BLOCK_UB) elements. DataCopyPad only pads to
    // CeilAlign(actualElem, BLOCK_UB/sizeof(ValueType)) which may be smaller.
    // Pre-fill with Duplicate to cover the full aligned region, then overwrite valid data.
    if (actualElem != blockSortSize_) {
        uint32_t alignTile = Ops::Base::CeilAlign(actualElem, BLOCK_UB);
        Duplicate(xLocal, defaultValue, alignTile);
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    DataCopyExtParams dataCopyParam{1, static_cast<uint32_t>(actualElem * sizeof(ValueType)), 0, 0, 0};
    uint32_t currTileSizeAlign = Ops::Base::CeilAlign(actualElem, static_cast<uint32_t>(BLOCK_UB / sizeof(ValueType)));
    DataCopyPadExtParams<ValueType> padParams{true, 0, static_cast<uint8_t>(currTileSizeAlign - actualElem), defaultValue};
    DataCopyPad(xLocal, inputX[elemOffset], dataCopyParam, padParams);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::SortBlockToStruct(
    LocalTensor<ValueType> xLocal, LocalTensor<ValueType> sortedLocal,
    uint32_t actualElem, uint32_t baseOffset)
{
    uint32_t alignSize = (actualElem == blockSortSize_) ? blockSortSize_ : Ops::Base::CeilAlign(actualElem, BLOCK_UB);
    uint32_t sortRepeatTimes = (actualElem == blockSortSize_) ? sortRepeatTimes_ : Ops::Base::CeilDiv(alignSize, DEALING_SORT_NUM_ONCE);
    uint32_t concatRepeatTimes = (actualElem == blockSortSize_) ? concatRepeatTimes_ : Ops::Base::CeilDiv(alignSize, DEALING_CONCAT_NUM_ONCE);

    LocalTensor<ValueType> concatTmpLocal = concatTmpBuf_.Get<ValueType>();
    LocalTensor<ValueType> sortTmpLocal = sortTmpBuf_.Get<ValueType>();
    LocalTensor<uint32_t> indexTmpLocal = indexTmpBuf_.Get<uint32_t>();
    ArithProgression<int32_t>(indexTmpLocal.template ReinterpretCast<int32_t>(), baseOffset, 1, actualElem);

    // Flip sign bit so that ascending bit-order matches ascending numeric order for signed floats
    // Note: Keep sign-flipped data for merge, flip back only in ExtractAndCopyChunk
    if constexpr (!IsDescend) {
        LocalTensor<int32_t> castTensor = xLocal.template ReinterpretCast<int32_t>();
        Adds(castTensor, castTensor, 0x80000000, alignSize);  // Flip sign bit for ascending float order
    }

    LocalTensor<ValueType> concatLocal;
    Concat(concatLocal, xLocal, concatTmpLocal, concatRepeatTimes);
    AscendC::Sort<ValueType, true>(sortedLocal, concatLocal, indexTmpLocal, sortTmpLocal, sortRepeatTimes);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::CopyOutToCache(
    LocalTensor<ValueType> sortedLocal, uint32_t blockId, uint32_t actualElem)
{
    // Cache is reused per batch (only 1 batch's cache allocated per core)
    int64_t cacheOffset = static_cast<int64_t>(blockId) * blockSortLen_;

    uint32_t sortLen = (actualElem == blockSortSize_) ? blockSortLen_ :
        AscendC::GetSortLen<ValueType>(actualElem);
    // {blockCount, blockLen, srcStride, dstStride, rsv}
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sortLen * sizeof(ValueType)), 0, 0, 0};

    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    DataCopyPad(cacheGm_[cacheOffset], sortedLocal, copyParams);

    sortedOutQueue_.FreeTensor(sortedLocal);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline uint32_t SortMergeBigBatch<ValueType, IndexType, IsDescend>::MergeSingleBatch()
{
    uint32_t numBlocks = blocksPerRow_;
    // At any merge level, blocks 0..numBlocks-2 all have fullBlockElemCount elements,
    // and the last block (numBlocks-1) has lastBlockElemCount elements.
    uint32_t fullBlockElemCount = blockSortSize_;
    uint32_t fullBlockSortLen = blockSortLen_;
    uint32_t lastBlockElemCount = lastBlockSize_;

    uint32_t pingPongFlag = 0;
    uint32_t mergeRounds = 0;

    while (numBlocks > 1) {
        uint32_t cumulativeOffset = 0;
        uint32_t newNumBlocks = 0;
        uint32_t newFullBlockElemCount = 0;
        uint32_t newLastBlockElemCount = 0;

        for (uint32_t i = 0; i < numBlocks; i += MERGE_LIST_MAX_NUM) {
            uint32_t groupBlockCount = (i + MERGE_LIST_MAX_NUM <= numBlocks) ?
                MERGE_LIST_MAX_NUM : (numBlocks - i);

            uint32_t mergedGroupElemCount = 0;
            MergeOneGroup(i, groupBlockCount, fullBlockElemCount, fullBlockSortLen,
                lastBlockElemCount, numBlocks, pingPongFlag,
                cumulativeOffset, mergedGroupElemCount);

            // First full group establishes the new full block size
            if (newNumBlocks == 0) {
                newFullBlockElemCount = mergedGroupElemCount;
            }
            // Always update last — the final group in the loop becomes the last block
            newLastBlockElemCount = mergedGroupElemCount;
            newNumBlocks++;
        }

        // Defensive: prevent infinite loop if newNumBlocks doesn't decrease
        if (newNumBlocks == 0 || newNumBlocks >= numBlocks) {
            break;
        }
        
        numBlocks = newNumBlocks;
        // If only one group was produced, full and last are the same
        if (numBlocks == 1) {
            fullBlockElemCount = newLastBlockElemCount;
        } else {
            fullBlockElemCount = newFullBlockElemCount;
        }
        fullBlockSortLen = AscendC::GetSortLen<ValueType>(fullBlockElemCount);
        lastBlockElemCount = newLastBlockElemCount;
        pingPongFlag = 1 - pingPongFlag;
        mergeRounds++;
    }

    return (mergeRounds == 0) ? 0 : pingPongFlag;
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::MergeOneGroup(
    uint32_t groupStart, uint32_t groupBlockCount,
    uint32_t fullBlockElemCount, uint32_t fullBlockSortLen, uint32_t lastBlockElemCount,
    uint32_t numBlocks, uint32_t pingPongFlag, uint32_t& cumulativeOffset,
    uint32_t& mergedGroupElemCount)
{
    MergeListContext ctx;
    ctx.listCount = groupBlockCount;
    
    int64_t srcRegionOffset = (pingPongFlag == 0) ? 0 : batchSortLen_;
    int64_t dstRegionOffset = (pingPongFlag == 0) ? batchSortLen_ : 0;

    for (uint32_t j = 0; j < groupBlockCount; j++) {
        uint32_t blockIdx = groupStart + j;
        ctx.srcOffsets[j] = srcRegionOffset + blockIdx * fullBlockSortLen;
        ctx.elemCounts[j] = (blockIdx < numBlocks - 1) ? fullBlockElemCount : lastBlockElemCount;
    }

    if (groupBlockCount > 1) {
        DoIncrementalMerge(dstRegionOffset + cumulativeOffset, ctx);

        uint32_t totalElem = 0;
        for (uint32_t j = 0; j < groupBlockCount; j++) {
            totalElem += ctx.elemCounts[j];
        }
        mergedGroupElemCount = totalElem;
        cumulativeOffset += AscendC::GetSortLen<ValueType>(totalElem);
    } else if (groupBlockCount == 1) {
        mergedGroupElemCount = ctx.elemCounts[0];
        if (ctx.elemCounts[0] > 0) {
            int64_t srcOffset = ctx.srcOffsets[0];
            uint32_t remainElems = ctx.elemCounts[0];
            uint32_t srcChunkOffset = 0;
            while (remainElems > 0) {
                uint32_t chunkSize = (remainElems > blockSortSize_) ? blockSortSize_ : remainElems;
                if (chunkSize == 0) break;
                CopyBlockChunk(srcOffset + AscendC::GetSortLen<ValueType>(srcChunkOffset),
                    dstRegionOffset + cumulativeOffset + AscendC::GetSortLen<ValueType>(srcChunkOffset), chunkSize);
                remainElems -= chunkSize;
                srcChunkOffset += chunkSize;
            }
            cumulativeOffset += AscendC::GetSortLen<ValueType>(ctx.elemCounts[0]);
        }
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::CopyBlockChunk(
    int64_t srcAddr, int64_t dstAddr, uint32_t elemCount)
{
    uint32_t sortLen = AscendC::GetSortLen<ValueType>(elemCount);
    LocalTensor<ValueType> srcLocal = mergeInQueue_.AllocTensor<ValueType>();
    DataCopyExtParams loadParams{1, static_cast<uint32_t>(sortLen * sizeof(ValueType)), 0, 0, 0};
    DataCopyPad(srcLocal, cacheGm_[srcAddr], loadParams, {false, 0, 0, 0});
    mergeInQueue_.EnQue(srcLocal);

    srcLocal = mergeInQueue_.DeQue<ValueType>();
    LocalTensor<ValueType> outLocal = mergeOutQueue_.AllocTensor<ValueType>();
    Copy(outLocal, srcLocal, sortLen);
    mergeOutQueue_.EnQue(outLocal);
    mergeInQueue_.FreeTensor(srcLocal);

    outLocal = mergeOutQueue_.DeQue<ValueType>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sortLen * sizeof(ValueType)), 0, 0, 0};
    DataCopyPad(cacheGm_[dstAddr], outLocal, copyParams);
    mergeOutQueue_.FreeTensor(outLocal);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::DoIncrementalMerge(
    int64_t dstOffsetBase, MergeListContext& ctx)
{
    if (ctx.listCount > MERGE_LIST_MAX_NUM) return;

    for (uint32_t i = 0; i < ctx.listCount; i++) {
        ctx.remains[i] = ctx.elemCounts[i];
    }
    
    uint32_t dstCumulativeOffset = 0, loopGuard = 0;
    while (true) {
        uint32_t activeLists = 0;
        for (uint32_t i = 0; i < ctx.listCount; i++) {
            if (ctx.remains[i] > 0) activeLists++;
        }
        if (activeLists <= 1) break;

        LocalTensor<ValueType> ubMainInput = mergeInQueue_.AllocTensor<ValueType>();
        uint16_t elementCountList[MERGE_LIST_MAX_NUM] = {0, 0, 0, 0};
        uint32_t remainListNum = LoadListsToUb(ubMainInput, elementCountList, ctx);
        mergeInQueue_.EnQue(ubMainInput);

        LocalTensor<ValueType> ubMainInputCalc = mergeInQueue_.DeQue<ValueType>();
        LocalTensor<ValueType> dstLocal = mergeOutQueue_.AllocTensor<ValueType>();

        uint32_t listSortedNums[MERGE_LIST_MAX_NUM] = {0, 0, 0, 0};
        uint32_t mergedCount = ExecuteMrgSort(dstLocal, ubMainInputCalc,
            elementCountList, listSortedNums, remainListNum);
        if (mergedCount == 0) {
            // FreeTensor can release AllocTensor'd buffers without EnQue/DeQue;
            // it only returns the buffer handle to the idle pool.
            mergeInQueue_.FreeTensor(ubMainInputCalc);
            mergeOutQueue_.FreeTensor(dstLocal);
            break;
        }

        mergeOutQueue_.EnQue(dstLocal);
        mergeInQueue_.FreeTensor(ubMainInputCalc);

        LocalTensor<ValueType> dstLocalOut = mergeOutQueue_.DeQue<ValueType>();
        DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(AscendC::GetSortLen<ValueType>(mergedCount) * sizeof(ValueType)), 0, 0, 0};
        DataCopyPad(cacheGm_[dstOffsetBase + dstCumulativeOffset], dstLocalOut, outCopyParams);
        mergeOutQueue_.FreeTensor(dstLocalOut);

        uint32_t j = 0;
        for (uint32_t i = 0; i < ctx.listCount; i++) {
            if (ctx.remains[i] > 0) {
                ctx.gmOffsets[i] += AscendC::GetSortLen<ValueType>(listSortedNums[j]);
                ctx.remains[i] -= listSortedNums[j];
                j++;
            }
        }
        dstCumulativeOffset += AscendC::GetSortLen<ValueType>(mergedCount);
        if (++loopGuard > maxMergeIterations_) break;
    }

    CopyRemainingList(ctx, dstOffsetBase, dstCumulativeOffset);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline uint32_t SortMergeBigBatch<ValueType, IndexType, IsDescend>::LoadListsToUb(
    LocalTensor<ValueType> ubMainInput,
    uint16_t elementCountList[MERGE_LIST_MAX_NUM], const MergeListContext& ctx)
{
    uint32_t remainListNum = 0;
    for (uint32_t i = 0; i < ctx.listCount; i++) {
        if (ctx.remains[i] > 0) {
            uint32_t loadCount = (ctx.remains[i] > blockSortSize_) ? blockSortSize_ : ctx.remains[i];
            elementCountList[remainListNum] = static_cast<uint16_t>(loadCount);
            
            int64_t srcAddr = ctx.srcOffsets[i] + ctx.gmOffsets[i];

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(AscendC::GetSortLen<ValueType>(loadCount) * sizeof(ValueType)), 0, 0, 0};
            DataCopyPadExtParams<ValueType> padParams{false, 0, 0, 0};
            DataCopyPad(ubMainInput[blockSortLen_ * remainListNum], cacheGm_[srcAddr],
                       copyParams, padParams);
            
            remainListNum++;
        }
    }
    return remainListNum;
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline uint32_t SortMergeBigBatch<ValueType, IndexType, IsDescend>::ExecuteMrgSort(
    LocalTensor<ValueType> dstLocal, LocalTensor<ValueType> ubMainInput,
    uint16_t elementCountList[MERGE_LIST_MAX_NUM],
    uint32_t listSortedNums[MERGE_LIST_MAX_NUM], uint32_t listCount)
{
    LocalTensor<ValueType> tmpUbInputs[MERGE_LIST_MAX_NUM];
    for (uint32_t j = 0; j < listCount; j++) {
        tmpUbInputs[j] = ubMainInput[blockSortLen_ * j];
    }
    // Fill unused slots with duplicates; MrgSortSrcList requires 4 args, validBitTail controls actual participation
    for (uint32_t j = listCount; j < MERGE_LIST_MAX_NUM; j++) {
        tmpUbInputs[j] = tmpUbInputs[0];
        elementCountList[j] = 0;
    }
    
    uint16_t validBitTail = (1 << listCount) - 1;

    MrgSortSrcList sortList(tmpUbInputs[0], tmpUbInputs[1], tmpUbInputs[2], tmpUbInputs[3]);
    MrgSort<ValueType, true>(dstLocal, sortList, elementCountList, listSortedNums, validBitTail, 1);

    uint32_t mergedCount = 0;
    for (uint32_t j = 0; j < listCount; j++) {
        mergedCount += listSortedNums[j];
    }
    return mergedCount;
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::CopyRemainingList(
    MergeListContext& ctx, int64_t dstOffsetBase, uint32_t& dstCumulativeOffset)
{
    for (uint32_t listIdx = 0; listIdx < ctx.listCount; listIdx++) {
        while (ctx.remains[listIdx] > 0) {
            uint32_t loadCount = (ctx.remains[listIdx] > blockSortSize_) ? blockSortSize_ : ctx.remains[listIdx];
            if (loadCount == 0) break;
            
            CopyBlockChunk(ctx.srcOffsets[listIdx] + ctx.gmOffsets[listIdx],
                dstOffsetBase + dstCumulativeOffset, loadCount);

            ctx.gmOffsets[listIdx] += AscendC::GetSortLen<ValueType>(loadCount);
            ctx.remains[listIdx] -= loadCount;
            dstCumulativeOffset += AscendC::GetSortLen<ValueType>(loadCount);
        }
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::ExtractAndCopyOut(int64_t batchIdx, uint32_t resultRegion)
{
    int64_t outputOffset = batchIdx * sortAxisNum_;

    // resultRegion: 0 = Ping (offset 0), 1 = Pong (offset batchSortLen_)
    int64_t cacheBatchOffset = (resultRegion == 1) ? batchSortLen_ : 0;

    uint32_t totalElem = sortAxisNum_;
    uint32_t elemProcessed = 0;
    uint32_t cacheOffset = 0;

    while (elemProcessed < totalElem) {
        uint32_t elemCount = (elemProcessed + extractChunkSize_ <= totalElem) ?
                             extractChunkSize_ : (totalElem - elemProcessed);
        if (elemCount == 0) break;

        ExtractAndCopyChunk(cacheBatchOffset, cacheOffset, outputOffset, elemProcessed, elemCount);

        elemProcessed += elemCount;
        cacheOffset += AscendC::GetSortLen<ValueType>(elemCount);
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeBigBatch<ValueType, IndexType, IsDescend>::ExtractAndCopyChunk(
    int64_t cacheBatchOffset, uint32_t cacheOffset,
    int64_t outputOffset, uint32_t elemProcessed, uint32_t elemCount)
{
    LocalTensor<ValueType> cacheLocal = extractInQueue_.AllocTensor<ValueType>();
    int64_t currentCacheOffset = cacheBatchOffset + cacheOffset;
    DataCopyExtParams loadParams{1, static_cast<uint32_t>(AscendC::GetSortLen<ValueType>(elemCount) * sizeof(ValueType)), 0, 0, 0};
    DataCopyPad(cacheLocal, cacheGm_[currentCacheOffset], loadParams, {false, 0, 0, 0});
    extractInQueue_.EnQue(cacheLocal);

    cacheLocal = extractInQueue_.DeQue<ValueType>();
    LocalTensor<ValueType> valueLocal = outValueQueue_.AllocTensor<ValueType>();
    LocalTensor<uint32_t> indexLocal = outIdxQueue_.AllocTensor<uint32_t>();
    Extract(valueLocal, indexLocal, cacheLocal, Ops::Base::CeilDiv(elemCount, DEALING_EXTRACT_NUM_ONCE));

    // Flip back sign bit for ascending order (was flipped in SortBlockToStruct)
    if constexpr (!IsDescend) {
        Adds(valueLocal.template ReinterpretCast<int32_t>(), valueLocal.template ReinterpretCast<int32_t>(), 0x80000000, elemCount);
    }

    outValueQueue_.EnQue(valueLocal);
    outIdxQueue_.EnQue(indexLocal);
    extractInQueue_.FreeTensor(cacheLocal);
    valueLocal = outValueQueue_.DeQue<ValueType>();
    indexLocal = outIdxQueue_.DeQue<uint32_t>();

    DataCopyExtParams outParams{1, static_cast<uint32_t>(elemCount * sizeof(ValueType)), 0, 0, 0};
    DataCopyPad(outValueGm_[outputOffset + elemProcessed], valueLocal, outParams);

    LocalTensor<int32_t> indexInt32 = indexLocal.template ReinterpretCast<int32_t>();
    if constexpr (IsSameType<int64_t, IndexType>::value) {
        LocalTensor<int64_t> indexInt64 = outIdxInt64Queue_.AllocTensor<int64_t>();
        Cast(indexInt64, indexInt32, RoundMode::CAST_NONE, Ops::Base::CeilAlign(elemCount, 4u));
        outIdxInt64Queue_.EnQue(indexInt64);
        indexInt64 = outIdxInt64Queue_.DeQue<int64_t>();
        outParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(int64_t));
        DataCopyPad(outIdxGm_[outputOffset + elemProcessed], indexInt64, outParams);
        outIdxInt64Queue_.FreeTensor(indexInt64);
    } else {
        outParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(int32_t));
        DataCopyPad(outIdxGm_[outputOffset + elemProcessed], indexInt32, outParams);
    }
    outIdxQueue_.FreeTensor(indexLocal);
    outValueQueue_.FreeTensor(valueLocal);
}

} // namespace Sort

#endif // SORT_MERGE_BIG_BATCH_H
