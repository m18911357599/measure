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
 * \file sort_apt.cpp
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/sort_tiling_key.h"
#include "arch35/sort_tiling_data.h"
#include "arch35/sort_radix_sort_more_core.h"
#include "arch35/sort_radix_sort_one_core.h"
#include "arch35/sort_merge_sort.h"
#include "arch35/merge_sort_big_size.h"
#include "arch35/sort_merge_big_batch.h"

using namespace AscendC;
using namespace Sort;

template <uint64_t schId, uint64_t isInt32, uint64_t isDescend>
__global__ __aicore__ void sort(
    GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SortRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(SortRegBaseTilingData, tilingData, tiling);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipe;
    if constexpr (schId == 2) {
        if constexpr(sizeof(DTYPE_X) == 1) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint8_t, uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            } else {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint8_t, int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
        if constexpr(sizeof(DTYPE_X) == 2) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint16_t,uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            } else {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint16_t,int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
        if constexpr(sizeof(DTYPE_X) == 4) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint32_t,uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            } else {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint32_t,int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
        if constexpr(sizeof(DTYPE_X) == 8) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint64_t, uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            } else {
                SortRadixMoreCore<DTYPE_X, DTYPE_Y2, uint64_t, int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
        return;
    }
    if constexpr (schId == 1) {
        if constexpr (isDescend == 1) {
            SortRadixOneCore<DTYPE_X, DTYPE_Y2, true> op;
            op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
            op.Process();
        } else {
            SortRadixOneCore<DTYPE_X, DTYPE_Y2, false> op;
            op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
            op.Process();
        }
        return;
    }
    if constexpr (schId == 0) {
        if constexpr (IsSameType<bfloat16_t, DTYPE_X>::value) {
            MergeSort<DTYPE_X, DTYPE_Y2, float, isDescend> op;
            op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
            op.Process();
        }
        if constexpr (IsSameType<float, DTYPE_X>::value) {
            MergeSort<DTYPE_X, DTYPE_Y2, DTYPE_X, isDescend> op;
            op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
            op.Process();
        }
        if constexpr (IsSameType<half, DTYPE_X>::value) {
            MergeSort<DTYPE_X, DTYPE_Y2, DTYPE_X, isDescend> op;
            op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
            op.Process();
        }
        return;
    }
    if constexpr (schId == 3) {
        if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value) {
            if constexpr (isDescend == 1) {
                MergeSortBigSize<DTYPE_X, float, true, DTYPE_Y2> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            } else {
                MergeSortBigSize<DTYPE_X, float, false, DTYPE_Y2> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
		if constexpr (IsSameType<float, DTYPE_X>::value || IsSameType<half, DTYPE_X>::value) {
            if constexpr (isDescend == 1) {
                MergeSortBigSize<DTYPE_X, DTYPE_X, true, DTYPE_Y2> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            } else {
                MergeSortBigSize<DTYPE_X, DTYPE_X, false, DTYPE_Y2> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
        return;
    }
    if constexpr (schId == 4) {
        if constexpr (IsSameType<float, DTYPE_X>::value) {
            if constexpr (isDescend == 1) {
                Sort::SortMergeBigBatch<DTYPE_X, DTYPE_Y2, true> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            } else {
                Sort::SortMergeBigBatch<DTYPE_X, DTYPE_Y2, false> op;
                op.Init(x, y1, y2, usrWorkspace, &tilingData, &pipe);
                op.Process();
            }
        }
        return;
    }
}
