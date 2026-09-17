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
 * \file sort_tiling_data.h
 * \brief sort_tiling_data.h
 */

#ifndef _SORT_REGBASE_TILING_DATA_H_
#define _SORT_REGBASE_TILING_DATA_H_

struct SortRegBaseTilingData {
    uint32_t numTileDataSize; // h轴ub一次处理个数
    uint32_t unsortedDimParallel; // b轴使用的核数
    uint32_t lastDimTileNum; // h轴循环次数
    uint32_t sortLoopTimes; // b轴循环次数
    uint32_t lastDimNeedCore; // h轴需要的核数
    // keyParamsxxx 预留参数
    // keyParams0 radix分支表示globalHistGmWk_使用核数，radix_one_core代表inqueX的ub大小,merge sort 表示一次处理几个h
    uint32_t keyParams0;
    uint32_t keyParams1; // radix分支清零excusiveBinsGmWk_的核， radix_one_core y2OutQue需要的ub大小, merge xQue ub大小
    uint32_t keyParams2; // radix分支清零的一次ub数据量, radix_one_core 输出int64时，一半的ub偏移, merge y2OutQue的ub大小
    uint32_t keyParams3; // radix分支清零globalHistGmWk_ ub循环次数, merge表示32个数对齐的alginH
    uint32_t keyParams4; // radix：清零chunk大小；merge_sort(sch0)：队列buffer数；merge_big_batch(sch4)：extract chunk大小
    uint32_t keyParams5; // radix：清零chunk大小；merge_big_batch(sch4)：最大归并迭代次数
    uint32_t tmpUbSize; // sort高级api需要的临时ub大小
    int64_t lastAxisNum; // h轴大小
    int64_t unsortedDimNum; // b轴大小
};
#endif
