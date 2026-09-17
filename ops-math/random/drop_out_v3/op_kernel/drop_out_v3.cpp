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
 * \file drop_out_v3.cpp
 * \brief
 */
#define FP32_TILING_KEY 1001
#define FP16_TILING_KEY 1002
#define BF16_TILING_KEY 1003

#include "arch35/drop_out_v3_impl.h"

using namespace DropOutV3;

extern "C" __global__ __aicore__  void drop_out_v3(
    GM_ADDR x, GM_ADDR noiseShape, GM_ADDR p, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(FP32_TILING_KEY)) {
        DropOutV3::DropOutV3Impl<float, DTYPE_P> op; 
        op.Init(p, mask, workspace, &tilingData, &pipe);
        op.Process(x, y, mask, &tilingData);
    } else if (TILING_KEY_IS(FP16_TILING_KEY)) {
        DropOutV3::DropOutV3Impl<half, DTYPE_P> op;
        op.Init(p, mask, workspace, &tilingData, &pipe);
        op.Process(x, y, mask, &tilingData);
    } else if (TILING_KEY_IS(BF16_TILING_KEY)) {
        DropOutV3::DropOutV3Impl<bfloat16_t, DTYPE_P> op;
        op.Init(p, mask, workspace, &tilingData, &pipe);
        op.Process(x, y, mask, &tilingData);
    }
}