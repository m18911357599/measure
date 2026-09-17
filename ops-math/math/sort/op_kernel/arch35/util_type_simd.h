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
 * \file util_type_simd.h
 * \brief util simd impl
 */
#ifndef UTIL_TYPE_SIMD_H
#define UTIL_TYPE_SIMD_H
#include "kernel_operator.h"
#include "op_kernel/platform_util.h" // 不引入导致 GetUbBlockSize 方法无法使用

__aicore__ inline uint32_t ROUND_UP_AGLIN(uint32_t x)
{
    const int32_t oneBlock = Ops::Base::GetUbBlockSize();
    if (oneBlock == 0) {
        return x;
    }
    return (x + oneBlock - 1) / oneBlock * oneBlock;
}

template <typename T> struct DoubleBufferSimd {
    AscendC::GlobalTensor<T> doubleBuffer_[2];
    int selector_ = 0;
    __aicore__ inline DoubleBufferSimd() {}
    __aicore__ inline void SetDoubleBuffer(AscendC::GlobalTensor<T> currentBuffer,
        AscendC::GlobalTensor<T> alternateBuffer)
    {
        selector_ = 0;
        doubleBuffer_[0] = currentBuffer;
        doubleBuffer_[1] = alternateBuffer;
    }
    __aicore__ inline AscendC::GlobalTensor<T> Current() const
    {
        return doubleBuffer_[selector_];
    }
    __aicore__ inline AscendC::GlobalTensor<T> Alternate() const
    {
        return doubleBuffer_[selector_ ^ 1];
    }
    __aicore__ inline void UpdateSelect()
    {
        selector_ = selector_ ^ 1;
    }
};
#endif