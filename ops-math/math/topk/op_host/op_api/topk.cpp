/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topk.h"
#include "math/sort_with_index/op_api/sort_with_index.h"
#include "aclnn_kernels/cast.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TopKV2);
OP_TYPE_REGISTER(TopKV3);
OP_TYPE_REGISTER(TopK);

const int64_t MAX_AICORE_CALC_INPUTSIZE = 32768;
const int64_t MAX_AICORE_CALC_DIM = 8;
const int64_t MAX_K = 16;

constexpr int64_t TWO_THOUSAND = 2000;

static const std::initializer_list<op::DataType> ANCIENT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> FUTURE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64,  op::DataType::DT_INT32,   op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT64, op::DataType::DT_UINT32,  op::DataType::DT_UINT16, op::DataType::DT_UINT8,
    op::DataType::DT_BF16,   op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor* self, int64_t k)
{
    SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
    // 在1980上，需要判断inputsize和k的大小，因为在k较小时，aicpu性能比aicore更好；但1971无此场景；
    if (version == SocVersion::ASCEND910) {
        auto inputShape = self->GetViewShape();
        int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
        int64_t inputSize = 1;
        for (int64_t i = 0; i < tmpDim; i++) {
            inputSize *= inputShape.GetDim(i);
        }
        if (inputSize > MAX_AICORE_CALC_INPUTSIZE && k < MAX_AICORE_CALC_DIM) {
            return false;
        }
    }
    if (version == SocVersion::ASCEND310P) {
        OP_LOGW("l0op::TopK use ANCIENT_DTYPE_SUPPORT_LIST for socVerison[%d]", static_cast<int32_t>(version));
        return CheckType(self->GetDataType(), ANCIENT_DTYPE_SUPPORT_LIST);
    } else if (version == SocVersion::ASCEND910B || version == SocVersion::ASCEND910_93) {
        OP_LOGW("l0op::TopK use CURRENT_DTYPE_SUPPORT_LIST for socVerison[%d]", static_cast<int32_t>(version));
        return CheckType(self->GetDataType(), CURRENT_DTYPE_SUPPORT_LIST);
    } else if (IsRegBase()) {
        OP_LOGW("l0op::TopK use FUTURE_DTYPE_SUPPORT_LIST for socVerison[%d]", static_cast<int32_t>(version));
        return CheckType(self->GetDataType(), FUTURE_DTYPE_SUPPORT_LIST);
    } else {
        // 非以上平台，使用旧有逻辑处理。
    }

    if ((version >= SocVersion::ASCEND910B && version <= SocVersion::ASCEND910E) ||
        (version >= SocVersion::ASCEND310P && version <= SocVersion::ASCEND310C) ||
        (version == SocVersion::ASCEND610LITE)) {
        return CheckType(
            self->GetDataType(), {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16});
    }
    // 910、310芯片
    return self->GetDataType() == op::DataType::DT_FLOAT16;
}

// 根据芯片类型、dtype判断算子是否支持走TopKV3
static bool IsAscendCSupport(const aclTensor* self, int64_t k)
{
    SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
    if (version == SocVersion::ASCEND310P && CheckType(self->GetDataType(), {op::DataType::DT_FLOAT16}) && k < MAX_K) {
        return true;
    }
    return false;
}

// 根据芯片类型、k 和 sotrd 判断是否需要额外增加 SortWithIndex
static bool IsSortWithIndex(int64_t k, bool sorted)
{
    return (IsRegBase()) && (k > TWO_THOUSAND) && (sorted == true);
}

// AICORE算子kernel
std::tuple<aclTensor*, aclTensor*> TopkV2AiCore(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopkV2AiCore, self, k, dim, largest, sorted, values, indices);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore TopKV2算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(TopKV2, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest));
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

std::tuple<aclTensor*, aclTensor*> TopkV2AiCoreForDavid(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, op::DataType indicesDType, aclOpExecutor* executor)
{
    L0_DFX(TopkV2AiCoreForDavid, self, k, dim, largest, sorted, values, indices, indicesDType);
    ADD_TO_LAUNCHER_LIST_AICORE(
        TopKV2, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest, indicesDType));
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

// 950 TopK + SortWithIndex
std::tuple<aclTensor*, aclTensor*> TopKAndSort(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopKAndSort, self, k, dim, largest, sorted, values, indices);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore TopKV2算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(
        TopKV2, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest, op::DataType::DT_INT32));
    return SortWithIndex(values, indices, dim, largest, true, executor);
}

std::tuple<aclTensor*, aclTensor*> TopkV3(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopkV3, self, k, dim, largest, sorted, values, indices);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore TopKV3算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(TopKV3, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest));
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

// AICPU算子kernel
std::tuple<aclTensor*, aclTensor*> TopkAiCpu(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopkAiCpu, self, k, dim, largest, sorted, values, indices);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu TopK算子加入任务队列
    static internal::AicpuTaskSpace space("TopK");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        TopK, OP_ATTR_NAMES({"sorted", "largest", "dim"}), OP_INPUT(self, k), OP_OUTPUT(values, indices),
        OP_ATTR(sorted, largest, dim));
    if (ret != ACLNN_SUCCESS) {
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}
std::tuple<aclTensor*, aclTensor*> Topk(
    const aclTensor* self, int64_t k, int64_t dim, bool largest, bool sorted, op::DataType indicesDType,
    aclOpExecutor* executor)
{
    op::Shape outShape = self->GetStorageShape();
    outShape.SetDim(dim, k);

    const aclScalar* kScalar = executor->AllocScalar(k);
    if (kScalar == nullptr) {
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    const aclTensor* kTensor = executor->ConvertToTensor(kScalar, op::ToOpDataType(ACL_INT32));
    auto valuesOut = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    aclTensor* indicesOut = nullptr;
    if ((IsRegBase()) && !IsSortWithIndex(k, sorted)) {
        indicesOut = executor->AllocTensor(outShape, indicesDType, self->GetStorageFormat());
    } else {
        indicesOut = executor->AllocTensor(outShape, op::DataType::DT_INT32, self->GetStorageFormat());
    }

    if (IsAiCoreSupport(self, k)) {
        if (IsAscendCSupport(self, k)) {
            return TopkV3(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
        } else if (IsSortWithIndex(k, sorted)) {
            return TopKAndSort(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
        } else {
            if (IsRegBase()) {
                return TopkV2AiCoreForDavid(
                    self, kTensor, dim, largest, sorted, valuesOut, indicesOut, indicesDType, executor);
            } else {
                return TopkV2AiCore(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
            }
        }
    } else {
        return TopkAiCpu(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
    }
}
} // namespace l0op
