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
 * \file sort_tiling_key.h
 * \brief sort tiling key declare
 */

#ifndef _SORT_TILING_KEY_DECL_H_
#define _SORT_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define SORT_SCHID_2 2
#define SORT_SCHID_3 3
#define SORT_SCHID_4 4

#define SORT_TPL_KEY_DECL()                                                                                 \
    ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, SORT_SCHID_2, SORT_SCHID_3,   \
                          SORT_SCHID_4),                                                                    \
    ASCENDC_TPL_UINT_DECL(isInt32, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1),                            \
    ASCENDC_TPL_UINT_DECL(isDescend, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1)                        \

#define SORT_TPL_RADIX_MORE_CORE_KEY_SEL()                                                                  \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SORT_SCHID_2),                                         \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 0, 1),                                           \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)                                          \

#define SORT_TPL_MERGE_BIG_BATCH_KEY_SEL()                                                                  \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, SORT_SCHID_4),                                         \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 0, 1),                                           \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)                                          \

#define SORT_TPL_ELSE_KEY_SEL()                                                                             \
    ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_LIST, 0, 1, SORT_SCHID_3),                                   \
        ASCENDC_TPL_UINT_SEL(isInt32, ASCENDC_TPL_UI_LIST, 1),                                              \
        ASCENDC_TPL_UINT_SEL(isDescend, ASCENDC_TPL_UI_LIST, 0, 1)                                          \

ASCENDC_TPL_ARGS_DECL(Sort, SORT_TPL_KEY_DECL());

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(SORT_TPL_RADIX_MORE_CORE_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(SORT_TPL_MERGE_BIG_BATCH_KEY_SEL()),
    ASCENDC_TPL_ARGS_SEL(SORT_TPL_ELSE_KEY_SEL()));
#endif
