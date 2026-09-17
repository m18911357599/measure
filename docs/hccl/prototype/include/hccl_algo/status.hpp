#pragma once

#include <cstdint>
#include <string>

namespace hccl_algo {

enum class Status : int { kOk = 0, kError = 1 };

inline bool ok(Status s) { return s == Status::kOk; }

struct Slice {
    int offset = 0;  // in blocks
    int count = 1;   // in blocks
};

inline int ceil_log2(int n)
{
    int s = 0;
    int v = 1;
    while (v < n) {
        v <<= 1;
        ++s;
    }
    return s;
}

}  // namespace hccl_algo
