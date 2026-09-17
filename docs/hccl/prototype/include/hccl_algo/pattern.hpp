#pragma once

#include "world.hpp"

namespace hccl_algo {

struct AllGatherPat {
    static constexpr bool kReduceOnRecv = false;
    static int ringSendIdx(int rank, int p, int step) { return (rank - step + p) % p; }
    static int ringRecvIdx(int rank, int p, int step) { return (rank - step - 1 + p) % p; }
};

struct ReduceScatterPat {
    static constexpr bool kReduceOnRecv = true;
    // Dual of AllGather: after p-1 steps, fully reduced block (rank+1)%p sits on this rank.
    static int ringSendIdx(int rank, int p, int step) { return (rank - step + p) % p; }
    static int ringRecvIdx(int rank, int p, int step) { return (rank - step - 1 + p) % p; }
};

}  // namespace hccl_algo
