#pragma once

#include "types.hpp"

namespace hccl_refactor {

// Collective operators are stateless policy classes.  They only describe
// *what* happens to a received chunk; they do not know the topology.
// Hot-path methods are static and intended to be fully inlined (CRTP/policy,
// no virtual dispatch).

struct ReduceScatterOp {
    static constexpr Collective kId = Collective::kReduceScatter;
    static constexpr bool kReduce = true;
    static constexpr const char* kName = "ReduceScatter";

    // HCCL ReduceScatterRing: first tx = rank-1, first rx = rank-2, then --
    static int tx_chunk(int rank, int n, int step)
    {
        return wrap(rank - 1 - step, n);
    }
    static int rx_chunk(int rank, int n, int step)
    {
        return wrap(rank - 2 - step, n);
    }
};

struct AllGatherOp {
    static constexpr Collective kId = Collective::kAllGather;
    static constexpr bool kReduce = false;
    static constexpr const char* kName = "AllGather";

    // HCCL AllGatherRing: first tx = rank, first rx = rank-1, then walk backward
    static int tx_chunk(int rank, int n, int step)
    {
        return wrap(rank - step, n);
    }
    static int rx_chunk(int rank, int n, int step)
    {
        return wrap(rank - 1 - step, n);
    }
};

struct ScatterOp {
    static constexpr Collective kId = Collective::kScatter;
    static constexpr bool kReduce = false;
    static constexpr const char* kName = "Scatter";

    static int tx_chunk(int rank, int n, int step)
    {
        return wrap(rank - 1 - step, n);
    }
    static int rx_chunk(int rank, int n, int step)
    {
        return wrap(rank - 2 - step, n);
    }
};

struct BroadcastOp {
    static constexpr Collective kId = Collective::kBroadcast;
    static constexpr bool kReduce = false;
    static constexpr const char* kName = "Broadcast";

    static int tx_chunk(int rank, int n, int step)
    {
        (void)rank;
        (void)n;
        (void)step;
        return 0;
    }
    static int rx_chunk(int rank, int n, int step)
    {
        (void)rank;
        (void)n;
        (void)step;
        return 0;
    }
};

struct ReduceOp {
    static constexpr Collective kId = Collective::kReduce;
    static constexpr bool kReduce = true;
    static constexpr const char* kName = "Reduce";

    static int tx_chunk(int rank, int n, int step)
    {
        return wrap(rank - 1 - step, n);
    }
    static int rx_chunk(int rank, int n, int step)
    {
        return wrap(rank - 2 - step, n);
    }
};

// Interconnect policies.  Clos is *not* a collective algorithm; it is the
// network type of COMM_LEVEL1/LEVEL2 on 910B/910_93 (see RankGraph).
struct OnChipMeshInterconnect {
    static constexpr InterconnectKind kKind = InterconnectKind::kOnChipMesh;
    static constexpr bool kFullyConnected = true;
    static constexpr bool kPreferSparseAlgo = false;  // mesh / fullmesh
    static constexpr const char* kName = "mesh";
};

struct ClosInterconnect {
    static constexpr InterconnectKind kKind = InterconnectKind::kClos;
    static constexpr bool kFullyConnected = true;     // full-bisection permutation
    static constexpr bool kPreferSparseAlgo = true;   // ring / nhr / hd / ahc
    static constexpr const char* kName = "clos";
};

struct RingInterconnect {
    static constexpr InterconnectKind kKind = InterconnectKind::kRingLinks;
    static constexpr bool kFullyConnected = false;
    static constexpr bool kPreferSparseAlgo = true;
    static constexpr const char* kName = "ring-links";
};

// Compile-time default algorithm pick: dense interconnect -> Mesh, sparse -> Ring.
template <typename Interconnect, typename DenseTopo, typename SparseTopo>
using DefaultTopoFor = std::conditional_t<Interconnect::kPreferSparseAlgo, SparseTopo, DenseTopo>;

}  // namespace hccl_refactor
