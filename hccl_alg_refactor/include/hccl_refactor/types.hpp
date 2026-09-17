#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <type_traits>

namespace hccl_refactor {

using Value = std::int64_t;

enum class Collective {
    kReduceScatter,
    kAllGather,
    kAllReduce,
    kBroadcast,
    kReduce,
    kScatter
};

enum class TopoKind {
    kMesh,
    kRing,
    kDoubleRing,
    kNhr,
    kClos,       // interconnect, not a collective kernel
    kHierarchical
};

enum class InterconnectKind {
    kOnChipMesh,  // COMM_TOPO_1DMESH / 910_93 intra-server
    kClos,        // COMM_TOPO_CLOS   / inter-server, inter-superpod
    kRingLinks    // logical ring neighbors only
};

struct StepPlan {
    int send_to = -1;
    int recv_from = -1;
    std::vector<int> tx_chunks;
    std::vector<int> rx_chunks;
};

struct Slice {
    int offset = 0;
    int size = 0;
};

inline int wrap(int v, int n)
{
    int r = v % n;
    return r < 0 ? r + n : r;
}

inline int ceil_log2(int n)
{
    int s = 0;
    int t = n - 1;
    while (t > 0) {
        t >>= 1;
        ++s;
    }
    return s;
}

inline const char* collective_name(Collective c)
{
    switch (c) {
        case Collective::kReduceScatter: return "ReduceScatter";
        case Collective::kAllGather: return "AllGather";
        case Collective::kAllReduce: return "AllReduce";
        case Collective::kBroadcast: return "Broadcast";
        case Collective::kReduce: return "Reduce";
        case Collective::kScatter: return "Scatter";
    }
    return "Unknown";
}

}  // namespace hccl_refactor
