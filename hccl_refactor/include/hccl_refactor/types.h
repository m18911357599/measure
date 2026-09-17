#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hccl_refactor {

enum class Collective : uint8_t {
    AllGather = 0,
    ReduceScatter,
    AllReduce,
    Broadcast,
    Scatter,
    Reduce,
};

enum class TopologyKind : uint8_t {
    Mesh = 0,
    Clos,
    ClosMesh,
    Ring,
    DoubleRing,
    Nhr,
};

enum class Policy : uint8_t {
    Sequence = 0,
    Parallel,
};

enum class BufferKind : uint8_t {
    Input = 0,
    Ccl,
    Output,
};

struct Transfer {
    uint32_t step = 0;
    uint32_t channel = 0;
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t slotOwner = 0;
    uint64_t bytes = 0;
    uint32_t level = 0;
};

struct CommPlan {
    Collective collective = Collective::AllGather;
    TopologyKind topology = TopologyKind::Mesh;
    Policy policy = Policy::Sequence;
    uint32_t rankSize = 0;
    uint32_t myRank = 0;
    uint32_t channels = 1;
    uint32_t level = 0;
    uint32_t stepCount = 0;
    uint64_t sliceBytes = 0;
    std::vector<Transfer> transfers;
    std::string name;
};

struct RankState {
    uint32_t rank = 0;
    std::vector<uint8_t> slots;  // 1 if this rank currently holds owner i's reduced/gathered data
};

inline uint32_t CeilLog2(uint32_t n)
{
    uint32_t steps = 0;
    uint32_t v = 1;
    while (v < n) {
        v <<= 1;
        ++steps;
    }
    return steps;
}

inline const char* TopologyName(TopologyKind k)
{
    switch (k) {
        case TopologyKind::Mesh:
            return "mesh";
        case TopologyKind::Clos:
            return "clos";
        case TopologyKind::ClosMesh:
            return "clos+mesh";
        case TopologyKind::Ring:
            return "ring";
        case TopologyKind::DoubleRing:
            return "double-ring";
        case TopologyKind::Nhr:
            return "nhr";
    }
    return "unknown";
}

inline const char* CollectiveName(Collective c)
{
    switch (c) {
        case Collective::AllGather:
            return "AllGather";
        case Collective::ReduceScatter:
            return "ReduceScatter";
        case Collective::AllReduce:
            return "AllReduce";
        case Collective::Broadcast:
            return "Broadcast";
        case Collective::Scatter:
            return "Scatter";
        case Collective::Reduce:
            return "Reduce";
    }
    return "Unknown";
}

}  // namespace hccl_refactor
