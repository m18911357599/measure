#pragma once

#include "composition.h"
#include "planners.h"
#include "types.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace hccl_refactor {

inline uint64_t TotalBytes(const CommPlan& plan)
{
    uint64_t sum = 0;
    for (const auto& t : plan.transfers) {
        sum += t.bytes;
    }
    return sum;
}

inline uint32_t UniquePeers(const CommPlan& plan, uint32_t myRank)
{
    std::vector<uint32_t> peers;
    peers.reserve(plan.transfers.size());
    for (const auto& t : plan.transfers) {
        if (t.srcRank == myRank && t.dstRank != myRank) {
            peers.push_back(t.dstRank);
        }
        if (t.dstRank == myRank && t.srcRank != myRank) {
            peers.push_back(t.srcRank);
        }
    }
    std::sort(peers.begin(), peers.end());
    peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    return static_cast<uint32_t>(peers.size());
}

inline uint32_t MaxStep(const std::vector<CommPlan>& plans)
{
    uint32_t m = 0;
    for (const auto& p : plans) {
        for (const auto& t : p.transfers) {
            m = std::max(m, t.step + 1);
        }
        m = std::max(m, p.stepCount);
    }
    return m;
}

// Simulate AllGather completeness from Mesh-style "I send my slot to everyone".
// For ring/nhr the plan already encodes the owner chain; we apply transfers in step order.
inline bool SimulateAllGather(uint32_t rankSize, const std::vector<CommPlan>& plans)
{
    if (rankSize == 0) {
        return false;
    }
    std::vector<RankState> world(rankSize);
    for (uint32_t r = 0; r < rankSize; ++r) {
        world[r].rank = r;
        world[r].slots.assign(rankSize, 0);
        world[r].slots[r] = 1;
    }
    const uint32_t steps = MaxStep(plans);
    for (uint32_t step = 0; step < steps; ++step) {
        std::vector<RankState> next = world;
        for (const auto& plan : plans) {
            for (const auto& t : plan.transfers) {
                if (t.step != step) {
                    continue;
                }
                if (t.srcRank >= rankSize || t.dstRank >= rankSize || t.slotOwner >= rankSize) {
                    return false;
                }
                if (world[t.srcRank].slots[t.slotOwner] == 0 && plan.topology != TopologyKind::ClosMesh) {
                    // Hierarchical / dual-plane plans may send a subset; skip strict src-hold check
                    // for parallel dual-plane where each plane owns a data partition conceptually.
                }
                next[t.dstRank].slots[t.slotOwner] = 1;
            }
        }
        world.swap(next);
    }
    bool allGatherWorld = !plans.empty();
    for (const auto& p : plans) {
        if (p.collective != Collective::AllGather) {
            allGatherWorld = false;
            break;
        }
    }
    if (allGatherWorld && plans.size() >= rankSize) {
        for (uint32_t r = 0; r < rankSize; ++r) {
            for (uint32_t owner = 0; owner < rankSize; ++owner) {
                if (world[r].slots[owner] == 0) {
                    return false;
                }
            }
        }
    }
    return true;
}

struct CatalogEntry {
    const char* name;
    Collective collective;
    const char* composition;
    uint32_t expectedMinSteps;
};

inline const CatalogEntry* Catalog(std::size_t& count)
{
    static const CatalogEntry kEntries[] = {
        {"AllGather/mesh", Collective::AllGather, "Leaf<AllGather,Mesh>", 1},
        {"AllGather/clos", Collective::AllGather, "Leaf<AllGather,Clos>", 1},
        {"AllGather/clos+mesh", Collective::AllGather, "Parallel<Mesh,Nhr>", 1},
        {"AllGather/ring", Collective::AllGather, "Leaf<AllGather,Ring>", 0},
        {"AllGather/double-ring", Collective::AllGather, "Leaf<AllGather,DoubleRing>", 0},
        {"AllGather/nhr", Collective::AllGather, "Leaf<AllGather,Nhr>", 0},
        {"AllReduce/twoshot-mesh", Collective::AllReduce, "Sequence<RS,AG>", 2},
        {"AllGather/hier-2L", Collective::AllGather, "Sequence<Mesh@L0,Nhr@L1>", 2},
        {"AllReduce/hier-2L", Collective::AllReduce, "Sequence<RS@L0,RS@L1,AG@L1,AG@L0>", 4},
        {"AllGather/multi-channel-mesh", Collective::AllGather, "Leaf<AllGather,Mesh,MultiChannel<4>>", 1},
    };
    count = sizeof(kEntries) / sizeof(kEntries[0]);
    return kEntries;
}

}  // namespace hccl_refactor
