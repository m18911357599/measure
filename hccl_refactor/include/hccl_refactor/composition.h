#pragma once

#include "planners.h"
#include "policies.h"
#include "types.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace hccl_refactor {

struct LeafDesc {
    Collective collective;
    TopologyKind topology;
    uint32_t channels;
    uint32_t level;
};

template <typename CollTag, typename Topo, typename Ch = SingleChannel, typename Lv = L0>
struct Leaf {
    using CollectiveTag = CollTag;
    using Topology = Topo;
    using Channel = Ch;
    using LevelTag = Lv;
    static constexpr Collective kCollective = CollTag::kCollective;
    static constexpr TopologyKind kTopology = Topo::kind;
    static constexpr uint32_t kChannels = Ch::kChannels;
    static constexpr uint32_t kLevel = Lv::kLevel;
};

struct AllGatherOp {
    static constexpr Collective kCollective = Collective::AllGather;
};
struct ReduceScatterOp {
    static constexpr Collective kCollective = Collective::ReduceScatter;
};
struct AllReduceOp {
    static constexpr Collective kCollective = Collective::AllReduce;
};

template <typename... Stages>
struct Sequence {
    static constexpr Policy kPolicy = Policy::Sequence;
    static constexpr std::size_t kArity = sizeof...(Stages);
};

template <typename... Branches>
struct Parallel {
    static constexpr Policy kPolicy = Policy::Parallel;
    static constexpr std::size_t kArity = sizeof...(Branches);
};

inline void AppendPlan(std::vector<CommPlan>& out, CommPlan plan)
{
    out.push_back(std::move(plan));
}

template <typename Topo, typename Ch, typename Lv>
CommPlan PlanLeaf(Collective coll, const PlannerInput& base)
{
    PlannerInput in = base;
    in.channels = Ch::kChannels;
    in.level = Lv::kLevel;
    switch (coll) {
        case Collective::ReduceScatter:
            return PlanReduceScatter<Topo>(in);
        case Collective::AllGather:
        default:
            return PlanAllGather<Topo>(in);
    }
}

template <typename Node>
struct Compose;

template <typename CollTag, typename Topo, typename Ch, typename Lv>
struct Compose<Leaf<CollTag, Topo, Ch, Lv>> {
    static std::vector<CommPlan> Build(const PlannerInput& in)
    {
        std::vector<CommPlan> out;
        AppendPlan(out, PlanLeaf<Topo, Ch, Lv>(CollTag::kCollective, in));
        return out;
    }
};

template <typename... Stages>
struct Compose<Sequence<Stages...>> {
    static std::vector<CommPlan> Build(const PlannerInput& in)
    {
        std::vector<CommPlan> out;
        (void)std::initializer_list<int>{(AppendStage<Stages>(in, out), 0)...};
        uint32_t stepBase = 0;
        for (auto& p : out) {
            p.policy = Policy::Sequence;
            for (auto& t : p.transfers) {
                t.step += stepBase;
            }
            if (p.stepCount > 0) {
                stepBase += p.stepCount;
            }
        }
        return out;
    }

private:
    template <typename Stage>
    static void AppendStage(const PlannerInput& in, std::vector<CommPlan>& out)
    {
        auto part = Compose<Stage>::Build(in);
        out.insert(out.end(), part.begin(), part.end());
    }
};

template <typename... Branches>
struct Compose<Parallel<Branches...>> {
    static std::vector<CommPlan> Build(const PlannerInput& in)
    {
        std::vector<CommPlan> out;
        const std::size_t n = sizeof...(Branches);
        uint64_t remaining = in.sliceBytes;
        std::size_t idx = 0;
        auto take = [&](PlannerInput& child) {
            if (idx + 1 == n) {
                child.sliceBytes = remaining;
            } else {
                child.sliceBytes = in.sliceBytes / static_cast<uint64_t>(n);
                if (child.sliceBytes > remaining) {
                    child.sliceBytes = remaining;
                }
                remaining -= child.sliceBytes;
            }
            ++idx;
        };
        (void)std::initializer_list<int>{(AppendBranch<Branches>(in, take, out), 0)...};
        for (auto& p : out) {
            p.policy = Policy::Parallel;
        }
        return out;
    }

private:
    template <typename Branch, typename Take>
    static void AppendBranch(const PlannerInput& in, Take& take, std::vector<CommPlan>& out)
    {
        PlannerInput child = in;
        take(child);
        auto part = Compose<Branch>::Build(child);
        out.insert(out.end(), part.begin(), part.end());
    }
};

// TwoShot AllReduce = ReduceScatter then AllGather on the same topology.
template <typename Topo, typename Ch = SingleChannel, typename Lv = L0>
using TwoShotAllReduce = Sequence<Leaf<ReduceScatterOp, Topo, Ch, Lv>, Leaf<AllGatherOp, Topo, Ch, Lv>>;

// 2-level hierarchical AllGather: intra Mesh then inter NHR.
template <typename Ch0 = SingleChannel, typename Ch1 = SingleChannel>
using HierarchicalAllGather2L =
    Sequence<Leaf<AllGatherOp, Mesh, Ch0, L0>, Leaf<AllGatherOp, Nhr, Ch1, L1>>;

// 3-level hierarchical AllGather: Mesh -> NHR -> NHR.
template <typename Ch0 = SingleChannel, typename Ch1 = SingleChannel, typename Ch2 = SingleChannel>
using HierarchicalAllGather3L = Sequence<Leaf<AllGatherOp, Mesh, Ch0, L0>, Leaf<AllGatherOp, Nhr, Ch1, L1>,
                                         Leaf<AllGatherOp, Nhr, Ch2, L2>>;

// Hierarchical AllReduce: RS L0 -> RS L1 -> AG L1 -> AG L0.
template <typename Intra = Mesh, typename Inter = Nhr>
using HierarchicalAllReduce2L =
    Sequence<Leaf<ReduceScatterOp, Intra, SingleChannel, L0>, Leaf<ReduceScatterOp, Inter, SingleChannel, L1>,
             Leaf<AllGatherOp, Inter, SingleChannel, L1>, Leaf<AllGatherOp, Intra, SingleChannel, L0>>;

// CLOS+Mesh concurrent AllGather.
template <typename Ch = MultiChannel<2>>
using ClosMeshConcurrentAllGather =
    Parallel<Leaf<AllGatherOp, Mesh, Ch, L0>, Leaf<AllGatherOp, Nhr, Ch, L0>>;

}  // namespace hccl_refactor
