#pragma once

#include "channel.h"
#include "policies.h"
#include "types.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace hccl_refactor {

struct PlannerInput {
    uint32_t rankSize = 1;
    uint32_t myRank = 0;
    uint32_t channels = 1;
    uint32_t level = 0;
    uint32_t portBudget = 4;
    uint64_t sliceBytes = 1;
    uint32_t root = 0;
};

inline void AppendTransfer(CommPlan& plan, uint32_t step, uint32_t channel, uint32_t src, uint32_t dst,
                           uint32_t owner, uint64_t bytes)
{
    Transfer t;
    t.step = step;
    t.channel = channel;
    t.srcRank = src;
    t.dstRank = dst;
    t.slotOwner = owner;
    t.bytes = bytes;
    t.level = plan.level;
    plan.transfers.push_back(t);
    plan.stepCount = std::max(plan.stepCount, step + 1);
}

inline void StripeAndSend(CommPlan& plan, uint32_t step, uint32_t src, uint32_t dst, uint32_t owner, uint64_t bytes)
{
    for (uint32_t c = 0; c < plan.channels; ++c) {
        const uint64_t part = ChannelSliceBytes(bytes, plan.channels, c);
        if (part == 0) {
            continue;
        }
        AppendTransfer(plan, step, c, src, dst, owner, part);
    }
}

inline CommPlan MakePlan(Collective coll, TopologyKind topo, const PlannerInput& in, const char* suffix = "")
{
    CommPlan plan;
    plan.collective = coll;
    plan.topology = topo;
    plan.rankSize = in.rankSize;
    plan.myRank = in.myRank;
    plan.channels = FabricChannels(topo, in.channels, in.portBudget);
    plan.level = in.level;
    plan.sliceBytes = in.sliceBytes;
    plan.name = std::string(CollectiveName(coll)) + "/" + TopologyName(topo);
    if (suffix && suffix[0] != '\0') {
        plan.name += suffix;
    }
    return plan;
}

// ---- Mesh: all-to-all, one logical step, rankSize-1 concurrent peers ----
inline CommPlan PlanMeshAllGather(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::AllGather, TopologyKind::Mesh, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    for (uint32_t dst = 0; dst < in.rankSize; ++dst) {
        if (dst == in.myRank) {
            continue;
        }
        StripeAndSend(plan, 0, in.myRank, dst, in.myRank, in.sliceBytes);
    }
    plan.stepCount = 1;
    return plan;
}

inline CommPlan PlanMeshReduceScatter(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::ReduceScatter, TopologyKind::Mesh, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    for (uint32_t src = 0; src < in.rankSize; ++src) {
        if (src == in.myRank) {
            continue;
        }
        StripeAndSend(plan, 0, src, in.myRank, in.myRank, in.sliceBytes);
    }
    plan.stepCount = 1;
    return plan;
}

// ---- CLOS: same peer set as Mesh, but channel count is port-capped ----
inline CommPlan PlanClosAllGather(const PlannerInput& in)
{
    CommPlan plan = PlanMeshAllGather(in);
    plan.topology = TopologyKind::Clos;
    plan.channels = FabricChannels(TopologyKind::Clos, in.channels, in.portBudget);
    plan.name = std::string(CollectiveName(Collective::AllGather)) + "/clos";
    return plan;
}

inline CommPlan PlanClosReduceScatter(const PlannerInput& in)
{
    CommPlan plan = PlanMeshReduceScatter(in);
    plan.topology = TopologyKind::Clos;
    plan.channels = FabricChannels(TopologyKind::Clos, in.channels, in.portBudget);
    plan.name = std::string(CollectiveName(Collective::ReduceScatter)) + "/clos";
    return plan;
}

// ---- Ring: neighbor exchange, rankSize-1 steps ----
inline CommPlan PlanRingAllGather(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::AllGather, TopologyKind::Ring, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    const uint32_t next = (in.myRank + 1) % in.rankSize;
    const uint32_t prev = (in.myRank + in.rankSize - 1) % in.rankSize;
    for (uint32_t step = 0; step < in.rankSize - 1; ++step) {
        const uint32_t sendOwner = (in.myRank + in.rankSize - step) % in.rankSize;
        const uint32_t recvOwner = (prev + in.rankSize - step) % in.rankSize;
        StripeAndSend(plan, step, in.myRank, next, sendOwner, in.sliceBytes);
        StripeAndSend(plan, step, prev, in.myRank, recvOwner, in.sliceBytes);
    }
    return plan;
}

inline CommPlan PlanRingReduceScatter(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::ReduceScatter, TopologyKind::Ring, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    const uint32_t next = (in.myRank + 1) % in.rankSize;
    const uint32_t prev = (in.myRank + in.rankSize - 1) % in.rankSize;
    for (uint32_t step = 0; step < in.rankSize - 1; ++step) {
        const uint32_t sendOwner = (in.myRank + 1 + step) % in.rankSize;
        StripeAndSend(plan, step, in.myRank, next, sendOwner, in.sliceBytes);
        StripeAndSend(plan, step, prev, in.myRank, sendOwner, in.sliceBytes);
    }
    return plan;
}

// ---- Double-Ring: two opposite rings, each carries half the payload ----
inline CommPlan PlanDoubleRingAllGather(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::AllGather, TopologyKind::DoubleRing, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    const uint64_t half = (in.sliceBytes + 1) / 2;
    const uint64_t rest = in.sliceBytes - (in.sliceBytes / 2);
    const uint32_t fwd = (in.myRank + 1) % in.rankSize;
    const uint32_t bwd = (in.myRank + in.rankSize - 1) % in.rankSize;
    for (uint32_t step = 0; step < in.rankSize - 1; ++step) {
        const uint32_t fwdOwner = (in.myRank + in.rankSize - step) % in.rankSize;
        const uint32_t bwdOwner = (in.myRank + step) % in.rankSize;
        StripeAndSend(plan, step, in.myRank, fwd, fwdOwner, half);
        StripeAndSend(plan, step, in.myRank, bwd, bwdOwner, rest);
    }
    return plan;
}

inline CommPlan PlanDoubleRingReduceScatter(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::ReduceScatter, TopologyKind::DoubleRing, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    const uint64_t half = (in.sliceBytes + 1) / 2;
    const uint32_t fwd = (in.myRank + 1) % in.rankSize;
    const uint32_t bwd = (in.myRank + in.rankSize - 1) % in.rankSize;
    for (uint32_t step = 0; step < in.rankSize - 1; ++step) {
        const uint32_t fwdOwner = (in.myRank + 1 + step) % in.rankSize;
        const uint32_t bwdOwner = (in.myRank + in.rankSize - 1 - step) % in.rankSize;
        StripeAndSend(plan, step, in.myRank, fwd, fwdOwner, half);
        StripeAndSend(plan, step, in.myRank, bwd, bwdOwner, half);
    }
    return plan;
}

// ---- NHR: ceil(log2(N)) neighbor-distance steps (distance = 2^k) ----
inline CommPlan PlanNhrAllGather(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::AllGather, TopologyKind::Nhr, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    const uint32_t nSteps = CeilLog2(in.rankSize);
    uint32_t hold = 1;
    for (uint32_t step = 0; step < nSteps; ++step) {
        const uint32_t dist = 1u << step;
        const uint32_t dst = (in.myRank + dist) % in.rankSize;
        const uint32_t src = (in.myRank + in.rankSize - dist) % in.rankSize;
        const uint64_t bytes = in.sliceBytes * hold;
        for (uint32_t k = 0; k < hold; ++k) {
            const uint32_t sendOwner = (in.myRank + in.rankSize - k) % in.rankSize;
            const uint32_t recvOwner = (src + in.rankSize - k) % in.rankSize;
            StripeAndSend(plan, step, in.myRank, dst, sendOwner, in.sliceBytes);
            StripeAndSend(plan, step, src, in.myRank, recvOwner, in.sliceBytes);
            (void)bytes;
        }
        hold = std::min(hold * 2, in.rankSize);
    }
    return plan;
}

inline CommPlan PlanNhrReduceScatter(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::ReduceScatter, TopologyKind::Nhr, in);
    if (in.rankSize <= 1) {
        return plan;
    }
    const uint32_t nSteps = CeilLog2(in.rankSize);
    uint32_t sendCount = in.rankSize / 2;
    if (sendCount == 0) {
        sendCount = 1;
    }
    for (uint32_t step = 0; step < nSteps; ++step) {
        const uint32_t dist = 1u << (nSteps - 1 - step);
        const uint32_t dst = (in.myRank + dist) % in.rankSize;
        const uint32_t src = (in.myRank + in.rankSize - dist) % in.rankSize;
        const uint32_t nSend = std::max(1u, sendCount);
        for (uint32_t k = 0; k < nSend; ++k) {
            const uint32_t owner = (in.myRank + k) % in.rankSize;
            StripeAndSend(plan, step, in.myRank, dst, owner, in.sliceBytes);
            StripeAndSend(plan, step, src, in.myRank, owner, in.sliceBytes);
        }
        sendCount = std::max(1u, sendCount / 2);
    }
    return plan;
}

// ---- CLOS+Mesh: dual-plane PARALLEL, Mesh plane + CLOS/NHR plane ----
inline void MergePlans(CommPlan& dst, const CommPlan& src, uint32_t stepOffset)
{
    for (Transfer t : src.transfers) {
        t.step += stepOffset;
        t.level = dst.level;
        dst.transfers.push_back(t);
        dst.stepCount = std::max(dst.stepCount, t.step + 1);
    }
}

inline CommPlan PlanClosMeshAllGather(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::AllGather, TopologyKind::ClosMesh, in);
    plan.policy = Policy::Parallel;
    PlannerInput meshIn = in;
    PlannerInput closIn = in;
    const uint64_t meshBytes = in.sliceBytes / 2;
    const uint64_t closBytes = in.sliceBytes - meshBytes;
    meshIn.sliceBytes = meshBytes == 0 ? in.sliceBytes : meshBytes;
    closIn.sliceBytes = closBytes == 0 ? in.sliceBytes : closBytes;
    closIn.channels = FabricChannels(TopologyKind::Clos, in.channels, in.portBudget);
    MergePlans(plan, PlanMeshAllGather(meshIn), 0);
    MergePlans(plan, PlanNhrAllGather(closIn), 0);
    plan.stepCount = std::max(plan.stepCount, 1u);
    return plan;
}

inline CommPlan PlanClosMeshReduceScatter(const PlannerInput& in)
{
    CommPlan plan = MakePlan(Collective::ReduceScatter, TopologyKind::ClosMesh, in);
    plan.policy = Policy::Parallel;
    PlannerInput meshIn = in;
    PlannerInput closIn = in;
    meshIn.sliceBytes = in.sliceBytes / 2;
    closIn.sliceBytes = in.sliceBytes - meshIn.sliceBytes;
    if (meshIn.sliceBytes == 0) {
        meshIn.sliceBytes = in.sliceBytes;
    }
    if (closIn.sliceBytes == 0) {
        closIn.sliceBytes = in.sliceBytes;
    }
    MergePlans(plan, PlanMeshReduceScatter(meshIn), 0);
    MergePlans(plan, PlanNhrReduceScatter(closIn), 0);
    return plan;
}

template <typename Topo>
CommPlan PlanAllGather(const PlannerInput& in);

template <>
inline CommPlan PlanAllGather<Mesh>(const PlannerInput& in)
{
    return PlanMeshAllGather(in);
}
template <>
inline CommPlan PlanAllGather<Clos>(const PlannerInput& in)
{
    return PlanClosAllGather(in);
}
template <>
inline CommPlan PlanAllGather<ClosMesh>(const PlannerInput& in)
{
    return PlanClosMeshAllGather(in);
}
template <>
inline CommPlan PlanAllGather<Ring>(const PlannerInput& in)
{
    return PlanRingAllGather(in);
}
template <>
inline CommPlan PlanAllGather<DoubleRing>(const PlannerInput& in)
{
    return PlanDoubleRingAllGather(in);
}
template <>
inline CommPlan PlanAllGather<Nhr>(const PlannerInput& in)
{
    return PlanNhrAllGather(in);
}

template <typename Topo>
CommPlan PlanReduceScatter(const PlannerInput& in);

template <>
inline CommPlan PlanReduceScatter<Mesh>(const PlannerInput& in)
{
    return PlanMeshReduceScatter(in);
}
template <>
inline CommPlan PlanReduceScatter<Clos>(const PlannerInput& in)
{
    return PlanClosReduceScatter(in);
}
template <>
inline CommPlan PlanReduceScatter<ClosMesh>(const PlannerInput& in)
{
    return PlanClosMeshReduceScatter(in);
}
template <>
inline CommPlan PlanReduceScatter<Ring>(const PlannerInput& in)
{
    return PlanRingReduceScatter(in);
}
template <>
inline CommPlan PlanReduceScatter<DoubleRing>(const PlannerInput& in)
{
    return PlanDoubleRingReduceScatter(in);
}
template <>
inline CommPlan PlanReduceScatter<Nhr>(const PlannerInput& in)
{
    return PlanNhrReduceScatter(in);
}

}  // namespace hccl_refactor
