#pragma once

#include "compose.hpp"

namespace hccl_refactor {

// Compile-time registry: one type alias per (topo × op × composition) instead
// of one C++ class + REGISTER_EXEC + REGISTER_TEMPLATE per combination.
// Runtime selection still happens once per op (string / enum), then the
// chosen kernel is a fully specialized, non-virtual type.

struct AlgKey {
    TopoKind l0;
    TopoKind l1;
    Collective op;
    int channels;
};

template <TopoKind L0, Collective Op, int Channels = 1>
struct KernelPick;

template <>
struct KernelPick<TopoKind::kRing, Collective::kReduceScatter, 1> {
    using type = RingKernel<ReduceScatterOp>;
};
template <>
struct KernelPick<TopoKind::kRing, Collective::kAllGather, 1> {
    using type = RingKernel<AllGatherOp>;
};
template <>
struct KernelPick<TopoKind::kRing, Collective::kAllReduce, 1> {
    using type = RingAllReduce;
};
template <>
struct KernelPick<TopoKind::kMesh, Collective::kReduceScatter, 1> {
    using type = MeshKernel<ReduceScatterOp>;
};
template <>
struct KernelPick<TopoKind::kMesh, Collective::kAllGather, 1> {
    using type = MeshKernel<AllGatherOp>;
};
template <>
struct KernelPick<TopoKind::kMesh, Collective::kAllReduce, 1> {
    using type = MeshAllReduce;
};
template <>
struct KernelPick<TopoKind::kNhr, Collective::kReduceScatter, 1> {
    using type = NhrKernel<ReduceScatterOp>;
};
template <>
struct KernelPick<TopoKind::kNhr, Collective::kAllGather, 1> {
    using type = NhrKernel<AllGatherOp>;
};
template <>
struct KernelPick<TopoKind::kNhr, Collective::kAllReduce, 1> {
    using type = NhrAllReduce;
};
template <>
struct KernelPick<TopoKind::kDoubleRing, Collective::kAllReduce, 2> {
    using type = DoubleRingAllReduce;
};
template <>
struct KernelPick<TopoKind::kRing, Collective::kAllReduce, 2> {
    using type = DoubleRingAllReduce;
};

template <TopoKind L0, Collective Op, int Channels = 1>
using KernelT = typename KernelPick<L0, Op, Channels>::type;

inline const char* topo_name(TopoKind k)
{
    switch (k) {
        case TopoKind::kMesh: return "mesh";
        case TopoKind::kRing: return "ring";
        case TopoKind::kDoubleRing: return "double-ring";
        case TopoKind::kNhr: return "nhr";
        case TopoKind::kClos: return "clos";
        case TopoKind::kHierarchical: return "hierarchical";
    }
    return "?";
}

// Source-unit accounting used by the design doc / tests.
struct SourceBudget {
    static constexpr int kTopoEngines = 3;       // ring, mesh, nhr
    static constexpr int kOpPolicies = 5;        // RS, AG, Scatter, Bcast, Reduce
    static constexpr int kCompositors = 3;       // AllReduce, MultiChannel, Hierarchical
    static constexpr int kInterconnects = 2;     // mesh, clos
    static constexpr int kNewUnits = kTopoEngines + kOpPolicies + kCompositors + kInterconnects;

    // Cartesian product the current hcomm tree essentially enumerates by hand.
    static constexpr int kOldCollectives = 7;
    static constexpr int kOldTopos = 6;
    static constexpr int kOldVariants = 4;  // slim/direct/91093/pipeline-ish
    static constexpr int kOldUnits = kOldCollectives * kOldTopos * kOldVariants;
};

}  // namespace hccl_refactor
