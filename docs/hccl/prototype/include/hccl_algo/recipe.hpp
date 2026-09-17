#pragma once

#include "hierarchical.hpp"
#include "registry.hpp"

namespace hccl_algo {

using AllGatherMesh = Mesh;
using AllGatherRing = Ring;
using AllGatherNhr = Nhr;
using AllGatherClos = ClosNhr;
using AllGatherDoubleRing = DoubleRing;
using AllGatherClosMesh = Hierarchical<Mesh, ClosRing>;
using AllGatherHier3 = Hierarchical<Mesh, ClosNhr>;  // L0 mesh + L1 nhr (L2 via second group)

using ReduceScatterMesh = Mesh;
using ReduceScatterRing = Ring;

template <class RS, class AG>
struct AllReduce {
    static constexpr const char* kName = "allreduce";
};

}  // namespace hccl_algo
