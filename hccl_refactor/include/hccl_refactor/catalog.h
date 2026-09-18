#pragma once

#include "composition.h"

namespace hccl_refactor {

template <typename Topo, typename Ch = SingleChannel>
using AlgAllGather = Leaf<AllGatherOp, Topo, Ch, L0>;

template <typename Topo, typename Ch = SingleChannel>
using AlgReduceScatter = Leaf<ReduceScatterOp, Topo, Ch, L0>;

using MeshAllGather = AlgAllGather<Mesh>;
using ClosAllGather = AlgAllGather<Clos>;
using ClosMeshAllGather = AlgAllGather<ClosMesh>;
using RingAllGather = AlgAllGather<Ring>;
using DoubleRingAllGather = AlgAllGather<DoubleRing>;
using NhrAllGather = AlgAllGather<Nhr>;

using MeshAllGather4Ch = AlgAllGather<Mesh, MultiChannel<4>>;
using NhrAllGather2Ch = AlgAllGather<Nhr, MultiChannel<2>>;

using MeshTwoShotAllReduce = TwoShotAllReduce<Mesh>;
using RingTwoShotAllReduce = TwoShotAllReduce<Ring>;
using NhrTwoShotAllReduce = TwoShotAllReduce<Nhr>;
using DoubleRingTwoShotAllReduce = TwoShotAllReduce<DoubleRing>;

using HierAllGather2L = HierarchicalAllGather2L<>;
using HierAllGather3L = HierarchicalAllGather3L<>;
using HierAllReduce2L = HierarchicalAllReduce2L<Mesh, Nhr>;
using ClosMeshConcurAG = ClosMeshConcurrentAllGather<>;

}  // namespace hccl_refactor
