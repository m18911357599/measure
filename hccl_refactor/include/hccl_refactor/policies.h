#pragma once

#include "types.h"

namespace hccl_refactor {

// Compile-time topology tags. These are empty policies: zero data, zero vtable.
struct Mesh {
    static constexpr TopologyKind kind = TopologyKind::Mesh;
    static constexpr bool isSwitchFabric = false;
    static constexpr bool dualPlane = false;
};

struct Clos {
    static constexpr TopologyKind kind = TopologyKind::Clos;
    static constexpr bool isSwitchFabric = true;
    static constexpr bool dualPlane = false;
};

struct ClosMesh {
    static constexpr TopologyKind kind = TopologyKind::ClosMesh;
    static constexpr bool isSwitchFabric = true;
    static constexpr bool dualPlane = true;
};

struct Ring {
    static constexpr TopologyKind kind = TopologyKind::Ring;
    static constexpr bool isSwitchFabric = false;
    static constexpr bool dualPlane = false;
};

struct DoubleRing {
    static constexpr TopologyKind kind = TopologyKind::DoubleRing;
    static constexpr bool isSwitchFabric = false;
    static constexpr bool dualPlane = true;
};

struct Nhr {
    static constexpr TopologyKind kind = TopologyKind::Nhr;
    static constexpr bool isSwitchFabric = false;
    static constexpr bool dualPlane = false;
};

template <uint32_t N>
struct MultiChannel {
    static constexpr uint32_t kChannels = N < 1 ? 1 : N;
};

using SingleChannel = MultiChannel<1>;

template <uint32_t L>
struct Level {
    static constexpr uint32_t kLevel = L;
};

using L0 = Level<0>;
using L1 = Level<1>;
using L2 = Level<2>;

}  // namespace hccl_refactor
