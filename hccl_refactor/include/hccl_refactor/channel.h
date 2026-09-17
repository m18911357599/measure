#pragma once

#include "types.h"

#include <algorithm>
#include <cstdint>

namespace hccl_refactor {

// Stripe a logical slice across C channels. Channel i takes
// [i * chunk, (i+1)*chunk) with the last channel absorbing the remainder.
inline uint64_t ChannelSliceBytes(uint64_t totalBytes, uint32_t channels, uint32_t channel)
{
    if (channels <= 1) {
        return totalBytes;
    }
    const uint64_t base = totalBytes / channels;
    if (channel + 1 == channels) {
        return totalBytes - base * (channels - 1);
    }
    return base;
}

inline uint32_t EffectiveChannels(uint32_t requested, uint32_t rankSize)
{
    (void)rankSize;
    return requested < 1 ? 1 : requested;
}

// CLOS switch fabrics share uplink ports. Model the usable concurrent
// channels as min(requested, portBudget). Mesh direct links keep requested.
inline uint32_t FabricChannels(TopologyKind topo, uint32_t requested, uint32_t portBudget)
{
    if (requested < 1) {
        requested = 1;
    }
    if (portBudget < 1) {
        portBudget = 1;
    }
    if (topo == TopologyKind::Clos || topo == TopologyKind::ClosMesh || topo == TopologyKind::Nhr) {
        return std::min(requested, portBudget);
    }
    return requested;
}

}  // namespace hccl_refactor
