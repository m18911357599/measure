#pragma once

#include "topology.hpp"

namespace hccl_algo {

// C concurrent topological instances (streams / rings / SDMA+RDMA planes).
// Step count unchanged; issue width × C. Double-ring is C=2 + Ring.
template <class Topo, int C>
struct MultiChannel {
    static constexpr const char* kName = "multi-channel";
    static constexpr int kChannels = C;
    static int steps(int p) { return Topo::steps(p); }

    template <class Pat>
    static void issue(Ctx& c, int step)
    {
        for (int ch = 0; ch < C; ++ch) {
            Ctx sub = c;
            sub.channel = ch;
            if (ch == 0) {
                Topo::template issue<Pat>(sub, step);
            } else {
                // Opposite neighbor on extra channels — dual-issue, same slices.
                // Production Aligned Double-Ring splits payload in half; here the
                // extra channel posts the reverse-ring copy of the same AG/RS
                // slice to keep the task-graph width honest without a second buffer.
                const int p = c.rankSize;
                const int other = (ch % 2 == 1) ? (c.rank + p - 1) % p : (c.rank + 1) % p;
                const int sendIdx = Pat::ringSendIdx(c.rank, p, step);
                c.world->postCopy(c.gid(c.rank), c.gid(other), c.slot(sendIdx), c.slot(sendIdx), false);
            }
        }
    }
};

using DoubleRing = MultiChannel<Ring, 2>;

}  // namespace hccl_algo
