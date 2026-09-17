#pragma once

#include "pattern.hpp"
#include "world.hpp"

namespace hccl_algo {

inline int backward(int rank, int p, int round) { return (rank + p - round) % p; }

struct Mesh {
    static constexpr const char* kName = "mesh";
    static int steps(int /*p*/) { return 1; }

    template <class Pat>
    static void issue(Ctx& c, int /*step*/)
    {
        const int p = c.rankSize;
        for (int r = 1; r < p; ++r) {
            const int peer = backward(c.rank, p, r);
            if constexpr (Pat::kReduceOnRecv) {
                c.world->postCopy(c.gid(c.rank), c.gid(peer), c.slot(peer), c.slot(peer), true);
            } else {
                c.world->postCopy(c.gid(c.rank), c.gid(peer), c.slot(c.rank), c.slot(c.rank), false);
            }
        }
    }
};

struct Ring {
    static constexpr const char* kName = "ring";
    static int steps(int p) { return p > 0 ? p - 1 : 0; }

    template <class Pat>
    static void issue(Ctx& c, int step)
    {
        const int p = c.rankSize;
        const int next = (c.rank + 1) % p;
        const int sendIdx = Pat::ringSendIdx(c.rank, p, step);
        c.world->postCopy(c.gid(c.rank), c.gid(next), c.slot(sendIdx), c.slot(sendIdx), false);
        if constexpr (Pat::kReduceOnRecv) {
            const int prev = (c.rank + p - 1) % p;
            const int recvIdx = Pat::ringRecvIdx(c.rank, p, step);
            c.world->postCopy(c.gid(prev), c.gid(c.rank), c.slot(recvIdx), c.slot(recvIdx), true);
        }
    }
};

// Log-step AllGather / ReduceScatter (Bruck / NHR family). Works for any p.
struct Nhr {
    static constexpr const char* kName = "nhr";
    static int steps(int p) { return ceil_log2(p); }

    template <class Pat>
    static void issue(Ctx& c, int step)
    {
        const int p = c.rankSize;
        const int dist = 1 << step;
        if (dist >= p) {
            return;
        }
        const int dst = (c.rank + p - dist) % p;
        const int nSend = std::min(dist, p);
        if constexpr (Pat::kReduceOnRecv) {
            for (int k = 0; k < nSend; ++k) {
                const int blk = (dst + k) % p;
                c.world->postCopy(c.gid(c.rank), c.gid(dst), c.slot(blk), c.slot(blk), true);
            }
        } else {
            for (int k = 0; k < nSend; ++k) {
                const int blk = (c.rank + k) % p;
                c.world->postCopy(c.gid(c.rank), c.gid(dst), c.slot(blk), c.slot(blk), false);
            }
        }
    }
};

struct ClosFabric {
    static constexpr const char* kName = "clos";
    static constexpr bool kAnyToAny = true;
};

template <class Topo>
struct Bind {
    static constexpr const char* kName = Topo::kName;
    static int steps(int p) { return Topo::steps(p); }
    template <class Pat>
    static void issue(Ctx& c, int step)
    {
        Topo::template issue<Pat>(c, step);
    }
};

using ClosFullMesh = Bind<Mesh>;
using ClosNhr = Bind<Nhr>;
using ClosRing = Bind<Ring>;

}  // namespace hccl_algo
