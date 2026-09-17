#pragma once

#include "channel.hpp"
#include "topology.hpp"

#include <vector>

namespace hccl_algo {

// Run one topological plane in lockstep: all ranks issue step s, then flush.
template <class Topo, class Pat>
void runPlane(World& world, const Plane& plane)
{
    const int p = plane.size();
    if (p <= 1) {
        return;
    }
    const int nsteps = Topo::steps(p);
    for (int s = 0; s < nsteps; ++s) {
        for (int loc = 0; loc < p; ++loc) {
            Ctx c;
            c.world = &world;
            c.plane = &plane;
            c.rank = loc;
            c.rankSize = p;
            Topo::template issue<Pat>(c, s);
        }
        world.flush();
    }
}

// Symmetric hierarchical: L0 groups of n0, L1 groups of color (same local id).
// Clos+Mesh = Hierarchical<Mesh, Bind<Ring or Nhr>> with n0 = devices/server.
template <class L0, class L1>
struct Hierarchical {
    static constexpr const char* kName = "hierarchical";

    static std::vector<Plane> level0Planes(int p, int n0)
    {
        std::vector<Plane> out;
        for (int base = 0; base < p; base += n0) {
            Plane pl;
            pl.chunk = 1;
            const int n = std::min(n0, p - base);
            for (int i = 0; i < n; ++i) {
                pl.members.push_back(base + i);
            }
            out.push_back(std::move(pl));
        }
        return out;
    }

    static std::vector<Plane> level1Planes(int p, int n0)
    {
        std::vector<Plane> out;
        if (n0 <= 0) {
            return out;
        }
        const int n1 = p / n0;
        for (int d = 0; d < n0; ++d) {
            Plane pl;
            pl.chunk = n0;  // each color rank holds a whole server chunk after L0 AG
            for (int s = 0; s < n1; ++s) {
                pl.members.push_back(s * n0 + d);
            }
            out.push_back(std::move(pl));
        }
        return out;
    }

    // AllGather: intra L0 first (Mesh shares server blocks), then L1 (Clos).
    static void allGather(World& world, int p, int n0)
    {
        for (const auto& pl : level0Planes(p, n0)) {
            runPlane<L0, AllGatherPat>(world, pl);
        }
        // After L0, each server rank has the n0 server-local blocks at indices
        // [s*n0, s*n0+n0). L1 Ring/NHR must move a chunk of n0 blocks whose
        // logical index is the server id. slot(logical)=logical*n0 matches that
        // only if members[local]=s*n0+d and we index by local s — Plane::slot
        // uses logical * chunk, so slot(s) = s*n0, correct for every color.
        for (auto& pl : level1Planes(p, n0)) {
            runPlane<L1, AllGatherPat>(world, pl);
        }
    }

    static int steps(int p, int n0)
    {
        const int n1 = n0 ? p / n0 : p;
        return L0::steps(n0) + L1::steps(n1);
    }
};

// Three-level L0×L1×L2 (910_93): n0 intra-server, n1 servers per pod, n2 pods.
template <class L0, class L1, class L2>
struct Hierarchical3 {
    static constexpr const char* kName = "hierarchical3";

    static void allGather(World& world, int p, int n0, int n1)
    {
        const int pod = n0 * n1;
        if (pod <= 0 || p % pod != 0) {
            Hierarchical<L0, L1>::allGather(world, p, n0);
            return;
        }
        const int n2 = p / pod;
        for (const auto& pl : Hierarchical<L0, L1>::level0Planes(p, n0)) {
            runPlane<L0, AllGatherPat>(world, pl);
        }
        for (int podId = 0; podId < n2; ++podId) {
            const int podBase = podId * pod;
            for (int d = 0; d < n0; ++d) {
                Plane pl;
                pl.chunk = n0;
                pl.base = podBase;
                for (int s = 0; s < n1; ++s) {
                    pl.members.push_back(podBase + s * n0 + d);
                }
                runPlane<L1, AllGatherPat>(world, pl);
            }
        }
        for (int off = 0; off < pod; ++off) {
            Plane pl;
            pl.chunk = pod;
            for (int k = 0; k < n2; ++k) {
                pl.members.push_back(k * pod + off);
            }
            runPlane<L2, AllGatherPat>(world, pl);
        }
    }

    static int steps(int p, int n0, int n1)
    {
        const int n2 = (n0 > 0 && n1 > 0) ? p / (n0 * n1) : 1;
        return L0::steps(n0) + L1::steps(n1) + L2::steps(n2);
    }
};

}  // namespace hccl_algo
