#pragma once

#include "policies.hpp"
#include "world.hpp"

namespace hccl_refactor {

// ---------------------------------------------------------------------------
// Topology engines.  Each engine is a *schedule* of StepPlan.  Combining
// Engine × OpPolicy yields one collective kernel.  No virtual calls.
// ---------------------------------------------------------------------------

template <typename Topo, typename Op>
struct TopologyKernel {
    static constexpr Collective kCollective = Op::kId;
    static constexpr const char* kTopoName = Topo::kName;
    static constexpr const char* kOpName = Op::kName;

    static void run(World& w)
    {
        const int n = w.n_ranks;
        const int nsteps = Topo::template steps<Op>(n);
        for (int s = 0; s < nsteps; ++s) {
            std::vector<StepPlan> plans(static_cast<size_t>(n));
            for (int r = 0; r < n; ++r) {
                plans[static_cast<size_t>(r)] = Topo::template plan<Op>(r, n, s);
            }
            commit_step<Op>(w, plans);
        }
    }
};

// ----- Ring -----------------------------------------------------------------
struct RingTopo {
    static constexpr TopoKind kKind = TopoKind::kRing;
    static constexpr const char* kName = "ring";

    template <typename Op>
    static int steps(int n)
    {
        (void)sizeof(Op);
        return n > 1 ? n - 1 : 0;
    }

    template <typename Op>
    static StepPlan plan(int rank, int n, int step)
    {
        StepPlan p;
        p.send_to = wrap(rank + 1, n);
        p.recv_from = wrap(rank - 1, n);
        p.tx_chunks = {Op::tx_chunk(rank, n, step)};
        p.rx_chunks = {Op::rx_chunk(rank, n, step)};
        return p;
    }
};

template <typename Op>
using RingKernel = TopologyKernel<RingTopo, Op>;

// ----- Mesh / FullMesh ------------------------------------------------------
// HCCL AllGatherMesh / ReduceScatterMesh: round r talks to BackwardRank.
struct MeshTopo {
    static constexpr TopoKind kKind = TopoKind::kMesh;
    static constexpr const char* kName = "mesh";

    template <typename Op>
    static int steps(int n)
    {
        (void)sizeof(Op);
        return n > 1 ? n - 1 : 0;
    }

    template <typename Op>
    static StepPlan plan(int rank, int n, int step)
    {
        // Serialize the on-chip all-to-all as n-1 matchings of distance d=step+1:
        // send to rank+d, recv from rank-d.  Real HCCL Mesh launches all peers
        // concurrently on sub-streams; the matching is equivalent and inlines.
        const int d = step + 1;
        StepPlan p;
        p.send_to = wrap(rank + d, n);
        p.recv_from = wrap(rank - d, n);
        if (Op::kReduce) {
            p.tx_chunks = {p.send_to};
            p.rx_chunks = {rank};
        } else {
            p.tx_chunks = {rank};
            p.rx_chunks = {p.recv_from};
        }
        return p;
    }
};

template <typename Op>
using MeshKernel = TopologyKernel<MeshTopo, Op>;

// ----- NHR (Nonuniform Hierarchical Ring) -----------------------------------
// Step formula matches hcomm ReduceScatterNHR::GetStepInfo / AllGatherNHR.
struct NhrTopo {
    static constexpr TopoKind kKind = TopoKind::kNhr;
    static constexpr const char* kName = "nhr";

    template <typename Op>
    static int steps(int n)
    {
        (void)sizeof(Op);
        return n > 1 ? ceil_log2(n) : 0;
    }

    template <typename Op>
    static StepPlan plan(int rank, int n, int step)
    {
        const int nsteps = ceil_log2(n);
        StepPlan p;
        if (Op::kReduce) {
            // ReduceScatterNHR::GetStepInfo
            const int delta = 1 << step;
            p.send_to = wrap(rank - delta, n);
            p.recv_from = wrap(rank + delta, n);
            const int n_slices = (n - 1 + (1 << step)) / (1 << (step + 1));
            const int delta_slice = 1 << (step + 1);
            int tx = p.send_to;
            int rx = rank;
            for (int i = 0; i < n_slices; ++i) {
                p.tx_chunks.push_back(tx);
                p.rx_chunks.push_back(rx);
                tx = wrap(tx - delta_slice, n);
                rx = wrap(rx - delta_slice, n);
            }
        } else {
            // AllGatherNHR::GetStepInfo  (delta counted from the last step)
            const int delta = 1 << (nsteps - 1 - step);
            p.send_to = wrap(rank + delta, n);
            p.recv_from = wrap(rank - delta, n);
            const int n_slices = (n - 1 + delta) / (1 << (nsteps - step));
            const int delta_slice = 1 << (nsteps - step);
            int tx = rank;
            int rx = wrap(rank - delta, n);
            for (int i = 0; i < n_slices; ++i) {
                p.tx_chunks.push_back(tx);
                p.rx_chunks.push_back(rx);
                tx = wrap(tx - delta_slice, n);
                rx = wrap(rx - delta_slice, n);
            }
        }
        return p;
    }
};

template <typename Op>
using NhrKernel = TopologyKernel<NhrTopo, Op>;

// ----- AllReduce as compile-time composition RS ∘ AG ------------------------
template <typename Topo>
struct AllReduceByComposition {
    static constexpr Collective kCollective = Collective::kAllReduce;
    static constexpr const char* kTopoName = Topo::kName;
    static constexpr const char* kOpName = "AllReduce";

    static void run(World& w)
    {
        TopologyKernel<Topo, ReduceScatterOp>::run(w);
        // After RS, rank r owns fully reduced chunk r.  Seed AG from that chunk.
        TopologyKernel<Topo, AllGatherOp>::run(w);
    }
};

using RingAllReduce = AllReduceByComposition<RingTopo>;
using MeshAllReduce = AllReduceByComposition<MeshTopo>;
using NhrAllReduce = AllReduceByComposition<NhrTopo>;

}  // namespace hccl_refactor
