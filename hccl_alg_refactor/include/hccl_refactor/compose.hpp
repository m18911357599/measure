#pragma once

#include "engines.hpp"

namespace hccl_refactor {

// ---------------------------------------------------------------------------
// Multi-channel: C independent topology engines on disjoint data slices.
// 8P_RING (4 rings) and AnyPath (SDMA+RDMA) are instances of this.
// Double-ring is the C=2 specialization with opposite ring orientation.
// ---------------------------------------------------------------------------

// Split each chunk along the element axis into C channels, run Kernel on each
// channel, then stitch.  Opposite ring for odd channels (double-ring).
template <int C, typename Kernel, bool OppositeOddChannel = true>
struct MultiChannel {
    static_assert(C >= 1, "channel count");
    static constexpr int kChannels = C;
    static constexpr const char* kOpName = Kernel::kOpName;
    static constexpr TopoKind kKind = TopoKind::kDoubleRing;

    static void run(World& w)
    {
        if (C == 1) {
            Kernel::run(w);
            return;
        }
        const int base = w.chunk_size / C;
        const int rem = w.chunk_size % C;
        if (base == 0 && rem == 0) {
            Kernel::run(w);
            return;
        }

        World orig = w;
        std::vector<World> ch(static_cast<size_t>(C));
        for (int c = 0; c < C; ++c) {
            const int cs = base + (c < rem ? 1 : 0);
            World cw = World::make(w.n_ranks, cs > 0 ? cs : 1);
            if (cs == 0) {
                ch[static_cast<size_t>(c)] = std::move(cw);
                continue;
            }
            cw.n_chunks = w.n_chunks;
            cw.buf.assign(static_cast<size_t>(w.n_ranks),
                std::vector<Value>(static_cast<size_t>(w.n_chunks * cs), 0));
            const int off = c * base + std::min(c, rem);
            for (int r = 0; r < w.n_ranks; ++r) {
                for (int k = 0; k < w.n_chunks; ++k) {
                    for (int e = 0; e < cs; ++e) {
                        cw.at(r, k, e) = orig.at(r, k, off + e);
                    }
                }
            }
            if (OppositeOddChannel && (c % 2 == 1)) {
                reverse_rank_order(cw);
                Kernel::run(cw);
                reverse_rank_order(cw);
            } else {
                Kernel::run(cw);
            }
            ch[static_cast<size_t>(c)] = std::move(cw);
        }

        for (int c = 0; c < C; ++c) {
            const int cs = base + (c < rem ? 1 : 0);
            if (cs == 0) {
                continue;
            }
            const int off = c * base + std::min(c, rem);
            for (int r = 0; r < w.n_ranks; ++r) {
                for (int k = 0; k < w.n_chunks; ++k) {
                    for (int e = 0; e < cs; ++e) {
                        w.at(r, k, off + e) = ch[static_cast<size_t>(c)].at(r, k, e);
                    }
                }
            }
        }
    }

    static void reverse_rank_order(World& w)
    {
        // Opposite ring = reverse neighbor order.  Ranks *and* chunk ids must
        // use the same permutation so ReduceScatter still lands on chunk==rank.
        const int n = w.n_ranks;
        World tmp = w;
        for (int r = 0; r < n; ++r) {
            const int src_r = n - 1 - r;
            for (int c = 0; c < w.n_chunks; ++c) {
                const int src_c = (c < n) ? (n - 1 - c) : c;
                for (int e = 0; e < w.chunk_size; ++e) {
                    w.at(r, c, e) = tmp.at(src_r, src_c, e);
                }
            }
        }
    }
};

template <typename Kernel>
using DoubleRing = MultiChannel<2, Kernel, true>;

using DoubleRingAllReduce = DoubleRing<RingAllReduce>;
using DoubleRingReduceScatter = DoubleRing<RingKernel<ReduceScatterOp>>;
using DoubleRingAllGather = DoubleRing<RingKernel<AllGatherOp>>;

// ---------------------------------------------------------------------------
// Hierarchical (level0 intra-server + level1 inter-server [+ level2]).
// Mirrors CollAllReduceRingFor91093Executor::KernelRun:
//   L0 ReduceScatter -> L1 AllReduce -> L0 AllGather
// Clos+Mesh is this pattern with L0=Mesh / L1=Ring-or-NHR on Clos.
// ---------------------------------------------------------------------------

template <typename L0Topo, typename L1Topo>
struct HierarchicalAllReduce {
    static constexpr Collective kCollective = Collective::kAllReduce;
    static constexpr TopoKind kKind = TopoKind::kHierarchical;
    static constexpr const char* kOpName = "AllReduce";

    // n0 = devices per server (COMM_LEVEL0 size).  n must be divisible by n0.
    static void run(World& w, int n0)
    {
        const int n = w.n_ranks;
        if (n0 <= 1 || n0 >= n) {
            AllReduceByComposition<L0Topo>::run(w);
            return;
        }
        if (n % n0 != 0) {
            throw std::runtime_error("HierarchicalAllReduce: n not divisible by n0");
        }
        const int n1 = n / n0;

        // ---- phase 0: intra-server ReduceScatter (n0-way) ----
        for (int s = 0; s < n1; ++s) {
            World local = extract_group(w, server_ranks(s, n0));
            // Reindex chunks 0..n0-1 as the L0 RS domain; keep extra chunks
            // packed after.  For the HCCL pattern, data is sliced by L0 size:
            // we RS the n0 leading logical chunks within the server.
            World l0 = pack_l0(local, n0);
            TopologyKernel<L0Topo, ReduceScatterOp>::run(l0);
            unpack_l0(local, l0, n0);
            write_group(w, server_ranks(s, n0), local);
        }

        // ---- phase 1: inter-server AllReduce on the local slice ----
        for (int local = 0; local < n0; ++local) {
            World color = extract_group(w, color_ranks(local, n0, n1));
            World slice = isolate_chunk(color, local, n0);
            AllReduceByComposition<L1Topo>::run(slice);
            merge_chunk(color, slice, local, n0);
            write_group(w, color_ranks(local, n0, n1), color);
        }

        // ---- phase 2: intra-server AllGather ----
        for (int s = 0; s < n1; ++s) {
            World local = extract_group(w, server_ranks(s, n0));
            World l0 = pack_l0(local, n0);
            TopologyKernel<L0Topo, AllGatherOp>::run(l0);
            unpack_l0(local, l0, n0);
            write_group(w, server_ranks(s, n0), local);
        }
    }

    static std::vector<int> server_ranks(int server, int n0)
    {
        std::vector<int> v(static_cast<size_t>(n0));
        for (int i = 0; i < n0; ++i) {
            v[static_cast<size_t>(i)] = server * n0 + i;
        }
        return v;
    }

    static std::vector<int> color_ranks(int local, int n0, int n1)
    {
        std::vector<int> v(static_cast<size_t>(n1));
        for (int s = 0; s < n1; ++s) {
            v[static_cast<size_t>(s)] = s * n0 + local;
        }
        return v;
    }

    static World extract_group(const World& w, const std::vector<int>& ranks)
    {
        World g;
        g.n_ranks = static_cast<int>(ranks.size());
        g.n_chunks = w.n_chunks;
        g.chunk_size = w.chunk_size;
        g.buf.resize(ranks.size());
        for (size_t i = 0; i < ranks.size(); ++i) {
            g.buf[i] = w.buf[static_cast<size_t>(ranks[i])];
        }
        return g;
    }

    static void write_group(World& w, const std::vector<int>& ranks, const World& g)
    {
        for (size_t i = 0; i < ranks.size(); ++i) {
            w.buf[static_cast<size_t>(ranks[i])] = g.buf[i];
        }
    }

    static World pack_l0(const World& local, int n0)
    {
        World l0 = World::make(n0, local.chunk_size);
        l0.n_chunks = n0;
        l0.buf.assign(static_cast<size_t>(n0),
            std::vector<Value>(static_cast<size_t>(n0 * local.chunk_size), 0));
        // Each local rank holds n0 * n1 global chunks.  Fold every n0-strided
        // global chunk into the L0 domain by summing?  No: HCCL slices the
        // *byte payload* into n0 pieces.  Model: concatenate groups of n1
        // global chunks into one L0 chunk.
        const int n1 = local.n_chunks / n0;
        l0.chunk_size = local.chunk_size * n1;
        l0.buf.assign(static_cast<size_t>(n0),
            std::vector<Value>(static_cast<size_t>(n0 * l0.chunk_size), 0));
        for (int r = 0; r < n0; ++r) {
            for (int c0 = 0; c0 < n0; ++c0) {
                for (int j = 0; j < n1; ++j) {
                    const int gc = c0 * n1 + j;
                    for (int e = 0; e < local.chunk_size; ++e) {
                        l0.buf[static_cast<size_t>(r)][static_cast<size_t>(
                            c0 * l0.chunk_size + j * local.chunk_size + e)] =
                            local.at(r, gc, e);
                    }
                }
            }
        }
        return l0;
    }

    static void unpack_l0(World& local, const World& l0, int n0)
    {
        const int n1 = local.n_chunks / n0;
        for (int r = 0; r < n0; ++r) {
            for (int c0 = 0; c0 < n0; ++c0) {
                for (int j = 0; j < n1; ++j) {
                    const int gc = c0 * n1 + j;
                    for (int e = 0; e < local.chunk_size; ++e) {
                        local.at(r, gc, e) = l0.buf[static_cast<size_t>(r)][static_cast<size_t>(
                            c0 * l0.chunk_size + j * local.chunk_size + e)];
                    }
                }
            }
        }
    }

    static World isolate_chunk(const World& color, int local, int n0)
    {
        const int n1 = color.n_ranks;
        const int group = color.n_chunks / n0;  // n1
        (void)group;
        World s = World::make(n1, color.chunk_size);
        s.n_chunks = n1;
        // Each color rank owns (after L0 RS) the n1 global chunks belonging
        // to local-id `local`, i.e. chunks [local*n1, local*n1+n1).
        s.buf.assign(static_cast<size_t>(n1),
            std::vector<Value>(static_cast<size_t>(n1 * color.chunk_size), 0));
        for (int r = 0; r < n1; ++r) {
            for (int j = 0; j < n1; ++j) {
                const int gc = local * n1 + j;
                for (int e = 0; e < color.chunk_size; ++e) {
                    s.at(r, j, e) = color.at(r, gc, e);
                }
            }
        }
        return s;
    }

    static void merge_chunk(World& color, const World& slice, int local, int n0)
    {
        (void)n0;
        const int n1 = color.n_ranks;
        for (int r = 0; r < n1; ++r) {
            for (int j = 0; j < n1; ++j) {
                const int gc = local * n1 + j;
                for (int e = 0; e < color.chunk_size; ++e) {
                    color.at(r, gc, e) = slice.at(r, j, e);
                }
            }
        }
    }
};

// Named recipes matching the requested families.
using MeshHierarchical = HierarchicalAllReduce<MeshTopo, RingTopo>;
using ClosMeshHierarchical = HierarchicalAllReduce<MeshTopo, NhrTopo>;   // L0 mesh + L1 NHR on Clos
using ClosRingHierarchical = HierarchicalAllReduce<MeshTopo, RingTopo>;  // L0 mesh + L1 ring on Clos
using RingHierarchical = HierarchicalAllReduce<RingTopo, RingTopo>;
using DoubleRingHierarchical = HierarchicalAllReduce<RingTopo, NhrTopo>;

}  // namespace hccl_refactor
