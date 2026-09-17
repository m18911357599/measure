#include "hccl_refactor.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

using namespace hccl_refactor;

static int g_failed = 0;
static int g_passed = 0;

#define CHECK(cond, msg)                                                                 \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL  " << (msg) << "  (" << __FILE__ << ":" << __LINE__       \
                      << ")\n";                                                          \
            ++g_failed;                                                                  \
        } else {                                                                         \
            ++g_passed;                                                                  \
        }                                                                                \
    } while (0)

static void require_allreduce(const World& w, const char* tag)
{
    for (int r = 0; r < w.n_ranks; ++r) {
        for (int c = 0; c < w.n_chunks; ++c) {
            for (int e = 0; e < w.chunk_size; ++e) {
                const Value got = w.at(r, c, e);
                const Value exp = w.expected_allreduce(w.n_ranks, c, e);
                if (got != exp) {
                    std::cerr << "FAIL  " << tag << " rank=" << r << " chunk=" << c
                              << " e=" << e << " got=" << got << " exp=" << exp << "\n";
                    ++g_failed;
                    return;
                }
            }
        }
    }
    ++g_passed;
    std::cout << "PASS  " << tag << "  n=" << w.n_ranks << " chunk=" << w.chunk_size << "\n";
}

static void require_reducescatter(const World& w, const char* tag)
{
    for (int r = 0; r < w.n_ranks; ++r) {
        for (int e = 0; e < w.chunk_size; ++e) {
            const Value got = w.at(r, r, e);
            const Value exp = w.expected_allreduce(w.n_ranks, r, e);
            if (got != exp) {
                std::cerr << "FAIL  " << tag << " rank=" << r << " own-chunk e=" << e
                          << " got=" << got << " exp=" << exp << "\n";
                ++g_failed;
                return;
            }
        }
    }
    ++g_passed;
    std::cout << "PASS  " << tag << "  n=" << w.n_ranks << "\n";
}

static void require_allgather(const World& w, const char* tag)
{
    // AG from identity: rank r originally owned chunk r.  After AG every rank
    // holds chunk c == original rank-c value.
    for (int r = 0; r < w.n_ranks; ++r) {
        for (int c = 0; c < w.n_chunks; ++c) {
            for (int e = 0; e < w.chunk_size; ++e) {
                const Value got = w.at(r, c, e);
                const Value exp = static_cast<Value>(c + 1) * 1000 + c * 10 + e;
                if (got != exp) {
                    std::cerr << "FAIL  " << tag << " rank=" << r << " chunk=" << c
                              << " e=" << e << " got=" << got << " exp=" << exp << "\n";
                    ++g_failed;
                    return;
                }
            }
        }
    }
    ++g_passed;
    std::cout << "PASS  " << tag << "  n=" << w.n_ranks << "\n";
}

static World make_ag_input(int n, int cs = 2)
{
    World w = World::make(n, cs);
    w.fill_rank_identity();
    // AG starts with only the local chunk valid.
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            if (c == r) {
                continue;
            }
            for (int e = 0; e < cs; ++e) {
                w.at(r, c, e) = 0;
            }
        }
    }
    return w;
}

static void test_ring()
{
    for (int n : {2, 3, 4, 8}) {
        World w = World::make(n, 3);
        w.fill_rank_identity();
        RingKernel<ReduceScatterOp>::run(w);
        require_reducescatter(w, "ring-RS");

        World ag = make_ag_input(n, 3);
        RingKernel<AllGatherOp>::run(ag);
        require_allgather(ag, "ring-AG");

        World ar = World::make(n, 3);
        ar.fill_rank_identity();
        RingAllReduce::run(ar);
        require_allreduce(ar, "ring-AR");
    }
}

static void test_mesh()
{
    for (int n : {2, 4, 8}) {
        World w = World::make(n, 2);
        w.fill_rank_identity();
        MeshKernel<ReduceScatterOp>::run(w);
        require_reducescatter(w, "mesh-RS");

        World ag = make_ag_input(n, 2);
        MeshKernel<AllGatherOp>::run(ag);
        require_allgather(ag, "mesh-AG");

        World ar = World::make(n, 2);
        ar.fill_rank_identity();
        MeshAllReduce::run(ar);
        require_allreduce(ar, "mesh-AR");
    }
}

static void test_nhr()
{
    for (int n : {2, 3, 4, 5, 8}) {
        World w = World::make(n, 2);
        w.fill_rank_identity();
        NhrKernel<ReduceScatterOp>::run(w);
        require_reducescatter(w, "nhr-RS");

        World ag = make_ag_input(n, 2);
        NhrKernel<AllGatherOp>::run(ag);
        require_allgather(ag, "nhr-AG");

        World ar = World::make(n, 2);
        ar.fill_rank_identity();
        NhrAllReduce::run(ar);
        require_allreduce(ar, "nhr-AR");
    }
}

static void test_double_ring()
{
    for (int n : {4, 8}) {
        World w = World::make(n, 4);  // even chunk_size so two channels split cleanly
        w.fill_rank_identity();
        DoubleRingAllReduce::run(w);
        require_allreduce(w, "double-ring-AR");

        World rs = World::make(n, 4);
        rs.fill_rank_identity();
        DoubleRingReduceScatter::run(rs);
        require_reducescatter(rs, "double-ring-RS");
    }
}

static void test_hierarchical()
{
    // 4 ranks, 2 per server: L0 mesh + L1 ring  (clos+mesh style)
    {
        World w = World::make(4, 2);
        w.fill_rank_identity();
        ClosRingHierarchical::run(w, /*n0=*/2);
        require_allreduce(w, "clos+mesh hierarchical AR (L0 mesh, L1 ring)");
    }
    {
        World w = World::make(4, 2);
        w.fill_rank_identity();
        ClosMeshHierarchical::run(w, /*n0=*/2);
        require_allreduce(w, "clos+mesh hierarchical AR (L0 mesh, L1 nhr)");
    }
    {
        World w = World::make(8, 2);
        w.fill_rank_identity();
        RingHierarchical::run(w, /*n0=*/4);
        require_allreduce(w, "ring hierarchical AR (L0=4, L1=2)");
    }
    {
        World w = World::make(8, 2);
        w.fill_rank_identity();
        MeshHierarchical::run(w, /*n0=*/2);
        require_allreduce(w, "mesh hierarchical AR (L0=2, L1=4)");
    }
}

static void test_interconnect_policy()
{
    using L0 = DefaultTopoFor<OnChipMeshInterconnect, MeshTopo, RingTopo>;
    using L1 = DefaultTopoFor<ClosInterconnect, MeshTopo, RingTopo>;
    static_assert(std::is_same<L0, MeshTopo>::value, "on-chip mesh prefers dense mesh");
    static_assert(std::is_same<L1, RingTopo>::value, "clos prefers sparse ring");
    CHECK((std::is_same<L0, MeshTopo>::value), "L0 default is Mesh");
    CHECK((std::is_same<L1, RingTopo>::value), "L1 default is Ring on Clos");
    std::cout << "PASS  interconnect policy  mesh->" << L0::kName << "  clos->" << L1::kName
              << "\n";
}

static void test_source_budget()
{
    CHECK(SourceBudget::kNewUnits < SourceBudget::kOldUnits / 4,
        "new source units should be << cartesian product of old kernels");
    std::cout << "PASS  source budget  old=" << SourceBudget::kOldUnits
              << " new=" << SourceBudget::kNewUnits << "\n";
}

static void test_registry_instantiation()
{
    World w = World::make(4, 2);
    w.fill_rank_identity();
    KernelT<TopoKind::kRing, Collective::kAllReduce>::run(w);
    require_allreduce(w, "registry Ring AR");

    World m = World::make(4, 2);
    m.fill_rank_identity();
    KernelT<TopoKind::kMesh, Collective::kAllReduce>::run(m);
    require_allreduce(m, "registry Mesh AR");

    World d = World::make(4, 4);
    d.fill_rank_identity();
    KernelT<TopoKind::kDoubleRing, Collective::kAllReduce, 2>::run(d);
    require_allreduce(d, "registry DoubleRing AR");
}

static void test_no_virtual_hot_path()
{
    // Policy/engine methods are static; this would fail to compile if they were virtual.
    static_assert(!std::is_polymorphic<RingTopo>::value, "RingTopo must not be polymorphic");
    static_assert(!std::is_polymorphic<MeshTopo>::value, "MeshTopo must not be polymorphic");
    static_assert(!std::is_polymorphic<NhrTopo>::value, "NhrTopo must not be polymorphic");
    static_assert(!std::is_polymorphic<ReduceScatterOp>::value, "ops must not be polymorphic");
    CHECK(true, "no virtual in topo/op policies");
    std::cout << "PASS  no-virtual hot path (compile-time static_assert)\n";
}

int main()
{
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    try {
    test_no_virtual_hot_path();
    test_interconnect_policy();
    test_source_budget();
    test_ring();
    test_mesh();
    test_nhr();
    test_double_ring();
    test_hierarchical();
        test_registry_instantiation();
    } catch (const std::exception& ex) {
        std::cerr << "EXCEPTION  " << ex.what() << "\n";
        return 1;
    }

    std::cout << "\n" << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
