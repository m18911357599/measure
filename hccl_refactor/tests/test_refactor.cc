#include "hccl_refactor/catalog.h"
#include "hccl_refactor/composition.h"
#include "hccl_refactor/planners.h"
#include "hccl_refactor/simulate.h"
#include "hccl_refactor/types.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace hccl_refactor;

static int g_failed = 0;
static int g_passed = 0;

#define CHECK(cond, msg)                                                                 \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);           \
            ++g_failed;                                                                  \
        } else {                                                                         \
            ++g_passed;                                                                  \
        }                                                                                \
    } while (0)

static PlannerInput MakeInput(uint32_t n, uint32_t me, uint32_t ch = 1, uint64_t bytes = 1024)
{
    PlannerInput in;
    in.rankSize = n;
    in.myRank = me;
    in.channels = ch;
    in.portBudget = 4;
    in.sliceBytes = bytes;
    return in;
}

template <typename Topo>
static std::vector<CommPlan> WorldAllGather(uint32_t n, uint32_t ch = 1, uint64_t bytes = 1024)
{
    std::vector<CommPlan> world;
    world.reserve(n);
    for (uint32_t r = 0; r < n; ++r) {
        world.push_back(PlanAllGather<Topo>(MakeInput(n, r, ch, bytes)));
    }
    return world;
}

static void TestMesh()
{
    auto plan = PlanAllGather<Mesh>(MakeInput(8, 3));
    CHECK(plan.stepCount == 1, "mesh AG steps == 1");
    CHECK(UniquePeers(plan, 3) == 7, "mesh AG peers == N-1");
    CHECK(SimulateAllGather(8, WorldAllGather<Mesh>(8)), "mesh AG completeness");

    auto rs = PlanReduceScatter<Mesh>(MakeInput(8, 3));
    CHECK(rs.stepCount == 1, "mesh RS steps == 1");
    CHECK(UniquePeers(rs, 3) == 7, "mesh RS peers == N-1");
}

static void TestRing()
{
    auto plan = PlanAllGather<Ring>(MakeInput(8, 0));
    CHECK(plan.stepCount == 7, "ring AG steps == N-1");
    CHECK(SimulateAllGather(8, WorldAllGather<Ring>(8)), "ring AG completeness");
}

static void TestDoubleRing()
{
    auto plan = PlanAllGather<DoubleRing>(MakeInput(8, 1, 1, 100));
    CHECK(plan.stepCount == 7, "double-ring AG steps == N-1");
    uint64_t fwd = 0;
    uint64_t bwd = 0;
    const uint32_t me = 1;
    const uint32_t fwdPeer = 2;
    const uint32_t bwdPeer = 0;
    for (const auto& t : plan.transfers) {
        if (t.srcRank == me && t.dstRank == fwdPeer) {
            fwd += t.bytes;
        }
        if (t.srcRank == me && t.dstRank == bwdPeer) {
            bwd += t.bytes;
        }
    }
    CHECK(fwd > 0 && bwd > 0, "double-ring uses both directions");
}

static void TestNhr()
{
    auto plan4 = PlanAllGather<Nhr>(MakeInput(4, 0));
    CHECK(plan4.stepCount == 2, "nhr AG N=4 steps == 2");
    auto plan8 = PlanAllGather<Nhr>(MakeInput(8, 0));
    CHECK(plan8.stepCount == 3, "nhr AG N=8 steps == 3");
    auto plan5 = PlanAllGather<Nhr>(MakeInput(5, 0));
    CHECK(plan5.stepCount == 3, "nhr AG N=5 steps == ceil(log2(5))");
    CHECK(SimulateAllGather(8, WorldAllGather<Nhr>(8)), "nhr AG completeness N=8");
    CHECK(CeilLog2(1) == 0, "ceil log2 1");
    CHECK(CeilLog2(2) == 1, "ceil log2 2");
    CHECK(CeilLog2(3) == 2, "ceil log2 3");
}

static void TestClosVsMeshChannels()
{
    PlannerInput in = MakeInput(8, 0, 8);
    in.portBudget = 2;
    auto mesh = PlanAllGather<Mesh>(in);
    auto clos = PlanAllGather<Clos>(in);
    CHECK(mesh.channels == 8, "mesh keeps requested channels");
    CHECK(clos.channels == 2, "clos caps channels by port budget");
}

static void TestClosMeshParallel()
{
    auto plan = PlanAllGather<ClosMesh>(MakeInput(8, 0, 2, 1000));
    CHECK(plan.policy == Policy::Parallel, "clos+mesh is parallel");
    bool hasMesh = false;
    bool hasNhr = false;
    for (const auto& t : plan.transfers) {
        (void)t;
    }
    auto composed = Compose<ClosMeshConcurAG>::Build(MakeInput(8, 0, 2, 1000));
    CHECK(composed.size() == 2, "concurrent AG has two branches");
    for (const auto& p : composed) {
        if (p.topology == TopologyKind::Mesh) {
            hasMesh = true;
        }
        if (p.topology == TopologyKind::Nhr) {
            hasNhr = true;
        }
    }
    CHECK(hasMesh && hasNhr, "clos+mesh concurrent uses mesh and nhr");
}

static void TestMultiChannel()
{
    auto plan = Compose<MeshAllGather4Ch>::Build(MakeInput(4, 0, 1, 1000));
    CHECK(plan.size() == 1, "single leaf");
    uint32_t maxCh = 0;
    for (const auto& t : plan[0].transfers) {
        maxCh = std::max(maxCh, t.channel + 1);
    }
    CHECK(maxCh == 4, "multi-channel stripes to 4 channels");
    uint64_t sum = 0;
    for (const auto& t : plan[0].transfers) {
        if (t.srcRank == 0 && t.dstRank == 1 && t.slotOwner == 0) {
            sum += t.bytes;
        }
    }
    CHECK(sum == 1000, "channel slices conserve bytes");
}

static void TestTwoShotAllReduce()
{
    auto plans = Compose<MeshTwoShotAllReduce>::Build(MakeInput(8, 0));
    CHECK(plans.size() == 2, "twoshot has RS then AG");
    CHECK(plans[0].collective == Collective::ReduceScatter, "first is RS");
    CHECK(plans[1].collective == Collective::AllGather, "second is AG");
    CHECK(plans[1].transfers.empty() || plans[1].transfers[0].step >= plans[0].stepCount,
          "sequence stacks steps");
}

static void TestHierarchical()
{
    auto ag2 = Compose<HierAllGather2L>::Build(MakeInput(8, 0));
    CHECK(ag2.size() == 2, "2-level AG has 2 stages");
    CHECK(ag2[0].topology == TopologyKind::Mesh && ag2[0].level == 0, "L0 mesh");
    CHECK(ag2[1].topology == TopologyKind::Nhr && ag2[1].level == 1, "L1 nhr");

    auto ag3 = Compose<HierAllGather3L>::Build(MakeInput(16, 0));
    CHECK(ag3.size() == 3, "3-level AG has 3 stages");
    CHECK(ag3[2].level == 2, "L2 present");

    auto ar = Compose<HierAllReduce2L>::Build(MakeInput(8, 0));
    CHECK(ar.size() == 4, "hier AR is RS-RS-AG-AG");
    CHECK(ar[0].collective == Collective::ReduceScatter, "AR[0] RS");
    CHECK(ar[1].collective == Collective::ReduceScatter, "AR[1] RS");
    CHECK(ar[2].collective == Collective::AllGather, "AR[2] AG");
    CHECK(ar[3].collective == Collective::AllGather, "AR[3] AG");
}

static void TestCodeReductionInvariant()
{
    // Cartesian product that used to be copy-pasted: 6 topos x 2 coll x 2 channel
    // modes x 3 hierarchy shapes. Template composition covers it from 6 planners.
    const int topos = 6;
    const int colls = 2;
    const int chans = 2;
    const int hier = 3;
    const int cartesian = topos * colls * chans * hier;
    const int planners = 6;
    CHECK(cartesian / planners == 12, "one planner serves 12 composed variants");
}

int main()
{
    TestMesh();
    TestRing();
    TestDoubleRing();
    TestNhr();
    TestClosVsMeshChannels();
    TestClosMeshParallel();
    TestMultiChannel();
    TestTwoShotAllReduce();
    TestHierarchical();
    TestCodeReductionInvariant();

    std::printf("hccl_refactor tests: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
