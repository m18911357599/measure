#include "hccl_algo/recipe.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace hccl_algo;

static void fail(const char* msg)
{
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

static void expect(bool cond, const char* msg)
{
    if (!cond) {
        fail(msg);
    }
}

static void initOwnBlock(World& w)
{
    const int p = w.nRanks();
    for (int r = 0; r < p; ++r) {
        std::memset(w.ptr(r), 0, static_cast<size_t>(w.nBlocks() * w.blockBytes()));
        w.ptr(r)[r * w.blockBytes()] = static_cast<uint8_t>(r + 1);
    }
}

static void checkAllGather(const World& w, const char* name)
{
    const int p = w.nRanks();
    for (int r = 0; r < p; ++r) {
        for (int b = 0; b < p; ++b) {
            const uint8_t got = w.ptr(r)[b * w.blockBytes()];
            const uint8_t exp = static_cast<uint8_t>(b + 1);
            if (got != exp) {
                std::fprintf(stderr, "FAIL %s rank %d block %d got %u expected %u\n",
                    name, r, b, got, exp);
                std::exit(1);
            }
        }
    }
}

static void initFullOnes(World& w)
{
    const int p = w.nRanks();
    for (int r = 0; r < p; ++r) {
        for (int b = 0; b < p; ++b) {
            w.ptr(r)[b * w.blockBytes()] = 1;
        }
    }
}

template <class Topo>
static World runAG(int p, const char* name)
{
    World w;
    w.reset(p, p, 1);
    initOwnBlock(w);
    runPlane<Topo, AllGatherPat>(w, identityPlane(p));
    checkAllGather(w, name);
    std::printf("  %-28s p=%d steps=%d tx=%d OK\n", name, p, Topo::steps(p), w.txCount());
    return w;
}

static World runClosMeshAG(int p, int n0, const char* name)
{
    World w;
    w.reset(p, p, 1);
    initOwnBlock(w);
    Hierarchical<Mesh, ClosRing>::allGather(w, p, n0);
    checkAllGather(w, name);
    std::printf("  %-28s p=%d n0=%d steps=%d tx=%d OK\n", name, p, n0,
        Hierarchical<Mesh, ClosRing>::steps(p, n0), w.txCount());
    return w;
}

static World runHier3AG(int p, int n0, int n1, const char* name)
{
    World w;
    w.reset(p, p, 1);
    initOwnBlock(w);
    Hierarchical3<Mesh, ClosRing, ClosNhr>::allGather(w, p, n0, n1);
    checkAllGather(w, name);
    std::printf("  %-28s p=%d n0=%d n1=%d steps=%d tx=%d OK\n", name, p, n0, n1,
        Hierarchical3<Mesh, ClosRing, ClosNhr>::steps(p, n0, n1), w.txCount());
    return w;
}

static void testReduceScatterMesh(int p)
{
    World w;
    w.reset(p, p, 1);
    initFullOnes(w);
    runPlane<Mesh, ReduceScatterPat>(w, identityPlane(p));
    for (int r = 0; r < p; ++r) {
        const uint8_t got = w.ptr(r)[r];
        if (got != static_cast<uint8_t>(p)) {
            std::fprintf(stderr, "FAIL RS-Mesh rank %d got %u expected %d\n", r, got, p);
            std::exit(1);
        }
    }
    std::printf("  %-28s p=%d steps=%d tx=%d OK\n", "ReduceScatterMesh", p, Mesh::steps(p), w.txCount());
}

static void testAllReduceMesh(int p)
{
    World w;
    w.reset(p, p, 1);
    initFullOnes(w);
    runPlane<Mesh, ReduceScatterPat>(w, identityPlane(p));
    // After RS, rank r holds reduced block r. Seed AG from that slot.
    World ag;
    ag.reset(p, p, 1);
    for (int r = 0; r < p; ++r) {
        std::memset(ag.ptr(r), 0, static_cast<size_t>(p));
        ag.ptr(r)[r] = w.ptr(r)[r];
    }
    runPlane<Mesh, AllGatherPat>(ag, identityPlane(p));
    for (int r = 0; r < p; ++r) {
        for (int b = 0; b < p; ++b) {
            if (ag.ptr(r)[b] != static_cast<uint8_t>(p)) {
                std::fprintf(stderr, "FAIL AR-Mesh rank %d block %d got %u\n", r, b, ag.ptr(r)[b]);
                std::exit(1);
            }
        }
    }
    std::printf("  %-28s p=%d (RS-Mesh + AG-Mesh) OK\n", "AllReduceClosMesh", p);
}

HCCL_REGISTER_RECIPE("AllGatherMeshOpbaseExecutor", "mesh", stepsMesh);
HCCL_REGISTER_RECIPE("AllGatherRingExecutor", "ring", stepsRing);
HCCL_REGISTER_RECIPE("AlignedAllGatherDoubleRingFor91093Executor", "double-ring", stepsDoubleRing);
HCCL_REGISTER_RECIPE("AllGatherNhr", "nhr", stepsNhr);
HCCL_REGISTER_RECIPE("AllGatherClosMesh", "clos+mesh", stepsClosMesh);

int main()
{
    std::printf("== topology primitives ==\n");
    runAG<Mesh>(8, "AllGatherMesh");
    runAG<Ring>(8, "AllGatherRing");
    runAG<Nhr>(8, "AllGatherNhr");
    runAG<ClosNhr>(8, "AllGatherClos(NHR)");
    runAG<ClosFullMesh>(8, "AllGatherClos(FullMesh)");
    runAG<DoubleRing>(8, "AllGatherDoubleRing");

    std::printf("== variable rank size ==\n");
    for (int p : {2, 3, 4, 5, 8}) {
        World w;
        w.reset(p, p, 1);
        initOwnBlock(w);
        runPlane<Ring, AllGatherPat>(w, identityPlane(p));
        checkAllGather(w, "Ring-varp");
        World w2;
        w2.reset(p, p, 1);
        initOwnBlock(w2);
        runPlane<Nhr, AllGatherPat>(w2, identityPlane(p));
        checkAllGather(w2, "Nhr-varp");
        World w3;
        w3.reset(p, p, 1);
        initOwnBlock(w3);
        runPlane<Mesh, AllGatherPat>(w3, identityPlane(p));
        checkAllGather(w3, "Mesh-varp");
        std::printf("  var-p=%d ring/nhr/mesh OK (nhr steps=%d ring steps=%d)\n",
            p, Nhr::steps(p), Ring::steps(p));
    }

    std::printf("== clos+mesh hierarchical ==\n");
    runClosMeshAG(8, 4, "AllGatherClosMesh-2x4");
    runClosMeshAG(8, 2, "AllGatherClosMesh-4x2");
    runClosMeshAG(16, 8, "AllGatherClosMesh-2x8");

    std::printf("== 3-level hierarchical ==\n");
    runHier3AG(8, 2, 2, "AllGatherHier3-2x2x2");
    runHier3AG(16, 4, 2, "AllGatherHier3-4x2x2");

    std::printf("== reduce / allreduce ==\n");
    testReduceScatterMesh(4);
    testReduceScatterMesh(8);
    testAllReduceMesh(4);

    std::printf("== recipe registry (algName contract) ==\n");
    const char* names[] = {
        "AllGatherMeshOpbaseExecutor",
        "AllGatherRingExecutor",
        "AlignedAllGatherDoubleRingFor91093Executor",
        "AllGatherNhr",
        "AllGatherClosMesh",
    };
    for (const char* n : names) {
        const RecipeInfo* info = RecipeRegistry::instance().find(n);
        expect(info != nullptr, n);
        std::printf("  %-48s family=%-12s steps(8)=%d\n", n, info->family.c_str(), info->steps(8));
    }

    expect(Mesh::steps(8) == 1, "mesh O(1)");
    expect(Ring::steps(8) == 7, "ring O(p-1)");
    expect(Nhr::steps(8) == 3, "nhr O(log p)");
    expect(Nhr::steps(5) == 3, "nhr non-pow2");
    expect(DoubleRing::steps(8) == 7, "double-ring same steps as ring");
    expect(Hierarchical<Mesh, ClosRing>::steps(8, 4) == Mesh::steps(4) + Ring::steps(2),
        "clos+mesh step additivity");

    std::printf("\nAll prototype checks passed.\n");
    return 0;
}
