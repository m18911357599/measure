# HCCL 算法重构设计：C++ 模板化 + 组合

> 在不降低性能的前提下，把 Mesh、CLOS、CLOS+Mesh、Ring、Double-Ring、NHR、多 Channel、分层算法收成正交策略，用编译期模板组合替代按算子/引擎/编排复制的实现。
> 基线见 [01-algorithm-framework-design.md](./01-algorithm-framework-design.md)。

---

## 1. 目标与非目标

### 1.1 目标

| 编号 | 目标 | 验收 |
|------|------|------|
| G1 | 拓扑原语与集合语义、引擎后端解耦 | Mesh/Ring/NHR 各一份 CommPlanner，AllGather/RS/AR 复用 |
| G2 | 编排通用化 | Sole/Sequence/Parallel/OmniPipe 各一份 Composer，不再按算子复制 |
| G3 | 编译期组合、热路径零虚调用 | 拓扑/Channel/引擎均为 template 参数；CRTP 或概念约束 |
| G4 | 新增一层拓扑 ≈ 追加一个类型别名 | 3 层→4 层不再复制 `*_3level` Executor |
| G5 | 接入现有 `HcclExecOp` | 仍注册到 `CollAlgExecRegistryV2`，Selector 只换算法名 |
| G6 | 源码与开发量下降 | 同构 Template 合并；新算法以 `using` + `REGISTER_COMPOSED_ALG` 完成 |

### 1.2 非目标

- 不改 HCCL 公开 API、不改 HCOMM primitive 语义
- 本阶段不替换 CostModel/Tuner 的选择策略（只让候选算法变为组合产物）
- 不把 CCU 微码指令集改成模板；引擎差异收敛在 `EngineBackend` 适配层
- 不在热路径引入 `std::function` / `std::variant` 递归解释器

### 1.3 与官方 recursive_executor RFC 的关系

| 维度 | RFC `OpsExecutor`（运行时树） | 本方案（编译期组合） |
|------|------------------------------|----------------------|
| 算法描述 | `HcclAlgorithm` 运行时 `variant` 树 | `using Alg = Sequence<Layer0, Layer1, ...>` |
| 执行器 | 1 个通用解释器 | 1 个 `ComposedExecutor<Alg>` 特化，编排循环可内联 |
| 灵活性 | 运行时拼 4 层树 | 编译期拼；新组合需重新编译 |
| 小消息开销 | 递归 + 堆分配，RFC 自承风险 | 无虚表、无 variant，与手写 Sequence 同级 |
| 落地 | `experimental/recursive_executor` | 可与 RFC 的 CommPlanner / ranks 归属契约共用 |

**选用编译期方案的原因**：用户约束是“不降低性能”。集合通信小消息路径对 Host/AICPU 标量开销敏感；编译期组合保持现有 V2 `Executor<Temp0,Temp1>` 的内联能力，同时把组合粒度从“整类 Template”降到“拓扑策略 + Channel 策略 + 引擎后端”。

RFC 的 **CommPlanner**、**ranksForInputData/OutputData**、**SEQUENCE/PARALLEL/OMNIPIPE** 语义直接复用，只是把运行时 `children` 向量换成模板参数包。

---

## 2. 正交轴：把复制来源拆开

现有类爆炸来自六个轴的笛卡尔积。重构后每个轴是独立 policy，算法是轴的笛卡尔**类型积**。

```text
Algorithm =
    Collective     ×   Topology      ×   ChannelPolicy   ×   Engine   ×   Orchestration   ×   Hierarchy
    AllGather          Mesh              Single              AICPU        Sole                1-level
    ReduceScatter      CLOS              MultiChannel        CCU_MS       Sequence            2/3/4-level
    AllReduce          CLOS+Mesh         MultiJetty          CCU_SCHED    Parallel            AHC
    Broadcast          Ring              MultiLink           AIV          OmniPipe
    Scatter            DoubleRing
    Reduce             NHR
```

约束：

- Topology 只负责 **peer 序列 + 每步切片归属**，不申请 Channel、不发 Notify
- ChannelPolicy 只负责 **同一 peer 上切几条 Channel、数据如何分条**
- Engine 只负责 **Write/Read/Reduce/Notify 的落地**
- Orchestration 只负责 **多层/多轴的顺序、切分、流水**
- Collective 只负责 **PreCopy/PostCopy/LocalReduce 的算子语义**（AllGather vs ReduceScatter 的归属变化）

---

## 3. 核心抽象

### 3.1 拓扑策略（Topology Policy）

拓扑策略是无状态的 traits + 静态算法，全部 `inline`/`constexpr`，可在 AICPU 上无异常使用。

```cpp
enum class TopoKind { Mesh, Clos, Ring, DoubleRing, Nhr };

struct StepPeer {
    u32 peerRank;      // 子通信域内逻辑 rank
    u32 distance;      // ring/nhr 步距；mesh 无意义
    u32 step;          // 第几步
};

template <typename T>
concept TopologyPolicy = requires(u32 n, u32 me, u32 step) {
    { T::kind } -> std::convertible_to<TopoKind>;
    { T::StepCount(n) } -> std::same_as<u32>;
    { T::PeersOfStep(me, n, step, /*out*/ std::declval<std::vector<StepPeer>&>()) };
};

// Mesh：一步、N-1 个 peer 全互连
struct Mesh1D {
    static constexpr TopoKind kind = TopoKind::Mesh;
    static constexpr bool concurrentPeers = true;
    static u32 StepCount(u32) { return 1; }
    static void PeersOfStep(u32 me, u32 n, u32 /*step*/, std::vector<StepPeer>& out) {
        out.clear();
        for (u32 p = 0; p < n; ++p) if (p != me) out.push_back({p, 0, 0});
    }
};

// Ring：N-1 步，每步 prev/next
struct Ring1D {
    static constexpr TopoKind kind = TopoKind::Ring;
    static constexpr bool concurrentPeers = false;
    static u32 StepCount(u32 n) { return n - 1; }
    static u32 Next(u32 me, u32 n) { return (me + 1) % n; }
    static u32 Prev(u32 me, u32 n) { return (me + n - 1) % n; }
    static void PeersOfStep(u32 me, u32 n, u32 step, std::vector<StepPeer>& out) {
        out.assign(1, StepPeer{Next(me, n), 1, step});
    }
};

struct ReverseRing1D : Ring1D {
    static u32 Next(u32 me, u32 n) { return Ring1D::Prev(me, n); }
    static u32 Prev(u32 me, u32 n) { return Ring1D::Next(me, n); }
};

// Double-Ring = 两个 Ring 策略的编译期对
struct DoubleRing1D {
    static constexpr TopoKind kind = TopoKind::DoubleRing;
    using Main = Ring1D;
    using Sub  = ReverseRing1D;
    static u32 StepCount(u32 n) { return Main::StepCount(n); }
};

// NHR：ceil(log2(N)) 步，距离 1<<step，非 2 幂走 NHRBase 同款重排
struct Nhr1D {
    static constexpr TopoKind kind = TopoKind::Nhr;
    static u32 StepCount(u32 n) { return n <= 1 ? 0 : 32 - __builtin_clz(n - 1); }
    static void BuildSliceMap(u32 n, bool keepOrder, std::vector<u32>& map);
    static void PeersOfStep(u32 me, u32 n, u32 step, std::vector<StepPeer>& out);
};

// CLOS：不是第三套步生成器，而是“在 CLOS 织物上跑哪个内嵌拓扑”
struct ClosFabric {
    static constexpr TopoKind kind = TopoKind::Clos;
    using DefaultInner = Nhr1D;          // 950 默认
    using FallbackInner = Ring1D;        // 小规模 / 拥塞
};
```

CLOS+Mesh 不是新的步公式，而是**同一 physical level 上两个 Topology 实例**：

```cpp
struct ClosPlusMesh {
    using MeshAxis = Mesh1D;
    using ClosAxis = ClosFabric::DefaultInner; // NHR
};
```

### 3.2 Channel 策略

```cpp
struct SingleChannel {
    static constexpr u32 kMinChannels = 1;
    static u32 ChannelCount(const ChannelInventory& inv) { return 1; }
    static void SplitSlice(const DataSlice& in, u32 /*ch*/, u32 /*nCh*/, DataSlice& out) { out = in; }
};

struct MultiChannel {          // HCCL_UB_MULTI_CHANNEL_NUM / 多 QP
    static u32 ChannelCount(const ChannelInventory& inv);
    static void SplitSlice(const DataSlice& in, u32 ch, u32 nCh, DataSlice& out);
};

struct MultiJetty : MultiChannel {};   // UB jetty 版，申请路径不同
struct MultiLink  : MultiChannel {};   // CLOS 多端口版，portNum[0]+portNum[1]
```

Template 热路径：

```cpp
for (u32 ch = 0; ch < ChannelPolicy::ChannelCount(inv); ++ch) {
    DataSlice s;
    ChannelPolicy::SplitSlice(full, ch, nCh, s);
    Engine::Write(threads[ch], channels[peer][ch], s);
}
```

编译期 `if constexpr (std::is_same_v<ChannelPolicy, SingleChannel>)` 可把循环折叠成一次 Write，避免多 Channel 场景外的开销。

### 3.3 引擎后端

```cpp
struct AicpuBackend {
    static constexpr OpExecuteConfig cfg = OpExecuteConfig::AICPU;
    static HcclResult Write(...);
    static HcclResult Read(...);
    static HcclResult Reduce(...);
    static HcclResult NotifyRecord(...);
    static HcclResult NotifyWait(...);
};

struct CcuMem2MemBackend { /* URMA 风格 mem2mem */ };
struct CcuMsBackend      { /* CCU_MS kernel 提交 */ };
struct AivBackend        { /* Vector Core */ };
```

后端**不**实现 Mesh/NHR。现有 `CcuTempAllGatherNHR1DMem2Mem` 里与 AICPU 重复的 step 循环删掉，只留 CCU 指令发射。

### 3.4 集合语义（Collective Trait）

CommPlanner 需要知道“这一步之后归属如何变”，这是算子语义，不是拓扑：

```cpp
struct AllGatherOp {
    // 输入：当前已有 owner 集合；输出：并上 peer 持有的 owner
    static void UpdateOwnersAfterStep(const StepPeer& peer,
        std::vector<u32>& owners);
    static constexpr bool needLocalReduce = false;
};

struct ReduceScatterOp {
    static void UpdateOwnersAfterStep(...); // 收敛到本 rank 负责的子集
    static constexpr bool needLocalReduce = true;
};

struct AllReduceTwoShot {
    using Phase0 = ReduceScatterOp;
    using Phase1 = AllGatherOp;
};
```

AllReduce TwoShot 不再是 700 行专用 Template，而是 `Sequence<Primitive<ReduceScatterOp, Mesh1D>, Primitive<AllGatherOp, Mesh1D>>`。

### 3.5 CommPlanner：拓扑 × 集合 → 收发描述

```cpp
template <typename Collective, typename Topology>
struct CommPlanner {
    static HcclResult Run(const DataParams& p,
                          const std::vector<u32>& ranks,
                          u32 myRank,
                          std::vector<u32>& ranksForOutput,
                          std::vector<TxRxSlicesList>& plans) {
        std::vector<u32> owners = p.ranksForInputData;
        const u32 n = static_cast<u32>(ranks.size());
        const u32 steps = Topology::StepCount(n);
        plans.resize(steps);
        std::vector<StepPeer> peers;
        for (u32 s = 0; s < steps; ++s) {
            Topology::PeersOfStep(LogicalRank(myRank, ranks), n, s, peers);
            BuildTxRxForStep<Collective>(p, ranks, myRank, peers, owners, plans[s]);
            for (auto& pe : peers) Collective::UpdateOwnersAfterStep(pe, owners);
        }
        ranksForOutput = std::move(owners);
        return HCCL_SUCCESS;
    }
};
```

一份 `CommPlanner<AllGatherOp, Nhr1D>` 同时服务 AICPU/CCU/AIV。这对应 RFC 已实现的 `RunMeshAllGather` / `RunNhrAllGather`，并扩展到 Ring / Double-Ring。

Double-Ring 特化：同 step 对 Main/Sub 各出一份 plan，交给 Parallel 轴或双 thread 组。

### 3.6 Primitive Template：PreCopy + Planner + Engine

```cpp
template <typename Collective,
          typename Topology,
          typename ChannelPolicy = SingleChannel,
          typename Engine = AicpuBackend>
class PrimitiveTemplate : public InsAlgTemplateBase {
public:
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& data,
                         TemplateResource& res) override {
        CHK_RET(PreCopyIfNeeded<Collective>(data, res));
        std::vector<TxRxSlicesList> plans;
        std::vector<u32> owners;
        CHK_RET((CommPlanner<Collective, Topology>::Run(data, ranks_, myRank_, owners, plans)));
        CHK_RET((SendAll<ChannelPolicy, Engine, Collective>(plans, res)));
        CHK_RET(PostCopyIfNeeded<Collective>(data, res, owners));
        return HCCL_SUCCESS;
    }
};
```

`SendAll` 对 Mesh 走全 peer 并发（`Topology::concurrentPeers == true`）；对 Ring/NHR 走按 step 的 Notify 栅栏。这些分支是 `if constexpr`，不会在 Mesh 热路径留下 NHR 的 step 循环。

类型别名即产品算法：

```cpp
using AicpuAllGatherMesh =
    PrimitiveTemplate<AllGatherOp, Mesh1D, SingleChannel, AicpuBackend>;

using AicpuAllGatherNhrMultiCh =
    PrimitiveTemplate<AllGatherOp, Nhr1D, MultiChannel, AicpuBackend>;

using CcuRsMeshMem2Mem =
    PrimitiveTemplate<ReduceScatterOp, Mesh1D, SingleChannel, CcuMem2MemBackend>;
```

### 3.7 编排组合器（Orchestration Composer）

用参数包代替 `*_3level` 复制。Composer 实现 `InsCollAlgBase` 三接口。

```cpp
// 单层
template <typename Primitive>
class SoleComposer : public InsCollAlgBase { /* CalcRes/Orchestrate 调一次 Primitive */ };

// 多层串行：前层 output owners = 后层 input owners
template <typename... Layers>   // Layer = {Primitive, SubCommIndex}
class SequenceComposer : public InsCollAlgBase {
    HcclResult OrchestrateLoop(...) {
        AlgoExecDataDesc st = InitFromInput();
        u32 i = 0;
        (RunOneLayer<Layers>(st, i++, /*isLast*/ i + 1 == sizeof...(Layers)), ...);
        return HCCL_SUCCESS;
    }
};

// 数据并行：按 dataSplitRatio 切 owners/slice
template <u32... Ratio>
struct SplitRatio { static constexpr u32 value[] = {Ratio...}; };

template <typename Ratio, typename... Axes>
class ParallelComposer : public InsCollAlgBase { /* PreSync → fold 各轴 → PostSync */ };

// OmniPipe：恰好两轴，编译期约束
template <typename AxisX, typename AxisY>
class OmniPipeComposer : public InsCollAlgBase {
    static_assert(!std::is_same_v<AxisX, void> && !std::is_same_v<AxisY, void>);
};
```

层描述把 Template 和子通信域下标绑在一起：

```cpp
template <typename Primitive, int SubComm>
struct Layer {
    using P = Primitive;
    static constexpr int kSubComm = SubComm;
};

using AicpuAgSeqMeshNhr = SequenceComposer<
    Layer<PrimitiveTemplate<AllGatherOp, Mesh1D>, 0>,
    Layer<PrimitiveTemplate<AllGatherOp, Nhr1D>,  1>>;

using AicpuAgSeqMeshNhrNhr = SequenceComposer<
    Layer<PrimitiveTemplate<AllGatherOp, Mesh1D>, 0>,
    Layer<PrimitiveTemplate<AllGatherOp, Nhr1D>,  1>,
    Layer<PrimitiveTemplate<AllGatherOp, Nhr1D>,  2>>;

// 4 层：只加一个 Layer，不复制 Executor 源文件
using AicpuAgSeq4 = SequenceComposer<
    Layer<PrimitiveTemplate<AllGatherOp, Mesh1D>, 3>,
    Layer<PrimitiveTemplate<AllGatherOp, Nhr1D>,  2>,
    Layer<PrimitiveTemplate<AllGatherOp, Nhr1D>,  1>,
    Layer<PrimitiveTemplate<AllGatherOp, Mesh1D>, 0>>;
```

`sizeof...(Layers)` 在 `CalcRes` 里展开申请每层 Channel；3 层与 4 层共享同一份 `SequenceComposer` 实现（一个 `.h` 里的折叠表达式）。

### 3.8 注册宏

```cpp
#define REGISTER_COMPOSED_ALG(cmd, name, ComposerType, TopoMatch, ...) \
    REGISTER_EXEC_V2(cmd, name, ComposedExecutorAdaptor, TopoMatch, ComposerType); \
    REGISTER_ALG_ATTRS(name, __VA_ARGS__)
```

`ComposedExecutorAdaptor<TopoMatch, ComposerType>` 把 Composer 接到现有 `InsCollAlgBase`，**不改 `HcclExecOp`**。

---

## 4. 分族重构设计

### 4.1 Mesh

**保留**：一步 all-to-all、`threadNum = N-1`、scratch × N、2Die Z 轴绕行作为 Mesh 的可选 `DetourPolicy`。

**删掉/合并**：

| 现状 | 重构后 |
|------|--------|
| `InsTempAllGatherMesh1D` / `InsTempReduceScatterMesh1D` / `InsTempScatterMesh1D` / `InsTempBroadcastMesh1D` 各一份切片 | `CommPlanner<Collective, Mesh1D>` 一份 |
| AICPU / CCU Mem2Mem / CCU MS / AIV 四套 Mesh kernel 各写 peer 循环 | 同一 Planner + 四个 `EngineBackend` |
| `MeshOneShot` / `MeshTwoShot` / `MeshChunk` | TwoShot = `Sequence<RS, AG>`；Chunk = `LoopPolicy`（外层 loop 已在 Executor） |
| `Mesh2Die` | `Mesh1D` + `TwoDieLayout` 策略（peer 集合按 die 分组，Z 轴 detour 是 Channel 拓扑而非新步公式） |

**性能**：Mesh 热路径保持“无 step 循环、peer 全并发”。`if constexpr (Topology::concurrentPeers)` 保证 Ring/NHR 代码不会编进 Mesh 实例。CCU Mem2Mem 仍可手写 kernel 内层搬运，但 **peer 列表由 Planner 传入**，避免 C++ 与微码各维护一份公式。

```cpp
using AicpuAllGatherSoleMesh =
    SoleComposer<PrimitiveTemplate<AllGatherOp, Mesh1D, SingleChannel, AicpuBackend>>;
```

### 4.2 CLOS

CLOS 不新增 `PeersOfStep`。它是 **Fabric 约束 + 默认内嵌拓扑 + Cost 模型**：

```cpp
struct ClosCostModel {
    static constexpr float kBw = 112.f; // 与 OMIN_CLOS_BW 对齐
    static float EqBw(u32 eqRankSize) { return kBw / (eqRankSize - 1); }
    static u32 RttStep(bool isPod) { return isPod ? 4 : 2; }
};

template <typename Inner = Nhr1D>
struct ClosOn { using Topology = Inner; using Cost = ClosCostModel; };
```

算法别名：

```cpp
using AicpuAllGatherSoleClosNhr =
    SoleComposer<PrimitiveTemplate<AllGatherOp, ClosOn<Nhr1D>::Topology, MultiLink, AicpuBackend>>;

using AicpuAllGatherSoleClosRing =  // 小规模 / HCCL_ALGO=ring
    SoleComposer<PrimitiveTemplate<AllGatherOp, ClosOn<Ring1D>::Topology, SingleChannel, AicpuBackend>>;
```

`REGISTER_ALG_ATTRS(..., topo.supportLevel0Topos = LEVEL0_TOPO_CLOS)`。Selector/Cost 继续用 CLOS 带宽，不再出现 `if (level0Topo == CLOS) select NHR` 的算子级拷贝——Attrs 已经表达“CLOS 上跑 NHR”。

### 4.3 CLOS + Mesh

映射为 **Parallel 两轴**，切分比来自端口数（Concurrent）或固定比（Parallel）：

```cpp
using MeshAxis = Layer<PrimitiveTemplate<AllGatherOp, Mesh1D, SingleChannel, AicpuBackend>, 0>;
using ClosAxis = Layer<PrimitiveTemplate<AllGatherOp, Nhr1D,  MultiLink,    AicpuBackend>, 1>;

using AicpuAgConcurMeshNhr = ParallelComposer<PortSplitRatio, MeshAxis, ClosAxis>;
using AicpuAgParallelMeshNhr = ParallelComposer<SplitRatio<1, 1>, MeshAxis, ClosAxis>;
```

UBX 判定（Mesh 数 == CLOS 数、CLOS 为 Mesh 倍数）留在 **AlgAttrs.topoCustomCheck**，不要写进 Composer。PCIE-SW（`level0PcieMix` 且 Mesh 非全连）用 Attrs 把 CCU Mesh 滤掉，AICPU Sequence/Parallel 自然接管。

OmniPipe 的 Mesh×CLOS 流水：

```cpp
using AicpuAgOmniPipeMeshNhr = OmniPipeComposer<MeshAxis, ClosAxis>;
```

现有 `InsV2AllGatherOmniPipeExecutor`（954 行）与 `AllReduce`/`ReduceScatter`/`Broadcast`/`Scatter` 的同构文件合并为这一份 `OmniPipeComposer`。

### 4.4 Ring

从 HCOMM `alg_template/temp_*_ring` 抽出步公式，放入 `Ring1D`：

```text
step s = 0..N-2
  AllGather:    recv from prev  the slice that prev owned at s-1; send to next own sliding window
  ReduceScatter: send to next; recv from prev + LocalReduce
  Broadcast:    root 沿环转发
```

HCCL V1 `ScatterRing` / `ScatterRingDirect` 并入 `PrimitiveTemplate<ScatterOp, Ring1D>`，Direct 作为 `DmaReducePolicy = true` 的模板参数（减少 round-trip，不改变 peer 序列）。

跨 Server 的 `HCCL_ALGO=level1:ring` 绑定：

```cpp
using Level1Ring = Layer<PrimitiveTemplate<AllGatherOp, Ring1D>, 1>;
using AgMeshThenRing = SequenceComposer<Layer<...Mesh1D, 0>, Level1Ring>;
```

这样 950 新栈与 A2/A3 Ring 走同一 Planner，HCOMM 里按算子复制的 `all_gather_ring.cc` / `reduce_scatter_ring.cc` 可逐步退役。

**性能**：Ring 步循环是算法固有的 `N-1`，不能消掉。组合器不得在每步之间插入额外虚调用或 `std::vector` 重新分配——`PeersOfStep` 对 Ring 返回 1 个 peer，可放栈上 `StepPeer[1]`。Planner 提供 `Ring` 特化，避免通用 `vector` 路径。

```cpp
template <typename Collective>
struct CommPlanner<Collective, Ring1D> {
    static HcclResult Run(...) {
        StepPeer peer;
        for (u32 s = 0; s < n - 1; ++s) {
            peer = {Ring1D::Next(me, n), 1, s};
            EmitStep<Collective>(..., peer);   // 栈上，无 heap
        }
    }
};
```

### 4.5 Double-Ring

定义为 **同 step 双环 Parallel**，而不是新的步公式：

```cpp
template <typename Collective, typename ChannelPolicy, typename Engine>
using DoubleRingPrimitive = ParallelComposer<
    SplitRatio<1, 1>,
    Layer<PrimitiveTemplate<Collective, Ring1D,        ChannelPolicy, Engine>, 0>,
    Layer<PrimitiveTemplate<Collective, ReverseRing1D, ChannelPolicy, Engine>, 0>
>;
```

910_93 的 aligned 切片、`DOUBLE_RING_STREAM_NUM = 3`（主流 + 两环）放入 `DoubleRingResourcePolicy`：

```cpp
struct AlignedDoubleRingResource {
    static constexpr u32 extraMainThreads = 1;
    static constexpr u32 rings = 2;
    static void AlignSlices(u64 count, u32 n, std::vector<Slice>& main, std::vector<Slice>& sub);
};
```

`AlignedAllGatherDoubleRingFor91093Executor` 变为：

```cpp
using AlignedAgDoubleRing = SoleComposer<
    PrimitiveTemplate<AllGatherOp, DoubleRing1D, SingleChannel, AicpuBackend,
                      AlignedDoubleRingResource>>;
```

`DoubleRing1D` 的 Planner 特化在同一步同时 emit 两条 TxRx，避免真正跑两遍 `N-1` 串行。这与手写 double-ring 的时序一致，带宽路径保持双环并发。

FastDoubleRing（910_93 大数据）若有额外 pipeline 与 DMA Reduce，用 `DmaReducePolicy` + `PipelinePolicy` 叠加，而不是第三套 Executor。

### 4.6 NHR

把 HCOMM `NHRBase::GetRankMapping` / `GetStepNumInterServer` 和 HCCL `InsTemp*NHR` 的 step 循环收成 `Nhr1D`。

分层：

| 模块 | 职责 |
|------|------|
| `Nhr1D::BuildSliceMap` | 非 2 幂重排（现 `NHRBase`） |
| `Nhr1D::PeersOfStep` | 第 s 步左右 peer |
| `CommPlanner<Collective, Nhr1D>` | 每步 Tx/Rx 切片 + owners 更新 |
| `LastStepOverlapPolicy` | 末步 Rx 与 PostCopy 并行（现 `CHANNEL_DOUBLE_FACTOR`） |
| `ChannelPolicy` | Single / MultiChannel / MultiJetty |

`NHR_V1` 若仍需兼容，做成 `NhrV1` 策略（根复杂度距离），不要在 `InsTemp*NHR` 里 `if (v1)`。

NHR 热路径允许 `log N` 次同步，这是算法复杂度；组合器只保证：

- step 间栅栏与手写 Template 相同（每 step NotifyWait 全集 peer）
- MultiChannel 切分在 Planner 之后、Engine 之前，不增加 step 数
- `ranksForOutputData` 在末步之后一次性给出，供 Sequence 下一层使用

### 4.7 多 Channel

独立于拓扑。任何 `PrimitiveTemplate<..., ChannelPolicy>` 可切换：

```text
Mesh  + SingleChannel     → 每 peer 1 条，N-1 线程
Mesh  + MultiChannel(k)   → 每 peer k 条，线程 k*(N-1) 或 k 条复用
NHR   + MultiChannel(k)   → 每 step 每 peer k 条
Ring  + MultiChannel(k)   → 每步 next/prev 各 k 条（少见，但零成本可表达）
CLOS  + MultiLink         → portNum 求和作为 k
UBX   + MultiJetty        → CalcChannelRequestNhrMultiJettyUbx
```

`HCCL_UB_MULTI_CHANNEL_NUM` 在 `MultiChannel::ChannelCount` 读取，SingleChannel 编译期不碰该环境变量。

资源申请：

```cpp
template <typename Topology, typename ChannelPolicy>
u32 ThreadNum(u32 rankSize, const ChannelInventory& inv) {
    const u32 ch = ChannelPolicy::ChannelCount(inv);
    if constexpr (Topology::concurrentPeers) {
        return (rankSize - 1) * ch;          // Mesh
    } else {
        return ch;                            // Ring/NHR：按 channel 流水，step 串行
    }
}
```

这替换了 NHR Template 里 `isMultiChannel ? portNum.size() : 1` 与 Mesh Detour 里 `remotes * (isClosMultiChannel ? 2 : 1)` 的散落公式。

### 4.8 分层算法

分层 = `SequenceComposer` + `TopoMatchNLevel`，叶子是任意拓扑策略。

```text
2 层 AllGather:   Sequence(Mesh@L0, NHR@L1)
3 层 AllGather:   Sequence(Mesh@L0, NHR@L1, NHR@L2)
4 层 AllGather:   Sequence(Mesh@L3, NHR@L2, NHR@L1, Mesh@L0)   // 与 RFC 示例同构
3 层 AllReduce:   Sequence(RS-Mesh@L0, RS-NHR@L1, AG-NHR@L1, AG-Mesh@L0)
AHC:              Sequence(InnerCollective@group, Concat@bridge)  // 非对称分组作为特殊 TopoMatch
```

数据契约（编译期与 RFC 一致）：

```text
next.ranksForInputData = previous.ranksForOutputData
中间层 outputBuffer = CCL
末层 outputBuffer   = OUTPUT
```

`SequenceComposer` 用折叠表达式展开，层数是编译期常量。4 层不会比 3 层多一份源文件，只多一个 `using` 与一行 `REGISTER_COMPOSED_ALG`。

AHC（Asymmetric Hierarchical Concatenate）今天在 HCOMM `AlgTypeLevel1::AHC`。重构时 AHC 是 **TopoMatch**（非对称子组）+ Sequence，而不是新的 Mesh 变体。子组内仍复用 Mesh/NHR/Ring Primitive。

Pipeline（`HCCL_ALGO=pipeline`，机内与机间链路并发）映射为 `ParallelComposer` 或 `OmniPipeComposer`，与 CLOS+Mesh 共用编排，不再为 pipeline 单独写 Executor。

---

## 5. 目录与模块边界

建议在 HCCL 仓新增（可先放 `experimental/`，成熟后进 `src/ops/op_common`）：

```text
src/ops/op_common/algorithm/compose/
├── topology/
│   ├── mesh_1d.h
│   ├── ring_1d.h
│   ├── double_ring_1d.h
│   ├── nhr_1d.h              // 迁入 NHRBase 重排
│   └── clos_fabric.h
├── channel/
│   ├── single_channel.h
│   ├── multi_channel.h
│   └── multi_jetty.h
├── engine/
│   ├── aicpu_backend.h
│   ├── ccu_mem2mem_backend.h
│   ├── ccu_ms_backend.h
│   └── aiv_backend.h
├── collective/
│   ├── all_gather_op.h
│   ├── reduce_scatter_op.h
│   ├── scatter_op.h
│   ├── broadcast_op.h
│   └── reduce_op.h
├── planner/
│   ├── comm_planner.h        // 主模板 + Mesh/Ring/NHR 特化
│   └── owners.h              // ranksForInput/Output
├── template/
│   └── primitive_template.h
├── orchestrate/
│   ├── sole_composer.h
│   ├── sequence_composer.h
│   ├── parallel_composer.h
│   └── omnipipe_composer.h
└── register/
    └── register_composed_alg.h
```

各算子目录退化为：

```text
src/ops/all_gather/algorithm/composed_algs.cc   // using + REGISTER_COMPOSED_ALG 列表
src/ops/all_gather/selector/                    // 逐步改为只调 CostModel
```

不再出现 `ins_v2_all_gather_sequence_executor_3level.cc` 这种按层复制的文件。

---

## 6. 性能约束（必须守住）

| 风险 | 约束 |
|------|------|
| 模板泛化导致 Ring 走 `vector<StepPeer>` | Ring/NHR 提供 Planner 特化，peer 固定 1～2 个走栈 |
| Composer 折叠表达式在 AICPU 上代码膨胀 | Primitive 的 `KernelRun` 放 `.cc` 显式实例化；Composer 头文件只做层间编排 |
| 多 Channel 的 `if` 污染 SingleChannel | `if constexpr` + SingleChannel 特化 `SendAll` |
| Sequence 层间多余同步 | 与现网一致：仅在 `sizeof...(Layers) > 1` 时插 Pre/PostSync；Sole 零额外同步 |
| CCU 微码路径变慢 | CCU Backend 仍提交现有 kernel；Planner 只在 Host/AICPU 算 plan，可预计算进 FastLaunch ctx |
| 小消息标量开销 | 禁止热路径 `std::function`、虚函数、`std::map`；Channel 表继续用现有 `map<u32, vector<ChannelInfo>>` 直到单独优化 |

对照方法：同一 CostModel 选中的算法，重构前后 Profiling 的 task 数、Notify 序、数据切片必须一致。允许 Host 编排 C++ 指令数下降，不允许通信步数上升。

显式实例化清单（写在各 `composed_algs.cc`，控制 AICPU 二进制体积）：

```cpp
template class PrimitiveTemplate<AllGatherOp, Mesh1D, SingleChannel, AicpuBackend>;
template class PrimitiveTemplate<AllGatherOp, Nhr1D,  MultiChannel,  AicpuBackend>;
template class SequenceComposer<Layer<...,0>, Layer<...,1>>;
// ... 只实例化 CostModel 会选到的组合
```

未实例化的组合不会进入 `libhccl.so`。

---

## 7. 源码量与开发量估算

以 AllGather 为例（数量级，用于说明结构，不是精确 loc）：

| 现状 | 重构后 |
|------|--------|
| Sole/Sequence/Parallel/Concurrent/OmniPipe/2d/3level Executor 约 7 个文件、~5000 行 | `composed_algs.cc` 注册表 ~200 行 + 共用 Composer |
| AICPU Mesh + NHR Template ~1000 行；CCU 再各 500+ | Planner ~400 行 + 4 个 Backend 适配 |
| Selector 内 MESH/CLOS/UBX 分支 ~400 行 | Attrs lambda 下沉，Selector 只留引擎优先级 |

全算子推广后，编排层从“每算子每编排一份”变为 4 个 Composer；拓扑层 5 个策略（Mesh/CLOS-as-fabric/Ring/DoubleRing/NHR）；Channel 3 个；引擎 4 个。新算法的开发路径：

```text
1. 若只需新组合：using + REGISTER_COMPOSED_ALG + Attrs   （小时级）
2. 若需新拓扑（如 Recursive Doubling）：实现 TopologyPolicy + Planner 特化
3. 若需新引擎：实现 EngineBackend
```

这对应 RFC 的场景 A/B，但组合发生在编译期。

---

## 8. 迁移步骤

采用**按拓扑族切入、注册表双轨**，避免一次替换 119 个 V2 注册点。

### Phase 0 — 基础设施（不改选择结果）

1. 落地 `topology/` + `CommPlanner<AllGatherOp, Mesh1D/Nhr1D>`，用现有 experimental planner 对拍切片
2. 落地 `PrimitiveTemplate` + `SoleComposer`，先注册**影子算法名** `AicpuAllGatherSoleMeshV3`，用 UT 对比 task dump
3. `REGISTER_COMPOSED_ALG` 接到 `CollAlgExecRegistryV2`

### Phase 1 — Mesh / NHR / 分层 Sequence（AllGather）

1. 用 `SequenceComposer` 替换 `InsV2AllGatherSequenceExecutor` 与 `_3level`
2. Selector 在 2/3 层分支改指向新名（或同名替换，旧 `.cc` 删除）
3. 对拍：单层 Mesh、单层 NHR、2 层 Sequence、3 层 Sequence；count 整除/不整除；多 Loop

### Phase 2 — CLOS、CLOS+Mesh、多 Channel

1. `MultiChannel` / `MultiJetty` / `MultiLink` policy
2. `ParallelComposer` 替换 Concurrent/Parallel Executor
3. UBX / PCIE-SW 条件只留在 Attrs
4. `HCCL_UB_MULTI_CHANNEL_NUM` 接到 `MultiChannel::ChannelCount`

### Phase 3 — Ring / Double-Ring 迁入新栈

1. 从 HCOMM 抽出 `Ring1D` / `DoubleRing1D` Planner，HCCL V1 ScatterRing 改用 Primitive
2. `AlignedDoubleRing` 用 Dual-plan 特化替换 910_93 专用 Executor
3. `HCCL_ALGO=level1:ring` 绑定 `ClosOn<Ring1D>` 或 Layer1 Ring

### Phase 4 — 其余算子与 OmniPipe / TwoShot / AHC

1. ReduceScatter / AllReduce / Broadcast / Scatter 只加 Collective trait + 注册表
2. AllReduce TwoShot = `Sequence<RS, AG>`
3. OmniPipeComposer 替换各算子 OmniPipe 大文件
4. AHC TopoMatch 接入 Sequence

每阶段要求：旧算法名可并行注册，环境变量 `HCCL_COMPOSED_ALG=1` 切换；对拍失败则切回旧注册。

---

## 9. 测试与对拍

| 级别 | 内容 |
|------|------|
| Planner UT | 给定 ranks/myRank/count，Mesh/Ring/NHR 的 `TxRxSlicesList` 与 golden 一致 |
| Composer UT | Sequence owners 传递；Parallel 余量落在最后一轴；OmniPipe steps≤5 |
| 引擎 ST | AICPU/CCU/AIV 各跑 Sole Mesh、Sole NHR、Sequence Mesh+NHR |
| 拓扑 ST | `MESH_1D` / `CLOS` / `MESH_1D_CLOS`（含 pcieMix） |
| 多 Channel ST | `HCCL_UB_MULTI_CHANNEL_NUM=1/2/4` 结果一致、带宽不回退 |
| Ring ST | 单环、双环 aligned、与 910_93 旧 Executor bit 级任务序对比（允许 Notify 编号重排，不允许步数变化） |
| 回归 | `build.sh -u`；Selector 在默认 CostModel 下选中的算法集合不变（Phase 1–2） |

性能门禁：小消息（≤1MB）时延回退 < 2%；大消息带宽回退 < 1%。超出则查 Composer 是否引入额外 Notify 或未走 Planner 特化。

---

## 10. 风险与否决项

| 风险 | 应对 |
|------|------|
| 模板实例化导致 AICPU so 膨胀 | 显式实例化清单；Composer 与 Primitive 分离编译 |
| 双轨注册遗漏 Attrs，CostModel 选空 | 注册宏强制 Attrs；CI 扫描“有 EXEC 无 ATTRS” |
| Double-Ring 对齐切片语义与通用 Parallel 不完全等价 | DoubleRing Planner 特化，不复用通用 Parallel 切分 |
| CCU kernel 仍手写一份 Mesh | Phase 1 允许 Backend 内保留 kernel；Phase 4 再把 peer 列表参数化 |
| 与 RFC recursive_executor 两套组合模型并存 | CommPlanner / owners 契约共享；只保留一套 Executor（本方案 Composer），RFC 解释器不进生产热路径 |

**否决**：

- 用运行时 `vector<unique_ptr<Template>>` 解释 Sequence 作为生产默认（违反 G3）
- 继续为 4 层拓扑复制 `*_4level.cc`
- 把 CLOS 再实现成第三套 peer 循环（CLOS 只作为 Fabric + Inner Topology）

---

## 11. 结论

现有 HCCL V2 已经用 `Executor<TopoMatch, Temps...>` 做了浅层模板化，但组合粒度停在“整类 Template”，拓扑、Channel、引擎、编排仍按算子复制。

重构把算法拆成五类可组合策略：

1. **Mesh** — 一步全互连 Planner  
2. **CLOS** — Fabric + 默认 Inner（NHR/Ring）  
3. **CLOS+Mesh** — Parallel/OmniPipe 两轴  
4. **Ring / Double-Ring** — 单环步公式 + 双环同 step 双 plan  
5. **NHR** — 对数步 Planner  
6. **多 Channel** — 正交 ChannelPolicy  
7. **分层** — `SequenceComposer<Layer...>` + TopoMatchNLevel  

用 C++ 模板参数包在编译期展开编排，热路径与手写代码同级；新增层级或组合只加类型别名与注册，从结构上消掉 Executor/Template 的笛卡尔积。
