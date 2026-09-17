# HCCL 算法分析与模板化 + 组合重构设计

> 配套文档：[01-existing-algorithm-framework.md](./01-existing-algorithm-framework.md)  
> 可编译原型：`docs/hccl/prototype/`  
> 约束：**不降低步进热路径性能**（无额外虚调用、无逐步堆分配、任务图与现网一致）。

## 1. 问题：笛卡尔展开

算法层（Executor + Template）合计约 **76k 行、560+ 文件、128 个 Executor、90+ Template**。真正不同的数学对象只有少数几个：

| 真正变化的轴 | 现网取值 | 应有的抽象 |
|---|---|---|
| 集合原语 | AG / RS / AR / Bcast / Reduce / Scatter / Gather / A2A | 数据布局 + 是否 Reduce |
| 拓扑原语 | Mesh / Ring / NHR / HD / NB / Pairwise | 编译期 Topology |
| 互联织物 | HCCS Mesh、Clos 交换、RoCE、SuperPod | Fabric（Clos 是织物不是步进公式） |
| 通道复用 | 单流、Mesh 多流、双环、SDMA+RDMA、多 QP | ChannelPolicy |
| 分层 | L0×L1×L2、AHC 非对称 | Hierarchical 组合器 |
| 性能特化 | DMA 消减、对齐、atomic、pipeline、AIV | Trait / Policy，不是新算法 |

当前实现把这些轴在 **叶子类** 上做乘积。例如 AllGather 仅 Mesh 就有 Opbase / Graph / Pipeline / AIV / 跨机 AIV 五六份，Ring 又按 910 / 910B / 910_93 / 310P / ZeroCopy 再拆。新增一种拓扑（或一种通道）需要改所有 Op 的 Operator + Executor + Template。

AHC 已经用「组内 Template + 组间 Template」证明组合可行，但只覆盖非对称分层，没有覆盖 Mesh/Ring/NHR/Clos。

## 2. 设计原则

1. **拓扑与原语正交**：Ring 的 `p-1` 步邻居通信只写一次；AllGather / ReduceScatter 只替换「本步搬运的切片」和「是否 reduce」。
2. **组合发生在编译期**：`Hierarchical<Mesh, Nhr, Ring>`、`MultiChannel<Ring, DoubleRingPolicy>` 都是类型，内层 `Step()` 可内联。
3. **虚函数停在编排边界**：`SelectAlg` / `CalcResRequest` / `Orchestrate` 仍可虚。步进循环内禁止 `AlgTemplateBase*` 虚调用。
4. **任务图等价**：stream 数、notify 数、建链平面、Tx/Rx 序、barrier 位置与现网叶子算法一致，才能保证带宽与时延不回退。
5. **保留 algName 字符串**：AICPU 配置表、`HCCL_ALGO`、profiling tag 继续可用；内部由类型注册表生成同名 Executor。
6. **特化是 Policy 不是分叉**：DMA 消减、atomic、对齐双环通过 trait 切换地址与同步，不复制拓扑循环。

## 3. 算法族分析（按用户指定分类）

### 3.1 Mesh

**物理**：Server / Die 内 HCCS 全连接，任意两点直达，复杂度 O(1) 步、O(p-1) 并发链路。

**现网代码**：`COMM_TAG_MESH` + `AllGatherMesh*` / `ReduceScatterMesh*` / `AllReduceMesh*`，多从流（`streamNum = rankSize-1` 或 `rankSize-2`），`LocalNotify` 主从同步。变体：

- Direct：用户指针直达（DMA 消减）
- Atomic：目的端 atomic reduce，省同步
- Mix / SingleStream：降流数
- Opbase vs Graph：CCL buffer vs user mem

**重构**：一个 `MeshTopology` 原语 + `StreamPerPeer` Channel + `DmaReduce`/`AtomicReduce` Policy。AllGather Mesh 与 ReduceScatter Mesh 共享同一轮询对端的循环。

```
for round in 1 .. p-1:
    peer = Backward(rank, round)
    stream = streams[round] or main
    TxAck / RxAck on stream
    Put/Get slice[rank] ↔ slice[peer]
LocalNotify join
```

耗时：$\alpha + n\beta / p$（AG）或附加 $(p-1)n\gamma / p$（RS）。

### 3.2 Clos

**物理**：机间 / SuperPod 内的 Clos（胖树）交换。任意 server 对可通过叶子/脊交换机互通，但：

- 完全 FullMesh 并发会在 uplink 拥塞（「一打多」）
- 小消息延迟敏感，适合对数步算法（NHR/NB/HD）
- 大消息带宽敏感，适合 Ring 或分阶段 Pairwise

**现网**：没有 `CLOS` 类型。Clos 上跑的是：

| 规模 / 消息 | 现网选择 | Clos 原语映射 |
|---|---|---|
| 小、节点多 | NHR / NB / HD | `Clos<Nhr>` / `Clos<Nb>` / `Clos<Hd>` |
| 大、节点少 | Ring | `Clos<Ring>` |
| AlltoAll 大消息避一打多 | Pairwise staged | `Clos<Pairwise>` |
| AlltoAll 直达 | Direct FullMesh | `Clos<FullMesh>` |

**重构**：`Clos` 是 **Fabric trait**（提供「任意点对可达 + 拥塞模型」），不是第三套步进公式。默认步进由消息大小策略绑定到 NHR 或 Ring，与现网 `AutoSelectAlgTypeLevel1` 一致。

这样「在 Clos 上跑 NHR」写成 `Bind<ClosFabric, NhrTopology>`，而不是再复制一份 `all_gather_nhr_for_clos.cc`。

### 3.3 Clos + Mesh（分层流水）

**物理**：L0 = 机内 Mesh（高带宽 HCCS），L1 = 机间 Clos。CANN 文章中的细粒度分级流水：Server 间 Ring/NHR 的第 i 步结果，立刻在 Server 内 Mesh 传播，隐藏机内拷贝。

**现网**：

- 朴素分层：`KernelRun` 里先 L2、再 L1、再 L0（910_93 Ring Executor）
- Pipeline：`AllGatherMeshOpbasePipelineExecutor`、`ReduceScatterMeshOpbasePipelineExecutor`、GraphPipeline
- 并发：`*DoubleRingConcurrentExecutor`（机内双环 + 机间 SDMA/RDMA 并发）

**重构**：`Hierarchical<Mesh, ClosBound>`。两种调度策略：

| Policy | 语义 | 对应现网 |
|---|---|---|
| `SeqSchedule` | L2 → L1 → L0 串行 | 910_93 RingFor91093 |
| `PipelineSchedule` | L1 第 k 步 ∥ L0 Mesh 传播 | MeshOpbasePipeline |
| `ConcurrentSchedule` | L0 与 L1 按数据块比例同时跑 | RDMA+SDMA concurrent |

切片公式（AllGather）一次写在 Hierarchical 组合器里：

```
offset(l2, l1, l0) = ((l2 * n1 + l1) * n0 + l0) * chunk
```

现网每个 910_93 Executor 都在手写 `PrepareSlicesL0/L1/L2`，重构后所有 Op 共用。

### 3.4 Ring

**物理**：逻辑环，每 rank 只与 prev/next 通信，步数 `p-1`，每步 `n/p`。抗拥塞，适合 Clos 上中大消息、节点不太多。

**现网**：单环 Template ~400–500 行/Op；910_93 又包一层分层 Executor ~460–800 行。`AllReduceRing` 已组合 RS+AG，说明原语组合在 Template 层也出现过，但没有推广。

**重构**：`RingTopology` 只保留：

```
prev = (rank + p - 1) % p
next = (rank + 1) % p
for step in 0 .. p-2:
    send_slice = pattern.send_index(rank, step)
    recv_slice = pattern.recv_index(rank, step)
    Tx(next, send_slice); Rx(prev, recv_slice); [Reduce]
```

AllGather：`send = (rank - step) % p`  
ReduceScatter：方向相反且 Rx 后 Reduce。

### 3.5 Double-Ring

**物理**：910_93 / A3 机内两套反向环，数据对半切到两环，步数仍 `p-1` 但每步数据减半，吃满两套 HCCS/SIO 平面。`DOUBLE_RING_NUM = 2`，从流 1～3 条。

**现网**：`Aligned*DoubleRing*` Template 700+ 行；`CollCommExecutor::MultiRing*` 再做 slice/order；Semi-Ring / FastDoubleRing / Concurrent 又是近亲复制。

**重构**：Double-Ring = **Channel 策略**，不是新拓扑：

```
using DoubleRing = MultiChannel<RingTopology, RingPairPolicy>;
```

`RingPairPolicy`：

- 两条环的 rank 序（主环 0→1→…，副环反向或 nicList 重排）
- slice 对半、对齐（现网 Aligned 变体）
- 主从流 + notify（现网 `GetSubStreamInfoOnOneRing`）
- 可选：与 RDMA 平面再并发（Concurrent）

热路径仍是 `RingTopology::Step` 内联两次，外层只做切分。对齐拷贝、serial local copy 是 `LocalCopyPolicy`，对应现网 `TEMPLATE_REDUCESCATTER_DB_RING_SLC`。

### 3.6 NHR

**物理**：非均衡层次环。构造 `⌈log2 p⌉` 步通信关系，相邻物理节点流量更大，非 2 幂也不退化成 RHD 的额外拷贝。小消息可降为单棵树（oneshot）。

**现网**：NHR / NHR_V1 / NHR_Oneshot 在 AG/RS/AR/Bcast/Scatter/Reduce 各写一份；`NHRBase::GetRankMapping` 已是共享映射，但 Tx/Rx 循环仍复制。

**重构**：

```
NhrTopology::steps(p) -> vector<StepRec{peer, txSlices, rxSlices}>
```

映射只算一次；各 Op 提供 slice 合并规则（AG 合并已收到块，RS 相反）。V1 / Oneshot 是 `MappingPolicy`。NHR 常用在 Clos 的 L1/L2，因此默认 `Bind<ClosFabric, NhrTopology>`。

耗时（RS）：$\lceil log_2 p\rceil\alpha + \frac{p-1}{p}n(\beta+\gamma)$。

### 3.7 多 Channel

「Channel」在现网里至少有四层含义，重构必须拆开，避免再长出 `*Concurrent*For91093*` 叶子：

| Channel 种类 | 现网机制 | 组合器 |
|---|---|---|
| 多从流 | Mesh `rankSize-1` 条 stream | `StreamPerPeer` |
| 多环 | Double-Ring / 8P Ring 多 nic | `MultiRingOrder` |
| 双平面并发 | `COMM_LEVEL*_ANYPATH_SDMA/RDMA` | `SdmaRdmaSplit`（按 87/90% 切） |
| 多 QP / 多 NIC | `nicList`、多 qp 刷新 hugeData | `NicStripe` |

共同模式：**把 payload 切成 C 份，C 个独立拓扑实例各跑一份，用 notify 对齐阶段**。

```
MultiChannel<Topo, SplitPolicy>::run:
    chunks = SplitPolicy::split(slices, C)
    parallel for c in 0..C-1:
        Topo::run(plane[c], stream[c], chunks[c])
    join
```

`C=2` 且 Topo=Ring → Double-Ring。  
`C=2` 且一个 SDMA 一个 RDMA → Concurrent。  
`C=p-1` 且 Topo=Mesh → 现网 Mesh 多流。

### 3.8 分层算法（含 AHC）

**对称分层**（卡数一致）：L0 Mesh/Double-Ring × L1 Clos(NHR/Ring) × L2 Clos(NHR/Ring)。

**非对称分层 AHC**：分组大小不同，LCM 切片 + 逻辑同号卡。组内/组间仍是 NB/NHR/Ring。现网 AHC 已组合 Template，重构后变成：

```
AhcHierarchical<Inner=Nhr, Inter=Nhr>
```

Broke 变体只换分组对齐策略（`CommAHCAlignInfo` vs Broke）。

**AllReduce 分层经典拆分**（现网 AllReduce Ring/Mesh 都是这条）：

```
L0 ReduceScatter(Mesh|Ring) → L1 AllReduce(NHR|HD|Ring) → L0 AllGather(Mesh|Ring)
```

写成：

```
using AR = HierarchicalAllReduce<
    ReduceScatter<Mesh>, AllReduce<Nhr>, AllGather<Mesh>>;
```

这就是 CLOS+Mesh 上 AllReduce 的全部代码量级——类型别名，而不是三个 400 行 Executor。

## 4. 目标架构

```
                    ┌─────────────────────────────────────┐
                    │  CollAlgOperator (仍按 Op 选择)      │
                    │  输出：algName + 类型擦除后的 Recipe │
                    └──────────────────┬──────────────────┘
                                       │ 编译期 Recipe
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   Topology primitives          Channel combinators           Collective patterns
   Mesh / Ring / NHR            MultiChannel<>                AllGather<>
   ClosFabric + Bind<>          Hierarchical<>                ReduceScatter<>
   HD / NB / Pairwise           PipelineSchedule<>            AllReduce<> = RS+AG
                                AhcHierarchical<>             Broadcast<> = Scatter+AG
         └─────────────────────────────┬─────────────────────────────┘
                                       ▼
                          Executor adaptor（薄封装）
                          CalcResRequest ← Recipe::resource()
                          KernelRun      ← Recipe::run(ctx)
```

### 4.1 零开销原语接口

热路径只依赖下面这组 **trait**（概念），全部 `static` / 模板成员，无 vtable：

```cpp
template <class Transport>
struct TopologyConcept {
    static constexpr int kStepsHint;          // Mesh=1, Ring=p-1, NHR=ceil(log p)
    static int steps(int p);
    template <class Pattern, class Channels>
    static Status run(Ctx&, Pattern&, Channels&);
};

template <class Topo>
struct PatternConcept {   // AllGather / ReduceScatter / ...
    Slice send(int rank, int p, int step) const;
    Slice recv(int rank, int p, int step) const;
    static constexpr bool kReduceOnRecv;
};
```

`run` 内 for-step 由编译器对 `Mesh`/`Ring`/`Nhr` 特化生成，与手写循环同一数量级的 Tx/Rx 调用。

### 4.2 Recipe 类型即算法

```cpp
// 机内 Mesh AllGather（对应 AllGatherMeshOpbaseExecutor）
using AG_Mesh = AllGather<Mesh, StreamPerPeer, DmaReduceOn>;

// 机内双环（对应 AlignedAllGatherDoubleRingFor91093）
using AG_DR   = AllGather<MultiChannel<Ring, RingPairAligned>, DmaReduceOn>;

// CLOS 上 NHR（对应 level1 NHR）
using AG_ClosNhr = AllGather<Bind<ClosFabric, Nhr>>;

// CLOS+Mesh 分层（对应 910B Mesh + level1 NHR，或文章中的分级流水）
using AG_ClosMesh = Hierarchical<AllGather<Mesh>, AllGather<Bind<ClosFabric, Nhr>>>;

// 细粒度 pipeline
using AG_ClosMeshPipe = Hierarchical<AllGather<Mesh>, AllGather<Ring>, PipelineSchedule>;

// 对称三层 910_93
using AG_A3 = Hierarchical<
    AllGather<MultiChannel<Ring, RingPairAligned>>,
    AllGather<Bind<ClosFabric, Nhr>>,
    AllGather<Bind<ClosFabric, Ring>>>;

// AHC
using AG_AHC = AhcHierarchical<AllGather<Nhr>, AllGather<Nhr>, AhcAlign>;
```

新增「910_93 + ZeroCopy + NHR L1」不再复制 Executor，只改类型别名和资源 trait（ZeroCopy 改变 intra/inter 分段，不改变拓扑循环）。

### 4.3 注册表收敛

保留三层注册，但 **Executor 叶子不再手写**：

| 注册 | 现网 | 重构后 |
|---|---|---|
| `REGISTER_OP` | 按 `HcclCMDType` | 不变 |
| `REGISTER_EXEC(algName, Class)` | 128 个手写类 | `REGISTER_RECIPE(algName, Recipe)` 生成适配器 |
| `REGISTER_TEMPLATE` | 90+ 个类 | 仅保留需要 AICPU/自定义插件的擦除入口；内部默认走 Recipe |

Operator 的 `SelectAlg` 仍返回字符串（兼容 AICPU），`GetAlgExec` 用同一字符串找到 Recipe 适配器。

## 5. 各算法族的模板设计

### 5.1 Mesh

```cpp
struct Mesh {
    static int steps(int p) { return 1; } // 一轮全交换
    template <class Pattern, class Ch>
    static Status run(Ctx& c, Pattern& pat, Ch& ch) {
        const int p = c.rankSize;
        for (int r = 1; r < p; ++r) {
            int peer = backward(c.rank, p, r);
            auto& st = ch.stream(r - 1);          // 最后一轮回主流动
            ch.tx_ack(peer, st);
            put(c, peer, pat.send(c.rank, p, r), st);
            get(c, peer, pat.recv(c.rank, p, r), st);
            ch.rx_ack(peer, st);
        }
        ch.join();
        return Status::ok();
    }
};
```

Policy：

- `StreamPerPeer`：现网默认
- `SingleStream`：`ReduceScatterMeshMixSingleStream`
- `AtomicReduce`：目的端 atomic，去掉部分 ack
- `DmaReduceOn`：user ptr 作为 src/dst，对应 Opbase Direct

### 5.2 Ring 与 Double-Ring

```cpp
struct Ring {
    static int steps(int p) { return p - 1; }
    template <class Pattern, class Ch>
    static Status run(Ctx& c, Pattern& pat, Ch& ch) {
        int prev = (c.rank + c.rankSize - 1) % c.rankSize;
        int next = (c.rank + 1) % c.rankSize;
        for (int s = 0; s < c.rankSize - 1; ++s) {
            auto tx = pat.send(c.rank, c.rankSize, s);
            auto rx = pat.recv(c.rank, c.rankSize, s);
            tx_async(c, next, tx, ch.main());
            rx_async(c, prev, rx, ch.main());
            wait(c, ch.main());
            if constexpr (Pattern::kReduceOnRecv) reduce(c, rx);
        }
        return Status::ok();
    }
};

struct RingPairAligned {
    static constexpr int kChannels = 2;
    static RingOrder order(int ch, NicList const&);
    static void split(Slices const& in, Slices out[2]); // 对齐切半
};
```

Double-Ring 不出现独立 `for` 拓扑，只实例化两次 `Ring::run`（或展开为双流发行，notify 对齐每步，与现网 Aligned Double Ring 任务图一致）。**性能要点**：保持现网的「每步双环并发 + 对齐 offset」，不要退化成两次串行 Ring。

### 5.3 NHR

```cpp
struct Nhr {
    static int steps(int p) { return ceil_log2(p); }
    static Mapping map(int p, NhrVariant v); // V0 保序 / V1 / Oneshot
    template <class Pattern, class Ch>
    static Status run(Ctx& c, Pattern& pat, Ch& ch) {
        auto m = map(c.rankSize, Ch::variant);
        for (int s = 0; s < m.nsteps; ++s) {
            auto rec = m.rec(c.rank, s);
            tx_vector(c, rec.peer, pat.gather_tx(rec), ch.main());
            rx_vector(c, rec.peer, pat.gather_rx(rec), ch.main());
            wait(c, ch.main());
            if constexpr (Pattern::kReduceOnRecv) reduce_vec(c, rec);
        }
        return Status::ok();
    }
};
```

`GetRankMapping` 从现网 `NHRBase` 原样迁入（已验证正确性），禁止重写映射以免性能回退。

### 5.4 Clos 与 Clos+Mesh

```cpp
struct ClosFabric {
    static constexpr bool kAnyToAny = true;
    static constexpr bool kCongestion = true;
};

template <class Topo>
struct Bind {
    using Inner = Topo;
    template <class Pattern, class Ch>
    static Status run(Ctx& c, Pattern& p, Ch& ch) {
        return Topo::template run<Pattern, Ch>(c, p, ch);
    }
};

template <class L0, class L1, class Sched = SeqSchedule>
struct Hierarchical {
    static Resource resource(TopoInfo const&);
    static Status run(Ctx& c) {
        return Sched::template apply<L0, L1>(c);
    }
};
```

`SeqSchedule`：先 L1（Clos 平面 `COMM_LEVEL1`）再 L0（`COMM_LEVEL0` MESH）。  
`PipelineSchedule`：L1 每完成一块，L0 Mesh 立刻 allgather 该块（对应细粒度分级流水）。  
资源：L0 stream = `p0-1`，L1 stream 由内层拓扑决定；与现网 `CalcStreamNum` 对齐。

Clos 默认绑定策略（写入 Operator，而不是复制 Executor）：

```
if (msg < pipeline_min && servers > 8) L1 = Bind<ClosFabric, Nhr>;
else if (pow2(servers))               L1 = Bind<ClosFabric, Hd>;
else                                  L1 = Bind<ClosFabric, Ring>;
```

与现网 `AutoSelectAlgTypeLevel1` 数值阈值保持一致。

### 5.5 多 Channel 组合器

```cpp
template <class Topo, class Split>
struct MultiChannel {
    static Resource resource(TopoInfo const& t) {
        auto r = Topo::resource(t);
        r.streams *= Split::kChannels;
        r.planes  *= Split::kChannels;
        return r;
    }
    template <class Pattern, class Ch>
    static Status run(Ctx& c, Pattern& pat, Ch& ch) {
        auto sub = Split::bind(c, ch);
        Status st = Status::ok();
        // 发行到多 stream；无堆分配，sub 是栈上 view
        for (int i = 0; i < Split::kChannels; ++i) {
            auto pat_i = Split::slice_pattern(pat, i);
            st |= Topo::template run(sub.ctx(i), pat_i, sub.ch(i));
        }
        sub.join();
        return st;
    }
};
```

特化 `Split`：

- `RingPairAligned`：双环
- `SdmaRdmaSplit<87>`：AnyPath concurrent（现网 `BEST_SPLIT_VALUE_SR = 87`）
- `StreamPerPeer`：Mesh
- `NicStripe`：按 nicList 条带

### 5.6 集合原语 Pattern

```cpp
struct AllGatherPat {
    static constexpr bool kReduceOnRecv = false;
    Slice send(int rank, int p, int step) const {
        // Ring: (rank - step) % p ；Mesh: 本 rank 块
    }
    Slice recv(int rank, int p, int step) const;
};

struct ReduceScatterPat {
    static constexpr bool kReduceOnRecv = true;
    // 切片方向与 AG 对偶
};

template <class RS, class AG>
struct AllReducePat {
    static Status run(Ctx& c) {
        CHK(RS::run(c));
        CHK(AG::run(c));
        return Status::ok();
    }
};
```

Broadcast = Scatter + AllGather，Reduce = ReduceScatter + Gather，与现网文档中的耗时公式一致，代码也一致。

## 6. Executor / Operator 薄适配

叶子 Executor 变成：

```cpp
template <class Recipe>
class CollRecipeExecutor : public CollNativeExecutorBase {
    HcclResult CalcStreamNum(u32& n) override {
        n = Recipe::resource(topoAttr_).streams;
        return HCCL_SUCCESS;
    }
    HcclResult CalcLevel0CommInfo(...) override {
        return Recipe::template fill_plane<0>(...);
    }
    HcclResult KernelRun(OpParam const& p, ExecMem& m) override {
        Ctx ctx = bind_ctx(p, m, algResResp_);
        return Recipe::run(ctx);
    }
};

REGISTER_RECIPE("AllGatherMeshOpbaseExecutor", AG_Mesh);
REGISTER_RECIPE("AlignedAllGatherDoubleRingFor91093Executor", AG_DR);
REGISTER_RECIPE("AllGatherRingFor91093Executor", AG_A3);
```

`SelectAlg` **字符串表保持不变**，行为兼容。内部实现从「128 份 KernelRun」变成「128 行类型别名」。

Operator 侧仍按芯片分支，但只返回 algName；禁止在 Operator 里再复制一套拓扑判断（现网 910B/91093 的 AHC/Pipeline/FFTS 容量检查可收成 `Selector` 策略函数）。

## 7. 性能保障（不降性能的具体约束）

| 风险 | 约束 |
|---|---|
| 模板间接层加虚调用 | `Topology::run` / `Pattern::send` 全部 `static` 或 CRTP；禁止 `AlgTemplateBase*` 出现在 step 循环 |
| 每步 `std::vector` 分配 | Slice view 用 `Span`；NHR 映射在 `Prepare` 算一次，步进只读 |
| Double-Ring 退化成串行 | `MultiChannel` 必须先对所有 channel `TxAsync` 再统一 wait，与现网 Aligned 发行序一致 |
| Pipeline 分级流水依赖被破坏 | `PipelineSchedule` 按现网文章的「L1 步完成 → L0 Mesh 该块」的依赖边生成 notify，不得改成 bulk barrier |
| 内联导致 I-cache 膨胀 | 只对 Mesh/Ring/NHR/HD 四个拓扑 × 7 个 Pattern 实例化；芯片特化放在 Policy，不按 910/910B/910_93 复制拓扑 |
| 建链平面变化 | `Recipe::fill_plane` 必须输出与现网相同的 `CommParaInfo`（MESH vs RING_INNER vs NHR vs AHC） |
| AIV / AICPU | 不纳入本轮模板化热路径；algName 仍指向原 AIV Executor |

验收：同一 `HCCL_ALGO`、同一消息大小下，task 序列（stream id、link 对端、slice offset/size、reduce 与否）与重构前 diff 为空，再比带宽。

## 8. 代码量与开发工作量

按 AllGather+ReduceScatter+AllReduce 三条主路径估算（现网约 45k 行 Executor+Template）：

| 类别 | 现网 | 重构后 | 说明 |
|---|---:|---:|---|
| 拓扑原语 Mesh/Ring/NHR/HD/NB | ~15k（×Op 复制） | ~2.5k | 每种拓扑一份 |
| Double-Ring / Concurrent / 多流 | ~8k | ~1k | Channel 组合器 |
| 分层 L0/L1/L2 slice + KernelRun | ~10k（每个 910_93 Executor） | ~1.5k | Hierarchical 一份 |
| AHC | ~2.5k + 每 Op 薄封装 | ~2.5k（基本保持） | 已是组合 |
| Pattern（AG/RS/AR/…） | 与拓扑耦合 | ~1.5k | 切片公式 |
| Recipe 注册与 Selector | 128 个类文件 | ~128 个别名 + 一份 Selector | |
| 适配器 / 资源 | 散落在叶子 | ~1k 基类 | |

**主路径预计减少 60%–75% 行数**。新增一种拓扑（例如新 NHR 变体）只需 1 个 `Topology` + 挂到 Clos/Hierarchical，不再改 7 个 Op × 3 个芯片。

开发工作量从「复制最近的 Executor 再改 slice」变为「选 Topology × Channel × Schedule 填别名」。这与 AHC 已走通的 `GetInterAlgTemplateOpInstance` 同一方向，只是把运行期工厂换成编译期类型。

## 9. 迁移步骤

分四步，每步可单独合入且可用现网测试集对比 task dump：

1. **抽出 Topology 静态函数**  
   把 `AllGatherRing::RunAsync` 的循环收成 `Ring::run<AllGatherPat>`，Executor 仍走旧类，内部转调。用 AllGather Ring 单机对照。

2. **Pattern 对偶化**  
   ReduceScatter Ring 改为同一 `Ring::run<ReduceScatterPat>`。AllReduce Ring 改为 RS+AG 类型组合（现网已是运行期组合，改为类型组合）。

3. **Channel 组合器**  
   用 `MultiChannel<Ring, RingPairAligned>` 替换 `Aligned*DoubleRing`；用 `StreamPerPeer` 替换 Mesh 从流逻辑。`CollCommExecutor` 的 2381 行逐步删减。

4. **Hierarchical**  
   用组合器替换所有 `*For91093Executor` 的 L2/L1/L0 手写。Pipeline / Concurrent / AHC 作为 Schedule 插件接入。最后 `REGISTER_RECIPE` 替换 128 个叶子文件。

禁止「先推倒注册表再填实现」：algName 与建链平面是对外契约。

## 10. 与原型的对应

`docs/hccl/prototype/` 用 mock Transport 实现了：

- `Mesh` / `Ring` / `Nhr` / `ClosFabric`
- `MultiChannel`（Double-Ring、多流）
- `Hierarchical`（含 Clos+Mesh、三层）
- `AllGather` / `ReduceScatter` / `AllReduce` Pattern
- Recipe 注册（字符串 → 类型）

原型验证的是 **组合关系与步进次数/数据正确性**，不是 NPU 任务下发。正式迁移时把 mock `put/get` 换成现网 `link->TxAsync/RxAsync` 即可，拓扑循环不用重写。
