# 现有算法盘点与重复度分析

## 1. 关键结论

- **CLOS 不是集合算法**。它是 `CommTopo::COMM_TOPO_CLOS`，描述服务器间 / 超节点间的交换网络。集合算法跑在 Clos 平面上时，仍然是 Ring / NHR / NB / HD / AHC。
- **CLOS + Mesh** 是分层配方，不是新内核：L0 = 片上 1DMesh（FullMesh），L1 = Clos 上的稀疏算法。
- **多 channel** 已有实现，但散落在 `CollCommExecutor::MultiRing*` 与 AnyPath 双平面，没有类型化。
- 最大重复来自 **Ring step 循环** 和 **分层 Executor 的 KernelRun 骨架**，不是来自数学公式本身。

## 2. CLOS 在源码中的位置

`include/hccl/hccl_rank_graph.h`：

```cpp
typedef enum {
    COMM_TOPO_CLOS = 0,     // CLOS 互联
    COMM_TOPO_1DMESH = 1,   // 片上 1D Mesh
    COMM_TOPO_910_93 = 2,   // 910_93（HCCS + SIO）
    COMM_TOPO_310P = 3,
} CommTopo;
```

`RankGraph::GetInstTopoTypeByNetLayer`：

| 芯片 | L0 | L1 | L2 |
|---|---|---|---|
| 910B / 910 | `COMM_TOPO_1DMESH` | **`COMM_TOPO_CLOS`** | — |
| 910_93 | `COMM_TOPO_910_93` | **`COMM_TOPO_CLOS`** | **`COMM_TOPO_CLOS`** |
| 310P3 | `COMM_TOPO_310P` | — | — |

因此：

- **Mesh**：片上全连接（HCCS/SDMA），对应 `COMM_TAG_MESH` + `AllGatherMesh` / `ReduceScatterMesh*`
- **Clos**：机间/超节点间全双分带宽交换网，**允许任意置换通信**；算法侧仍选稀疏拓扑以降低步数或抗拥塞
- **Clos+Mesh**：`Hierarchical(L0=Mesh, L1=Ring|NHR|NB|HD|AHC)`，即 910B 多机的默认形态

源码中 **没有** `*clos*` 算法模板，也没有 fat-tree 步进实现。重构时 Clos 应建模为 **InterconnectPolicy**，而不是第四种 step engine。

## 3. 拓扑族 × 集合操作矩阵（HCOMM 模板 `.cc`）

| 拓扑引擎 | AG | AR | RS | Bcast | Scatter | Gather | Reduce | A2A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ring | 5 | 1 | 5 | 1 | 5 | 1 | 1 | — |
| Mesh | 4 | 3 | 5 | — | 1 | 1 | — | 2+2 |
| NHR | 1 | 2 | 1 | 2 | 1 | — | 1 | — |
| NHR_V1 | 1 | 1 | 1 | 1 | — | — | — | — |
| NB | 1 | 1 | 1 | 2 | 1 | — | — | — |
| HD / RHD | 3 | 7 | 3 | 3 | — | — | 2 | — |
| Double-ring / aligned | 1 | — | 2 | — | 1 | — | — | — |
| AHC | 2 | 2 | 2 | — | — | — | — | — |
| Pipeline | 4 | 2 | 3 | — | — | — | — | 4 |
| Star | — | — | — | 1 | — | 1 | — | — |
| Pairwise | — | — | — | — | — | — | — | 4 |

同一拓扑下，AG / RS / Scatter / Bcast 的 **步进循环几乎同构**，差别只在：

- 本步发送哪一片、接收落在哪一片
- 接收后是 `memcpy` 还是 `Reduce`
- 是否 root 门控（Bcast / Scatter / Reduce）

## 4. 各家族实现要点

### 4.1 Mesh / FullMesh

- 模板：`temp_all_gather/all_gather_mesh*.cc`、`reduce_scatter_mesh*.cc`、`all_reduce_mesh_*.cc`
- 建链：`CommMesh`，`COMM_TAG_MESH`
- 步进：对每个 peer 一轮（HCCL 用多 stream 并发，`meshStreams_` + `LocalNotify`）
- 适用：910B 节点内 HCCS 全连接；AlltoAll 的 direct fullmesh
- 变体膨胀：atomic / direct / mix / opbase / graph / AIV / small-count / DMA-elimination

### 4.2 Ring

- 模板：`all_gather_ring.cc`、`reduce_scatter_ring.cc`、`all_reduce_ring.cc`（**RS∘AG 组合**）、`broadcast_ring.cc`、`scatter_ring.cc`、`gather_ring.cc`、`reduce_ring.cc`
- 邻居：`prev = rank-1`，`next = rank+1`；`TxAsync` / `RxAsync` + `TxAck`/`RxAck`
- ReduceScatter 片号：首发 `rank-1`，首收 `rank-2`，之后递减
- AllGather 片号：首发 `rank`，首收 `rank-1`，之后 `ForwordRank`
- 变体：slim / concurrent-direct / zerocopy / 310P / 8P 多环

`AllReduceRing` 已经是组合，不应再为 NHR/NB 各写一份 AR 内核（它们同样是 RS∘AG，见 `all_reduce_nhr.cc`）。

### 4.3 Double-Ring

- HCOMM：`AlignedAllGatherDoubleRing`、`AlignedReduceScatterDoubleRing`，Executor `CollAligned*For91093Executor`；另有 semi / fast 薄子类
- cann-hccl：较老的 `*DoubleRingConcurrentExecutor`
- 拓扑：`ALG_LEVEL0_NP_DOUBLE_RING` / `TOPO_TYPE_NP_DOUBLE_RING`
- 本质：**2 channel × 反向环序 × 数据对半切分**，外加 910_93 SIO 对齐约束
- 重复：aligned AG↔RS 模板高度相似；semi/fast Executor 与 91093 ring Executor 骨架相同

### 4.4 NHR

- 基类：`NHRBase`（`GetStepNumInterServer`、`GetStepInfo`、`sliceMap_`）
- ReduceScatter 步公式（`reduce_scatter_nhr.cc`）：

```
nSteps = ceil(log2(n))
delta  = 1 << step
sendTo = (rank - delta) % n
recvFrom = (rank + delta) % n
nSlices = (n - 1 + (1<<step)) / (1<<(step+1))
```

- AllGather 的 `delta` 从最后一步往回数（`1 << (nSteps-1-step)`）
- 步数对数，适合 L1/L2 大规模；小数据走 `*_NHR_ONESHOT`
- NHR_V2 **不存在**；NHR_V1 是另一套 `GetRankMapping`

### 4.5 多 Channel

没有名为 MultiChannel 的算法族。实际机制：

| 机制 | 位置 | 通道含义 |
|---|---|---|
| `MultiRingAllReduce/ReduceScatter/AllGather` | `coll_comm_executor.h` | 8P_RING 4 环；`nicList` 决定环序 |
| `PrepareMultiRingSlice` | 同上 | 按环切 slice |
| AnyPath SDMA+RDMA | `COMM_LEVEL0_ANYPATH_*` | 双平面并发 |
| Double-ring | aligned 模板 | 2 环反向 |

`CollCommExecutor` 约 2300 行，是多通道的中枢，但每个集合操作各写一套 `MultiRing*`。

### 4.6 分层 / AHC / Pipeline

- **分层 Executor**：L0 RS → L1 AR（或 L1 RS + L2 AR + L1 AG）→ L0 AG。复制于 ring/mesh/91093/zerocopy/V。
- **AHC**：`AHCAlgTemplateBase`，组内 + 组间，Broke 为分组策略。AG/RS/AR 薄封装，重复度低，应保留。
- **Pipeline**：L0 mesh 与 L1 流量 overlap（多 stream + notify）。opbase / graph / A2A continuous 三套调度，可抽 `PipelineScheduler`。

## 5. 重复度与可消除量

按「若抽成模板+策略，可删的近拷贝」估计（不含 AIV 内核、310P 边角）：

| 优先级 | 热点 | 约可删 LOC | 抽法 |
|---|---:|---|
| P0 | Ring step（AG/RS/Scatter/Bcast/Gather/Reduce + concurrent-direct） | 4–6k | `TopologyKernel<RingTopo, OpPolicy>` |
| P0 | 分层 `KernelRun`（ring/mesh/DR × AG/AR/RS × 91093/zerocopy） | 5–8k | `HierarchicalAllReduce<L0,L1>` |
| P1 | aligned / semi / fast double-ring | 1–2k | `MultiChannel<2, RingKernel>` |
| P1 | Mesh AG/RS atomic/direct/mix | 2–3k | `TopologyKernel<MeshTopo, Op>` |
| P1 | Pipeline opbase AG/AR/RS | 1.5–2k | 调度器策略 |
| P2 | NHR/NB/RHD 各集合封装 | 1–2k | 同一 `GetStepInfo` + Op |
| P2 | AHC | 低 | 已组合，Broke 做策略开关 |

合计大约 **12–20k LOC** 可从算法目录消失，同时新算法从「复制一个 Executor+Template」变成「选 Topo × Op × 通道数 × 分层配方」。

## 6. 性能相关的现状约束

当前热路径：

```
RunAsync(虚) → for step: TxAck(虚) TxAsync(虚) RxAsync(虚) Reduce(虚) WaitDone(虚)
```

重构允许保留 **Transport 虚接口**（平台层，跨 SDMA/RDMA），但不应再叠加：

- 每 step 的 `virtual ApplyOp()`
- 每 byte 的 `std::function`
- 运行时多态的拓扑引擎（Ring/Mesh/NHR 在一次 KernelRun 内是确定的）

L1/L2 算法在 **KernelRun 开头** 用枚举选一次 Template（已经是运行时组合），选完后必须是单态类型。

下一篇：[03-refactor-design.md](03-refactor-design.md)
