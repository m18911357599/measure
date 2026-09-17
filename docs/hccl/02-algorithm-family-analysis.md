# 现有算法族分析

分析对象：新 HCCL `src/` + 旧 cann-hccl `algorithm/`。结论用于指导「按拓扑原语拆、按策略组合」的重构边界。

## 1. 总表

| 族 | 新 HCCL | 旧 cann-hccl | 本质 |
|---|---|---|---|
| Mesh | 主路径；AICPU/AIV/CCU 大量 `*mesh_1D*` | `*_mesh*` / `*_mesh_opbase*` / atomic | **叶子通信计划**：全互连，1 逻辑步、N-1 对端 |
| Clos | **无 `*clos*` 模板**；`Level0Shape::CLOS` + Channel/Cost | 无独立 Clos 模板 | **拓扑形态 / 交换结构**，不是叶子算法 |
| Clos+Mesh | `MESH_1D_CLOS` + Concurrent Mesh∥NHR | 无同名；近似 SIO/HCCS 双平面、double-ring concurrent | **Parallel 组合**：Mesh 平面 + CLOS 平面数据切开 |
| Ring | 基本退役（Scatter 残留） | L0/L1/L2 主路径 `*_ring*` | **叶子计划**：N-1 步、邻居交换 |
| Double-Ring | 无 | 910_93 aligned double-ring 关键路径 | **双环并发叶子**；新仓用 Mesh concurrent / MultiJetty 替代 |
| NHR | 跨 server / CLOS 主路径 | `nonuniform_hierarchical_ring_base` + `*_nhr*` | **叶子计划**：`ceil(log2 N)` 步 |
| 多 Channel | `channelsPerRank` / MultiJetty / MultiLink | stream / concurrent direct | **资源维度**，可叠加在任一叶子上 |
| 分层 | Sole / Sequence / Parallel / OmniPipe / TwoShot | `COMM_LEVEL0/1/2` 写在各 CollExecutor 里 | **组合策略**，不是拓扑 |

模板文件名匹配（新仓 `*/template/*`，含 CCU kernel）：

| 模式 | 数量（约） |
|---|---|
| `*mesh*` | 112 |
| `*nhr*` | 55 |
| `*ring*` | 2 |
| `*clos*` | 0 |
| `*double*` | 0 |

旧仓 `REGISTER_EXEC` **130**、`REGISTER_TEMPLATE` **98**，ring/mesh/nhr 按算子 × 产品形态展开。

## 2. Mesh

**实现**：`InsTemp*Mesh1D`、AIV/CCU `*mesh1d*`、Z-axis detour、one-shot/two-shot、meshchunk。Channel：`CalcChannelRequestMesh1D*`。

**选择**：`Aicpu*SoleMesh` 等，`supportLevel0Topos = MESH_1D | MESH_1D_CLOS`。Fullmesh 是 Channel 形态 / `HCCL_ALGO_TYPE_FULLMESH`，不是另一套模板族。2D 出现在 OmniPipe 2D 与 `CalcChannelRequestMesh2D`（die 轴），不是几何 2D mesh 算法库。

**重复**：同一套「对 N-1 个 peer 做 Write + Notify」按 AllGather / ReduceScatter / Broadcast / Scatter / Reduce / AlltoAll 和 AICPU/AIV/CCU 各写一遍。Cost 用 `CalcMeshParam`。

**性能要点**：peer 级并发、多 channel 切条、Pre/Post local copy、scratch = `rankSize` 个 slot。

## 3. Clos

**结论：CLOS 是 RankGraph 的 Fabric / `Level0Shape`，不是 CommPlanner。**

证据：

- `CommTopo::COMM_TOPO_CLOS`、`Level0Shape::CLOS`
- CostModel 在 `COMM_TOPO_CLOS` 上改端口共享与带宽
- `*NHRMultiLink` 的 AlgAttrs 限制 `LEVEL0_TOPO_CLOS`
- 两个仓都 **0** 个 `*clos*` 模板文件

跑在 CLOS 上的算法：

- **Mesh over switch**：全互连，Channel 走交换机链路
- **NHR over CLOS**：对数步，占用 CLOS 端口
- **MultiJetty**：一 rank 多 QP/Jetty 打满上行

重构时 Clos 应作为 **Channel/端口策略**（`isSwitchFabric=true`，channel 数 ≤ portBudget），而不是第七种叶子循环。

## 4. Clos + Mesh

`Level0Shape::MESH_1D_CLOS`：layer0 上 **两个 TopoInstance**（1DMesh + CLOS）。

执行：

- `TopoMatchConcurrentV2`
- `REGISTER_EXECUTOR_BY_TWO_TEMPS(..., MeshTemp, NhrTemp)`
- 名：`AicpuAllGatherConcurMeshNHR` 等
- 过滤：Mesh 规模与 CLOS 规模匹配，且 Rank 不超过 concurrent 上限

语义：**按端口比切开数据，Mesh 平面与 CLOS/NHR 平面同时发**，不是第三种拓扑原语。RFC 0003 把它归并进 `PARALLEL + dataSplitRatio`。

## 5. Ring

旧仓是 Server 内/间基础算法：`*_ring.cc`、`*_ring_direct`、`*_slim_ring`、`Coll*RingExecutor`、`*RingFor91093*`。步数 N-1，拥塞不敏感，小规模/中大数据量常用。

新仓几乎不用 Ring 做默认 L0（Server 内默认 Mesh）。残留：`scatter_ring.cc`、`ScatterRingFor91093Executor`。重构必须把 Ring 收回叶子计划器，否则旧产品与 `HCCL_ALGO=ring` 无法用同一套组合框架表达。

## 6. Double-Ring

旧仓 910_93 关键路径：`aligned_*_double_ring.cc`、`CollAligned*DoubleRingFor91093Executor`、dual ring order + subStream + LocalNotify + DMAReduce。

新仓 **无** double-ring 模板，带宽叠加改由：

- Mesh concurrent
- MultiJetty
- OmniPipe 双轴

重构把 Double-Ring 建模为 **双方向 Ring 叶子**（两环各带一半 payload，步数仍 N-1，并发度 2）。与 Clos+Mesh 的 Parallel 不同：Double-Ring 是同一拓扑上的双环，不是两个 Fabric。

## 7. NHR（Nonuniform Hierarchical Ring）

跨 Server / 超节点默认算法之一，步数 `ceil(log2 N)`。新仓：`ins_temp_*_nhr*`、DPU/CCU mem2mem、MultiJetty。旧仓：`nhr_base`、`*_nhr_v1`、oneshot。

非 2 次幂时部分步携带不均衡块（Nonuniform）。常把 `netType` 视为 CLOS 来算 cost。末步可直写 output（`CanReadLastStepToOutput`）。

**重复**：步进状态机（`GetNHRStepNum` / `NHRStepInfo`）在每个算子、每个引擎各一份。这是 CommPlanner 的首选抽取对象。

## 8. 多 Channel

叠加在 Mesh/NHR/CLOS 上：

| 机制 | 位置 |
|---|---|
| `channelsPerRank` 切条 | AICPU Mesh/NHR 模板 |
| MultiJetty | CCU NHR / AlltoAll，多端口 |
| `*MultiLink*` AlgAttrs | 常要求 CLOS |
| `GetUbMultiChannelNum` | op_common |

约束：Jetty/QP 上限、Notify/线程随 channel 涨、切条余量给最后一条 channel。重构用 `MultiChannel<N>` 策略类型，计划器只对每条 Transfer 做 stripe，不复制 Kernel。

## 9. 分层 / 多级

文档：`algo_intro.md`、`hierarchical_comm_principle.md`。

典型 AllReduce（两级）：

```text
Server 内 ReduceScatter（Mesh）
    → Server 间 AllReduce 或 ReduceScatter+AllGather（NHR）
    → Server 内 AllGather（Mesh）
```

AllGather：先内后外；ReduceScatter：先外后内。Broadcast ≈ Scatter + AllGather；Reduce ≈ ReduceScatter + Gather。

新仓用 Sequence/Parallel/OmniPipe/TwoShot **类**表达；旧仓把 `COMM_LEVEL0/1/2` 写进每个 CollExecutor。Four-level（+OCS）会把类数量再乘一截。

## 10. 算子 vs 原语

| 原语 | 回答的问题 | 被哪些算子用 |
|---|---|---|
| Mesh / Clos-as-mesh | 和谁全互连、一次搬多少 | AG/RS/Bcast/Scatter/Reduce/AlltoAll 叶子 |
| NHR | 对数邻接、每步块数 | 同上，偏 inter |
| Ring / DoubleRing | 线性邻接 / 双环 | 旧 L0/L1；AllReduce 常 RS-ring + AG-ring |
| Clos | 链路走哪类 Fabric、端口预算 | Channel + Cost，不出现在 Kernel 循环 |
| Sequence / Parallel / OmniPipe | 叶子怎么串/并/流水 | 所有集合算子编排 |
| MultiChannel | 一条逻辑边切几条物理通道 | 任意叶子 |

**AllReduce TwoShot** 今日有三种写法：专用大 Template、`TwoShotSole` 包 RS+AG、多层 Sequence。重构只保留 `Sequence<ReduceScatter(Topo), AllGather(Topo)>`。

## 11. 开发量痛点（量化）

1. **编排 × 算子**：~53 个 Executor，Orchestrate/CalcRes/同步大量拷贝；3 层→4 层每个算子再 +300–500 行。
2. **叶子 × 算子 × 引擎**：Mesh/NHR 计划重复；仅 mesh 文件 100+。
3. **AllReduce TwoShot** 把 RS/AG 再写进大模板。
4. **旧仓** 130 个 `REGISTER_EXEC`，产品形态（910B / 910_93 / 310P）再叉乘。

自然缝（与 RFC 0003 一致，并扩展 Ring/Clos）：

1. CommPlanner × {Mesh, Clos-channel, Ring, DoubleRing, NHR}
2. Template 骨架 PreCopy → Plan → SendAll/Reduce → PostCopy
3. 编排只保留 Sequence / Parallel / OmniPipe
4. TopoMatch 只负责子通信域
5. Clos 留在 topo/channel/cost
