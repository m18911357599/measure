# HCCL 现有算法注册、实现与调用框架

本文描述新 HCCL（`gitcode.com/cann/hccl`，分析提交 `b3a8fe9`）的算法栈，并对照旧 cann-hccl 的 Operator/Executor/Template 三层。路径均相对于新仓根目录，除非另行标明。

## 1. 分层总览

```text
公开 API (HcclAllGather / AllReduce / ...)
        │
        ▼
  Selector（按拓扑 / 数据量 / 引擎选算法名字符串）
        │  algName e.g. "AicpuAllGatherSoleMesh"
        ▼
  CollAlgExecRegistryV2  （algName → Executor 工厂）
        │
        ▼
  InsCollAlgBase  Executor
        │  CalcAlgHierarchyInfo / CalcRes / Orchestrate
        ├─ TopoMatch*     把 RankGraph 拆成逐层子通信域
        ├─ Channel 计算    Mesh1D / NHR / MultiJetty / MeshClos
        └─ InsAlgTemplate* 单层通信编排（DMA Write/Read + Notify）
                 │
                 ▼
          HCOMM Channel / Thread / Notify
```

对应目录：

| 层 | 目录 |
|---|---|
| 选择器 | `src/ops/<op>/selector/` + `src/ops/op_common/selector/` |
| 执行器 | `src/ops/<op>/algorithm/executor/` + `src/ops/op_common/algorithm/executor/` |
| 模板 | `src/ops/<op>/algorithm/template/` + `src/ops/op_common/algorithm/template/` |
| 拓扑匹配 | `src/ops/op_common/algorithm/topo_match/` |
| 拓扑信息 | `src/ops/op_common/topo_info/` |
| Channel | `src/ops/op_common/algorithm/executor/channel/` |

旧 cann-hccl 对应关系：

```text
REGISTER_OP(CMD, Operator)
    → Operator 按 AlgTypeLevel0/1/2 选 executor 名
    → REGISTER_EXEC("FooExecutor", tag, CollFooExecutor)
    → CollExecutor 调 AlgTemplateRegistry::GetAlgTemplate(TEMPLATE_*)
```

旧仓注册宏在：

- `src/domain/collective_communication/algorithm/impl/operator/registry/coll_alg_op_registry.h`（`REGISTER_OP`）
- `.../coll_executor/registry/coll_alg_exec_registry.h`（`REGISTER_EXEC`）

## 2. 算法注册

新仓同时存在 **V1 / V2** 两套表，商用路径以 V2 为主。

### 2.1 执行器注册（V2）

定义：`src/ops/op_common/algorithm/executor/registry/coll_alg_v2_exec_registry.h`

表结构：`HcclCMDType → algName → Creator`，Creator 返回 `InsCollAlgBase*`。

| 宏 | 实例化形态 | 用途 |
|---|---|---|
| `REGISTER_EXEC_V2(cmd, name, Exec, TopoMatch, Temp)` | `Exec<TopoMatch, Temp>` | 单层 Sole |
| `REGISTER_EXECUTOR_BY_TWO_TEMPS(..., T0, T1)` | `Exec<TopoMatch, T0, T1>` | Sequence / Parallel / Concurrent 两模板 |
| `REGISTER_EXECUTOR_BY_FOUR_TEMPS(..., T0..T3)` | 四模板 | 多层 Sequence / TwoShot |
| `REGISTER_EXEC_V2_MULTI(..., Ts...)` | 可变模板参数 | 3 层及以上 |
| `REGISTER_EXECUTOR_IMPL(cmd, name, Exec)` | 无模板实参 | Send 等特化类 |
| `REGISTER_EXECUTOR_IMPL_NO_TOPOMATCH(..., Temp)` | `Exec<Temp>` | 无拓扑匹配 |

静态初始化时还调用 `AddAlgToAllAlgos`，把「算法名 / 执行器类名 / 模板类名列表」写入全局 `AllAlgos`，供 CostModel 枚举候选。

示例（AllGather 单层 Mesh，CCU 引擎）：

```cpp
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLGATHER, CcuMSAllGatherSoleMesh2Die,
    InsV2AllGatherSoleExecutor, TopoMatchOneLevel, CcuTempAllGather2DiesMesh1D);
```

分析时统计（`src/`）：

| 宏 | 出现次数 |
|---|---|
| `REGISTER_EXEC_V2` | 94 |
| `REGISTER_EXEC_V2_MULTI` | 27 |
| `REGISTER_EXECUTOR_BY_TWO_TEMPS` | 52 |
| `REGISTER_EXECUTOR_BY_FOUR_TEMPS` | 21 |
| `REGISTER_EXEC`（V1） | 5 |
| `REGISTER_TEMPLATE_V2` | 21 |
| `REGISTER_TEMPLATE`（V1） | 6 |
| `REGISTER_ALG_ATTRS` | 198 |

### 2.2 算法属性注册

定义：`src/ops/op_common/selector/alg_attrs_registry.h` 的 `REGISTER_ALG_ATTRS`。

`AlgAttrs`（`alg_attrs.h`）描述「这个算法名在什么拓扑/引擎/数据类型下可跑」：

- `TopoAttrs`：`min/maxTopoLevelNum`、`supportLevel0Topos` 位图（`CLOS` / `MESH_1D` / `MESH_1D_CLOS`）、PCIe mix、最大 Rank、设备白名单、`topoCustomCheck`
- `OpAttrs`：inplace、保序、数据类型黑/白名单、`opCustomCheck`

`ParseAlgName` 从算法名字符串解析引擎前缀（`Aicpu` / `CcuSched` / `Aiv`…）与算法段（`Mesh` / `NHR` / `Concur` / `Sequence`…），再被 `REGISTER_ALG_ATTRS` 的语句覆盖。

Level0 拓扑位图：

```cpp
constexpr uint8_t LEVEL0_TOPO_CLOS         = 0x01; // Level0Shape::CLOS
constexpr uint8_t LEVEL0_TOPO_MESH_1D      = 0x02;
constexpr uint8_t LEVEL0_TOPO_MESH_1D_CLOS = 0x04; // 双 TopoInstance
constexpr uint8_t LEVEL0_TOPO_ANY          = 0xFF;
```

### 2.3 模板注册

- **V1**：`REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_RING, ScatterRing)` → `AlgTemplateRegistry` 按枚举取对象。
- **V2**：`REGISTER_TEMPLATE_V2("InsTempScatterNHRDPUInterNode", Class)` → `InsAlgTemplateRegistry` 按字符串取对象。

V2 主流并不走字符串取模板：Executor 把模板类型写进 **C++ 模板参数**，`Orchestrate` 里直接 `std::make_unique<InsAlgTemplate>()`。字符串注册主要用于 DFX / AllAlgos 列举。

### 2.4 选择器注册

`REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType, priority, XxxAutoSelector)` 把每算子的 `AutoSelectorBase` 挂到 `SelectorRegistry`。priority=18 为各算子默认 AutoSelector。

并行还有 `SelectorEngine`（CostModel + Tuner）：对白名单算子按代价表选最小 cost 的 algName，再按引擎前缀过滤。

## 3. 调用链

### 3.1 Host 侧主路径

`src/ops/op_common/op_common.cc` 中 `HcclExecOp`：

1. 若算法插件选中（`HCCL_ALGO_PLUGIN`），走 `PluginBroker::ExecuteAlg`，不再进内置 Executor。
2. 若存在 fallback ctx，改 algName 递归执行。
3. `CollAlgExecRegistryV2::GetAlgExec(opType, algName)` 构造 Executor。
4. `HcclGetAlgRes`：`CalcAlgHierarchyInfo` → `CalcRes` → 申请 Thread/Notify/Channel/Scratch，序列化为 `AlgResourceCtxSerializable`。可 `TryReuseResource`。
5. 按引擎分发：
   - **AICPU_TS / CPU**：`HcclAicpuKernelEntranceLaunch`，Device 侧再次 `GetAlgExec` + `Orchestrate`
   - **AIV**：`HcclAivKernelEntranceLaunch`
   - **CCU / 默认**：Host 直接 `executor->Orchestrate(param, resCtx)`

### 3.2 选择路径

两条并存：

```text
路径 A  AutoSelector（旧）
  ExecuteSelector::Run
    → SelectorRegistry 按 opType、priority 遍历
    → XxxAutoSelector::Select 写 selectAlgName

路径 B  SelectorEngine（新，CostModel）
  InitCostModel（通信域初始化时按 AlgAttrs 过滤拓扑）
    → 每次调用生成 CostTable（A/B/C/D α–β 模型）
    → Tuner 可改 cost
    → SelectMinCost 得到 algName
```

`SelectorEngine::IsOpSupported` 已覆盖 AllReduce / ReduceScatter / AllGather / Broadcast / Scatter / Reduce / AlltoAll* / Barrier / P2P 等。环境变量 `HCCL_ALGO` 可强制 level0/level1/level2 算法，此时自适应失效。

### 3.3 Executor 三个纯虚接口

`InsCollAlgBase`（`executor_v2_base.h`）：

| 接口 | 时机 | 职责 |
|---|---|---|
| `CalcAlgHierarchyInfo` | 资源计算前 | TopoMatch 把拓扑拆成 `AlgHierarchyInfoForAllLevel` |
| `CalcRes` | Host | 各层 Channel 请求 + 线程/notify/scratch 数量 |
| `Orchestrate` | Host 或 AICPU | Loop 切分 + 调 Template KernelRun |

可选：`CalcCostCoeff`、`GetAlgNetMeta`、`FastLaunch`（CCU 二次下发）。

### 3.4 Template 热路径

以 `InsTempAllGatherMesh1D` 为例：

1. `CalcRes` → `CalcChannelRequestMesh1D`，得到 `channelsPerRank`
2. `GetRes` → `threadNum ≈ (rankSize-1) * channelsPerRank`（多通道并发）
3. `KernelRun`：PreCopy（input→CCL/output）→ 对每个 peer DMA Write + Notify → PostCopy

NHR 模板则是 `GetNHRStepNum = ceil(log2(N))` 的步进循环，每步按 `NHRStepInfo` 收发。

## 4. 编排执行器矩阵（重复的根源）

V2 并不按拓扑复制 Executor，而是按 **编排策略 × 算子** 复制。RFC 0003 统计约 **53** 个继承 `InsCollAlgBase` 的集合通信执行器（不含 P2P）。

| 编排 | 语义 | 典型类 |
|---|---|---|
| Sole | 单层单模板 | `InsV2AllGatherSoleExecutor<TopoMatch, Temp>` |
| Sequence | 多层串行，前级输出喂后级 | `*SequenceExecutor`、`*Sequence*_3level` |
| Parallel | 数据按比例切开，两套模板同时跑 | `*ParallelExecutor`，常 50/50 Mesh∥NHR |
| Concurrent | Parallel 的特例：Mesh 平面 ∥ CLOS 平面，比例跟端口走 | `*ConcurrentExecutor` + `TopoMatchConcurrentV2` |
| OmniPipe | 两轴按 step 流水重叠 | `*OmniPipeExecutor` / `*OmniPipe2dExecutor` |
| TwoShot | AllReduce = RS Template → AG Template | `InsV2AllReduceTwoShotSoleExecutor` |

算法名语法：`{Engine}{Op}{Policy}{Prim…}`，例如 `AicpuAllGatherSequenceMeshConcurNHRNHR`。

代码规模（新仓 `src/ops`）：

- Executor `.cc`：**75** 个，约 **5.2 万行**
- Template `.cc/.h`：**197** 个文件，约 **8.1 万行**

新增一层拓扑（例如第 4 层 OCS）时，每个算子都要再做一个 Sequence/Parallel 变体，这是 RFC 0003 的直接动机。

## 5. Channel 与多通道

`channel.h` 按拓扑原语提供请求函数，而不是按算子：

- Mesh：`CalcChannelRequestMesh1D` / `FullMesh` / `Level0` / `Level1` / `Mesh2D`
- NHR：`CalcChannelRequestNhr` / `NhrMultiJetty` / `NhrMultiJettyUbx`
- CLOS 双平面：`CalcChannelRequestMeshClosMultiJetty`、`ProcessLinksForChannel` 可按 `priorityTopo` 选 Mesh 或 CLOS link

多通道本身不是独立算法族，而是 **Channel 维度**：

- AICPU Mesh/NHR：`channelsPerRank`、`PrepareDataSplitForMultiChannel`
- CCU：`*MultiJetty*`、`*MultiLink*`（常限制在 `LEVEL0_TOPO_CLOS` / `MESH_1D_CLOS`）
- 线程数随 channel 线性增加，Notify 数按线程配

## 6. 拓扑匹配

`TopoMatch*` 回答「当前 Rank 在每一层和谁一组」：

- `TopoMatchOneLevel` / `TwoLevel` / `ThreeLevel` / `3_level` / `Nlevel` / `Multilevel`
- `TopoMatchConcurrent` / `ConcurrentV2`（MESH_1D_CLOS）
- `TopoMatchUbx` / `Ubx1D` / `PcieMix` / `Squeeze2D`
- 试验：`experimental/.../topo_match_four_level.*`

`physicalLevels` 的互联形态必须回查，不能用层下标推断（`GetPhysicalLevelTopoType` 注释）。一层 `netLayer` 可能贡献一级或两级 TopoInstance。

## 7. 试验路径：RFC 0003 recursive_executor

`docs/zh/rfcs/0003-executor-template-refactor.md` 与 `experimental/ops/op_common/recursive_executor/`：

- 用 `HcclAlgorithm` 树描述 SEQUENCE / PARALLEL / OMNIPIPE
- 一个 `OpsExecutor` 递归解释，替代 53 个特化 Executor
- Mesh/NHR 抽成 `CommPlanner`，Template 只做 PreCopy / SendAll / PostCopy
- `REGISTER_ALG` 把 `AdaptorExecutor` 挂进现有 `CollAlgExecRegistryV2`
- 默认 **关闭**（`recursiveExecutorEnabled = false`），需编译开关 + `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR=true`

与本仓库重构方案的差异见 [03-template-composition-refactor.md](./03-template-composition-refactor.md)：RFC 是运行时解释器；本方案用编译期类型组合，并把 ring / double-ring / clos 收成一等原语。

## 8. 插件路径

`src/algo_plugin` + RFC 0002：Selector 可返回自定义算法名，`HcclExecOp` 改走 PluginBroker。内置注册表不感知插件实现。自定义算法仍建议复用 Template/CommPlanner，否则会再次复制编排。
