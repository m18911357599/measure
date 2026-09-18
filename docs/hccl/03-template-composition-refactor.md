# 重构设计：C++ 模板化 + 组合

目标：在 **不降低热路径性能** 的前提下，把算法源码从「算子 × 编排 × 拓扑 × 引擎」笛卡尔积，改成 **拓扑原语 × 通信语义 × 组合策略** 的直积；新增一层拓扑或一种并发切分时，只加类型别名，不复制 Executor。

对照仓内 RFC 0003（`OpsExecutor` 运行时递归解释算法树）：本方案采用 **编译期类型树**。组合关系在类型上固定后，C++ 内联掉编排，避免解释器虚调用与 Device 侧堆分配。Selector 边界仍可用字符串 algName，与现网注册表兼容。

可运行原型：`hccl_refactor/`（`make test`）。

## 1. 设计原则

1. **拓扑是空策略类型**（tag dispatch），无 vtable、无数据成员。
2. **通信计划与执行分离**：Planner 只产出 `Transfer{step,channel,src,dst,owner,bytes}`；Send/Reduce/Notify 仍走现有 HCOMM。
3. **算子语义与拓扑正交**：AllGather/ReduceScatter 描述「slot 归属如何变化」；Mesh/Ring/NHR 描述「和谁、分几步」。
4. **编排只有三种**：`Sequence`（数据依赖）、`Parallel`（数据切开）、`OmniPipe`（双轴流水）。Sole = 单叶子 Sequence；TwoShot = `Sequence<RS,AG>`；Concurrent = 带不同 `subComm` 的 Parallel。
5. **多 channel 是策略，不是算法**：`MultiChannel<N>` 对每条逻辑 Transfer 切条。
6. **Clos 是 Fabric 约束**：同样的 Mesh/NHR 计划，channel 数 `min(requested, portBudget)`。
7. **引擎特化只留在 SendAll**：AICPU/AIV/CCU 差异不进入 Planner。

## 2. 类型积木

```cpp
// 拓扑原语
struct Mesh; struct Clos; struct ClosMesh;
struct Ring; struct DoubleRing; struct Nhr;

template<uint32_t N> struct MultiChannel;
template<uint32_t L> struct Level;   // L0/L1/L2

struct AllGatherOp; struct ReduceScatterOp; struct AllReduceOp;

template<typename Op, typename Topo, typename Ch = MultiChannel<1>, typename Lv = Level<0>>
struct Leaf;

template<typename... Stages>  struct Sequence;
template<typename... Branches> struct Parallel;
```

叶子执行（编译期绑定 Planner）：

```cpp
template<typename Topo>
CommPlan PlanAllGather(const PlannerInput&);   // 全特化：Mesh/Clos/Ring/DoubleRing/Nhr/ClosMesh

template<typename Topo, typename Ch, typename Lv>
CommPlan PlanLeaf(Collective c, const PlannerInput& in) {
    PlannerInput x = in;
    x.channels = Ch::kChannels;
    x.level    = Lv::kLevel;
    return c == Collective::ReduceScatter ? PlanReduceScatter<Topo>(x)
                                          : PlanAllGather<Topo>(x);
}
```

`Compose<AlgoTree>::Build(in)` 展开类型树，得到有序 `vector<CommPlan>`。`Sequence` 把后级 `step` 叠在前级之后；`Parallel` 按分支数切开 `sliceBytes`（余量给最后一支）。

## 3. 八族如何落地

### 3.1 Mesh

- **计划**：1 个逻辑 step；本 rank 与其余 N-1 个 rank 各一条（或 N-1 条并发）逻辑边；每边再 stripe 到 C 个 channel。
- **复杂度**：时延 α，带宽项 `(N-1)/N * n / C`（全互连并发）。
- **对应现状**：`CalcChannelRequestMesh1D` + `InsTemp*Mesh1D::KernelRun`。
- **开发**：AllGather/RS 各一份归属规则，Mesh 计划器一份，所有算子共用。

### 3.2 Clos

- **不是第三套步进循环**。`PlanAllGather<Clos>` = Mesh 对端集合 + `FabricChannels(CLOS, requested, portBudget)`。
- Channel 层继续调用 `CalcChannelRequestMeshClosMultiJetty` / priority CLOS link。
- Cost 继续走 CLOS 端口共享模型。

### 3.3 Clos + Mesh

类型：

```cpp
using ClosMeshConcurAG =
    Parallel< Leaf<AllGatherOp, Mesh, MultiChannel<2>, L0>,
              Leaf<AllGatherOp, Nhr,  MultiChannel<2>, L0> >;
```

或叶子 `ClosMesh` 内部直接 Merge Mesh 计划与 NHR 计划（原型两种都有）。`dataSplitRatio` 由端口比在 **算法构建期** 算好，执行期不再分支。

对应现状：`TopoMatchConcurrentV2` + `*ConcurrentExecutor`。

### 3.4 Ring

- **计划**：N-1 step；每 step 向 `rank+1` 发送当前块、从 `rank-1` 接收。
- AllGather：step k 发送 owner `(myRank - k)`。
- ReduceScatter：反向把各 owner 块递到目标 rank。
- 对应旧仓 `*_ring` / `HCCL_ALGO=ring`。新仓补回该 Planner 即可重新挂 Selector，无需 `ScatterRingFor91093Executor` 这种按产品抄类。

### 3.5 Double-Ring

- **计划**：正向环 + 反向环同时走，各带 ≈½ payload，step 仍 N-1。
- 与 Clos+Mesh 的区别：同一组 Rank、同一层、两套环序；不是两个 Fabric。
- 对应旧仓 aligned double-ring；Notify/双 stream 放在 SendAll 策略（`DualStream` tag），不进 Planner。

```cpp
using AgDoubleRing = Leaf<AllGatherOp, DoubleRing, MultiChannel<1>, L0>;
using ArDoubleRing = Sequence<
    Leaf<ReduceScatterOp, DoubleRing>,
    Leaf<AllGatherOp, DoubleRing>>;
```

### 3.6 NHR

- **计划**：`steps = ceil(log2 N)`；step k 距离 `2^k`（ReduceScatter 反向从大距离收到 1）。
- 非 2 次幂：hold 块数 `min(2^k, N)`，与现状 Nonuniform 一致的抽象（产品级再替换 `NhrStepTable`）。
- MultiJetty = `Leaf<..., Nhr, MultiChannel<J>>`，不要 `*NHRMultiLink` 复制类。

### 3.7 多 Channel

```cpp
template<uint32_t N> struct MultiChannel;
// StripeAndSend: channel i 取 total/N，最后一 channel 吃余量
```

- Mesh：C 条并发打满 HCCS。
- NHR/CLOS：`C = min(requested, portBudget)`。
- 线程/Notify：`GetRes` 用 `channels * peers` 公式，与现状 `GetThreadNum` 同构，但只写一次。

### 3.8 分层

```cpp
// 2 级 AllGather
using HierAG2 = Sequence<
    Leaf<AllGatherOp, Mesh, MultiChannel<4>, L0>,
    Leaf<AllGatherOp, Nhr,  MultiChannel<2>, L1>>;

// 3 级 AllGather（RFC 四层只需再加一个 Leaf）
using HierAG3 = Sequence<
    Leaf<AllGatherOp, Mesh, SingleChannel, L0>,
    Leaf<AllGatherOp, Nhr,  SingleChannel, L1>,
    Leaf<AllGatherOp, Nhr,  SingleChannel, L2>>;

// 2 级 AllReduce
using HierAR2 = Sequence<
    Leaf<ReduceScatterOp, Mesh, SingleChannel, L0>,
    Leaf<ReduceScatterOp, Nhr,  SingleChannel, L1>,
    Leaf<AllGatherOp,     Nhr,  SingleChannel, L1>,
    Leaf<AllGatherOp,     Mesh, SingleChannel, L0>>;
```

4 层 OCS：在 `HierAG3` 上再 `Leaf<AllGatherOp, Mesh /*或 Ocs*/, ..., L3>`，**不新增 Executor 类**。

`subCommIndex` / `Level<L>` 与 TopoMatch 的 `infos[L]` 对齐；越界仍由 TopoMatch 报错。

## 4. 与现网框架对接（零开销路径）

```text
Selector / CostModel  （仍返回 algName 字符串，AlgAttrs 不变）
        │
        ▼
REGISTER_EXEC_V2(cmd, algName, TemplatedAdaptor<AlgoTree>, TopoMatch, void)
        │  或 REGISTER_ALG 风格：Creator 绑定 AlgoTree
        ▼
TemplatedAdaptor<AlgoTree> : InsCollAlgBase
        CalcAlgHierarchyInfo → TopoMatch
        CalcRes             → 对 Compose<AlgoTree> 每片调对应 Channel 函数
        Orchestrate         → 展开 CommPlan，inline SendAll（无递归解释器）
```

`TemplatedAdaptor<AlgoTree>` 是 **一个类模板**，53 个 Sole/Sequence/... 类变成一次实例化。热路径：

- `Compose<AlgoTree>::Build` 可做成 `constexpr` 元数据 + 运行期填 rank/bytes；或直接 inline 特化函数。
- 禁止在 AICPU Kernel 里对编排走虚函数。
- `std::vector<Transfer>` 可换成预分配 scratch（按 `maxSteps * maxPeers * maxChannels`），避免 Device 堆分配（RFC 0003 风险项）。

与 RFC 0003 对照：

| 项 | RFC 0003 | 本方案 |
|---|---|---|
| 算法结构 | 运行时 `HcclAlgorithm` 树 | 编译期 `Sequence/Parallel/Leaf<...>` |
| 执行器 | 1 个 `OpsExecutor` 递归 | 1 个类模板，实例化后无递归 |
| 小消息开销 | 有解释器税 | 与手写 Sole 同量级（inline） |
| 插件/4 级灰度 | 很好（改树不改代码） | Selector 仍可对特殊 algName 走 RFC 解释器 |
| Ring / Double-Ring / Clos | 未作为一等原语 | 一等 Planner |
| 落地 | `experimental/recursive_executor` | `hccl_refactor` 原型 + 可迁入 src 的 Adaptor |

建议 **双轨**：商用热路径用模板实例化；试验 4 级 / 插件用 RFC 树。两者共享同一套 Planner 函数。

## 5. 源码与工作量对比

以 AllGather/ReduceScatter × 6 拓扑 × 2 种 channel × 3 种层级为例：

| | 现状（复制） | 重构后 |
|---|---|---|
| 叶子 Planner | ~每个算子每拓扑每引擎一份（mesh 文件 100+） | **6 个拓扑计划器** + 2 个算子归属规则 |
| 编排 Executor | ~53 个类，3→4 层再 +12–18 个 | **1 个 `TemplatedAdaptor<Tree>`** |
| 新增 4 层 AG | 新 Sequence 类 ~300–500 行 | 类型别名 + 一行 `REGISTER` ~10 行 |
| 新增 Double-Ring AG | 新模板+执行器（旧仓整文件复制） | `Leaf<AllGatherOp, DoubleRing>` |
| 新增 4 channel NHR | 新 `*MultiJetty*` 类 | `Leaf<..., Nhr, MultiChannel<4>>` |

原型用 6 个计划器覆盖 6×2×2×3 = **72** 种组合中的通信结构（测试里 `cartesian/planners == 12`）。

不在削减范围内、应保持特化的部分：

- AIV/CCU Kernel 指令序列（硬件绑定）
- Order-preserved 归约顺序（Template 层约束）
- 非对称 AHC、Pairwise AlltoAll、BIRS 等独立算法
- Cost 公式系数（仍按拓扑+引擎表）

## 6. 性能约束（必须守住）

1. **步进次数与对端集合不变**：Mesh 1 步 N-1 peer；Ring N-1 步邻居；NHR `ceil(log2 N)`；Double-Ring 双环并发。原型测试已锁 step/peer。
2. **Channel 切条守恒**：各 channel bytes 之和 = 逻辑边 payload。
3. **分层数据契约**：Sequence 前级 `ranksForOutput` = 后级 `ranksForInput`（接入 src 时接 RFC 的归属表，避免再靠 repeatStride 推导）。
4. **无额外 Device 分配**：Transfer 表用 `CalcRes` 阶段算好的上限预分配。
5. **同步次数不增**：Parallel 前后各一次 subcomm mask 同步，与现 `*ConcurrentExecutor` 相同。
6. **Benchmark 门禁**：Sole Mesh/NHR 小消息与现状对比，回归阈值建议时延 < 1%。若解释器路径打开，仅允许用在 4 层试验 algName。

## 7. 迁移步骤

1. **抽出 Planner**（行为不变）：先把 `InsTempAllGatherMesh1D` / `*Nhr` 的对端+切片改成调用 `PlanAllGather<Mesh/Nhr>`，Kernel 仍用旧 Send。
2. **补 Ring / Double-Ring / Clos channel 策略**，用旧仓 ST 对齐。
3. **用 `TemplatedAdaptor<Leaf<...>>` 替换 SoleExecutor** 一类，AlgAttrs/算法名不变。
4. **Sequence/Parallel 类型树** 替换 `*Sequence*` / `*Concurrent*`；先 AllGather，再 RS、TwoShot AllReduce。
5. **MultiChannel 策略** 替换各 `*MultiJetty*` 复制。
6. OmniPipe 仍可先走 RFC `OMNIPIPE` 解释器（步进切片公式复杂），稳定后再做成 `OmniPipe<X,Y>` 模板特化。

## 8. 原型如何验证设计

`hccl_refactor/tests/test_refactor.cc`（`make test`）覆盖：

| 用例 | 断言 |
|---|---|
| Mesh AllGather | 1 step，N-1 peer，全 rank 模拟后 slot 齐全 |
| Ring AllGather | N-1 step，全 rank 模拟齐全 |
| Double-Ring | N-1 step，正反向都有流量 |
| NHR | N=4/5/8 的 step = ceil(log2 N)，N=8 模拟齐全 |
| Clos vs Mesh | Mesh 保留 8 channel；Clos 被 portBudget=2 裁剪 |
| Clos+Mesh | Parallel 两支：Mesh + NHR |
| MultiChannel<4> | 4 条 channel，切条字节守恒 |
| TwoShot AllReduce | Sequence RS→AG，step 叠加 |
| 分层 AG 2L/3L、AR 2L | 阶段数与 L0/L1/L2 拓扑绑定 |

这不是上板带宽测试；它锁定 **计划结构**。接入 src 后应用现有 ST（AllGather/RS/AR mesh/nhr）做正确性，再用 hccl_test 看带宽。

## 9. 推荐目录（合入 hccl 仓时）

```text
src/ops/op_common/algorithm/
  primitive/          # Mesh/Ring/DoubleRing/Nhr 计划器（与算子无关）
  channel_policy/     # MultiChannel、Clos port cap
  compose/            # Sequence/Parallel/OmniPipe 类型与 Adaptor
  catalog/            # using 别名 + REGISTER_EXEC_V2 / REGISTER_ALG_ATTRS
src/ops/<op>/selector/   # 仍只返回 algName
```

算子目录不再放 `ins_v2_*_sequence_executor.cc` 这类编排副本；只保留该算子特有的 PreCopy/Reduce 语义（例如 ReduceScatter 的 LocalReduce）。
