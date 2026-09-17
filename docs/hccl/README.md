# HCCL 算法框架分析与重构设计

本目录基于开源仓 [cann-hccl](https://gitee.com/ascend/cann-hccl)（`version.info = 8.3.T5.0`，分析时拉取 `master`）完成。

HCCL（Huawei Collective Communication Library）是昇腾 NPU 的集合通信库。开源仓覆盖**通信框架**与**通信算法**两层；通信平台层（Transport / Notify / 建链）不在本仓。

## 文档

| 文档 | 内容 |
|---|---|
| [01-existing-algorithm-framework.md](./01-existing-algorithm-framework.md) | 现有算法**注册、实现、调用**框架设计文档 |
| [02-algorithm-analysis-and-template-composition-refactor.md](./02-algorithm-analysis-and-template-composition-refactor.md) | 现有算法分析，以及 **C++ 模板化 + 组合** 重构设计（mesh / clos / clos+mesh / ring / double-ring / nhr / 多 channel / 分层） |
| [prototype/](./prototype/) | 可编译的零开销组合原型（mock Transport，不依赖 CANN） |

## 源码规模（算法层）

| 模块 | 文件数 | 行数（约） | 注册条目 |
|---|---:|---:|---:|
| CollExecutor | 287 | 38,562 | 128 个 `REGISTER_EXEC` |
| AlgTemplate | 275 | 38,019 | 90+ 个 `TemplateType` |
| CollAlgOperator | 约 20 | — | 15 个 `REGISTER_OP`（按 `HcclCMDType`） |

## 结论摘要

1. 现有框架已经是「**算子选择 → Executor 资源/编排 → Template 步进**」三层，但 **Op × 拓扑 × 芯片 × 模式 × 数据量** 在 Executor 层做笛卡尔展开，导致 128 个几乎同构的 Executor。
2. CLOS 在源码中**不是独立算法类**，而是机间/超节点间的交换拓扑（Clos fabric）。机内 Mesh + 机间 Clos 上的 Ring/NHR/FullMesh，就是文档与白皮书中的 **CLOS+Mesh 分层**。
3. 重构目标：把 **Mesh / Clos / Ring / Double-Ring / NHR** 做成编译期拓扑原语，把 **多 Channel 与分层** 做成组合器；集合原语（AllGather / ReduceScatter / AllReduce…）只描述数据切片与 Reduce 语义。热路径无额外虚调用、无逐步堆分配。
