# HCCL 算法框架分析与模板化重构设计

本目录基于公开 HCCL 源码仓（见 [SOURCE.md](./SOURCE.md)）完成两件事：

1. **现状设计文档**：算法如何注册、如何实现、如何被调用。
2. **重构设计**：用 **C++ 模板化 + 组合** 把 mesh / clos / clos+mesh / ring / double-ring / nhr、多 channel、分层算法拆成可复用积木，在不牺牲热路径性能的前提下压缩源码与开发量。

| 文档 | 内容 |
|---|---|
| [SOURCE.md](./SOURCE.md) | 源码仓地址与分析用提交 |
| [01-existing-algorithm-framework.md](./01-existing-algorithm-framework.md) | 注册 / 选择 / 执行 / 模板调用链 |
| [02-algorithm-family-analysis.md](./02-algorithm-family-analysis.md) | 八类算法族现状与重复度 |
| [03-template-composition-refactor.md](./03-template-composition-refactor.md) | 模板 + 组合重构方案 |
| [../../hccl_refactor/](../../hccl_refactor/) | 可编译原型（`make test`） |

结论先行：

- **新 HCCL**（gitcode.com/cann/hccl）已经用 C++ 模板参数把 `Executor<TopoMatch, Template...>` 拼起来，但 **编排类仍按「算子 × Sole/Sequence/Parallel/Concurrent/OmniPipe」复制**，Mesh/NHR 通信计划仍按算子各写一份。
- **CLOS 不是算法模板**，是 RankGraph 上的拓扑形态；clos+mesh 是双平面 **Parallel**。
- **Ring / Double-Ring** 在新仓基本退役，完整实现在旧 cann-hccl。
- 仓内 RFC 0003 给出 **运行时算法树 + OpsExecutor 解释器**；本方案在其组合思想上改为 **编译期类型树**，避免递归解释器的小消息开销，并补齐 ring / double-ring / clos 原语。
