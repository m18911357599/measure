# HCCL 集合通信算法：现状分析与重构设计

本目录基于开源 HCCL / HCOMM 源码分析，给出两份设计文档：

| 文档 | 内容 |
|------|------|
| [01-algorithm-framework-design.md](./01-algorithm-framework-design.md) | 现有算法**注册、实现、调用**框架：分层、注册宏、Selector、Executor、Template、调用链 |
| [02-algorithm-refactor-design.md](./02-algorithm-refactor-design.md) | **C++ 模板化 + 组合**重构：Mesh / CLOS / CLOS+Mesh / Ring / Double-Ring / NHR / 多 Channel / 分层算法 |

分析源码来源（只读克隆，未合入本仓）：

- HCCL 算子仓：`https://github.com/hicann/hccl`（对应 [gitcode.com/cann/hccl](https://gitcode.com/cann/hccl)）
- HCOMM 通信基础仓：`https://github.com/kakukaops/hcomm`（对应 [gitcode.com/cann/hcomm](https://gitcode.com/cann/hcomm)，含 Ring / Double-Ring / NHR 历史算法模板）

分析口径以 HCCL `src/ops/` 当前主干为准，HCOMM `src/algorithm/` 用于补充 Ring / Double-Ring / 分层 AHC 等仍在生产路径上的算法族。
