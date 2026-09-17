# HCCL 集合通信算法：现有框架与重构设计

本目录基于开源 **HCOMM / cann-hccl** 源码（分析时使用 `kakukaops/hcomm` 完整树与 `xwqtju/cann-hccl` 开源子集），整理两件事：

1. **现有算法如何注册、实现、被调用**
2. **用 C++ 模板化 + 组合重构 mesh / clos / clos+mesh / ring / double-ring / nhr / 多 channel / 分层算法**，在不降低性能的前提下压缩源码与开发工作量

| 文档 | 内容 |
|---|---|
| [01-existing-framework.md](01-existing-framework.md) | 注册表、选择器、调用链、分层通信域 |
| [02-algorithm-analysis.md](02-algorithm-analysis.md) | 各拓扑族实现盘点、重复度、CLOS 的真实含义 |
| [03-refactor-design.md](03-refactor-design.md) | 模板 + 组合重构方案、性能约束、落地步骤 |

可编译的设计原型（CPU 模拟，验证组合语义而非 CANN 任务下发）在仓库根目录：

[`hccl_alg_refactor/`](../../hccl_alg_refactor/README.md)

```bash
make -C hccl_alg_refactor test
```
