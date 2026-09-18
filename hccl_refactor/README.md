# HCCL 算法重构原型（C++ 模板 + 组合）

本目录是对 HCCL 现有算法栈的**可编译设计原型**，不依赖 CANN / HCCL 运行时。目标是证明：

- 6 个拓扑原语（mesh / clos / clos+mesh / ring / double-ring / nhr）可以各自只写一份通信计划器
- 多 channel、分层 Sequence / Parallel 用类型组合表达，而不是再复制一套 Executor
- 热路径全部模板内联，没有虚函数编排开销

## 构建

```bash
cd hccl_refactor
make test
```

需要 g++（C++17）。

## 映射关系

| 原型类型 | 对应 HCCL 现状 |
|---|---|
| `Leaf<AllGatherOp, Mesh>` | `InsTempAllGatherMesh1D` + `InsV2*SoleExecutor` |
| `Leaf<AllGatherOp, Nhr>` | `InsTemp*Nhr` |
| `Leaf<AllGatherOp, Clos>` | Mesh 计划 + CLOS 端口裁剪的 Channel |
| `Leaf<AllGatherOp, ClosMesh>` / `Parallel<Mesh,Nhr>` | `*ConcurrentExecutor` + `MESH_1D_CLOS` |
| `Leaf<AllGatherOp, Ring>` | 旧仓 `*_ring` template |
| `Leaf<AllGatherOp, DoubleRing>` | 旧仓 aligned double-ring |
| `Sequence<RS, AG>` | `TwoShotSoleExecutor` |
| `Sequence<Mesh@L0, Nhr@L1, Nhr@L2>` | `*Sequence*3Level` |
| `MultiChannel<N>` | `channelsPerRank` / MultiJetty |

设计说明见 `docs/hccl/`。
