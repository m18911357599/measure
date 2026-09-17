# HCCL 算法模板化 + 组合 原型

本目录是设计文档 [`docs/hccl-alg/`](../docs/hccl-alg/README.md) 的可编译验证，用 CPU 锁步模拟集合通信，证明：

- **拓扑引擎**（Ring / Mesh / NHR）与 **集合策略**（ReduceScatter / AllGather）正交
- **AllReduce = RS ∘ AG**
- **Double-ring = MultiChannel\<2, Ring\>**（奇通道反向）
- **Clos 是互联策略**，不是步进内核；**Clos+Mesh = Hierarchical\<Mesh, Ring|NHR\>**
- 热路径类型 **没有虚表**

它 **不** 链接 CANN，也不下发 SQE。合入 HCOMM 时把 `commit_step` 换成 `Transport::TxAsync/RxAsync` 即可，片号公式与现网 `GetStepInfo` / Ring slice 下标一致。

## 构建

```bash
make -C hccl_alg_refactor test
```

需要 C++17。

## 映射

| 头文件 | 职责 |
|---|---|
| `include/hccl_refactor/policies.hpp` | OpPolicy + Clos/Mesh 互联策略 |
| `include/hccl_refactor/engines.hpp` | `TopologyKernel<Topo, Op>`，Ring/Mesh/NHR，AR 组合 |
| `include/hccl_refactor/compose.hpp` | `MultiChannel`、`HierarchicalAllReduce` |
| `include/hccl_refactor/registry.hpp` | 编译期 `KernelPick<Topo, Op, Channels>` |

## 与现网公式的对应

- Ring RS/AG 片号：`reduce_scatter_ring.cc` / `all_gather_ring.cc`
- NHR RS：`ReduceScatterNHR::GetStepInfo`
- NHR AG：`AllGatherNHR::GetStepInfo`（delta 从最后一步回数）
- 分层顺序：`CollAllReduceRingFor91093Executor::KernelRun`
