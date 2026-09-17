# HCCL 算法模板化 + 组合原型

本目录是设计文档的**可编译样例**，用 CPU mock Transport 验证：

- 拓扑原语：Mesh / Ring / NHR / ClosFabric
- 组合器：Double-Ring（多 channel）、Hierarchical（含 Clos+Mesh）、多 channel
- 集合 Pattern：AllGather / ReduceScatter / AllReduce
- 字符串 Recipe 注册（兼容现网 `algName`）

不依赖 CANN。热路径是编译期模板，无虚函数。

## 构建与测试

```bash
cd docs/hccl/prototype
make test
```

需要 C++17。`make test` 会跑通 8 类算法的正确性断言，并打印各 Recipe 的步进次数（用于对照现网复杂度）。

## 文件

| 路径 | 内容 |
|---|---|
| `include/hccl_algo/*.hpp` | 零开销组合库 |
| `tests/test_compose.cpp` | 正确性 + 步进计数 |
| `Makefile` | `make test` |

## 与现网对应

| 原型 Recipe 名 | 现网近似 |
|---|---|
| `AllGatherMesh` | `AllGatherMeshOpbaseExecutor` |
| `AllGatherRing` | `AllGatherRingExecutor` |
| `AllGatherDoubleRing` | `AlignedAllGatherDoubleRingFor91093Executor` |
| `AllGatherNhr` | level1 `TEMPLATE_ALL_GATHER_NHR` |
| `AllGatherClos` | Clos 织物上的 FullMesh/NHR |
| `AllGatherClosMesh` | 机内 Mesh + 机间 Clos（分级） |
| `AllGatherHier3` | 910_93 L0/L1/L2 |
| `AllReduceClosMesh` | RS Mesh + AG Mesh，L1 走 Clos |
