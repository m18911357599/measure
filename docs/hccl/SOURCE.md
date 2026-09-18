# HCCL 源码来源

本设计文档与重构原型基于以下公开 HCCL 源码仓分析，**未将第三方源码整仓合入本仓库**（许可证与体积限制）。分析时使用的提交如下。

| 仓库 | 地址 | 提交 | 角色 |
|---|---|---|---|
| 新 HCCL（主分析对象） | https://gitcode.com/cann/hccl | `b3a8fe96684c4cf736e052c60e7d04749431b374`（`aivonly dfx fix`） | V2 算法注册 / Selector / Executor / Template / Channel / RFC 0003 |
| 旧 cann-hccl（GitHub 镜像） | https://github.com/lightclock/cann-hccl | `94e81756279088062f07f7315870eae1d38922d0` | Ring / Double-Ring / NHR / Mesh 旧栈 |
| 旧 cann-hccl（Gitee 官方） | https://gitee.com/ascend/cann-hccl | `2d685b084a60366b47ad8e50a563ee5c4dbef7fe` | 与 GitHub 镜像算法树基本一致 |

本地分析副本默认拉到 `.hccl-src/`（已 gitignore），不提交进本仓。

```bash
# 从 gitcode.com/cann/hccl 获取源码
bash scripts/fetch_hccl.sh

# 同时拉取 github.com/m18911357599/hccl 工作副本
bash scripts/fetch_hccl.sh --github
```

仓内关键文档（新 HCCL）：

- `docs/zh/architecture/architecture-brief.md` — RankGraph / Channel / 引擎分层
- `docs/zh/user_guide/coll_algo_intro/algo_intro.md` — Mesh / Ring / Double-Ring / NHR 等算法简介
- `docs/zh/user_guide/coll_algo_intro/hierarchical_comm_principle.md` — 分级通信
- `docs/zh/rfcs/0003-executor-template-refactor.md` — Executor 统一算法结构 RFC
- `experimental/ops/op_common/recursive_executor/` — RFC 0003 试验落地（默认关闭）
