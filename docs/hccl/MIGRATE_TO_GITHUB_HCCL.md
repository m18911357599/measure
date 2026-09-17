# 迁移到 m18911357599/hccl

本窗口的 GitHub token 只能写入 `m18911357599/measure`，**不能 push** 到 [m18911357599/hccl](https://github.com/m18911357599/hccl)。已把 cann/hccl 源码 + 分析文档打成可完整恢复的 git bundle。

Bundle 文件：

- 本仓：`docs/hccl/m18911357599_hccl_sync.bundle`
- 分支：`cursor/sync-cann-hccl-analysis-8203`
- 提交：`36977e8` *Import cann/hccl source plus algorithm analysis overlay.*
- 上游快照：gitcode.com/cann/hccl `b3a8fe96684c4cf736e052c60e7d04749431b374`

在有 `hccl` 仓写权限的环境执行：

```bash
git clone https://github.com/m18911357599/hccl.git
cd hccl
git fetch ../measure/docs/hccl/m18911357599_hccl_sync.bundle \
  cursor/sync-cann-hccl-analysis-8203:cursor/sync-cann-hccl-analysis-8203
git checkout cursor/sync-cann-hccl-analysis-8203
git push -u origin cursor/sync-cann-hccl-analysis-8203
```

恢复后仓库将包含：

| 路径 | 内容 |
|---|---|
| `src/` `include/` `experimental/` 等 | cann/hccl 源码 |
| `UPSTREAM.md` | 同步提交与许可证说明 |
| `docs/analysis/` | 算法注册/调用框架与重构设计 |
| `hccl_refactor/` | `make test` 原型（39 项通过） |

也可把 Cursor Cloud Agent 直接开在 **m18911357599/hccl** 上，该 Agent 的 token 即可 push。
