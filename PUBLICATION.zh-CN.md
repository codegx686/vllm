# 发布记录与后续操作

2026-10-24 已按提交人的明确授权完成公开发布。

- Draft PR：[vllm-project/vllm#63458](https://github.com/vllm-project/vllm/pull/63458)
- 标题：`[Feature][Scheduler] Add adaptive token budget scheduler example`
- 代码分支：[codegx686/vllm:feat/adaptive-budget-example](https://github.com/codegx686/vllm/tree/feat/adaptive-budget-example)
- 代码提交：`ca2728850d3add8d6c7066b4a6fa4ae62e7a3247`
- 验证基线：`e9acf48614a2c0d027d7aa2f0e38d963117617a6`
- 公开证据：[adaptive-budget-evidence-20261024](https://github.com/codegx686/vllm/tree/adaptive-budget-evidence-20261024)
- PR 只包含 6 个文件、395 行新增；实验工具与原始测量放在独立证据分支。

## 作者和签署

作者为 `codegx686 <147488460+codegx686@users.noreply.github.com>`。
共同作者为 `Pig with dreams <114994886+PigHaveDreamS@users.noreply.github.com>`，并保留 `Co-authored-by: OpenAI Codex`。
GitHub 已识别两个用户账号，DCO 检查通过。Signed-off-by 只使用 codegx686 的署名；没有替共同作者签署 DCO。

## 发布时的验证状态

5090 上实验测试 54 项通过；本次上游新增测试 16 项通过，包括真实模型与 Prefix Cache。
完整 pre-commit 的 17 项适用检查通过，19 项无匹配文件；真正提交时再次运行 hook 和 sign-off 检查均通过。
性能证据包含 30 轮、1440 个请求、0 请求失败，但性能数据来自 vLLM 0.22.1，不能当作当前 main 的性能结果。

截至 2026-10-24 08:20 UTC，线上 CI 尚未通过：

1. [pre-run-check](https://github.com/vllm-project/vllm/actions/runs/40077252853/job/120823176914) 因新贡献者 CI 门槛失败。账号在该仓库的已合并 PR 数为 0；工作流要求至少 4 个已合并 PR，或 PR 有 `verified`、`ready`、`ready-run-all-tests` 之一的标签。后续 pre-commit 被跳过，没有运行代码检查。
2. [Read the Docs 构建](https://app.readthedocs.org/projects/vllm/builds/35374293/) 在 `docs/pre_run_check.sh` 中读取上述失败结果后退出；尚未执行文档编译。因此线上文档预览仍未验证。
3. DCO 已通过。完整上游 GPU CI、其他 GPU/AMD 验证仍未完成。

这两项失败的日志没有指出本补丁的代码或文档编译错误；也不能据此称整个 CI 已通过。
原始日志见 `logs/publication-pre-run-check.log` 和 `logs/publication-readthedocs.log`。

## 接下来怎么做

当前 PR 保持 Draft。等待维护者评估是否接受这个示例，并按项目流程允许 CI 运行；不能通过修改本补丁绕过该门槛。
后续处理审查意见、完成线上 CI 与文档预览检查后，再由提交人决定将同一个 PR 标为 Ready for review。
无需再次 Fork、创建重复 PR 或重复签署已有提交。

Windows 本地代码 checkout 在 `vllm_submission/`，已经跟踪远端贡献分支；5090 上源码 checkout 位于 `/root/autodl-tmp/vllm-budget-upstream`。
若修改代码，先在 Linux CUDA 环境运行 `SUBMIT.zh-CN.md` 中的测试与完整 pre-commit，再提交并推送到 `feat/adaptive-budget-example`；GitHub 会自动更新同一个 PR。

可以在已登录 GitHub CLI 的环境查看进展：

```bash
gh pr view 63458 --repo vllm-project/vllm --web
gh pr checks 63458 --repo vllm-project/vllm
```

没有额外发送 RFC 评论、请求评审或把 Draft 改为 Ready。
