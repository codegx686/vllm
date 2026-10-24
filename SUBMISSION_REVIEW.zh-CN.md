# 提交前人工审查

拟提交到 `vllm-project/vllm` 的 Draft PR：
`[Feature][Scheduler] Add adaptive token budget scheduler example`。
使用现有 Fork `codegx686/vllm`，代码分支 `feat/adaptive-budget-example`。
作者为 `codegx686 <147488460+codegx686@users.noreply.github.com>`；
共同作者为 `Pig with dreams <114994886+PigHaveDreamS@users.noreply.github.com>`。
保留 `Co-authored-by: OpenAI Codex`，并在 PR 描述披露 AI 辅助。

## 需要理解的改动

上游补丁只有 6 个文件、395 行新增：

1. `examples/features/adaptive_budget/scheduler.py`：可选的同步单卡文本调度器。
   Pydantic 验证配置；当 Prefill/Decode 混合且 Prefill 达到阈值时，从高预算
   降到低预算。最短驻留避免频繁切换，停滞时限频提高预算。原生 Scheduler
   继续负责排队和 KV 分配，每步预算在 `finally` 中恢复。
2. `tests/plugins_tests/test_scheduler_plugins.py`：新增 16 个实例化测试项，
   覆盖配置边界、驻留、guard 限频、真实 KV 分配与取消，以及真实 Qwen 生成
   和 Prefix Cache。生成比较使用已有 top-5 helper，不保证 token 序列恒等。
3. `tests/v1/core/utils.py`：现有 fixture 增加可选 additional_config 参数，
   以便测试通过真实入口传递策略配置。
4. `docs/features/adaptive_budget.md`：使用命令、默认值、边界和性能取舍。
5. `docs/features/README.md`：提供上述文档的入口。
6. `.buildkite/test_areas/plugins.yaml`：补充路径依赖，使现有 plugin 测试 job
   在调度示例或相关源码改变时被选择；没有创建新的昂贵 CI job。

## 必须保留的限制

- 这是示例，不是默认策略。总 token 上限同时影响 Prefill 和 Decode。
- 单 GPU、同步、文本 decoder、Chunked Prefill；不声称支持多模态、分布式、
  投机解码、LoRA 或远程 KV。
- guard 只提供推进机会，不能保证无饥饿；每步扫描请求，成本随队列增长。
- 0.22.1 的突发实验中，P99 token 间隔改善约 70%，但 E2E 增加约 8.4%；
  不能把旧版本性能数据当成最新 main 的性能证明。
- 本扩展限制总 token，与 P-PAS 的 aggregate prefill 限制不同。

## 本人运行相关测试

在 5090 上进入 `/root/autodl-tmp/vllm-budget-upstream` 后执行：

```bash
VLLM_MODEL_REDIRECT_PATH=/root/autodl-tmp/model-redirect-budget.json \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_BATCH_INVARIANT=1 \
  .venv/bin/python -m pytest tests/plugins_tests/test_scheduler_plugins.py \
  -k adaptive_budget -v
```

审核本地 `vllm_submission` 的已暂存差异。`git diff --cached` 可查看全部内容。
本人理解、审查并运行测试的要求来自上游 AGENTS.md；代理运行的测试不应被
表述为你亲自运行。

## DCO 与公开材料

阅读 `vllm_submission/DCO`。`git commit -s` 会以你确认的姓名和邮箱添加
Signed-off-by，表示你同意该贡献许可和来源声明；它不是登录认证，也不是 GPG 签名。
共同作者署名不等于替对方添加 DCO 签署。本次只拟添加你的 Signed-off-by。

为了让维护者核对数据，拟在同一 Fork 的独立分支
`adaptive-budget-evidence-20261024` 发布已审计的实验源码、报告及原始结果，
并从 PR 链接过去。该证据分支不会进入上游代码 diff。
材料不包含 SSH 凭据、模型权重或虚拟环境；完整文件以交付包 manifest 为准。

确认上述人工审核、测试和 DCO 后，再执行签署、推送及 Draft PR 发布。
