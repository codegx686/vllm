# 如何向 vLLM 提交

这份交付包含可复现的实验项目和一个针对 vLLM 的独立补丁。真正提交给
`vllm-project/vllm` 的是 `upstream.patch`，不是整个实验目录、历史报告、模型或日志。
建议标题：`[Feature][Scheduler] Add adaptive token budget scheduler example`。

补丁基于 `e9acf48614a2c0d027d7aa2f0e38d963117617a6`（2026-10-24 main）。
它把策略放在 `examples/features/adaptive_budget/`，新增用户文档，并扩展原有
scheduler plugin 测试与 Buildkite 的路径选择。它没有更改核心调度器或默认策略。

## 1. 先理解贡献范围与重复工作

阅读 `REVISION_REPORT.md`、`PR_CHECKLIST.md` 和上游示例源码。
原始 P-PAS 作者已提出 [RFC #52906](https://github.com/vllm-project/vllm/issues/52906)。
本项目限制总 token，P-PAS 限制 Prefill token；请明确说明这个区别。
已有的 EWSJF #33392、FlowPrefill #44794 和 defer #61080 使用不同的控制机制。
`results/revision-20261024/duplicate-search.json` 保存本次查重结果。

提交当天再次查重，读取相关 issue 的最新讨论：

```bash
gh issue view 52906 --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search '52906 in:body'
gh pr list --repo vllm-project/vllm --state open --search 'adaptive budget'
gh pr list --repo vllm-project/vllm --state open --search 'adaptive scheduler'
```

建议先在该 RFC 中说明本示例的范围、证据和区别，询问维护者是否接受示例贡献。
`RFC_COMMENT.md` 是可自行审阅发送的草稿。当前没有替你发布评论或创建 PR。
如果出现等价的开放 PR，优先协作，不重复投稿。

## 2. Fork 并应用补丁

把 `YOUR_LOGIN` 换成自己的 GitHub 用户名，路径改为实际解压位置。
以下命令在 Linux/WSL 上执行；需要 GitHub CLI 已登录。

```bash
gh repo fork vllm-project/vllm --clone=false
git clone https://github.com/YOUR_LOGIN/vllm.git
cd vllm
git remote add upstream https://github.com/vllm-project/vllm.git
git fetch upstream main
git switch -c feat/adaptive-budget-example e9acf48614a2c0d027d7aa2f0e38d963117617a6
git apply --check /path/to/vllm_adaptive_budget/upstream.patch
git apply --index /path/to/vllm_adaptive_budget/upstream.patch
git diff --cached --stat
git diff --cached
```

逐行阅读差异。不要直接把 `upstream_payload/` 覆盖到任意版本的 checkout；
它保存的是上述基线上的完整文件，补丁更适合审查和解决版本差异。

## 3. 运行检查并准备提交

按照上游当前 `AGENTS.md` 和贡献指南创建独立环境。当前源码使用预编译安装：

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/lint.txt
uv pip install -r requirements/test/cuda.in
pre-commit install
.venv/bin/python -m pytest tests/plugins_tests/test_scheduler_plugins.py -k adaptive_budget -v
pre-commit run --files \
  examples/features/adaptive_budget/scheduler.py \
  tests/plugins_tests/test_scheduler_plugins.py tests/v1/core/utils.py \
  docs/features/adaptive_budget.md docs/features/README.md \
  .buildkite/test_areas/plugins.yaml
```

真实 GPU 测试需要 Qwen/Qwen3-1.7B 权重。已有本地模型时，可按照 vLLM 的
`VLLM_MODEL_REDIRECT_PATH` 机制映射模型名；不需要把权重提交到 Git。
本次测试支持在进入 LLM 前显式解析该映射，因此可以在离线机器运行：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
Path("/tmp/adaptive-model-redirect.json").write_text(json.dumps({
    "Qwen/Qwen3-1.7B": "/absolute/path/to/Qwen3-1.7B"
}))
PY
VLLM_MODEL_REDIRECT_PATH=/tmp/adaptive-model-redirect.json \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_BATCH_INVARIANT=1 \
  .venv/bin/python -m pytest tests/plugins_tests/test_scheduler_plugins.py -k adaptive_budget -v
```

5090 实测使用 cu130、与基线提交匹配的官方预编译 wheel，以及
`dill==0.4.0`、`networkx==3.4.2`；本机解析到的更新版本曾出现导入错误。
如需复现同一环境，参考 `requirements-upstream-tested.txt` 和安装日志。
测试依赖、CUDA wheel 和模型可能占用数十 GB；本次使用 uv hardlink 模式复用缓存。
完整实测环境、执行命令及任何未通过项以 `REVISION_REPORT.md` 为准。

将上游 `.agents/skills/pr-checklist/SKILL.md` 再检查一遍。重点检查：
新测试实际运行、输出差异和性能退化已披露、最新 main 上无冲突、文档预览可访问。
`PR_BODY.md` 已按本次上游 PR 模板组织；在发送前更新当天的测试和查重结果。

## 4. 用本人身份签署 DCO、推送、创建 Draft PR

vLLM 要求提交人理解所有改动，并亲自审查和运行相关测试。
这是上游 [AGENTS.md](https://github.com/vllm-project/vllm/blob/main/AGENTS.md)
与 [贡献指南](https://docs.vllm.ai/en/latest/contributing/) 的要求。
本交付不代替你的审核，也不替你签署 DCO。

```bash
git config user.name "你的真实姓名"
git config user.email "你的 GitHub 邮箱"
git add examples/features/adaptive_budget/scheduler.py \
  tests/plugins_tests/test_scheduler_plugins.py tests/v1/core/utils.py \
  docs/features/adaptive_budget.md docs/features/README.md \
  .buildkite/test_areas/plugins.yaml
git commit -s -m "[Feature][Scheduler] Add adaptive token budget scheduler example" \
  -m "Co-authored-by: OpenAI Codex"
git fetch upstream main
git rebase upstream/main
# 若有新提交或冲突，解决后重跑上面的测试和 pre-commit。
git push -u origin feat/adaptive-budget-example
gh pr create --draft --repo vllm-project/vllm --base main \
  --head YOUR_LOGIN:feat/adaptive-budget-example \
  --title "[Feature][Scheduler] Add adaptive token budget scheduler example" \
  --body-file /path/to/vllm_adaptive_budget/PR_BODY.md
```

CI 和人工审核通过后再转为 Ready for review；不要把 Draft 描述为已具备合并保证。
如果维护者倾向维持外置扩展，可以独立发布本项目，并在 RFC 中提供验证材料。
