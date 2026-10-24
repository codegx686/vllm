# 2026-10-24 修订与验证报告

本次交付把原有实验扩展整理为可供人工审核的 vLLM scheduler 示例补丁。
实验版保留多策略、日志和复现工具；上游版只有自适应策略，使用现有
`scheduler_cls` 和 `additional_config` 接口，不修改核心调度器及默认行为。
它展示延迟取舍，不能宣称普遍加速，也不能保证上游接受。

## 提交准备更新

本次提交基线已更新为 `e9acf48614a2c0d027d7aa2f0e38d963117617a6`。
新增 16 项测试在该基线上再次全部通过（60.62 秒、1 项原有测试未选中、
4 条 fork 弃用警告），见 `logs/upstream-tests-submission.log`。
与初始基线相比，上游新增的 11 文件改动未涉及调度器、配置、测试 fixture、
原生扩展源码、构建配置或依赖，因此保留原来安装的原生扩展并重新验证 Python 源码。
本地提交暂存区的 6 个 Git blob ID 已与 5090 上实测版本逐一核对一致。

完整 pre-commit 已通过：17 项适用检查通过，19 项无匹配文件、正常不运行，
没有使用 SKIP 或 --no-verify。日志为 `logs/precommit-final-submission.log`；
结构化检查记录为 `results/revision-20261024/submission-validation.json`。
通过 Windows 现有代理下载并校验官方 Go 1.27.3，再放入远程临时工具目录，
解决了 actionlint 工具链安装的网络障碍。最初的失败记录仍保留。
已授权使用现有 Fork `codegx686/vllm`；作者及共同作者姓名、GitHub 隐私邮箱
已确认。2026-10-24 提交人已确认逐行审查、理解改动、亲自运行相关测试，
同意 DCO，并授权以已确认署名签署、公开代码和证据分支、创建 Draft PR。
发布已完成：[Draft PR #63458](https://github.com/vllm-project/vllm/pull/63458)，
代码提交 `ca2728850d3add8d6c7066b4a6fa4ae62e7a3247`。DCO 检查通过。
线上 pre-commit 和文档构建受到新贡献者 CI 门槛限制，详情见 `PUBLICATION.zh-CN.md`。

## 修复与贡献范围

- 拒绝布尔值/非整数预算、非有限超时、无效阈值和容量越界；失败发生在调度状态变更前。
- 配置接入 `additional_config.adaptive_budget`。实验版兼容旧 `AB_CONFIG`，同时设置时报错。
- 跟踪 Request 对象身份，避免复用 request ID 继承前一请求的停滞时间。
- 将抢占后生成 token 的重算计为 prefill；排除语法/KV 阻塞状态。
- 转发原生 `schedule` 参数，在 `finally` 恢复初始化容量；取消和分配仍交给原生 Scheduler。
- 实验日志关闭时释放句柄；记录失败不会丢弃已经分配好的调度步骤；关闭 trace 时避免计时开销。
- 上游示例明确限制单 GPU、同步、纯文本 decoder、Chunked Prefill，拒绝未支持的功能组合。
- 新增用户文档、功能索引入口、真实 allocator/生成测试，以及现有 Buildkite plugin job 的路径依赖。

初始验证基线：`49d634bd6a3dc7433d8fb464821c95c76db85350`（2026-10-24 main）。
当前交付补丁基线以 `UPSTREAM_BASE.txt` 为准，已更新为上面的提交准备基线。
`upstream_payload/` 是基线上的完整文件，`upstream.patch` 才是提交差异。
不要把整个实验工程提交给 vLLM。

## 验证记录

硬件：RTX 5090，32607 MiB，驱动 595.71.05，单卡。
实验环境：Python 3.12、vLLM 0.22.1、PyTorch 2.11.0+cu130、Qwen3-1.7B BF16。
当前 main 使用独立虚拟环境和与上述提交匹配的官方 cu130 预编译扩展。
主环境实测为 PyTorch 2.14.0+cu130，日志确认使用 V2 Model Runner；官方 wheel
版本为 `0.32.1rc1.dev438+g49d634bd6`。浅克隆 editable 的版本字符串显示
`0.1.dev1+g49d634bd6.d20261024`，因此以 Git commit 而不是该字符串识别源码。
完整版本以日志和每轮 `metadata.json` 为准。

| 验证 | 结果与证据 |
| --- | --- |
| 原始测试基线 | 29 passed，`logs/baseline-20261024.log` |
| 新回归对原始 controller | 13 failed、1 passed、20 deselected；确认无效输入确有缺陷，`logs/red-20261024.log` |
| 修订实验版 | 54 passed，14 条 Torch 弃用警告，`logs/tests-final-20261024.log` |
| 策略差分 | 上述测试含 4 组配置各 1000 次随机观察，实验版与精简示例预算一致 |
| 实验源码 Ruff | check 和 format 通过，`logs/ruff-final-20261024.log` |
| 初始 main 新增测试 | 16 passed、1 deselected、4 条 fork 弃用警告，130.34 秒；`logs/upstream-tests.log` |
| 提交基线新增测试 | 同样 16 passed，60.62 秒；`logs/upstream-tests-submission.log` |
| 上游适用检查 | 14 个实际适用 hook 通过，见 `logs/upstream-changed-hooks.log` 与 `logs/upstream-ci-hooks.log` |
| 完整 pre-commit 包装命令 | 本次补跑通过；初次 actionlint 的 Go 环境下载 SSL EOF 已解决，保留历史失败日志 |
| 文档 | 独立 agent 从功能索引查找并审核配置/边界；本地可发现。发布后的文档构建因 pre-run-check 未通过而跳过，预览尚未验证 |

一次最终测试调用遗漏 `AB_TEST_MODEL`，产生 16 个 KeyError，保留在
`logs/tests-missing-model-env.log`；设置真实本地模型路径后 54 项全部通过。
远程 main 安装曾因磁盘空间不足中断，后续修复和实测见相关安装日志。
改用 uv hardlink 模式后安装成功；环境中 `dill 0.4.1` 和 `networkx 3.7`
曾出现导入错误，最终固定为 `dill==0.4.0`、`networkx==3.4.2`。
模型离线解析与重定向的执行顺序导致前两次 GPU 加载失败/等待网络，
测试现使用上游现有 `maybe_model_redirect` 在进入 LLM 前解析真实本地模型路径。
最终 16 项全部运行，没有跳过新增测试；唯一 deselected 是原有 OPT dummy scheduler 测试。
测试诊断的 90 秒 faulthandler 曾打印运行栈，最终测试正常完成，不是超时终止。

当前 main 的实际命令（在独立源码 checkout 中）：

```bash
VLLM_MODEL_REDIRECT_PATH=/root/autodl-tmp/model-redirect-budget.json \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_BATCH_INVARIANT=1 \
  .venv/bin/python -m pytest tests/plugins_tests/test_scheduler_plugins.py \
  -k adaptive_budget -v -s -o faulthandler_timeout=90
```

`requirements-upstream-tested.txt` 保存独立环境依赖；`logs/upstream-patch-check.log`
记录补丁在干净基线 index 上应用检查通过。最终差异为 6 文件、395 行新增。

适用的 hook：ruff-check、ruff-format、typos、markdownlint-cli2、mypy、
check-spdx-header、check-filenames、check-forbidden-imports、check-torch-cuda-call、
validate-config、check-dockerignore-tracked、check-boolean-context-manager、
check-buildkite-step-keys、check-test-tethering。未声称执行完整 vLLM 测试集、
Buildkite 云端 CI、MkDocs 全站构建、多 GPU 或 AMD 验证。

## 新性能实验：30 轮、1440 请求、0 失败

只汇总 `results/revision-20261024/sweep-{101,202,303}`，不与旧报告的 42 轮混合。
5 策略 × 2 场景 × 3 seed，每轮 48 请求。每个 seed 打乱策略执行顺序。
自适应组直接运行上游精简示例；其他组运行原生转发或固定运行时预算。
各组容量均为 8192，最大序列 64，32k 上下文，85% 显存，开启 CUDA Graph，
同步 Chunked Prefill，关闭 Prefix Cache 和 trace。每次重启后预热。

burst 每 6 秒一轮混合突发；短输入/输出 512/384 token，长输入 12288 或
16384、输出 128。steady 每 0.5 秒一个请求；短 512/256、长 8192/128。
输入是重复技术短句，固定生成长度。这是服务负载实验，不是模型质量评测。

下表为三个 seed 指标的平均；尾分位数是各轮分位数的平均，并非合并样本分位数。

| 场景 | 策略 | 平均 TTFT ms | P99 ITL ms | 平均 E2E ms | 输出 tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| burst | native8192 | 715.67 | 150.08 | 3237.28 | 651.50 |
| burst | static1024 | 987.94 | 42.22 | 3650.95 | 637.12 |
| burst | static2048 | 836.90 | 67.59 | 3435.55 | 644.79 |
| burst | static4096 | 751.19 | 103.73 | 3301.46 | 649.40 |
| burst | adaptive | 882.20 | 45.14 | 3508.79 | 641.99 |
| steady | native8192 | 82.15 | 12.14 | 1180.55 | 417.28 |
| steady | adaptive | 82.38 | 12.16 | 1180.20 | 417.23 |

对 burst 的配对 seed 比较：

- 相对 native8192，P99 ITL 改善 69.88%（各 seed 69.27%–70.95%）；
  平均 TTFT 增加 23.25%，E2E 增加 8.39%，吞吐下降 1.46%。
- 相对 static1024，平均 TTFT 改善 10.72%（10.15%–11.72%），
  E2E 改善 3.90%（3.49%–4.23%），吞吐改善 0.76%；P99 ITL 变差 6.90%。
- steady 与原生基本持平。没有一个策略在所有指标上支配其他策略。

长短请求拆分、所有固定预算、每个 seed 的变化范围见
`results/revision-20261024/summary/tables.md` 和 `paired_comparisons.json`。
三个 seed 不足以证明统计显著性；未锁 GPU 时钟；准备 main 环境的 CPU/网络工作
曾与压测重叠，但没有并发模型 GPU 负载。ITL 来自客户端 SSE 接收时间，
不是 GPU kernel 时间。30 轮共记录 1220 个包含多个 token 的 SSE chunk，
这些 token 共用接收时间，限制了 ITL 时间分辨率；最大客户端发送延迟为 5.71 ms。
没有测出生产 SLO、无饥饿保证或超大队列 CPU 成本。
性能数值来自 0.22.1，不能冒充当前 main 上的性能结论。

## 生成结果核对

除固定长度的吞吐实验外，单独使用自然短问题与两个长输入，生成 64 token、
greedy、top-5 logprobs；开启 Prefix Cache 的组重复执行以验证真实 cache hit。
证据在 `results/revision-20261024/generation-probabilities/`。

- off 转发与 native：3/3 token 序列完全一致。
- adaptive、上游 example、固定运行时 128、原生容量 128：各 2/3 完全一致；
  短问题在第 10 个 token 出现分歧。原生固定预算也存在相同现象。
- adaptive/example 开启 Prefix Cache 的两轮：各 6/6 完全一致。
- 所有对比在首个分歧处，两个获胜 token 均出现在对方 top-5 中，符合上游
  `check_logprobs_close` 使用的默认判据。这不证明整个后续序列或全部概率数值等价。
- trace 中所有实际调度 token 都不超过选择的预算，也不超过 512 初始化容量；
  自适应组确实执行了低预算步骤，并非只经过高预算路径。

批形状变化引起的浮点敏感性是上述观测的一致解释；并未把这一推断当成
严格数值等价的证明。上游生成测试采用已有概率比较 helper，保留输出长度和
cache hit 断言，没有用强制一致的假 token 或跳过失败来制造通过。

## 复现

在实验目录、已安装的 Linux CUDA 环境中执行：

```bash
AB_TEST_MODEL="$PWD/models/Qwen3-1.7B" HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
for seed in 101 202 303; do
  .venv/bin/python benchmarks/run_sweep.py \
    --policies native8192 static1024 static2048 static4096 adaptive \
    --scenarios burst steady --seeds "$seed" --order-seed "$seed" \
    --no-trace --adaptive-example --output "results/reproduction-$seed"
done
.venv/bin/python benchmarks/analyze.py results/reproduction-{101,202,303} \
  --output results/reproduction-summary
```

每轮保存实际启动命令、配置、源码 SHA-256、模型配置、依赖版本、逐请求 token
和时间数据。`artifacts/benchmark-source-20261024.tar.gz` 保存性能实测后的源码快照，
`artifacts/budget-baseline-20261024.tar.gz` 保存修订前源码。
交付源码在性能实验后另有格式化、未使用配置项校验和 trace 清理修复；
不应声称所有文件字节均与实测时相同。上游预算策略保持一致，并通过差分测试。

## 上游定位与提交前事项

[P-PAS RFC #52906](https://github.com/vllm-project/vllm/issues/52906) 限制的是
prefill token 总量；本扩展限制所有调度 token。旧 `ppas` 组只是共享压力条件的
比较组，不能称为论文实现或原始 P-PAS 的复现。查重还检查了
[EWSJF #33392](https://github.com/vllm-project/vllm/pull/33392)、
[FlowPrefill #44794](https://github.com/vllm-project/vllm/pull/44794) 和
[defer #61080](https://github.com/vllm-project/vllm/pull/61080)。
查重结果保存在 `duplicate-search.json`，状态可能随时间变化。

本人审核、测试确认及 DCO 授权已完成；已公开推送代码和独立证据分支，并创建 Draft PR。
维护者是否愿意接纳该示例仍需确认。后续步骤见 `PUBLICATION.zh-CN.md`；
复现提交步骤见 `SUBMIT.zh-CN.md`，已使用的 AI 辅助说明和 PR 正文保存在 `PR_BODY.md`。
没有另外发送 RFC 评论或请求审查。
