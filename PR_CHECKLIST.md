# vLLM PR checklist 审核记录

状态：已准备可审查的示例贡献；提交前仍需本人审核、签署和运行检查。
本次使用基线提交中的 `.agents/skills/pr-checklist/SKILL.md`，逐项结果如下。

## 1. Design Fit

通过现有 scheduler_cls 扩展点实现，所有策略代码置于 examples，未改核心
Scheduler/ModelRunner/defaults，未添加运行依赖。精简版移除了实验模式、trace、
环境变量配置和单独打包层，避免把研究工具带入引擎。

仍需维护者判断：是否需要维护该示例，或应继续作为外置实验。
调用具体 Scheduler 属性仍然与上游实现耦合；升级时需回归。
预算收紧会影响 prefill 和 decode，不保证无饥饿；观察器每步 O(请求数)。
不增加 GPU 张量读取、同步或 CPU/GPU 拷贝。
单卡同步文本范围外明确拒绝；没有声称覆盖异步、分布式或多模态路径。

查重：P-PAS #52906、EWSJF #33392、FlowPrefill #44794、defer #61080。
机制区别见报告。未发现等价的开放示例 PR；提交当天需要重查。

## 2. Testing and Validation

实验版原基线 29 passed；新回归在原 controller 上 13 failed/1 passed，
修复后完整实验测试 54 passed。使用真实 Request、Scheduler 和 KV allocator。
新增上游测试包含配置边界、状态驻留、guard 限频、混合分配与取消、容量恢复、
实际模型生成和 Prefix Cache；不是替换内部实现的 mock 测试。
4×1000 步差分核对实验与精简策略。生成比较采用已有 check_logprobs_close，
首分歧处 top-5 判据范围已披露；未宣称逐 token 恒等。

Buildkite 现有 plugin job 已选择此测试文件；补充 example、helper、scheduler
的 source_file_dependencies，check-test-tethering 和 step-key 校验通过。
本机 RTX 5090 外的 GPU/AMD 和云端 Buildkite 未运行，不能预先打勾声称 CI 通过。
当前 main 新增 16 项全部通过，包括 V2 Model Runner 的两项 GPU 生成测试，
原有 OPT dummy 测试未选中；命令、依赖与限制见 REVISION_REPORT.md 和日志。

30 轮/1440 请求/0 失败性能实验保留所有 baseline、长短切片和回归指标，
只支持有限负载下的延迟取舍，不足以将策略设为默认或称为普遍性能优化。

## 3. Code Quality and Style

遵循 SPDX 头、Ruff、Markdown、拼写、命名等检查。
初始 14 个适用 pre-commit hook 已实际通过。本次修复 Go 下载障碍后，
完整 pre-commit 也已通过：17 项检查通过、19 项无匹配文件，没有使用 SKIP。
没有进行与上游补丁无关的核心格式化。

用户文档包含可复制启动命令、所有配置默认值、兼容边界、退出方式及测量建议，
并从 docs/features/README.md 建立入口。按 docs 贡献说明进行独立 agent
查找验证；根据反馈补明同步/Chunked Prefill 的要求。
未发布分支，因此 GitHub/网站搜索可发现性及文档预览仍需提交后检查。

## 4. Pull Request Contents

PR_BODY.md 保留上游模板 Overview/Claims/Validation/Details/checklist，
包含 AI 辅助披露、性能退化、概率检查限度、硬件和版本范围。
RFC_COMMENT.md 仅为草稿；没有发送评论或重复投稿。
SUBMIT.zh-CN.md 提供应用补丁、复查、本人 DCO 签署、rebase、push、Draft PR 命令。

提交人需要完成：

1. 逐行理解并审核代码，在自己的身份和环境下运行相关验证。
2. 与 RFC 作者/维护者确认示例定位，复查当天的重复工作。
3. 更新到当天 main，解决冲突并重跑测试和完整 pre-commit。
4. 用本人身份签署 DCO，更新 PR 描述，上传可公开访问的证据链接。
5. 检查 CI 与文档预览后再转为 Ready for review。

没有生成替代本人签署的 commit，也没有替用户勾选人工审查或 CI 完成项。
