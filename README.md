# vLLM 自适应 token 预算调度扩展

已于 2026-10-24 发布 [vLLM Draft PR #63458](https://github.com/vllm-project/vllm/pull/63458)。提交署名、线上检查状态和后续操作见 [发布记录](PUBLICATION.zh-CN.md)。

在 vLLM 0.22.1 的同步 V1 Scheduler 上实现可配置的运行时 token 预算控制，用于研究长 Prefill 与 Decode 混合负载的延迟取舍。代码、测试和 RTX 5090 实测记录一并保存。

性能、消融与已知退化见 [RTX 5090 实验报告](REPORT.md)，逐请求与逐调度步证据保存在 `results/`。

2026-10-24 修订：新增严格配置校验、`additional_config` 接入、请求 ID 复用清理、抢占后生成 token 重算识别、日志关闭与异常处理，以及真实模型/Prefix Cache 回归。上游示例、文档和 CI 补丁保存在 `upstream_payload/`；本次验证与提交方法分别见 [修订报告](REVISION_REPORT.md) 和 [提交指南](SUBMIT.zh-CN.md)。旧报告和源码审计描述的是 10 月 14 日版本，不能当作当前源码的逐行说明。

## 实现边界

- `src/adaptive_budget/controller.py`：独立状态机，支持原生、固定预算、借用 P-PAS 压力条件的比较组、自适应四种模式。进入压力阈值为 2 个 Prefill，退出阈值为 0；默认至少驻留 4 个调度步。没有混合工作时立即恢复高预算。
- `src/adaptive_budget/scheduler.py`：通过 `scheduler_cls` 继承原生 Scheduler，调用 `super().schedule()`；保留 KV 分配、抢占和请求生命周期。临时改变本步 `max_num_scheduled_tokens`，通过 `finally` 恢复容量；不修改引擎初始化的编译容量。
- 无进展保护：以单调时钟记录尚未完成 Prefill 的请求最近一次 `num_computed_tokens` 增长；停滞超过 250 ms 时，在压力态中每隔至少 8 步把当步预算提升到 4096。它增加推进机会，不重排队列，也不保证请求一定被接纳或永不饥饿。
- 默认高／低预算 8192／1024，调度预算不得超过引擎预先分配的容量。`off` 直接走原生调度，不做请求扫描和逐步日志。
- 验证范围：单 GPU、纯文本、同步 V1、Chunked Prefill、无投机解码。性能评测关闭 Prefix Caching；未声称验证多卡、异步、远程 KV、多模态或线上生产部署。

P-PAS 的压力条件来自 [2026 年论文](https://arxiv.org/abs/2608.15171) 及 [开源实现](https://github.com/TimoSaemann/ppas-vllm/tree/v1.0-paper)。本项目 `ppas` 模式只复用压力条件，限制的是**总调度 token**；[作者 RFC](https://github.com/vllm-project/vllm/issues/52906) 中 P-PAS 限制的是**总 Prefill token**，保持全局预算不变。因此 `ppas` 是规则比较组，不是 P-PAS 实现或论文复现，旧结果不能表述成“战胜原始 P-PAS”。

## 环境与运行

在 Linux CUDA 主机上运行。此次工作目录为 `/root/autodl-tmp/vllm-budget-lab`。Windows 工作区保存源码与结果，模型与虚拟环境仅保留在远程机器。

```bash
bash setup_remote.sh
.venv/bin/uv pip install --python .venv/bin/python -e .
.venv/bin/python download_model.py
AB_TEST_MODEL="$PWD/models/Qwen3-1.7B" .venv/bin/pytest -q
.venv/bin/python benchmarks/run_sweep.py \
  --policies static4096 adaptive ppas native8192 no_guard static2048 no_hysteresis \
  --scenarios burst steady --seeds 101 202 303 \
  --output results/reproduction
```

安装脚本使用阿里云 PyPI 镜像；模型由 ModelScope 下载。环境固定 vLLM 0.22.1、PyTorch 2.11.0+cu130，完整依赖版本见实测 `metadata.json`。`setup_remote.sh` 以自身目录为根目录，保留已有环境；可用 `UV_BIN` 和 `UV_INDEX_URL` 指定 uv 与包索引。

手动运行服务器时，推荐把配置放入原生 `additional_config`，并添加：

```text
--scheduler-cls adaptive_budget.scheduler.AdaptiveBudgetScheduler
--no-async-scheduling --enable-chunked-prefill
--max-num-batched-tokens 8192
--additional-config '{"adaptive_budget":{"high_budget":8192,"low_budget":1024,"guard_budget":4096}}'
```

配置字段见 `PolicyConfig`，可增加 `trace_path` 保存逐步状态、预算、切换次数、保护触发次数和决策耗时。旧的 `AB_CONFIG=/absolute/config.json` 仍可复现实验，但不能与 `additional_config.adaptive_budget` 同时设置。上游精简示例仅保留 adaptive 策略，不含实验模式、文件日志或环境变量入口；其默认策略与实验版通过差分测试核对。

## 评测方法

`benchmarks/run_sweep.py` 启动仅监听 `127.0.0.1:8000` 的服务器。每种策略使用相同模型、BF16、32k 上下文、8192 引擎容量、64 最大序列数、85% 显存比例和 CUDA Graph 配置。固定预算通过运行时控制器收紧，避免改变预分配／编译容量造成混杂。每次启动先预热，再以固定到达时间发起独立 HTTP/SSE 请求。

- burst：4 轮，每轮 12 个请求；短输入 512／输出 384 token，长输入 12288 或 16384／输出 128 token。长请求在短请求开始 Decode 后突发到达，每轮间隔 6 秒。
- steady：48 个请求，每 0.5 秒一个；每三个请求中一个长请求（8192／128），两个短请求（512／256）。
- 输入由公开技术短句重复拼接到精确 token 数，属于合成服务负载，不是语义质量基准。请求 ID 和长度由 seed 固定；`ignore_eos=True`、`temperature=0` 固定输出长度。
- SSE 返回实际 token ID；保存逐 token 接收时间、usage、输出哈希、失败和客户端发送延迟。TTFT 包括 HTTP／排队；TPOT 为首末 token 间隔除以 N−1；ITL 是客户端接收间隔，不能直接解释为 GPU kernel 时间。若一个 chunk 包含多 token，记录其数量并承认时间分辨率限制。
- 校准 seed 11 不计入正式汇总；正式 seed 101／202／303。均值、尾延迟、长短请求和吞吐都报告，保留退化结果；同一合成负载不能推出所有在线场景的加速比。
- `no_hysteresis` 同时去掉阈值滞回与最短驻留，是两个稳定机制的联合消融；`no_guard` 单独关闭无进展预算提升。
- 输出逐请求 JSONL、逐步调度 JSONL、启动命令、配置、源码 SHA-256、依赖版本与汇总 JSON。完成后脚本关闭自己启动的服务器，不关闭远程主机。

## 上游与署名

vLLM v0.22.1 上游提交：`0decac0d96c42b49572498019f0a0e3600f50398`。

P-PAS `v1.0-paper`：`a9ad4530d63f25a856af004ce2a9732495e7f92a`。只引用其压力规则，不把论文性能数据作为本项目结果。

测试中 Scheduler fixture 与抢占场景参考 vLLM `tests/v1/core/utils.py`、`test_scheduler.py`（Apache-2.0，Copyright contributors to the vLLM project）。模型为 Qwen/Qwen3-1.7B，保留下载的模型配置与版本信息。本目录不含 SSH 密码、模型权重或完整第三方源码副本。

对应许可文本见 [VLLM_LICENSE.txt](artifacts/VLLM_LICENSE.txt)。
