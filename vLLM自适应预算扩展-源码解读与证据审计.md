# vLLM 自适应 token 预算扩展：源码解读与证据审计

> 对象：`vllm_adaptive_budget` —— 在 vLLM 0.22.1 同步 V1 Scheduler 上加的一层「运行时 token 预算控制器」。
> 本文分两部分：**第一部分逐层读代码**（机制、状态机、边界、坑），**第二部分逐项核证据**（把仓库里的 JSON 结果独立重算一遍，验证 `REPORT.md` 的 14 个数字是否站得住）。
> 所有结论都标注了依据文件和行号。凡是我自己的判断而非代码/数据直接给出的，都写成「**判断**」。

---

## 0. 速览

| 项目 | 事实 |
|---|---|
| 代码体量 | 生产代码 3 个文件约 230 行（`controller.py` 126 行、`scheduler.py` 102 行、`__init__.py` 2 行） |
| 测试 | 29 项通过、0 跳过（`logs/tests.txt`），控制器单测 19 项 + 真实 KV 分配器集成测试 10 项 |
| 接入方式 | `--scheduler-cls adaptive_budget.scheduler.AdaptiveBudgetScheduler`，仅覆盖 `schedule()` 一个方法 |
| 实验规模 | 7 策略 × 2 负载 × 3 seed = 42 轮、2016 请求、0 失败 |
| 硬件 | 单卡 RTX 5090 32 GB，Qwen3-1.7B BF16，vLLM 0.22.1，PyTorch 2.11.0+cu130（`results/main/metadata.json`） |
| 证据自洽性 | `REPORT.md` 引用的 **14 个百分比全部能在 `results/main/summary.json` 上复现**（逐项核过，见 §7） |
| 最大问题 | 「相对原生 P99 ITL 降低 70.7%」这个头条数字的**基线选得极弱**；换成任何其他基线，这个优势降到 4% 量级甚至消失 |
| 第二大问题 | 默认配置下**退出阈值分支不可达**（§4.1），且**驻留门从未被任何测试真正验证过**（§5.2）——这两个机制是 README 列为「新增工作」的四个机制中的两个 |

一句话概括这个项目的真实取舍：**它用「平均 TTFT 变慢 25%、平均 E2E 变慢 10%」换来了「突发负载下 P99 输出间隔（ITL）从 155 ms 降到 45.5 ms」。** 这不是全面加速，是一次明确的尾延迟换均值延迟。

阅读本文的约定：凡属**代码或数据直接给出**的结论，都附了 `文件:行号` 或可复算的数字；凡属**我的推断**，都写成「**判断**」。

---

# 第一部分：源码解读

## 1. 分层结构：纯状态机 + 薄接入层

```
        ┌─────────────────────────────────────────────┐
        │  controller.py   （无 GPU、无时钟、无队列改动）│
        │  Observation ──► BudgetController.decide()   │
        │                        │                     │
        │                    Decision{budget,state,...} │
        └────────────────────────┬────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────┐
        │  scheduler.py    AdaptiveBudgetScheduler     │
        │  继承 vllm Scheduler，只重写 schedule()      │
        │  ① 扫请求 → Observation                      │
        │  ② 决策 → 临时改 self.max_num_scheduled_tokens│
        │  ③ super().schedule() → finally 还原         │
        │  ④ 写 trace / 断言不越界                      │
        └─────────────────────────────────────────────┘
```

这个分层是**这个项目最值得肯定的设计决策**：

- 控制器是 `@dataclass(frozen=True)` 的纯函数式状态机，输入 `Observation`、输出 `Decision`，不碰 GPU、不读时钟（时钟由调用方传入）、不改队列。所以它可以用 20 个纯 CPU 单测覆盖边界，包括 1 万步随机轨迹（[`tests/test_controller.py:79`](tests/test_controller.py#L79)）。
- 接入层只重写 `schedule()`，**不做任何队列重排、不预留配额、不碰 KV 分配**。原生调度器的 KV 分配、抢占、取消、请求生命周期全部原样保留。这是「可回退、可对比」的前提。

**判断**：对研究型扩展来说，这个边界划得很干净。它解释了为什么集成测试能直接复用 vLLM 的真实 KV 分配器而不用起 GPU（[`tests/test_scheduler.py:1-5`](tests/test_scheduler.py#L1-L5)）。

## 2. 配置与校验（`PolicyConfig`）

[`src/adaptive_budget/controller.py:10-44`](src/adaptive_budget/controller.py#L10-L44)

| 字段 | 默认 | 含义 | 校验 |
|---|---|---|---|
| `mode` | `adaptive` | `off` / `static` / `ppas` / `adaptive` | 必须属于这 4 个 |
| `high_budget` | 8192 | 无压力时的每步 token 预算 | `low ≤ high ≤ engine_capacity` |
| `low_budget` | 1024 | 有压力时的每步 token 预算 | `≥ 1` |
| `static_budget` | 8192 | 仅 `static` 模式使用 | `1 ≤ static ≤ engine_capacity` |
| `enter_prefills` | 2 | 进入压力态的 prefill 数阈值 | `exit < enter` |
| `exit_prefills` | 0 | 退出压力态的阈值 | `≥ 0` |
| `min_dwell_steps` | 4 | 两次切换之间至少间隔的调度步数 | `≥ 0` |
| `guard_enabled` | True | 是否启用无进展保护 | — |
| `max_stall_ms` | 250.0 | 判定「停滞」的毫秒阈值 | `> 0` |
| `guard_budget` | 4096 | 保护触发时的预算 | `low ≤ guard ≤ high` |
| `guard_interval_steps` | 8 | 保护最快触发间隔（步） | `≥ 1` |

校验逻辑有三个**按模式短路**的地方，值得注意：

- `mode == "off"` 时**立即 return**（[:29-30](src/adaptive_budget/controller.py#L29-L30)）——只查容量为正，其它字段一律不管。所以 `AB_CONFIG='{"mode":"off"}'` 是合法的。
- `mode == "static"` 时只查 `static_budget`（[:31-34](src/adaptive_budget/controller.py#L31-L34)）——`enter_prefills` 等字段随便填都不报错。
- 只有 `ppas` / `adaptive` 才走全套校验。

**坑（重要）**：`guard_budget` 的校验是 `low ≤ guard ≤ high`，而默认 `guard_budget=4096 > low_budget=1024`。因为 `Decision` 里写的是 `budget = max(budget, cfg.guard_budget)`（[:120](src/adaptive_budget/controller.py#L120)），所以**保护触发时预算永远是 `guard_budget`，`max()` 是冗余的**——在 `limited` 分支里 `budget` 恒等于 `low_budget`。这不影响行为，但会让读者误以为存在「保护叠加在当前预算上」的额外逻辑。**判断**：改成 `budget = cfg.guard_budget` 更诚实，或干脆允许 `guard_budget < low_budget`（当前校验禁止了这种配置）。

## 3. 观测层：`_ab_observe()`

[`src/adaptive_budget/scheduler.py:50-72`](src/adaptive_budget/scheduler.py#L50-L72)

每一步调度前扫一遍 `self.requests`，产出 4 个量：

| 字段 | 计算方式 | 用途 |
|---|---|---|
| `prefills` | 状态属于 `{WAITING, RUNNING, PREEMPTED}` 且 `num_computed_tokens < num_prompt_tokens` 的请求数 | **状态机主输入** |
| `decodes` | 状态为 `RUNNING` 且已算完 prompt 的请求数 | **状态机主输入** |
| `oldest_stall_ms` | 所有未完成 prefill 中最大的「距上次前进的毫秒数」 | 保护触发条件 |
| `pending_prefill_tokens` | 未完成的 prompt token 总数 | **只写进 trace，任何决策都不用** |

几个实现细节：

- **阻塞请求被排除**：`RequestStatus` 不在允许集合里的（例如 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`、已完成）直接 `continue`。测试专门验证了这一点（[`tests/test_scheduler.py:92-94`](tests/test_scheduler.py#L92-L94)）。
- **停顿计时用单调时钟**：`time.monotonic()` 由调用方传入（[:78](src/adaptive_budget/scheduler.py#L78)），控制器本身不读时钟——这正是纯状态机能够单测的原因。
- **抢占回滚不算前进**：`if req.num_computed_tokens > old_count` 才刷新 `last_progress`（[:64-66](src/adaptive_budget/scheduler.py#L64-L66)）。被抢占的请求 `num_computed_tokens` 会回退，如果写成 `!=`，回退会被误判成前进。这行写得对。
- **状态清理**：每步用 `self.ab_progress.keys() - live` 求差集后删除（[:70-71](src/adaptive_budget/scheduler.py#L70-L71)）。因为 `keys() - live` 先构造出**新集合**再迭代，不存在「迭代中改字典」的问题。

**坑 / 改进点**：
1. `pending_prefill_tokens` 是**死字段**——采了、记了、从不用。要么用它做压力信号（比如按剩余工作量而非请求个数判压力，这也正是 `REPORT.md` §后续工作 自己提出的方向），要么删掉以免误导。
2. 每步扫全部请求，复杂度 **O(R)**（R = 在飞请求数）。这里 `max-num-seqs=64`，实测决策中位数 12.5 μs，可忽略；但大队列下这是线性开销，且**没有任何上界或采样**。报告自己也承认「大队列开销需要另测」。

## 4. 决策层：`BudgetController.decide()`

[`src/adaptive_budget/controller.py:76-125`](src/adaptive_budget/controller.py#L76-L125)。调用顺序：

```
1. 负值检查（prefills/decodes/oldest_stall_ms/pending_prefill_tokens 任一 < 0 就抛错）
2. self.step += 1                      ← 注意：off/static 也照样加
3. mode == off      → 直接返回 Decision(engine_capacity, "OFF", "native")
4. mode == static   → 直接返回 Decision(static_budget, "STATIC", "fixed")
5. 以下仅 ppas / adaptive：
   a. ppas 模式：limited = (prefills >= 2 and decodes > 0)，无滞回无驻留
   b. adaptive：(
        decodes == 0 或 prefills == 0  → limited=False, reason="no_mixed_work"   ← 绕过驻留
        否则若 step - last_switch >= min_dwell_steps → 进入/退出/保持判定
        否则 → reason="residence_hold"（什么都不改）
      )
6. switched = (上一态的 limited != 当前 limited)；若切换则更新 last_switch、switches += 1
7. budget = low_budget if limited else high_budget
8. guard 判定（仅 adaptive / guard_enabled / limited / prefills>0 / 停滞超阈值 / 冷却已过）
9. 返回 Decision
```

### 4.1 三种策略的语义差异

| | 进入条件 | 退出条件 | 驻留门 |
|---|---|---|---|
| `ppas` | `prefills ≥ 2 且 decodes > 0` | 条件不成立即退出 | **无**（每步都重算） |
| `adaptive` | `prefills ≥ 2`，且距上次切换 ≥ 4 步 | `prefills ≤ 0`，且距上次切换 ≥ 4 步 | **有**（4 步） |
| `no_hysteresis`（实验用配置） | 同上 | `prefills ≤ 1` | 0 步 |

**关键澄清一**：`no_hysteresis` 这个消融配置**并没有真正去掉滞回**——它只是把「退出阈值从 0 改成 1」并「把驻留步数从 4 改成 0」。而 `enter_prefills` 恒为 2、`exit_prefills=0` 时，滞回窗口本来就等于 `[1, 2)`——只有一个整数宽的窗口。所以「去掉滞回」在数值上只剩下 `exit=1` 这一项实际起作用。**判断**：消融标签比实际改动更响亮，建议改名为 `no_dwell`，或拆成两个独立消融。

**关键澄清二（这条比上一条重要得多）**：在**出厂默认配置下，退出阈值这条分支永远不会被执行**。

```python
elif obs.decodes == 0 or obs.prefills == 0:      # :90  ← 一定先命中这里
    self.limited = False
    reason = "no_mixed_work"
elif self.step - self.last_switch >= cfg.min_dwell_steps:
    ...
    elif self.limited and obs.prefills <= cfg.exit_prefills:   # :98  ← 因此到不了
        self.limited = False
        reason = "exit_pressure"
```

因为 `prefills` 已被校验为非负（[:77](src/adaptive_budget/controller.py#L77)），`prefills <= exit_prefills` 在 `exit_prefills=0` 时**等价于 `prefills == 0`**——而 `prefills == 0` 必然先被 [:90](src/adaptive_budget/controller.py#L90) 的 `no_mixed_work` 分支吃掉。按代码逐行可判定：把 `limited` 预置为 `True`，喂 `prefills=0` 只会得到 `no_mixed_work`；喂 `prefills=1` 得到 `hysteresis_hold`；`exit_pressure` 不可能出现。只有把 `exit_prefills` 设成 ≥ 1 时它才会出现——也就是说，**唯一能让这条分支执行的配置，恰好只有 `no_hysteresis` 这一个消融组**。

由此得到一张读者最容易读错的表：

| 路径 | 是否受 `min_dwell_steps` 约束 |
|---|---|
| 进入压力态 `enter_pressure`（[:95-97](src/adaptive_budget/controller.py#L95-L97)） | ✅ 受约束 |
| 退出压力态 `exit_pressure`（[:98-100](src/adaptive_budget/controller.py#L98-L100)） | ✅ 受约束，但**默认配置下不可达** |
| 混合工作消失即释放（[:90-93](src/adaptive_budget/controller.py#L90-L93)） | ❌ **绕过驻留直接释放** |

也就是说，`REPORT.md:34` 里「最短驻留 4 步」在日常工况下**只约束"进入"这一个方向**：一旦混合工作消失，驻留门就被绕过。代码注释 `# No mixed-phase interference: bypass residence and release capacity.`（[:91](src/adaptive_budget/controller.py#L91)）说明这是有意设计，也有测试固定住（[`tests/test_controller.py:44-48`](tests/test_controller.py#L44-L48)：`min_dwell_steps=100` 时，一个无混合工作的观测仍立刻返回 8192），但它没有在任何文档里被点明。

**这条澄清直接改写了消融结果的含义**，见 §10.3。

### 4.2 驻留门的真实效果（实测切换计数）

先看实测。`diagnostics.json` 里三种策略的切换计数：

| 策略 | 切换次数 | 保护触发 |
|---|---|---|
| `adaptive` | 36 | 50 |
| `no_hysteresis` | 38 | 51 |
| `ppas`（无驻留） | 38 | 0 |

（来自 [`results/main/diagnostics.json`](results/main/diagnostics.json)）

**真实负载下 `adaptive` 和 `no_hysteresis` 的切换次数只差 2 次（36 vs 38）**，与完全没有稳定机制的 `ppas`（38 次）也几乎一样。这说明在 burst/steady 这两种负载下**阈值抖动根本没有发生**，滞回与驻留没有起作用的机会。

另一处反直觉的小设计：初始化时 `last_switch = -min_dwell_steps`（[:71](src/adaptive_budget/controller.py#L71)），默认值 4 时即 `-4`，所以第 1 步 `step - last_switch = 1 + 4 = 5 ≥ 4`，**第一步就能切换**。这是有意的（避免冷启动被门卡住），但没有注释，读者容易误以为「前 4 步一定不能切换」。`last_guard` 同理（[:72](src/adaptive_budget/controller.py#L72)），所以保护首次触发也不受冷却约束。

### 4.3 无进展保护（guard）

触发条件（[:111-118](src/adaptive_budget/controller.py#L111-L118)）：

```python
guard = (cfg.mode == "adaptive" and cfg.guard_enabled and self.limited
         and obs.prefills > 0
         and obs.oldest_stall_ms >= cfg.max_stall_ms        # 默认 250 ms
         and self.step - self.last_guard >= cfg.guard_interval_steps)  # 默认 8 步
```

效果：本步预算抬到 `guard_budget`（4096），`last_guard = self.step`，`reason = "stalled_prefill_boost"`，**`state` 不变（仍是 LIMITED）、`switched` 不变**。

代码里有两处很克制的注释，值得引用：

- 类文档：`A pure state machine: no GPU calls, wall-clock assumptions, or queue mutation.`（[:1](src/adaptive_budget/controller.py#L1)）
- 返回前：`# A boost creates an opportunity; it is not an admission/fairness guarantee.`（[:124](src/adaptive_budget/controller.py#L124)）

**这两句话必须认真读**：guard **不重排队列、不预留配额、不保证被停滞的请求一定会被接纳**。它只是「这一步把闸门开大一点，让原生调度器有机会带上它」。`REPORT.md` 说「不会重排队列或预留专属配额，不能保证过载下不饥饿」（§正确性与调度开销），与此一致。

**坑一（限速但没有上界）**：保护的冷却单位是**调度步数**，不是时间——`self.step - self.last_guard >= cfg.guard_interval_steps`（[:117](src/adaptive_budget/controller.py#L117)），而 `self.step` 每调用一次 `decide()` 就 +1（[:80](src/adaptive_budget/controller.py#L80)），每次 `schedule()` 调用一次 `decide()`。一个长期拿不到调度的请求会让 `oldest_stall_ms` 单调增长（它只在真正前进时才重置，抢占回滚不算前进，[:62-67](src/adaptive_budget/scheduler.py#L62-L67)），于是**每 8 步触发一次，永不停止，且没有次数上限或退避**。在 burst 轨迹里触发 50 次，正说明它在长期反复触发。这是「持续性提升」而非「一次性补救」。另外，`guard_interval_steps=8` 是步数，`max_stall_ms=250` 是墙钟毫秒——引擎转得快时 8 步可能远小于 250 ms，所以「限速」不是时间意义上的限速。

**坑二（纯 prefill 场景下保护结构性失效）**：触发条件里有 `and self.limited`（[:114](src/adaptive_budget/controller.py#L114)），而 `limited` 在 `decodes == 0` 时会被 [:90-93](src/adaptive_budget/controller.py#L90-L93) 强制置为 `False`（`no_mixed_work` 分支）。因此**当一个纯长 prefill 独占引擎、没有任何 decode 在跑时，保护永远不会触发**——而这恰恰是最容易发生长 prefill 饥饿的场景。`prefills > 0` 这个条件也因此在 `limited` 为真的前提下恒成立（因为 `no_mixed_work` 已排除 `prefills == 0`），属于冗余条件。

**坑三（保护覆盖了 reason）**：触发时 `reason = "stalled_prefill_boost"`（[:123](src/adaptive_budget/controller.py#L123)）会**覆盖**掉原本的 `hysteresis_hold` / `residence_hold`。所以 trace 里看到 `stalled_prefill_boost` 时，无法知道同一时刻滞回或驻留是处于什么判断。`state` 和 `switched` 不受影响。

反向证据：`no_guard` 的 `max_observed_stall_ms` 是 1862.9 ms，`adaptive` 是 1650.0 ms（[`results/main/diagnostics.json`](results/main/diagnostics.json)）。也就是说**保护把观测到的最长停滞从 1.86 s 压到 1.65 s，只降了 11%**。这从侧面支持了作者自己「不保证不饥饿」的措辞。

## 5. 接入层：`AdaptiveBudgetScheduler.schedule()`

[`src/adaptive_budget/scheduler.py:74-102`](src/adaptive_budget/scheduler.py#L74-L102)

```python
def schedule(self):
    if self.ab_config.mode == "off":
        return super().schedule()                      # 零开销旁路
    started = time.perf_counter_ns()
    obs = self._ab_observe(time.monotonic())
    decision = self.ab_controller.decide(obs)
    decision_us = (time.perf_counter_ns() - started) / 1000
    saved_budget = self.max_num_scheduled_tokens
    self.max_num_scheduled_tokens = decision.budget   # ← 临时收紧
    try:
        result = super().schedule()                   # ← 原生调度原样执行
    finally:
        self.max_num_scheduled_tokens = saved_budget  # ← 必定还原
    scheduled = sum(result.num_scheduled_tokens.values())
    if scheduled > decision.budget:
        raise AssertionError("Native scheduler exceeded selected token budget")
    ... 写 trace ...
    return result
```

这段 28 行代码里有 4 个正确的工程决策，值得单独点出来：

1. **`try/finally` 还原容量**。这是整个集成里最危险的一处状态改写，用 `finally` 兜住，且**有专门测试**模拟 `super().schedule()` 抛异常后容量是否还原（[`tests/test_scheduler.py:71-78`](tests/test_scheduler.py#L71-L78)）。这是我认为写得最好的一处。
2. **越界断言**。每一步都验证原生调度器返回的 token 总数没超过本步预算，把「预算被绕过」变成硬失败而不是静默偏差。实测 19652 步、0 次越界。
3. **`off` 模式旁路逐步观测与逐步日志**。[:75-76](src/adaptive_budget/scheduler.py#L75-L76) 在 `_ab_observe`（[:78](src/adaptive_budget/scheduler.py#L78)）和 trace 块（[:90-101](src/adaptive_budget/scheduler.py#L90-L101)）之前就返回，所以 README 说的「不做请求扫描和逐步日志」是准确的。**但严格说不等于「不产生 trace 文件」**：`__init__` 里 [:41-48](src/adaptive_budget/scheduler.py#L41-L48) 与 mode 无关，只要配了 `trace_path`，`off` 模式仍会建目录、开文件、写入**一行 `{"type":"config",...}`**、flush 并注册 `atexit` 关闭。所以 `off` 组可能留下一个只有配置行的 `.scheduler.jsonl`——`inspect_evidence.py` 恰好会跳过它（因为没有 `type == "step"` 的行），但这个空壳文件容易被误读。同时 `off` 模式是唯一跳过 [:30-38](src/adaptive_budget/scheduler.py#L30-L38) 那 4 项负向校验的模式（因为 `validate()` 在 `off` 时直接返回，[:29-30](src/adaptive_budget/controller.py#L29-L30)），也就是说 `off` 下允许异步调度、无 chunked prefill、多卡、投机解码。
4. **容量来源**：`self.ab_capacity = self.max_num_scheduled_tokens`（[:28](src/adaptive_budget/scheduler.py#L28)），即把引擎初始化时的每步 token 上限当作不可超越的天花板，绝不去改引擎预分配/编译容量。

### 5.1 接入所依赖的私有属性（强推断，但未读到上游源码）

上面整段逻辑**成立的前提是 `Scheduler.max_num_scheduled_tokens` 真的是调度器每步读取的字段**。本机没有 vLLM，所以这条因果链无法用「读上游源码」的方式闭合（完整讨论见 §9）。这里先给出旁证与它的性质。

支持它的旁证相当强：

- 集成测试直接断言 `s.max_num_scheduled_tokens` 的值和还原行为（[`tests/test_scheduler.py:60-68`](tests/test_scheduler.py#L60-L68)）——如果这个属性不存在，[`src/adaptive_budget/scheduler.py:28`](src/adaptive_budget/scheduler.py#L28) 会在每次构造时 `AttributeError`；
- 5 档预算测试（512/1024/2048/4096/8192）在 12000 token 的长 prompt 上，断言 `out.num_scheduled_tokens == {'long': budget}`（[:59-64](tests/test_scheduler.py#L59-L64)）。**如果原生调度器只读 `scheduler_config.max_num_batched_tokens`，这 5 档都会得到 8192，其中 4 档必然失败**——它们通过，是「`super().schedule()` 确实读这个实例属性」的因果证据；
- `--max-num-batched-tokens 8192` 与 `ab_capacity` 实测一致（trace 第一行 `"capacity": 8192`）；
- 19652 步中 `scheduled_tokens > budget` 出现 0 次。

同时要如实说明它的**性质**：`logs/scheduler-help.txt` 是 0.22.1 `SchedulerConfig --help` 的实拍，里面只有 `--max-num-batched-tokens`（第 44 行），**没有 `--max-num-scheduled-tokens`**。也就是说 `max_num_scheduled_tokens` 不是公开配置面，**改它是一个私有 API 触碰**，跨 vLLM 版本不保证稳定——这一点值得在 README 里写明。

### 5.2 测试覆盖的真实边界

29 项测试是**真的在跑**（`logs/tests.txt`：`29 passed, 14 warnings in 7.31s`，且 0 skipped），而且覆盖率在同类研究代码里算好的：非法配置 8 组参数化、1 万步随机轨迹、负观测拒绝、真实 KV 分配器上的 5 档预算、小 KV 容量下的抢占/恢复/缓存归还、取消清理、异常恢复。

但那个最显眼的「滞回降低抖动」测试，需要大幅降低预期：

```python
for p in [2, 1] * 20:          # 交替 2 → 1 → 2 → 1 …
    adaptive.decide(Observation(p, 4))
    ppas.decide(Observation(p, 4))
assert adaptive.switches == 1
assert ppas.switches == 40
```

[`tests/test_controller.py:24-31`](tests/test_controller.py#L24-L31)

**这个测试完全没有检验到驻留门。** 序列里 `prefills` 在 2 和 1 之间交替，永远到不了 0，所以退出分支（需要 `prefills <= exit_prefills == 0`）**永远不可能成立**，控制器一旦进入 LIMITED 就不会想离开；驻留门在这种情况下只能产生几条 `residence_hold` 标签（40 步里有 3 步，其余 1 步 `enter_pressure`、36 步 `hysteresis_hold`），其判断结果与不设驻留时完全相同。也就是说：**把 `min_dwell_steps` 设成 0、1、4 还是 1000，`switches` 都等于 1。**

所以 `adaptive.switches == 1` 这个断言**检验的是退出阈值的非对称性（死区），与 `min_dwell_steps` 无关**，但测试名会让人把功劳同时记给驻留机制。而 `ppas.switches == 40`（= 20 次交替 × 2）只是在验证计数器本身。

它**没有**触及：退出路径、`exit_prefills ≥ 1`、保护（该测试里被 `guard_enabled=False` 关掉）、预算与状态断言、无混合工作旁路。综合 §4.1 的「退出分支在默认配置下不可达」和这里的「驻留门未被测到」，可以下一个明确结论：**这个项目没有任何证据表明 `min_dwell_steps` 曾经发挥作用——既没有性能证据（§10.3），也没有贴近真实抖动的功能证据。**

`REPORT.md:97` 自己写得很清醒：「不能把人工 40 步切换测试的降幅写成服务性能收益」。这句自我约束是这份报告可信度的一个重要来源。

### 5.3 从未被任何测试执行到的代码

- **整个 trace 写盘分支**（[`src/adaptive_budget/scheduler.py:87-101`](src/adaptive_budget/scheduler.py#L87-L101)）在测试里从不执行：fixture 只写 `dict(mode=mode, static_budget=budget)`（[`tests/test_scheduler.py:25`](tests/test_scheduler.py#L25)），从不设 `trace_path`，所以每个测试里 `self.ab_trace is None`。这意味着 `**asdict(obs)` 合并、`json.dumps`、以及 `len(result.preempted_req_ids)`（[:95](src/adaptive_budget/scheduler.py#L95)）**全都没有测试覆盖**——它只被 benchmark 在带外跑到过。**判断**：建议补一个带 `trace_path` 的测试（指向 `tmp_path`），否则上游哪天把 `preempted_req_ids` 改成 `None`，每一步都会 `TypeError` 而测试全绿。
- **退出路径**在默认配置下不可达（§4.1），所以 `exit_prefills=0` 的实际行为没有测试断言。
- **`no_hysteresis` 的配置组合**（`exit_prefills=1` 且 `min_dwell_steps=0`）只在压测脚本里出现，控制器单测没有覆盖这个组合。

### 5.4 两个更细的语义坑

- **`switches_total` 混入了非阈值事件**：`switched = previous != self.limited`（[:106](src/adaptive_budget/controller.py#L106)）是在 `no_mixed_work` 分支可能已经把 `limited` 置为 `False`（[:92](src/adaptive_budget/controller.py#L92)）**之后**求值的。所以「负载从混合变成不混合」这一次释放也会被记成一次 `switched`，并重置 `last_switch`。而 trace 字段名是 `switches_total`（[:96](src/adaptive_budget/scheduler.py#L96)），文档也管它叫「状态切换次数」——**它其实是"阈值切换 + 负载变单纯"的混合计数**。§4.2 里 36/38/38 这三个数字因此不能直接当作「阈值抖动次数」来读。
- **状态抖动的请求永远不会触发保护**：`_ab_observe` 对不在 `{WAITING, RUNNING, PREEMPTED}` 里的请求会删掉进度记录（[:56](src/adaptive_budget/scheduler.py#L56)、[:70-71](src/adaptive_budget/scheduler.py#L70-L71)），而重新进入时又通过 `.get(req_id, (..., now))` 把 `last_progress` 重置为当前时刻（[:62](src/adaptive_budget/scheduler.py#L62)）。所以一个在阻塞状态（例如 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`）之间来回跳的请求**永远累积不出停滞时间，也就永远等不到保护**。这是有意为之（[`tests/test_scheduler.py:92-94`](tests/test_scheduler.py#L92-L94) 固定了该行为），但没有在任何地方说明它是保护的一个盲区。
- **trace 刷盘条件像是写反了**：`if not obs.prefills or self.ab_controller.step % 128 == 0: flush()`（[:100](src/adaptive_budget/scheduler.py#L100)）——`not obs.prefills` 在**每个纯 decode 步**都为真，等于在最常见的稳定态里每步都 flush，把 [:44](src/adaptive_budget/scheduler.py#L44) 设的 64 KiB 缓冲完全抵消。语义上更像是「有 prefill 时反而应该及时刷」。严重性低（只是 I/O 开销），但值得改。

## 6. 运行与配置入口

- **配置注入**：`AB_CONFIG` 环境变量指向一个 JSON 文件路径（[:23-27](src/adaptive_budget/scheduler.py#L23-L27)）。`trace_path` 被 `pop` 出来单独处理，其余字段直接 `PolicyConfig(**raw)` 展开——**多写一个字段就是 `TypeError`，少写就取默认值**。12 个配置字段在每份 `*.config.json` 里都是全量写出的，可复现性好。
- **端口**：`run_sweep.py` 把服务固定监听 `127.0.0.1:8000`。
- **环境负向校验**（[:30-38](src/adaptive_budget/scheduler.py#L30-L38)）：非 `off` 模式下会主动拒绝 4 种情况——开了异步调度、有投机解码配置、没开 chunked prefill、`world_size != 1`。**这是很好的做法**：把「验证范围」写成运行时的硬约束，而不是只写在 README 里。
- **日志**：`atexit.register(self.ab_trace.close)`（[:48](src/adaptive_budget/scheduler.py#L48)），`buffering=65536`，`flush` 条件是「本步没有 prefill」或「每 128 步」（[:100-101](src/adaptive_budget/scheduler.py#L100-L101)）——既不至于丢关键状态，也不至于每步 fsync。
- **断言**：`scheduled > decision.budget` 会抛 `AssertionError`（[:88-89](src/adaptive_budget/scheduler.py#L88-L89)）。在生产里这会变成 500，但对实验代码这是有意的 fail-loud。

---

# 第二部分：证据审计

这一部分是我把仓库里的 JSON 独立重算的结果。**方法声明**：所有重算只用 `results/main/summary.json` 作为原始输入，不依赖仓库脚本的输出；并且额外做了一次「用仓库自己的 `benchmarks/analyze.py` 在临时目录重跑」的交叉验证（§8）。

## 7. `REPORT.md` 的 14 个数字，逐项核验

| `REPORT.md` 的说法 | 报告值 | 我从 `summary.json` 重算 | 结论 |
|---|---|---|---|
| 相对 P-PAS 突发平均 E2E 降低 | −3.3% | −3.32% | ✅ |
| 相对 P-PAS 突发 P95 TTFT 降低 | −8.4% | −8.44% | ✅ |
| 相对原生突发 P99 ITL 降低 | −70.7% | −70.66% | ✅ |
| 相对原生突发平均 E2E 增加 | +10.2% | +10.19% | ✅ |
| `no_guard` 消融：平均 E2E 降低 | −3.1% | −3.06% | ✅ |
| `no_guard` 消融：P95 TTFT 降低 | −8.6% | −8.61% | ✅ |
| `no_guard` 消融：P99 ITL 增加 | +3.4% | +3.39% | ✅ |
| `no_hysteresis`：平均 E2E 增加 | +0.3% | +0.26% | ✅ |
| `no_hysteresis`：P95 TTFT 增加 | +0.4% | +0.35% | ✅ |
| 保护代价：长请求 P99 ITL 增加 | +120.4% | +120.36% | ✅ |
| 相对固定 4096：平均 E2E 增加 | +7.1% | +7.08% | ✅ |
| 相对固定 4096：P99 ITL 降低 | −56.8% | −56.76% | ✅ |
| 平稳负载：平均 E2E 增加 | +1.4% | +1.38% | ✅ |
| 平稳负载：平均 TTFT 增加 | +0.7% | +0.68% | ✅ |

**14/14 全部复现，误差都在四舍五入范围内。** 同时 `results/summary/aggregate.json` 与 `summary.json` 的逐栏重算**完全一致（0 处不符）**，`aggregate.csv` 在重跑时**逐字节相同**。

结论：**这份报告里的数字没有编造，引用是忠实的。** 这在同类项目里不常见，值得明确肯定。

## 8. 数据链的三重一致性

1. **聚合层**：`aggregate.json` / `aggregate.csv` / `tables.md` 与 `summary.json` 重算一致。
2. **配对层**：`paired_comparisons.json` 的 **192 个 `mean_percent`、192 个 `min_percent`、192 个 `max_percent`、192 组 `per_seed_percent`** 全部与 `benchmarks/analyze.py:50-53` 声明的公式一致：
   ```python
   direction = 1 if metric == 'output_tokens_per_s' else -1
   changes = [direction * (candidate[s][metric] / base[s][metric] - 1) * 100 for s in common]
   ```
   即**正值 = 自适应更好**。这一点我做了独立交叉验证：用仓库自己的 `analyze.py` 在临时目录重跑，重生成的 `paired_comparisons.json` 与提交版本在 **576 个标量上 0 处不同**。
3. **源码层**：`results/main/metadata.json` 里记录的 6 个源文件 SHA-256，与当前磁盘上的文件**全部 MATCH**。也就是说「跑的代码 = 仓库里的代码」。

> 说明：`aggregate.json` 与 `paired_comparisons.json` 对同一个比较会给出略有差异的百分比（例：`short_itl_ms_p99` 在 `adaptive vs native8192/burst` 上是 **配对法 −23.6%**，而用 `aggregate.json` 两个聚合值直接相除是 **+4.2%**）。这不是 bug，而是**两种不同的估计量**：配对法是「先算每个 seed 的变化率，再对 3 个 seed 取平均」，聚合相除是「先对 3 个 seed 取均值，再算一次变化率」。7 个指标上这个差异超过 0.05 个百分点，其中 `short_itl_ms_p99` 差到 27.8 个百分点——**这个指标在两种口径下符号相反**。原因见 §10.2。

## 9. 待确认项：`max_num_scheduled_tokens` 是否为上游真实字段

**结论：因果链的强度比我最初判断的高得多，但严格意义上仍是「强推断」而非「读源码确认」。**

我专门为此做了一轮核查，确认 **vLLM 0.22.1 在这台机器上确实没有安装**（`import vllm` 直接 `ModuleNotFoundError`；`D:\minconda`、其 10 个子环境、`C:\Users\NslipWrind\anaconda3` 下均无 vllm 包；全盘搜 `scheduler.py` 只命中本仓库、一个重复副本和 `nanovllm`）。虚拟环境按 `README.md:19` 只保留在远程机器（`logs/pip-freeze.txt` 记录的是 `/root/autodl-tmp/vllm-budget-lab/.venv` 里的 `vllm==0.22.1`）。**所以上游源码无法逐行阅读。**

支持「该属性真实且被调度器读取」的证据（按强度排序）：

1. **最强的一条**：集成测试在 12000 token 的长 prompt 上，对 5 档预算断言 `out.num_scheduled_tokens == {'long': budget}`（[`tests/test_scheduler.py:59-64`](tests/test_scheduler.py#L59-L64)），而 fixture 里 `max_num_batched_tokens=8192` 是固定的。**如果原生调度器只读 `scheduler_config.max_num_batched_tokens`，这 5 档都会返回 8192，其中 4 档必然断言失败。** 它们通过，只能解释为 `super().schedule()` 读的是被临时改写的实例属性。
2. `logs/tests.txt` 是 `29 passed, 14 warnings`，**0 skipped**。也就是说 `tests/test_scheduler.py` 这 10 项（含上面那条）确实在装有 vLLM 的环境里执行过，而不是被 `importorskip` 跳过。
3. 属性不存在就会立刻崩：`self.ab_capacity = self.max_num_scheduled_tokens`（[`src/adaptive_budget/scheduler.py:28`](src/adaptive_budget/scheduler.py#L28)）会在每次构造时 `AttributeError`。
4. trace 首行 `"capacity": 8192` 与 `--max-num-batched-tokens 8192` 一致，说明 `__init__` 时该属性取值正确。
5. 19652 步中 `scheduled_tokens > budget` 出现 **0 次**，说明收紧确实传导到了调度结果。

**但有一条必须如实写上**：`logs/scheduler-help.txt`（0.22.1 `SchedulerConfig --help` 的实拍）里只有 `--max-num-batched-tokens`（第 44 行），**没有 `--max-num-scheduled-tokens`**。所以 `max_num_scheduled_tokens` 不是公开配置面——**改它属于触碰私有 API**，跨版本不保证稳定。

**建议**：在能访问上游的环境里跑一句 `grep -rn "max_num_scheduled_tokens" $(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/v1/`，把输出记进 README。另外在 README 里明说「依赖一个非公开属性」，这对读者比藏起来更有用。

## 10. 方法论问题与证据缺口

### 10.1 头条数字的基线选得极弱（最重要）

「相对原生 8192 预算，P99 输出间隔降低 70.7%」——这个数字来自 `native8192` 在 burst 下的 ITL P99 = **155.21 ms**。但同一张表里：

| 策略 | burst P99 ITL (ms) | burst 平均 E2E (ms) |
|---|---:|---:|
| `native8192` | **155.21** | 3285.2 |
| `static4096` | 105.31 | 3380.6 |
| `static2048` | 68.79 | 3528.1 |
| `ppas` | 44.07 | 3744.2 |
| `no_guard` | **44.04** | 3733.9 |
| `adaptive` | 45.54 | 3619.8 |

**原生 8192 在这个负载下的 P99 ITL 是全场最差，比次差的 `static4096` 还差 47%。** 而 `adaptive` 的 ITL（45.54）**比它自己的消融组 `no_guard`（44.04）还差 3.4%**。

**判断**：70.7% 这个数字在算术上完全正确，但它衡量的是「自适应 vs 一个在 ITL 上表现异常的基线」。任何「把预算压到 1024~4096」的配置都能拿到接近的 ITL 收益——`static2048` 没有控制器、没有保护、没有滞回，就拿到了 55.7% 的 ITL 改善（68.79 vs 155.21），而平均 E2E 反而比 `adaptive` 更好（3528 vs 3620）。

**如果读者只记住一个数字是 70.7%，他记住的是一个由基线选择主导的数字。** 建议报告把头条改成对 `static2048`/`static4096` 的比较——因为那才是「必须有一个控制器才能做到的事」。

### 10.2 切片指标（`short_itl_ms_p99`）在两种口径下符号相反

该指标的每个 run 值是「该 run 内所有短请求 token 间隔的 99 分位」。从 `REPORT.md` 按长短拆分表看，burst 短请求的 ITL P99 只有约 30~46 ms，也就是说**每个 run 里真正进入 99 分位的样本极少，这个量接近「最大值」统计量，方差极大**。证据是逐 seed 值：

| seed | adaptive | native8192 | 配对变化率 |
|---|---:|---:|---:|
| 101 | 43.21 | **19.83** | −117.9% |
| 202 | 44.26 | 51.86 | +14.6% |
| 303 | 44.89 | 66.51 | +32.5% |
| **平均** | **44.12** | **46.07** | **−23.6%** |

`native8192` 的 seed 101 是 19.83、seed 303 是 66.51——**同一个策略、同一负载、不同 seed 差 3.35 倍**。而 `adaptive` 三个 seed 都稳定在 43~45。

**判断**：
- 这个指标在两个 seed 上说 adaptive 更好、在一个 seed 上说差 118%，**均值 −23.6% 是被 seed 101 的离群值主导的**，不构成稳定结论。
- 用聚合值相除（`aggregate.json` 口径）反而得到 +4.2%（adaptive 更好），与配对法符号相反。两个口径都不该被当作结论。
- 建议：要么放弃 p99/切片切片交叉的指标，要么报告「达标的请求比例」而不是分位数，正如 `REPORT.md` §后续工作 自己提出的「明确 ITL SLO 并报告达标率」。

### 10.3 消融的真实结论：两个机制都没兑现，而且驻留门从未被测到

把消融结果正面摆出来，而不是只看标题：

| 对比（burst） | 平均 E2E | P95 TTFT | P99 ITL |
|---|---:|---:|---:|
| `adaptive` vs `no_guard`（关掉保护） | +3.06% | +8.61% | +3.39% |
| `adaptive` vs `no_hysteresis`（关掉滞回+驻留） | +0.26% | +0.35% | +0.10% |
| `adaptive` vs `ppas`（关掉两者） | +3.32% | +8.42% | −3.31% |

**滞回／驻留的净收益约 0.26%——低于 1%，与运行间波动同量级。** 这是本项目最重要的负面发现，`REPORT.md:97` 也诚实写出来了：「本组负载没有显示滞回／驻留带来明显端到端收益」。

但把源码结论叠加进来之后，这个否定可以下得更重、更精确。三条独立证据指向同一件事：

1. **性能上**：真实负载下切换次数 36（adaptive）vs 38（no_hysteresis）vs 38（ppas）——抖动根本没发生，机制没有触发机会（§4.2）。
2. **结构上**：默认 `exit_prefills=0` 时退出分支**不可达**，而退出路径正是唯一受驻留门约束、且真正会降低预算切换的那条路径（§4.1）。所以「驻留门限制退出」这件事在出厂配置下不存在。
3. **测试上**：唯一声称验证驻留机制的单测，其断言值在 `min_dwell_steps ∈ {0, 1, 4, 1000}` 下**完全相同**（§5.2）——它测的是退出阈值死区，不是驻留。

**判断**：所以本项目真正有效的稳定机制**只有「退出阈值取值」这一个参数**（`exit: 0 → 1` 也就是 `no_hysteresis` 组的唯一实质改动），而它带来的差异是 0.26%（E2E）/ 0.35%（P95 TTFT）——在只有 3 个 seed、没有锁 GPU 时钟的实验里，这个量级无法与系统性漂移区分。**建议要么补一个会真正跨越进入/退出两个阈值的轨迹把驻留门测出来，要么把 `min_dwell_steps` 从结论与 README 的卖点里撤下来。** 目前把它列为「新增工作」的四个机制之一，证据是不足的。

### 10.4 保护的代价落在长请求上，报告已正确披露

`adaptive` vs `ppas`：长请求 P99 ITL 从 46.67 ms → 102.84 ms（**+120.4%**），而全请求 P99 ITL 只从 44.07 → 45.54（+3.3%）。**全请求口径完全掩盖了长请求组 2.2 倍的劣化。**

`REPORT.md:98` 明确写了这一点，并给出解释：保护触发时抬高单步预算，会把这次调度的负载集中到长请求上。**判断**：这是全文方法学上最扎实的一段——它主动拆开了一个对自己不利的分组结果。

### 10.5 统计与系统性偏差

- **只有 3 个 seed**，`mean_min_seed` / `mean_max_seed` 就是极值，报告没有给置信区间或显著性检验（也没有声称）。
- **不同策略是串行跑的，且没有锁 GPU 时钟**（`REPORT.md:48`）。策略顺序是 `static4096 → adaptive → ppas → native8192 → no_guard → static2048 → no_hysteresis`，每策略连跑 3 个 seed。**判断**：`adaptive`/`ppas` 与 `native8192` 之间相隔了数分钟的 GPU 热状态漂移；而 §10.3 里那些 0.26% 级别的差异远小于这种漂移可能的量级。要支撑「滞回无收益」这类小效应结论，必须先锁频并做交错（interleaved）重跑。
- **负载是合成且完全确定的**：prompt 由一句技术短句重复拼接而成、`temperature=0`、`ignore_eos=True`、固定输出长度。报告明确承认这不是语义质量基准。合理。

### 10.6 仓库完整性的两个小缺口

- `results/summary/source-verification.json` **在仓库里没有任何生成脚本**（全仓 grep 只在该 JSON 自身的引用处命中）。它的内容是 6 个 `true`，看起来是「源码哈希校验通过」的结论，但**产生它的代码不在仓库里**。我自己重算这 6 个 SHA-256 确实全部 MATCH，所以结论是对的，只是**这条证据链缺一环**。
- `benchmarks/package_evidence.py` 会生成 `evidence.zip` 与 `results/model-manifest.json`，但 **`evidence.zip` 不在工作区**，且该脚本需要本地存在 `models/Qwen3-1.7B`（模型权重按 README 只留在远程机器）。所以「打包证据」这一步实际没有产出，虽然 `REPORT.md` 把它列为复现步骤的一环。
- `metadata.json` 的 `source_sha256` 只覆盖 `src/`、`benchmarks/`、`tests/` 下的 6 个文件，**`benchmarks/analyze.py`、`inspect_evidence.py`、`package_evidence.py` 三个分析脚本没有被哈希**——而分析脚本恰恰是「原始数据 → 报告数字」的那一环。**判断**：建议把分析脚本也纳入哈希范围，否则无法排除「事后改分析口径」。

---

## 11. 值得直接采用的优点（避免只讲问题）

1. **控制器零副作用**：不碰 GPU、不读全局时钟、不改队列。这是它能被 20 个纯 CPU 单测覆盖、包括 1 万步随机轨迹的原因。
2. **`try/finally` 还原 + 专门测试**：唯一改写原生状态的地方，还原行为有测试兜底。
3. **每步越界断言**：把「预算被绕过」变成硬失败，实测 19652 步 0 次越界。
4. **负向运行期校验**：异步调度、投机解码、非 chunked prefill、多卡四种情况在启动时直接拒绝，而不是写进文档了事。
5. **贪心输出哈希对拍**：7 种策略在同一批请求上的完整输出哈希与原生**全部一致（288/288）**。这是「按预算改变 batch 形状不改变输出」的有力正确性证据（虽然是贪心、固定长度、合成输入下的）。
6. **报告主动披露不利结果**：长请求 ITL 劣化 120%、滞回无收益、稳定负载轻微退化、`off` 组不给同口径开销、不声称统计显著性、不把论文数字写成自己的。这些自我约束显著提高了整份材料的可信度。
7. **数据可复现性**：`summary.json` 能重算出全部 `aggregate.*`，`paired_comparisons.json` 能被仓库脚本逐标量重建，源码哈希 MATCH。

## 12. 如果我来改，优先做哪些

按「性价比」排序：

1. **换掉头条基线**。用 `static2048`/`static4096` 作为主要对照，把 `native8192` 降为「未做预算控制」的参照。当前的头条数字经不起「为什么原生 ITL 这么差」这一问。
2. **锁 GPU 时钟 + 交错重跑**。否则所有 1% 以下的结论（尤其是「滞回无收益」）都不可靠。
3. **给消融重新命名并补齐**：当前 `no_hysteresis` 实际同时改了 `exit_prefills` 和 `min_dwell_steps`。拆成 `no_exit_hysteresis` 与 `no_dwell` 两个独立消融，才谈得上归因。
4. **换掉抖动测试场景**。把 `test_hysteresis_reduces_threshold_chatter` 的 `[2,1]*20` 换成会真正跨越进入/退出两个阈值的序列（例如 `[2,1,1,1,0,0,2,2]` 带随机噪声），或者直接用从真实 trace 里回放出来的 `prefills` 序列。**并且先断言 `min_dwell_steps` 会影响结果**——现在改这个参数对断言值毫无影响（§5.2）。
5. **决定 `exit_prefills=0` 的存废**：在这个默认值下退出分支不可达（§4.1），「最短驻留」实际上只约束进入方向。要么把默认改成 `exit_prefills=1` 让退出路径真正生效，要么在 README 里写明默认配置下退出只有一条无驻留的快速通道。
6. **决定 `pending_prefill_tokens` 的归宿**：要么用它做「剩余工作量加权」的压力信号（这是唯一能区别于「数请求个数」的信号，也是报告自己提出的后续方向），要么删掉——它现在只出现在 trace 里，非常容易被误读成决策依据。
7. **给 guard 加行为上界与解释**：当前它会每 8 步无上限重复触发（§4.3）；建议记录「触发后该请求是否真的前进了」的统计，否则无法回答「保护到底有没有用」。另外建议明确写出「纯 prefill 场景下保护不生效」这一限制。
8. **补 trace 路径的测试**：整个写盘分支从未被执行过（§5.3），加一个 `trace_path` 指向 `tmp_path` 的用例即可。
9. **把 `switches_total` 拆成两个计数**：把「阈值切换」和「负载转为不混合导致的释放」分开计数（§5.4），否则 36/38/38 这组数字会被误读成抖动次数。
10. **修正 trace 刷盘条件**：`if not obs.prefills or step % 128 == 0`（§5.4）在稳定态每步刷盘，抵消了 64 KiB 缓冲。
11. **补齐证据链**：把 `analyze.py` 等分析脚本纳入 `source_sha256`；给 `source-verification.json` 补上生成脚本；要么产出 `evidence.zip`，要么把 `package_evidence.py` 从复现步骤里去掉。
12. **切片指标降级或改造**：`short_itl_ms_p99` 在 3 个 seed 上跨 3.35 倍，两个口径符号相反。改成「ITL SLO 达标率」会比继续报分位数诚实得多。
13. **在 README 里声明私有 API 依赖**：`max_num_scheduled_tokens` 不在 `SchedulerConfig` 的公开参数里（§9），跨版本可能失效。

## 13. 复现清单（按依赖顺序）

```bash
# 1. 环境（远程 Linux CUDA 主机；脚本里的根目录需按机器改）
bash setup_remote.sh
.venv/bin/uv pip install --python .venv/bin/python -e .
.venv/bin/python download_model.py

# 2. 测试（需要模型目录存在，集成测试会读 config.json）
AB_TEST_MODEL="$PWD/models/Qwen3-1.7B" .venv/bin/pytest -q     # 期望 29 passed

# 3. 正式实验（约 42 轮，每轮起停一次服务器）
.venv/bin/python benchmarks/run_sweep.py \
  --policies static4096 adaptive ppas native8192 no_guard static2048 no_hysteresis \
  --scenarios burst steady --seeds 101 202 303 --output results/reproduction

# 4. 聚合与自检
.venv/bin/python benchmarks/analyze.py results/reproduction --output results/summary
.venv/bin/python benchmarks/inspect_evidence.py results/reproduction --save
```

配置注入示例（`AB_CONFIG` 指向的 JSON 必须 12 个字段齐全，多一个就 `TypeError`）：

```json
{
  "mode": "adaptive",
  "high_budget": 8192, "low_budget": 1024, "static_budget": 8192,
  "enter_prefills": 2, "exit_prefills": 0, "min_dwell_steps": 4,
  "guard_enabled": true, "max_stall_ms": 250.0,
  "guard_budget": 4096, "guard_interval_steps": 8,
  "trace_path": "/abs/path/adaptive.scheduler.jsonl"
}
```

服务器侧必须同时带上：`--scheduler-cls adaptive_budget.scheduler.AdaptiveBudgetScheduler --no-async-scheduling --enable-chunked-prefill --max-num-batched-tokens 8192`。

## 14. 附件索引

| 想了解 | 看哪里 |
|---|---|
| 机制设计说明 | [`README.md`](README.md) |
| 完整实验报告（含全部表格与自我披露） | [`REPORT.md`](REPORT.md) |
| 决策状态机（全部逻辑所在） | [`src/adaptive_budget/controller.py`](src/adaptive_budget/controller.py) |
| vLLM 接入层 | [`src/adaptive_budget/scheduler.py`](src/adaptive_budget/scheduler.py) |
| 纯 CPU 状态机测试 | [`tests/test_controller.py`](tests/test_controller.py) |
| 真实 KV 分配器集成测试 | [`tests/test_scheduler.py`](tests/test_scheduler.py) |
| 压测脚本（含合成轨迹定义） | [`benchmarks/run_sweep.py`](benchmarks/run_sweep.py) |
| 聚合与配对比较脚本 | [`benchmarks/analyze.py`](benchmarks/analyze.py) |
| 一步一行的控制器 trace（19653 行） | [`results/main/adaptive.scheduler.jsonl`](results/main/adaptive.scheduler.jsonl) |
| 逐请求 token ID 与时间戳 | `results/main/*.requests.jsonl` |
| 环境、版本、源码哈希 | [`results/main/metadata.json`](results/main/metadata.json) |
| 越界/切换/保护/决策耗时统计 | [`results/main/diagnostics.json`](results/main/diagnostics.json) |
| 聚合表（Markdown/CSV/JSON） | [`results/summary/tables.md`](results/summary/tables.md) |
| 逐 seed 配对变化率 | [`results/summary/paired_comparisons.json`](results/summary/paired_comparisons.json) |

---

## 附：本次审计使用的独立复算方法

为避免「用被审计的代码验证被审计的结果」，正文 §7 的数字来自一份只读 `summary.json` 的独立脚本，公式与 [`benchmarks/analyze.py`](benchmarks/analyze.py) 的声明一致但代码独立；§8 的交叉验证则是把仓库自己的 `analyze.py` 在临时目录**原样执行**（`exec`，不复制逻辑），把输出与提交版本逐标量比对。

§4.1、§4.3、§5.2 关于分支可达性与测试有效性的结论，来自另一条独立路径：在不导入 vLLM 的前提下直接对 [`src/adaptive_budget/controller.py`](src/adaptive_budget/controller.py) 与 [`tests/test_controller.py`](tests/test_controller.py) 做源码级核查与状态机复算（包括把 `limited`/`last_switch` 预置成特定值来逼出各分支、以及把 `min_dwell_steps` 取 {0, 1, 4, 1000} 重跑那个抖动测试的观测序列）。**需要说明的是**：本机没有 vLLM，无法运行 `tests/test_scheduler.py`；因此凡涉及真实调度器行为的结论（§5.1、§9）都是推断，并已在那两处标明。

一处环境说明：本次审计期间，该工作区根目录缺失 `WRITE_OWNER`，导致所有 `pwsh` 调用在启动阶段的沙箱授权就失败（`SetNamedSecurityInfoW failed (Win32 5)`）。给当前用户补上目录的完全控制项后已恢复；`_audit_signs.py` 一个临时文件因继承到该目录的低完整性标签而无法删除，属于同类权限问题，直接删除即可。
