## Overview

Add an opt-in custom scheduler example that adjusts the total per-step token
budget under mixed prefill/decode load. It uses the existing `scheduler_cls`
extension point and leaves the standard scheduler and defaults unchanged.
Related discussion: [P-PAS RFC #52906](https://github.com/vllm-project/vllm/issues/52906).

## Claims

- Provide a bounded, configurable example with pressure thresholds, minimum
  residence and a rate-limited budget boost for stalled prefills.
- Preserve native allocation, request order and lifecycle; restore the configured
  capacity after each scheduling call. Reject unsupported configurations.
- Document the trade-offs and exercise the example through the existing plugin
  test suite, including actual model generation with prefix caching on and off.

## Validation

Base: `e9acf48614a2c0d027d7aa2f0e38d963117617a6`. Hardware: one RTX 5090, 32 GB.
Model: Qwen/Qwen3-1.7B, BF16. On this main commit, all 16 added tests passed
(60.62 s, four fork deprecation warnings; the existing OPT dummy test was
deselected). This includes both GPU generation cases using Model Runner V2,
PyTorch 2.14.0+cu130 and official precompiled extensions from `49d634bd6`;
native sources, build configuration and dependencies are unchanged between the
two commits. The same tests also passed on that original base. Local model
redirect and `VLLM_BATCH_INVARIANT=1` were used. Exact commands, environment pins
and raw logs accompany the revision report linked below.

```bash
.venv/bin/python -m pytest tests/plugins_tests/test_scheduler_plugins.py -k adaptive_budget -v
pre-commit run --files \
  examples/features/adaptive_budget/scheduler.py \
  tests/plugins_tests/test_scheduler_plugins.py tests/v1/core/utils.py \
  docs/features/adaptive_budget.md docs/features/README.md \
  .buildkite/test_areas/plugins.yaml
```

The external experiment suite passes 54 tests on vLLM 0.22.1, including real
allocator lifecycle tests and 4×1000 controller differential steps. The full
pre-commit command passed on the submission base: 17 applicable checks passed,
and 19 had no matching files. No SKIP override was used. An earlier Go-download
TLS failure was resolved by installing a checksum-verified official toolchain.
Hosted CI has not yet run.

Exploratory serving measurements used vLLM 0.22.1, three seeds, randomized
policy order, 30 runs and 1440 requests with zero failures. The adaptive arm ran
this example. Runner capacity was held at 8192 tokens, max sequences 64, with
chunked prefill, CUDA graphs, and prefix caching/tracing disabled.
These are not current-main performance measurements.
Client timing included 1220 multi-token SSE chunks; tokens in such a chunk share
a timestamp, so these interarrival measurements have limited time resolution.

| Burst workload | Native 8192 | Fixed runtime 1024 | Adaptive |
| --- | ---: | ---: | ---: |
| Mean TTFT (ms) | 715.67 | 987.94 | 882.20 |
| P99 client token interarrival (ms) | 150.08 | 42.22 | 45.14 |
| Mean end-to-end latency (ms) | 3237.28 | 3650.95 | 3508.79 |
| Output tokens/s | 651.50 | 637.12 | 641.99 |

The three-seed paired ITL reduction versus native was 69.27–70.95%, with
7.95–8.63% worse end-to-end latency and 1.31–1.55% lower throughput. Relative
to fixed 1024, end-to-end latency improved 3.49–4.23% while ITL worsened
3.31–9.51%. Steady traffic was approximately unchanged from native. This is a
trade-off, not a universal speedup; three seeds do not establish significance.

Generation checks use the existing `check_logprobs_close` helper. Batch-shape
changes can alter greedy outputs: the separate experiment observed 2/3 exact
outputs without prefix caching and 6/6 with it; native fixed-capacity 128 also
produced the short-output divergence. At the first divergence the winning tokens
were mutually in the other run's top-5. This criterion does not establish full
sequence or probability equality.

Full evidence: [revision report](https://github.com/codegx686/vllm/blob/adaptive-budget-evidence-20261024/REVISION_REPORT.md),
[raw measurements and per-seed comparisons](https://github.com/codegx686/vllm/tree/adaptive-budget-evidence-20261024/results/revision-20261024),
[validation logs](https://github.com/codegx686/vllm/tree/adaptive-budget-evidence-20261024/logs),
and [source/evidence hashes](https://github.com/codegx686/vllm/blob/adaptive-budget-evidence-20261024/DELIVERY_MANIFEST.json).

## Details

The example scans runnable requests once per iteration and temporarily narrows
`max_num_scheduled_tokens`, restoring capacity in `finally`. Prefill replay after
preemption is included; request identity prevents reused IDs from inheriting a
previous request's progress timer. The native scheduler still owns allocation,
preemption and cancellation. No GPU synchronization is added.

Scope is synchronous, single-GPU text generation with chunked prefill. A boost
provides a progress opportunity rather than a starvation bound. Smaller total
budgets can delay decodes and worsen TTFT; queue scanning has linear CPU cost.
This relies on the concrete Scheduler implementation and must be checked on
upgrades. Documentation covers configuration, limitations and evaluation.

Unlike P-PAS's aggregate **prefill** cap, this example caps **all** scheduled
tokens. It is not a P-PAS implementation or reproduction. The existing plugin
Buildkite step already runs this test module; source dependencies are extended
to select it when the example or scheduler changes. Hosted CI and AMD validation
remain to be run. AI assistance was used to implement, review and validate this
contribution. The human submitter has reviewed the changes, confirmed the
relevant tests and approved DCO sign-off.

The 2026-10-24 duplicate-work search also checked EWSJF #33392, FlowPrefill
#44794 and long-prefill defer #61080. Those use different queue, chunk-execution
or defer mechanisms; no equivalent open custom-scheduler example was found.
The refreshed search evidence and RFC discussion are included in the report
bundle. Maintainer agreement on accepting this example is still needed.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.
- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.

</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>**
