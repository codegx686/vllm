# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one isolated, batch-invariant real-model correctness arm.

Run native, off, and adaptive in separate processes, then compare the saved
token IDs. This validates generation, not task-level model quality.
"""

import argparse
import json
import math
import os
from pathlib import Path


def main():
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--policy",
        choices=["native", "native128", "off", "adaptive", "example", "static128"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-caching", action="store_true")
    args = parser.parse_args()
    os.environ.pop("AB_CONFIG", None)
    trace = args.output.with_suffix(".trace.jsonl")
    kwargs = {}
    if args.policy not in ("native", "native128"):
        kwargs.update(
            scheduler_cls="adaptive_budget.scheduler.AdaptiveBudgetScheduler",
            additional_config={
                "adaptive_budget": {
                    "mode": args.policy,
                    "high_budget": 512,
                    "low_budget": 128,
                    "guard_budget": 256,
                    "enter_prefills": 1,
                    "trace_path": str(trace),
                }
            },
        )
    if args.policy == "example":
        kwargs["scheduler_cls"] = (
            "examples.features.adaptive_budget.scheduler.AdaptiveBudgetScheduler"
        )
        kwargs["additional_config"]["adaptive_budget"].pop("mode")
        kwargs["additional_config"]["adaptive_budget"].pop("trace_path")
    if args.policy == "static128":
        kwargs["additional_config"]["adaptive_budget"].update(
            mode="static", static_budget=128
        )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=128 if args.policy == "native128" else 512,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        async_scheduling=False,
        enable_prefix_caching=args.prefix_caching,
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        seed=0,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    short = tokenizer.encode("What is the capital of France? Explain briefly.")
    body = tokenizer.encode(
        "A scheduler shares compute between input and output tokens. "
    )
    prompts = [
        {"prompt_token_ids": short},
        {"prompt_token_ids": (body * 200)[:1536]},
        {"prompt_token_ids": (body * 220)[:2048]},
    ]
    sampling = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, logprobs=5)
    batches = []
    for _ in range(2 if args.prefix_caching else 1):
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        batch = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            assert completion.finish_reason == "length"
            assert len(completion.token_ids) == 64
            assert completion.logprobs is not None
            assert all(
                math.isfinite(step[token].logprob)
                for token, step in zip(completion.token_ids, completion.logprobs)
            )
            batch.append(
                {
                    "token_ids": list(completion.token_ids),
                    "logprobs": [
                        {str(token): value.logprob for token, value in step.items()}
                        for step in completion.logprobs
                    ],
                    "text": completion.text,
                    "cached_tokens": request_output.num_cached_tokens,
                }
            )
        batches.append(batch)
    if args.prefix_caching:
        assert sum(item["cached_tokens"] for item in batches[1]) > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "policy": args.policy,
                "prefix_caching": args.prefix_caching,
                "batches": batches,
            },
            indent=2,
        )
    )
    print(f"Validated {args.policy}: {sum(map(len, batches))} requests", flush=True)


if __name__ == "__main__":
    main()
