"""Open-loop HTTP/SSE benchmark. Run only on the authorized single-GPU host.

Every policy starts the same engine capacity, uses the same trace and warms up.
Records token IDs, per-token client receive times, actual usage and all failures.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer

from adaptive_budget.controller import PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8000"
POLICIES = {
    "native8192": {"mode": "off"},
    "static1024": {"mode": "static", "static_budget": 1024},
    "static2048": {"mode": "static", "static_budget": 2048},
    "static4096": {"mode": "static", "static_budget": 4096},
    "ppas": {"mode": "ppas"},
    "adaptive": {"mode": "adaptive"},
    "no_guard": {"mode": "adaptive", "guard_enabled": False},
    "no_hysteresis": {"mode": "adaptive", "exit_prefills": 1, "min_dwell_steps": 0},
}


def trace(seed, scenario):
    rng = random.Random(seed)
    rows = []
    if scenario == "burst":
        for cycle in range(4):
            for j in range(12):
                long = 4 <= j < 8
                rows.append(
                    {
                        "id": f"{seed}-{cycle}-{j}",
                        "at": cycle * 6
                        + (0 if j < 4 else 0.3 if j < 8 else 0.6)
                        + rng.uniform(0, 0.08),
                        "input_tokens": rng.choice([12288, 16384]) if long else 512,
                        "output_tokens": 128 if long else 384,
                        "kind": "long" if long else "short",
                    }
                )
    elif scenario == "steady":
        for j in range(48):
            long = j % 3 == 0
            rows.append(
                {
                    "id": f"{seed}-{j}",
                    "at": j * 0.5 + rng.uniform(0, 0.04),
                    "input_tokens": 8192 if long else 512,
                    "output_tokens": 128 if long else 256,
                    "kind": "long" if long else "short",
                }
            )
    elif scenario == "smoke":
        for j in range(4):
            rows.append(
                {
                    "id": f"smoke-{j}",
                    "at": j * 0.1,
                    "input_tokens": 512,
                    "output_tokens": 32,
                    "kind": "short",
                }
            )
    else:
        raise ValueError(scenario)
    return sorted(rows, key=lambda x: x["at"])


def prompt_tokens(tokenizer, row):
    prefix = tokenizer.encode(
        f"Record {row['id']}. Read the following technical notes. ",
        add_special_tokens=False,
    )
    body = tokenizer.encode(
        "A language model serves concurrent requests. Prefill processes input tokens and decode generates new tokens. Measure latency and throughput carefully. ",
        add_special_tokens=False,
    )
    n = row["input_tokens"]
    return (prefix + body * (n // len(body) + 1))[:n]


async def send(session, row, prompt, start):
    await asyncio.sleep(max(0, start + row["at"] - time.perf_counter()))
    sent = time.perf_counter()
    result = dict(
        row, sent_at=sent - start, dispatch_delay_ms=(sent - start - row["at"]) * 1000
    )
    ids, token_times, chunk_sizes = [], [], []
    usage = None
    try:
        payload = {
            "model": "benchmark-model",
            "prompt": prompt,
            "max_tokens": row["output_tokens"],
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
            "seed": 0,
        }
        async with session.post(URL + "/v1/completions", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:500]}"
                )
            async for raw in response.content:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                now = time.perf_counter()
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    new = choice.get("token_ids") or []
                    if new:
                        ids.extend(new)
                        token_times.extend([now - sent] * len(new))
                        chunk_sizes.append(len(new))
        ended = time.perf_counter()
        if (
            not usage
            or usage["completion_tokens"] != row["output_tokens"]
            or len(ids) != row["output_tokens"]
        ):
            raise AssertionError(f"Token mismatch: usage={usage}, received={len(ids)}")
        if usage["prompt_tokens"] != row["input_tokens"]:
            raise AssertionError(f"Input mismatch: {usage}")
        result.update(
            ok=True,
            ttft_ms=token_times[0] * 1000,
            tpot_ms=(token_times[-1] - token_times[0]) * 1000 / (len(ids) - 1),
            e2e_ms=(ended - sent) * 1000,
            finished_at=ended - start,
            token_times_s=token_times,
            token_ids=ids,
            chunk_sizes=chunk_sizes,
            usage=usage,
            output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        )
    except Exception as exc:  # noqa: BLE001 - preserve failures in benchmark evidence
        result.update(ok=False, error=str(exc), finished_at=time.perf_counter() - start)
    return result


def summarize(rows):
    good = [r for r in rows if r["ok"]]
    out = {
        "requests": len(rows),
        "succeeded": len(good),
        "failed": len(rows) - len(good),
    }
    if not good:
        return out
    elapsed = max(r["finished_at"] for r in rows) - min(r["sent_at"] for r in rows)
    out.update(
        elapsed_s=elapsed,
        output_tokens_per_s=sum(r["usage"]["completion_tokens"] for r in good)
        / elapsed,
        max_dispatch_delay_ms=max(r["dispatch_delay_ms"] for r in rows),
        multi_token_chunks=sum(sum(n > 1 for n in r["chunk_sizes"]) for r in good),
    )
    for label, subset in [
        ("all", good),
        ("short", [r for r in good if r["kind"] == "short"]),
        ("long", [r for r in good if r["kind"] == "long"]),
    ]:
        if not subset:
            continue
        for metric in ["ttft_ms", "tpot_ms", "e2e_ms"]:
            vals = [r[metric] for r in subset]
            for stat, value in [
                ("mean", np.mean(vals)),
                ("p50", np.percentile(vals, 50)),
                ("p95", np.percentile(vals, 95)),
                ("p99", np.percentile(vals, 99)),
            ]:
                out[f"{label}_{metric}_{stat}"] = float(value)
        itl = [v * 1000 for r in subset for v in np.diff(r["token_times_s"])]
        out[f"{label}_itl_ms_p99"] = float(np.percentile(itl, 99))
    return out


async def run_trace(tokenizer, rows, path):
    prompts = [prompt_tokens(tokenizer, r) for r in rows]
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=128)
    ) as session:
        start = time.perf_counter() + 0.1
        results = await asyncio.gather(
            *(send(session, r, p, start) for r, p in zip(rows, prompts))
        )
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    return summarize(results)


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3-1.7B"))
    ap.add_argument(
        "--policies",
        nargs="+",
        default=["native8192", "static2048", "ppas", "adaptive"],
    )
    ap.add_argument("--scenarios", nargs="+", default=["burst", "steady"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    ap.add_argument("--output", required=True)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--order-seed", type=int)
    ap.add_argument(
        "--adaptive-example",
        action="store_true",
        help="Use the upstream example for the adaptive arm",
    )
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    metadata = {
        "args": vars(args),
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for base in ["src", "benchmarks", "tests", "upstream_payload/examples"]
            for p in (ROOT / base).rglob("*.py")
        },
        "model_config": json.loads((Path(args.model) / "config.json").read_text()),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    summaries = []
    policies = list(args.policies)
    if args.order_seed is not None:
        random.Random(args.order_seed).shuffle(policies)
    metadata["policy_order"] = policies
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for policy in policies:
        cfg = asdict(PolicyConfig(**POLICIES[policy]))
        if not args.no_trace:
            cfg["trace_path"] = str(out / f"{policy}.scheduler.jsonl")
        config_path = out / f"{policy}.config.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        env = os.environ.copy()
        env.update(
            AB_CONFIG=str(config_path),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VLLM_USE_MODELSCOPE="False",
            TOKENIZERS_PARALLELISM="false",
        )
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--served-model-name",
            "benchmark-model",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "32768",
            "--max-num-batched-tokens",
            "8192",
            "--max-num-seqs",
            "64",
            "--gpu-memory-utilization",
            "0.85",
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--enable-chunked-prefill",
            "--scheduler-cls",
            "adaptive_budget.scheduler.AdaptiveBudgetScheduler",
            "--stream-interval",
            "1",
            "--seed",
            "0",
        ]
        if args.adaptive_example and policy == "adaptive":
            example_cfg = {
                k: v
                for k, v in cfg.items()
                if k not in {"mode", "static_budget", "trace_path"}
            }
            command[command.index("--scheduler-cls") + 1] = (
                "examples.features.adaptive_budget.scheduler.AdaptiveBudgetScheduler"
            )
            command.extend(
                ["--additional-config", json.dumps({"adaptive_budget": example_cfg})]
            )
            env.pop("AB_CONFIG", None)
            env["PYTHONPATH"] = (
                str(ROOT / "upstream_payload") + os.pathsep + env.get("PYTHONPATH", "")
            )
        if args.eager:
            command.append("--enforce-eager")
        (out / f"{policy}.command.json").write_text(json.dumps(command, indent=2))
        with (out / f"{policy}.server.log").open("w") as log:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"START {policy} pid={proc.pid}", flush=True)
            try:
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"Server exited {proc.returncode}, see {policy}.server.log"
                        )
                    try:
                        with urllib.request.urlopen(
                            URL + "/health", timeout=2
                        ) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        time.sleep(2)
                else:
                    raise TimeoutError("Server startup timeout")
                warm = [
                    {
                        "id": "warm-long",
                        "at": 0.0,
                        "input_tokens": 16384,
                        "output_tokens": 32,
                        "kind": "long",
                    },
                    {
                        "id": "warm-short",
                        "at": 0.1,
                        "input_tokens": 512,
                        "output_tokens": 128,
                        "kind": "short",
                    },
                ]
                w = asyncio.run(
                    run_trace(tokenizer, warm, out / f"{policy}.warmup.jsonl")
                )
                if w["failed"]:
                    raise RuntimeError(f"Warmup failed: {w}")
                for scenario in args.scenarios:
                    for seed in args.seeds:
                        name = f"{policy}.{scenario}.{seed}"
                        rows = trace(seed, scenario)
                        (out / f"{name}.trace.json").write_text(
                            json.dumps(rows, indent=2)
                        )
                        result = asyncio.run(
                            run_trace(tokenizer, rows, out / f"{name}.requests.jsonl")
                        )
                        result.update(policy=policy, scenario=scenario, seed=seed)
                        summaries.append(result)
                        (out / "summary.json").write_text(
                            json.dumps(summaries, indent=2)
                        )
                        print(json.dumps(result), flush=True)
                        if result["failed"]:
                            raise RuntimeError(f"Request failures in {name}")
            finally:
                stop(proc)
        time.sleep(2)
    (out / "DONE").write_text("complete\n")


if __name__ == "__main__":
    main()
