"""Inspect saved evidence without changing a running experiment."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    root = args.directory
    summary = (
        json.loads((root / "summary.json").read_text())
        if (root / "summary.json").exists()
        else []
    )
    print("policy scenario seed TTFT_mean TTFT_p95 TPOT_mean ITL_p99 E2E_mean tok/s")
    for r in summary:
        print(
            r["policy"],
            r["scenario"],
            r["seed"],
            *(
                round(r[m], 2)
                for m in [
                    "all_ttft_ms_mean",
                    "all_ttft_ms_p95",
                    "all_tpot_ms_mean",
                    "all_itl_ms_p99",
                    "all_e2e_ms_mean",
                    "output_tokens_per_s",
                ]
            ),
        )
    diagnostics = {
        "runs": len(summary),
        "requests": sum(r["requests"] for r in summary),
        "failed": sum(r["failed"] for r in summary),
        "max_dispatch_delay_ms": max(
            (r["max_dispatch_delay_ms"] for r in summary), default=0
        ),
        "policies": {},
    }
    for path in sorted(root.glob("*.scheduler.jsonl")):
        steps = []
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # An in-flight final line is not a complete observation.
            if r["type"] == "step":
                steps.append(r)
        if not steps:
            continue
        us = sorted(s["decision_us"] for s in steps)
        diagnostics["policies"][path.name.split(".")[0]] = {
            "steps": len(steps),
            "switches": steps[-1]["switches_total"],
            "guards": steps[-1]["guards_total"],
            "budget_violations": sum(
                s["scheduled_tokens"] > s["budget"] for s in steps
            ),
            "preemptions": sum(s.get("preempted_requests", 0) for s in steps),
            "decision_us_median": statistics.median(us),
            "decision_us_p99": us[int((len(us) - 1) * 0.99)],
            "max_observed_stall_ms": max(s["oldest_stall_ms"] for s in steps),
        }
    by_policy = {}
    for path in root.glob("*.requests.jsonl"):
        policy, scenario, seed = path.name.split(".")[:3]
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["ok"]:
                by_policy.setdefault(policy, {})[(scenario, seed, r["id"])] = r[
                    "output_sha256"
                ]
    reference = by_policy.get("native8192", {})
    diagnostics["greedy_output_comparison_to_native"] = {}
    for policy, requests in by_policy.items():
        common = reference.keys() & requests.keys()
        diagnostics["greedy_output_comparison_to_native"][policy] = {
            "matched": sum(reference[k] == requests[k] for k in common),
            "compared": len(common),
        }
    diagnostics["multi_token_chunks"] = sum(r["multi_token_chunks"] for r in summary)
    print(json.dumps(diagnostics, indent=2))
    if args.save:
        (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
