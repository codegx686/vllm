"""Aggregate independent seeds and retain paired comparisons and regressions."""

import argparse
import csv
import json
import statistics as stats
from pathlib import Path

METRICS = [
    "all_ttft_ms_mean",
    "all_ttft_ms_p95",
    "all_tpot_ms_mean",
    "all_e2e_ms_mean",
    "all_e2e_ms_p95",
    "all_itl_ms_p99",
    "short_ttft_ms_p95",
    "short_tpot_ms_mean",
    "short_itl_ms_p99",
    "short_e2e_ms_mean",
    "long_ttft_ms_mean",
    "long_ttft_ms_p95",
    "long_tpot_ms_mean",
    "long_itl_ms_p99",
    "long_e2e_ms_mean",
    "output_tokens_per_s",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directories", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [
        r
        for d in args.directories
        for r in json.loads((d / "summary.json").read_text())
    ]
    keys = [(r["policy"], r["scenario"], r["seed"]) for r in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate policy/scenario/seed; do not silently pool repeats")
    if any(r["failed"] for r in runs):
        raise ValueError(
            "Failed requests must be resolved before aggregate performance claims"
        )
    groups = sorted({(r["policy"], r["scenario"]) for r in runs})
    aggregate = []
    for policy, scenario in groups:
        rows = [r for r in runs if (r["policy"], r["scenario"]) == (policy, scenario)]
        row = {
            "policy": policy,
            "scenario": scenario,
            "seeds": [r["seed"] for r in rows],
            "requests": sum(r["requests"] for r in rows),
        }
        for m in METRICS:
            row[m] = stats.mean(r[m] for r in rows)
            row[m + "_min_seed"] = min(r[m] for r in rows)
            row[m + "_max_seed"] = max(r[m] for r in rows)
        aggregate.append(row)
    paired = []
    for scenario in sorted({r["scenario"] for r in runs}):
        candidate = {
            r["seed"]: r
            for r in runs
            if r["policy"] == "adaptive" and r["scenario"] == scenario
        }
        for policy in sorted({r["policy"] for r in runs} - {"adaptive"}):
            base = {
                r["seed"]: r
                for r in runs
                if r["policy"] == policy and r["scenario"] == scenario
            }
            common = sorted(candidate.keys() & base.keys())
            if not common:
                continue
            comparison = {
                "candidate": "adaptive",
                "baseline": policy,
                "scenario": scenario,
                "seeds": common,
            }
            for metric in METRICS:
                # Positive means candidate improves the named latency/throughput metric.
                direction = 1 if metric == "output_tokens_per_s" else -1
                changes = [
                    direction * (candidate[s][metric] / base[s][metric] - 1) * 100
                    for s in common
                ]
                comparison[metric] = {
                    "mean_percent": stats.mean(changes),
                    "min_percent": min(changes),
                    "max_percent": max(changes),
                    "per_seed_percent": changes,
                }
            paired.append(comparison)
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (args.output / "paired_comparisons.json").write_text(json.dumps(paired, indent=2))
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["policy", "scenario", "requests"] + METRICS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(aggregate)
    lines = [
        "# 实测汇总",
        "",
        "表中为各 seed 指标的算术平均；P95／P99 为每轮分位数的平均，不是将所有请求合并后的分位数。",
        "",
        "| 负载 | 策略 | 请求数 | 平均 TTFT ms | P95 TTFT ms | 平均 TPOT ms | P99 ITL ms | 平均 E2E ms | 输出 tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(aggregate, key=lambda x: (x["scenario"], x["policy"])):
        lines.append(
            f"| {r['scenario']} | {r['policy']} | {r['requests']} | {r['all_ttft_ms_mean']:.2f} | {r['all_ttft_ms_p95']:.2f} | {r['all_tpot_ms_mean']:.3f} | {r['all_itl_ms_p99']:.2f} | {r['all_e2e_ms_mean']:.2f} | {r['output_tokens_per_s']:.2f} |"
        )
    lines.extend(
        [
            "",
            "配对变化率及每个 seed 的范围见 `paired_comparisons.json`。正值表示改善，负值表示退化。",
            "",
        ]
    )
    lines.extend(
        [
            "## 按长短请求拆分",
            "",
            "| 负载 | 策略 | 短请求 P95 TTFT ms | 长请求 P95 TTFT ms | 短请求平均 E2E ms | 长请求平均 E2E ms | 短请求 P99 ITL ms | 长请求 P99 ITL ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in sorted(aggregate, key=lambda x: (x["scenario"], x["policy"])):
        lines.append(
            f"| {r['scenario']} | {r['policy']} | {r['short_ttft_ms_p95']:.2f} | {r['long_ttft_ms_p95']:.2f} | {r['short_e2e_ms_mean']:.2f} | {r['long_e2e_ms_mean']:.2f} | {r['short_itl_ms_p99']:.2f} | {r['long_itl_ms_p99']:.2f} |"
        )
    lines.append("")
    (args.output / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
