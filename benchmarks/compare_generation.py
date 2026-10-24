# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit exact matches and first-divergence top-5 compatibility separately."""

import argparse
import json
from pathlib import Path


def compare(reference, candidate):
    assert len(reference["batches"]) == len(candidate["batches"])
    comparisons = []
    for batch_id, (base, changed) in enumerate(
        zip(reference["batches"], candidate["batches"])
    ):
        assert len(base) == len(changed)
        for request_id, (a, b) in enumerate(zip(base, changed)):
            assert len(a["token_ids"]) == len(b["token_ids"]) == 64
            mismatch = next(
                (
                    i
                    for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"]))
                    if x != y
                ),
                None,
            )
            result = {
                "batch": batch_id,
                "request": request_id,
                "exact_match": mismatch is None,
                "first_difference": mismatch,
            }
            if mismatch is not None:
                x, y = str(a["token_ids"][mismatch]), str(b["token_ids"][mismatch])
                lp_a, lp_b = a["logprobs"][mismatch], b["logprobs"][mismatch]
                result["mutual_top5"] = x in lp_b and y in lp_a
                result["native_top5"] = lp_a
                result["candidate_top5"] = lp_b
                # Same first-divergence criterion as upstream check_logprobs_close.
                # This does not assert numerical equality or compare divergent tails.
                assert result["mutual_top5"], result
            comparisons.append(result)
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = {
        p.stem: json.loads(p.read_text())
        for p in args.directory.glob("*.json")
        if p.name != "comparison.json"
    }
    output = {}
    for name, candidate in files.items():
        reference = "native-prefix" if name.endswith("-prefix") else "native"
        if name == reference:
            continue
        output[name] = compare(files[reference], candidate)
    assert all(row["exact_match"] for row in output["off"])
    (args.directory / "comparison.json").write_text(json.dumps(output, indent=2))
    for name, rows in output.items():
        print(
            name,
            f"exact={sum(r['exact_match'] for r in rows)}/{len(rows)}",
            "first-divergence top-5 check passed",
        )


if __name__ == "__main__":
    main()
