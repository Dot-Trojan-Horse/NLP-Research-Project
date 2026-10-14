#!/usr/bin/env python3
"""Calculate PER/Levenshtein only for validated study references.

The hypothesis must come from an independent phone recognizer (Allosaurus).
No reference sequence is imposed on the audio. Streets without a validated
reference are reported as pending and excluded from numerical PER.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "results" / "assignment3" / "ground_truth" / "assignment3_ground_truth_master.csv"
DEFAULT_HYP = PROJECT_ROOT / "results" / "assignment3" / "phone_recognition" / "study_phone_recognition.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "assignment3" / "per"


def levenshtein(ref, hyp):
    # cell = (cost, substitutions, deletions, insertions)
    dp = [[(0, 0, 0, 0) for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        dp[0][j] = (j, 0, 0, j)
    for i, r in enumerate(ref, 1):
        for j, h in enumerate(hyp, 1):
            if r == h:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                dele = dp[i - 1][j]
                ins = dp[i][j - 1]
                dp[i][j] = min(
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                    (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                    (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                    key=lambda x: (x[0], x[1] + x[2] + x[3]),
                )
    return dp[-1][-1]


def map_reference(lwazi_sequence, mapping):
    tokens = [t for t in lwazi_sequence.split() if t]
    mapped, unmapped = [], []
    for t in tokens:
        if t in mapping:
            mapped.append(mapping[t])
        else:
            unmapped.append(t)
    return mapped, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--hypotheses", type=Path, default=DEFAULT_HYP)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    # Explicit common-inventory mapping. Extend only after verification.
    mapping = {
        "m": "m", "a": "a", "b_<": "ɓ", "E": "ɛ", "l": "l",
        "n": "n", "d": "d", "p_>": "pʼ",
    }

    with args.registry.open(encoding="utf-8-sig", newline="") as f:
        registry = {r["street_key"]: r for r in csv.DictReader(f)}
    with args.hypotheses.open(encoding="utf-8-sig", newline="") as f:
        hypotheses = list(csv.DictReader(f))

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for h in hypotheses:
        street = h["street_key"]
        rec_type = h["recording_type"]
        reg = registry.get(street, {})
        reference = (reg.get("reference_phones") or "").strip()
        if reg.get("validation_status") not in {"manual_validated", "validated_reference"} or not reference:
            results.append({
                "street_key": street,
                "recording_type": rec_type,
                "status": "reference_pending",
                "reference_lwazi": reference,
                "recognized_phones": h.get("recognized_phones", ""),
                "levenshtein_distance": "",
                "substitutions": "", "deletions": "", "insertions": "", "PER_percent": "",
            })
            continue

        ref_common, unmapped = map_reference(reference, mapping)
        hyp = [x for x in h.get("recognized_phones", "").split() if x]
        if unmapped:
            status = "reference_unmapped_tokens"
            results.append({
                "street_key": street, "recording_type": rec_type, "status": status,
                "reference_lwazi": reference, "recognized_phones": " ".join(hyp),
                "unmapped_reference": " ".join(unmapped),
                "levenshtein_distance": "", "substitutions": "", "deletions": "", "insertions": "", "PER_percent": "",
            })
            continue

        dist, s, d, i = levenshtein(ref_common, hyp)
        per = 100.0 * dist / len(ref_common) if ref_common else None
        results.append({
            "street_key": street, "recording_type": rec_type,
            "status": "experimental_independent_hypothesis",
            "reference_lwazi": reference,
            "reference_common": " ".join(ref_common),
            "recognized_phones": " ".join(hyp),
            "unmapped_reference": "",
            "levenshtein_distance": dist, "substitutions": s, "deletions": d, "insertions": i,
            "PER_percent": f"{per:.2f}" if per is not None else "",
        })

    output_file = args.output / "validated_per_results.csv"
    fields = sorted({k for r in results for k in r.keys()})
    preferred = ["street_key", "recording_type", "status", "reference_lwazi", "reference_common", "recognized_phones", "unmapped_reference", "levenshtein_distance", "substitutions", "deletions", "insertions", "PER_percent"]
    fields = [x for x in preferred if x in fields] + [x for x in fields if x not in preferred]
    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)

    print(f"PER rows written: {len(results)}")
    print(f"Output: {output_file.resolve()}")


if __name__ == "__main__":
    main()
