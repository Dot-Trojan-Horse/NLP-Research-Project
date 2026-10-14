#!/usr/bin/env python3
"""Project-compatible Lwazi G2P utility.

Adapted from the supplied starter package. Defaults point at the existing
NLP_Projects/data/corpus Lwazi resources and Assignment 3 output area.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = PROJECT_ROOT / "data" / "corpus" / "Lwazi_isiZulu_Pronunciation_Dictionary_1.2" / "g2p" / "isizulu.rules"
DEFAULT_PHONE_MAP = PROJECT_ROOT / "data" / "corpus" / "Lwazi_isiZulu_Pronunciation_Dictionary_1.2" / "g2p" / "isizulu.map.phones"


def load_map(path):
    mapping = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    return mapping


def load_rules(path):
    rules = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip("\n\r")
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split(";")
        if len(parts) < 6:
            continue
        grapheme, left, right, symbol, refinement, _count = parts[:6]
        try:
            refinement = int(refinement)
        except ValueError:
            refinement = 0
        rules.append((grapheme, left, right, symbol, refinement))
    return rules


def context_match(pattern, text, side):
    if pattern == "":
        return True
    try:
        return (re.search(f"(?:{pattern})$", text) if side == "left" else re.match(f"(?:{pattern})", text)) is not None
    except re.error:
        return text.endswith(pattern) if side == "left" else text.startswith(pattern)


def g2p_word(word, rules, phone_map):
    word = word.strip().lower()
    out = []
    i = 0
    while i < len(word):
        candidates = []
        for grapheme, left, right, symbol, refinement in rules:
            if word.startswith(grapheme, i):
                left_text = " " + word[:i]
                right_text = word[i + len(grapheme):] + " "
                if context_match(left, left_text, "left") and context_match(right, right_text, "right"):
                    candidates.append((refinement, len(grapheme), symbol))
        if not candidates:
            raise ValueError(f"No G2P rule matched {word!r} at position {i}")
        refinement, length, symbol = max(candidates, key=lambda x: (x[0], x[1]))
        if symbol != "0":
            out.append(phone_map.get(symbol, symbol))
        i += length
    return out

SUFFIXES = {"road", "rd", "street", "st", "avenue", "ave", "drive", "dr", "lane", "ln", "place", "pl", "close", "crescent", "way"}


def clean_name(name):
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", name)
    if words and words[-1].lower().rstrip(".") in SUFFIXES:
        words = words[:-1]
    return " ".join(words)


def phonemize_text(text, rules, phone_map):
    cleaned = clean_name(text)
    phones = []
    for word in cleaned.split():
        phones.extend(g2p_word(word, rules, phone_map))
    return cleaned, phones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--phone-map", type=Path, default=DEFAULT_PHONE_MAP)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--word")
    group.add_argument("--csv")
    ap.add_argument("--column", default="Street_Name")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "assignment3" / "lwazi_g2p" / "street_g2p.csv")
    args = ap.parse_args()

    rules = load_rules(args.rules)
    phone_map = load_map(args.phone_map)
    if args.word:
        cleaned, phones = phonemize_text(args.word, rules, phone_map)
        print(f"Input: {args.word}")
        print(f"G2P input: {cleaned}")
        print("Lwazi phones:", " ".join(phones))
        return

    rows = []
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row[args.column]
            try:
                cleaned, phones = phonemize_text(name, rules, phone_map)
                status, phone_string = "ok", " ".join(phones)
            except Exception as exc:
                cleaned, status, phone_string = clean_name(name), f"error: {exc}", ""
            rows.append({"road_name": name, "g2p_input": cleaned, "lwazi_phonemes": phone_string, "status": status})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["road_name", "g2p_input", "lwazi_phonemes", "status"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
