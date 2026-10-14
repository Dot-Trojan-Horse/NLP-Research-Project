#!/usr/bin/env python3
"""Map Lwazi/NCHLT isiZulu phone symbols to the Allosaurus target inventory."""
from __future__ import annotations
import argparse
from pathlib import Path

MAP = {
    "!\\": "ǃ", "!\\g_0": "ǃ", "!\\h": "ǃ",
    "E": "ɛ", "J": "ɲ", "K": "ɬ", "K\\": "ɮ", "N": "ŋ", "O": "ɔ", "S": "ʃ",
    "a": "a", "b": "b", "b_<": "ɓ", "d": "d", "d_0Z": "d͡ʒ", "dz": "d͡z",
    "f": "f", "g": "ɡ", "h": "h", "h\\": "ɦ", "i": "i", "j": "j", "k": "k",
    "kK_>": "t͡ɬ", "k_h": "kʰ", "l": "l", "m": "m", "n": "n", "p_>": "pʼ",
    "p_h": "pʰ", "r": "r", "s": "s", "tS_>": "t͡ʃʼ", "t_>": "tʼ", "t_h": "tʰ",
    "ts_>": "t͡sʼ", "u": "u", "v": "v", "w": "w", "z": "z",
    "|\\": "ǀ", "|\\g_0": "ǀ", "|\\h": "ǀ",
    "|\\|\\": "ǁ", "|\\|\\g_0": "ǁ", "|\\|\\h": "ǁ",
}


def convert_sequence(seq):
    out = []
    for phone in seq:
        if phone not in MAP:
            raise ValueError(f"Unknown Lwazi phone: {phone}")
        out.append(MAP[phone])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phones")
    ap.add_argument("--write-inventory", type=Path)
    args = ap.parse_args()
    if args.phones:
        print(" ".join(convert_sequence(args.phones.split())))
    if args.write_inventory:
        args.write_inventory.parent.mkdir(parents=True, exist_ok=True)
        args.write_inventory.write_text("\n".join(sorted(set(MAP.values()))) + "\n", encoding="utf-8")
        print(f"Wrote {len(set(MAP.values()))} phones to {args.write_inventory}")


if __name__ == "__main__":
    main()
