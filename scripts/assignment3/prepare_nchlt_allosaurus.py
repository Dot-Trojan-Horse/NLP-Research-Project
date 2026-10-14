#!/usr/bin/env python3
"""Prepare the user's NCHLT isiZulu corpus for Allosaurus adaptation.

This version is adapted to the existing NLP_Projects structure:
    data/corpus/nchlt_speech_corpus_zul/
    data/corpus/Lwazi_isiZulu_Pronunciation_Dictionary_1.2/
    results/assignment3/allosaurus_preparation/

The generated labels are canonical Lwazi-G2P labels converted to the target
Allosaurus phone inventory. They are training labels, not human phonetic
transcriptions and not the ground truth for the 33-street evaluation set.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from lwazi_g2p import clean_name, g2p_word, load_map, load_rules
    from lwazi_to_allosaurus import MAP, convert_sequence
except ImportError as exc:
    raise ImportError(
        "Run this script from the project's scripts directory so "
        "lwazi_g2p.py and lwazi_to_allosaurus.py are importable."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "data" / "corpus" / "nchlt_speech_corpus_zul" / "audio"
DEFAULT_XML = PROJECT_ROOT / "data" / "corpus" / "nchlt_speech_corpus_zul" / "transcriptions" / "nchlt_zul.trn.xml"
DEFAULT_RULES = PROJECT_ROOT / "data" / "corpus" / "Lwazi_isiZulu_Pronunciation_Dictionary_1.2" / "g2p" / "isizulu.rules"
DEFAULT_PHONE_MAP = PROJECT_ROOT / "data" / "corpus" / "Lwazi_isiZulu_Pronunciation_Dictionary_1.2" / "g2p" / "isizulu.map.phones"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "assignment3" / "allosaurus_preparation"


def local(tag: str) -> str:
    return tag.split("}")[-1].lower()


def attr_or_child(el: ET.Element, names: set[str]) -> str | None:
    names = {n.lower() for n in names}
    for key, value in el.attrib.items():
        if key.lower() in names and value:
            return value.strip()
    for child in el:
        if local(child.tag) in names and child.text:
            return child.text.strip()
    return None


def resolve_audio(audio_value: str, audio_root: Path) -> Path:
    """Resolve NCHLT XML audio references without changing the corpus."""
    raw = Path(audio_value)
    candidates = [
        audio_root / raw,
        audio_root / raw.name,
        audio_root / raw.parent.name / raw.name,
    ]
    # XML commonly contains nchlt_zul/audio/001/file.wav.
    m = re.search(r"(?:^|[/\\])([0-9]{3})[/\\][^/\\]+\.wav$", audio_value, re.I)
    if m:
        candidates.insert(0, audio_root / m.group(1) / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def parse_xml(xml_path: Path, audio_root: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for rec in root.iter():
        if local(rec.tag) != "recording":
            continue
        audio = attr_or_child(rec, {"audio", "wav", "file", "filename"})
        orth = attr_or_child(rec, {"orth", "text", "transcription", "prompt", "orthography"})
        if not audio or not orth or not audio.lower().endswith(".wav"):
            continue
        speaker_el = next((p for p in root.iter() if False), None)  # intentionally unused
        # Find nearest speaker element through a lightweight parent-free lookup.
        # NCHLT IDs are also encoded in filenames, so use that as the fallback.
        filename = Path(audio).name
        match = re.search(r"_(\d{3})[mf]_", filename, re.I)
        speaker = match.group(1) if match else "unknown"
        wav = resolve_audio(audio, audio_root)
        rows.append({
            "utterance_id": Path(audio).stem,
            "speaker_id": speaker,
            "audio_path": str(wav),
            "orthography": orth,
        })
    if not rows:
        raise RuntimeError(f"No recording elements parsed from {xml_path}")
    return rows


def g2p_text(text: str, rules, phone_map):
    # Keep this conservative: use the same G2P implementation supplied with
    # the starter package, but remove bracketed annotation and punctuation.
    text = re.sub(r"\[[^\]]+\]", " ", text.lower())
    words = re.findall(r"[a-zà-öø-ÿ'-]+", text)
    lwazi = []
    for word in words:
        lwazi.extend(g2p_word(word, rules, phone_map))
    return lwazi, convert_sequence(lwazi)


def write_split(folder: Path, rows):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "wave").open("w", encoding="utf-8") as wf, (folder / "text").open("w", encoding="utf-8") as tf:
        for row in rows:
            wf.write(f"{row['utterance_id']} {row['audio_path']}\n")
            tf.write(f"{row['utterance_id']} {' '.join(row['allosaurus_phones'])}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--phone-map", type=Path, default=DEFAULT_PHONE_MAP)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--validation-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for path in [args.train_xml, args.audio_root, args.rules, args.phone_map]:
        if not path.exists():
            raise FileNotFoundError(path)

    rules = load_rules(args.rules)
    phone_map = load_map(args.phone_map)
    raw = parse_xml(args.train_xml, args.audio_root)

    accepted = []
    rejected_missing_audio = 0
    rejected_g2p = 0
    for row in raw:
        wav = Path(row["audio_path"])
        if not wav.is_file():
            rejected_missing_audio += 1
            continue
        try:
            lwazi, allosaurus = g2p_text(row["orthography"], rules, phone_map)
            if not allosaurus:
                raise ValueError("empty phone sequence")
        except Exception:
            rejected_g2p += 1
            continue
        row["lwazi_phones"] = " ".join(lwazi)
        row["allosaurus_phones"] = allosaurus
        accepted.append(row)

    speakers = sorted({r["speaker_id"] for r in accepted if r["speaker_id"] != "unknown"})
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    n_val = max(1, round(len(speakers) * args.validation_fraction)) if speakers else 0
    val_speakers = set(speakers[:n_val])
    train = [r for r in accepted if r["speaker_id"] not in val_speakers]
    validate = [r for r in accepted if r["speaker_id"] in val_speakers]

    args.output.mkdir(parents=True, exist_ok=True)
    write_split(args.output / "train", train)
    write_split(args.output / "validate", validate)

    inventory = sorted(set(MAP.values()))
    (args.output / "zul_target_phones.txt").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    (args.output / "validation_speakers.txt").write_text("\n".join(sorted(val_speakers)) + "\n", encoding="utf-8")

    summary = [
        ["metric", "value"],
        ["xml_recordings_parsed", len(raw)],
        ["accepted_recordings", len(accepted)],
        ["train_recordings", len(train)],
        ["validation_recordings", len(validate)],
        ["unique_speakers", len(speakers)],
        ["validation_speakers", len(val_speakers)],
        ["rejected_missing_audio", rejected_missing_audio],
        ["rejected_g2p", rejected_g2p],
        ["target_phone_inventory", len(inventory)],
        ["seed", args.seed],
        ["validation_fraction", args.validation_fraction],
    ]
    with (args.output / "preparation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(summary)

    print("=" * 72)
    print("NCHLT -> ALLOSAURUS PREPARATION")
    print("=" * 72)
    for k, v in summary[1:]:
        print(f"{k:28}: {v}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
