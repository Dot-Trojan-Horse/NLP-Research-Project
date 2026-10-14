from pathlib import Path

LWAZI_ROOT = Path(
    r"C:\Users\Sam\Documents\NLP_Projects\data\corpus"
) / "Lwazi_isiZulu_Pronunciation_Dictionary_1.2"

MAP_FILE = LWAZI_ROOT / "g2p" / "isizulu.map.phones"
SOUNDS_DIR = LWAZI_ROOT / "dict" / "sounds"

print("=" * 80)
print("LWAZI PHONE TEMPLATE INSPECTION")
print("=" * 80)

print("\nPhone mapping:")
print("-" * 80)

with open(MAP_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) >= 2:
            symbol = parts[0]
            mapping = " ".join(parts[1:])

            print(f"{symbol:8} -> {mapping}")

print("\n" + "=" * 80)
print("AVAILABLE SOUND FILES")
print("=" * 80)

sound_files = sorted(SOUNDS_DIR.glob("*.wav"))

for wav in sound_files:
    print(wav.name)

print("\n" + "=" * 80)
print(f"Total phone sound files: {len(sound_files)}")
print("=" * 80)