from pathlib import Path

LWAZI_ROOT = Path(
    r"C:\Users\Sam\Documents\NLP_Projects\data\corpus"
) / "Lwazi_isiZulu_Pronunciation_Dictionary_1.2"

DICT_FILE = LWAZI_ROOT / "dict" / "isizulu.dict"
MAP_FILE = LWAZI_ROOT / "g2p" / "isizulu.map.phones"
SOUNDS_DIR = LWAZI_ROOT / "dict" / "sounds"

TARGET = "amabele"

# ---------------------------------------------------------
# Load phone mapping
# ---------------------------------------------------------

phone_map = {}

with open(MAP_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) >= 2:
            symbol = parts[0]
            phone = " ".join(parts[1:])
            phone_map[symbol] = phone

# ---------------------------------------------------------
# Find target dictionary entry
# ---------------------------------------------------------

entry = None

with open(DICT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        if parts and parts[0].lower() == TARGET.lower():
            entry = parts
            break

print("=" * 80)
print("LWAZI MABELE PHONE MAPPING")
print("=" * 80)

if entry is None:
    print(f"Entry '{TARGET}' not found.")
    raise SystemExit

word = entry[0]
symbols = entry[1:]

print("\nDictionary entry:")
print(" ".join(entry))

print("\nDictionary symbols:")
print(" -> ".join(symbols))

print("\nMapped phones:")
mapped = []

for symbol in symbols:
    phone = phone_map.get(symbol, "<UNKNOWN>")
    mapped.append(phone)
    print(f"{symbol:5} -> {phone}")

print("\nMapped phone sequence:")
print(" -> ".join(mapped))

# ---------------------------------------------------------
# Display relevant sound templates
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("AVAILABLE SOUND FILES")
print("=" * 80)

for wav in sorted(SOUNDS_DIR.glob("*.wav")):
    print(wav.name)

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)