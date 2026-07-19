# %%
import pandas as pd
import re
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "KZN_Street_Names_RAW.csv"
output_file = project_root / "output" / "KZN_Street_Names_CLASSIFIED.csv"

print("Loading dataset...")

df = pd.read_csv(input_file)

print(f"Original records: {len(df)}")

# --------------------------------------------------
# Remove duplicates
# --------------------------------------------------

df = df.drop_duplicates(subset=["Street_Name"])

# --------------------------------------------------
# Remove blank names
# --------------------------------------------------

df = df[df["Street_Name"].notna()]

df["Street_Name"] = df["Street_Name"].astype(str).str.strip()

# --------------------------------------------------
# Classification function
# --------------------------------------------------

english_words = [
    "street",
    "road",
    "drive",
    "avenue",
    "lane",
    "close",
    "crescent",
    "place",
    "way",
    "park",
    "circle",
    "boulevard"
]

zulu_prefixes = (
    "u",
    "um",
    "isi",
    "kwa",
    "ama",
    "imi",
    "in",
    "e"
)


def classify(name):

    lower = name.lower()

    # -----------------------
    # Numeric streets
    # -----------------------

    if re.match(r'^\d+', lower):
        return "Numeric"

    # -----------------------
    # Road codes
    # -----------------------

    if re.match(r'^[rmdp]\d+$', lower):
        return "Road_Code"

    # -----------------------
    # Likely isiZulu
    # -----------------------

    first_word = lower.split()[0]

    if first_word.startswith(zulu_prefixes):
        return "Likely_IsiZulu"

    # -----------------------
    # Mixed names
    # -----------------------

    zulu_patterns = [
        "ng",
        "hl",
        "dl",
        "kh",
        "bh",
        "nt",
        "sh",
        "mb",
        "nk"
    ]

    for pattern in zulu_patterns:
        if pattern in lower:
            return "Possible_IsiZulu"

    # -----------------------
    # English
    # -----------------------

    for word in english_words:
        if word in lower:
            return "English"

    return "Unknown"


# --------------------------------------------------
# Apply classification
# --------------------------------------------------

df["Category"] = df["Street_Name"].apply(classify)

# --------------------------------------------------
# Sort
# --------------------------------------------------

df = df.sort_values(["Category", "Street_Name"])

# --------------------------------------------------
# Save classified dataset
# --------------------------------------------------

output_file.parent.mkdir(exist_ok=True)

df.to_csv(output_file, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nClassification Summary\n")

print(df["Category"].value_counts())

print("\nSaved to:")

print(output_file)