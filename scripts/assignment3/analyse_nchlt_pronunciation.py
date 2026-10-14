from pathlib import Path
import argparse
import re
import pandas as pd


def normalise_word(word):
    """Normalise a dictionary word for basic lexical analysis."""
    if pd.isna(word):
        return ""
    word = str(word).strip().lower()
    return word


def parse_dictionary(dictionary_path):
    """
    Parse an NCHLT pronunciation dictionary.

    Expected general format:
        WORD    PHONE PHONE PHONE ...

    Lines beginning with #, ; or empty lines are ignored.
    """
    records = []
    malformed = []

    with open(dictionary_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#") or line.startswith(";"):
                continue

            parts = re.split(r"\s+", line)

            if len(parts) < 2:
                malformed.append({
                    "line_number": line_number,
                    "raw_line": raw_line.rstrip("\n")
                })
                continue

            word = normalise_word(parts[0])
            phones = parts[1:]

            if not word:
                malformed.append({
                    "line_number": line_number,
                    "raw_line": raw_line.rstrip("\n")
                })
                continue

            records.append({
                "line_number": line_number,
                "word": word,
                "phone_sequence": " ".join(phones),
                "phone_count": len(phones)
            })

    return pd.DataFrame(records), pd.DataFrame(malformed)


def build_phone_inventory(dictionary_df):
    """Create a phone-frequency table."""
    phone_counts = {}

    for sequence in dictionary_df["phone_sequence"]:
        phones = sequence.split()

        for phone in phones:
            phone_counts[phone] = phone_counts.get(phone, 0) + 1

    rows = [
        {
            "phone": phone,
            "word_count": count
        }
        for phone, count in phone_counts.items()
    ]

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["word_count", "phone"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return result


def build_sequence_summary(dictionary_df):
    """Summarise pronunciation sequence lengths."""
    if dictionary_df.empty:
        return pd.DataFrame()

    return pd.DataFrame([{
        "dictionary_entries": len(dictionary_df),
        "unique_words": dictionary_df["word"].nunique(),
        "minimum_phone_count": dictionary_df["phone_count"].min(),
        "maximum_phone_count": dictionary_df["phone_count"].max(),
        "mean_phone_count": dictionary_df["phone_count"].mean(),
        "median_phone_count": dictionary_df["phone_count"].median()
    }])


def build_phone_count_distribution(dictionary_df):
    """Count how many lexical entries have each pronunciation length."""
    result = (
        dictionary_df
        .groupby("phone_count")
        .size()
        .reset_index(name="word_count")
        .sort_values("phone_count")
    )

    return result


def build_longest_pronunciations(dictionary_df, n=100):
    """Return the longest pronunciation sequences."""
    return (
        dictionary_df[
            ["word", "phone_sequence", "phone_count"]
        ]
        .sort_values(
            ["phone_count", "word"],
            ascending=[False, True]
        )
        .head(n)
        .reset_index(drop=True)
    )


def build_shortest_pronunciations(dictionary_df, n=100):
    """Return the shortest pronunciation sequences."""
    return (
        dictionary_df[
            ["word", "phone_sequence", "phone_count"]
        ]
        .sort_values(
            ["phone_count", "word"],
            ascending=[True, True]
        )
        .head(n)
        .reset_index(drop=True)
    )


def build_dictionary_inventory_summary(
    dictionary_df,
    malformed_df,
    dictionary_path
):
    """Create one overall Script 3 summary."""
    phone_inventory = build_phone_inventory(dictionary_df)

    return pd.DataFrame([{
        "dictionary_file": str(dictionary_path),
        "dictionary_entries": len(dictionary_df),
        "unique_words": dictionary_df["word"].nunique()
        if not dictionary_df.empty else 0,
        "unique_phone_symbols": phone_inventory["phone"].nunique()
        if not phone_inventory.empty else 0,
        "minimum_phone_count": dictionary_df["phone_count"].min()
        if not dictionary_df.empty else 0,
        "maximum_phone_count": dictionary_df["phone_count"].max()
        if not dictionary_df.empty else 0,
        "mean_phone_count": dictionary_df["phone_count"].mean()
        if not dictionary_df.empty else 0,
        "median_phone_count": dictionary_df["phone_count"].median()
        if not dictionary_df.empty else 0,
        "malformed_lines": len(malformed_df)
    }])


def main():
    parser = argparse.ArgumentParser(
        description="Analyse the NCHLT isiZulu pronunciation dictionary."
    )

    parser.add_argument(
        "--dictionary",
        required=True,
        help="Path to the NCHLT pronunciation dictionary."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for Script 3 results."
    )

    args = parser.parse_args()

    dictionary_path = Path(args.dictionary)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not dictionary_path.exists():
        raise FileNotFoundError(
            f"Dictionary file not found: {dictionary_path}"
        )

    print("=" * 70)
    print("NCHLT isiZulu Pronunciation Dictionary Analysis")
    print("=" * 70)

    print(f"Dictionary: {dictionary_path}")
    print(f"Output:     {output_dir}")
    print()

    # ------------------------------------------------------------
    # 1. Parse dictionary
    # ------------------------------------------------------------

    print("Parsing pronunciation dictionary...")

    dictionary_df, malformed_df = parse_dictionary(dictionary_path)

    print(f"Dictionary entries: {len(dictionary_df):,}")
    print(f"Malformed lines:     {len(malformed_df):,}")
    print()

    # ------------------------------------------------------------
    # 2. Phone inventory
    # ------------------------------------------------------------

    print("Building phone inventory...")

    phone_inventory = build_phone_inventory(dictionary_df)

    print(
        f"Unique phone symbols: "
        f"{len(phone_inventory):,}"
    )
    print()

    # ------------------------------------------------------------
    # 3. Sequence statistics
    # ------------------------------------------------------------

    print("Calculating pronunciation sequence statistics...")

    sequence_summary = build_sequence_summary(dictionary_df)
    phone_distribution = build_phone_count_distribution(dictionary_df)

    print("Done.")
    print()

    # ------------------------------------------------------------
    # 4. Longest / shortest pronunciations
    # ------------------------------------------------------------

    longest = build_longest_pronunciations(dictionary_df)
    shortest = build_shortest_pronunciations(dictionary_df)

    # ------------------------------------------------------------
    # 5. Overall summary
    # ------------------------------------------------------------

    overall_summary = build_dictionary_inventory_summary(
        dictionary_df,
        malformed_df,
        dictionary_path
    )

    # ------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------

    print("Writing Script 3 outputs...")

    dictionary_df.to_csv(
        output_dir / "nchlt_script3_dictionary_entries.csv",
        index=False,
        encoding="utf-8"
    )

    malformed_df.to_csv(
        output_dir / "nchlt_script3_malformed_lines.csv",
        index=False,
        encoding="utf-8"
    )

    phone_inventory.to_csv(
        output_dir / "nchlt_script3_phone_inventory.csv",
        index=False,
        encoding="utf-8"
    )

    sequence_summary.to_csv(
        output_dir / "nchlt_script3_sequence_summary.csv",
        index=False,
        encoding="utf-8"
    )

    phone_distribution.to_csv(
        output_dir / "nchlt_script3_phone_count_distribution.csv",
        index=False,
        encoding="utf-8"
    )

    longest.to_csv(
        output_dir / "nchlt_script3_longest_pronunciations.csv",
        index=False,
        encoding="utf-8"
    )

    shortest.to_csv(
        output_dir / "nchlt_script3_shortest_pronunciations.csv",
        index=False,
        encoding="utf-8"
    )

    overall_summary.to_csv(
        output_dir / "nchlt_script3_overall_summary.csv",
        index=False,
        encoding="utf-8"
    )

    print()
    print("Script 3 completed successfully.")
    print()
    print("Outputs:")
    for file in sorted(output_dir.glob("*.csv")):
        print(f"  {file.name}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()