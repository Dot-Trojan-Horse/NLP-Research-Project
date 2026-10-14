from pathlib import Path
import argparse
import re
import pandas as pd


# ============================================================
# Utility functions
# ============================================================

def normalise_text(value):
    """Normalise text for comparison while retaining useful characters."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def normalise_lookup_word(value):
    """Normalise an individual lexical item."""
    text = normalise_text(value)

    # Remove surrounding punctuation
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)

    return text


# ============================================================
# Dictionary parser
# ============================================================

def parse_dictionary(dictionary_path):
    """
    Parse the NCHLT pronunciation dictionary.

    Expected format:

        word    phone phone phone ...

    Returns:
        dictionary_df
    """

    records = []

    with open(
        dictionary_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line_number, raw_line in enumerate(f, start=1):

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#") or line.startswith(";"):
                continue

            parts = re.split(r"\s+", line)

            if len(parts) < 2:
                continue

            word = normalise_lookup_word(parts[0])
            phones = parts[1:]

            if not word:
                continue

            records.append({
                "line_number": line_number,
                "word": word,
                "phone_sequence": " ".join(phones),
                "phone_count": len(phones)
            })

    return pd.DataFrame(records)


# ============================================================
# Identify street-name column
# ============================================================

def identify_street_column(df):
    """
    Attempt to identify the street-name column automatically.

    The script first checks common column names.
    """

    preferred_names = [
        "street_name",
        "street",
        "name",
        "streetname",
        "isiZulu_street_name",
        "isizulu_street_name",
        "word"
    ]

    normalised_columns = {
        normalise_text(column).replace(" ", "_"): column
        for column in df.columns
    }

    for preferred in preferred_names:

        if preferred in normalised_columns:
            return normalised_columns[preferred]

    # Secondary heuristic
    for column in df.columns:

        name = normalise_text(column)

        if "street" in name and "name" in name:
            return column

    return None


# ============================================================
# Dictionary lookup
# ============================================================

def lookup_street_name(street_name, dictionary_words):
    """
    Compare a street name against the dictionary.

    Matching levels:

    EXACT
        The complete street name is one dictionary entry.

    TOKEN_MATCH
        Individual tokens from the street name occur in the
        dictionary.

    NO_MATCH
        No useful lexical match was found.
    """

    normalised_street = normalise_text(street_name)

    if not normalised_street:
        return {
            "match_status": "EMPTY",
            "matched_dictionary_words": "",
            "candidate_phone_sequences": "",
            "matched_token_count": 0,
            "street_token_count": 0
        }

    # --------------------------------------------------------
    # Exact complete-name match
    # --------------------------------------------------------

    if normalised_street in dictionary_words:

        entry = dictionary_words[normalised_street]

        return {
            "match_status": "EXACT",
            "matched_dictionary_words": normalised_street,
            "candidate_phone_sequences": entry["phone_sequence"],
            "matched_token_count": 1,
            "street_token_count": 1
        }

    # --------------------------------------------------------
    # Token-level matching
    # --------------------------------------------------------

    tokens = [
        normalise_lookup_word(token)
        for token in normalised_street.split()
    ]

    tokens = [
        token for token in tokens
        if token
    ]

    matched_words = []
    phone_sequences = []

    for token in tokens:

        if token in dictionary_words:

            entry = dictionary_words[token]

            matched_words.append(token)
            phone_sequences.append(
                f"{token}: {entry['phone_sequence']}"
            )

    if matched_words:

        return {
            "match_status": "TOKEN_MATCH",
            "matched_dictionary_words": " | ".join(
                matched_words
            ),
            "candidate_phone_sequences": " | ".join(
                phone_sequences
            ),
            "matched_token_count": len(matched_words),
            "street_token_count": len(tokens)
        }

    # --------------------------------------------------------
    # No match
    # --------------------------------------------------------

    return {
        "match_status": "NO_MATCH",
        "matched_dictionary_words": "",
        "candidate_phone_sequences": "",
        "matched_token_count": 0,
        "street_token_count": len(tokens)
    }


# ============================================================
# Main analysis
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Assess NCHLT pronunciation-dictionary coverage "
            "for the experimental isiZulu street names."
        )
    )

    parser.add_argument(
        "--dictionary",
        required=True,
        help="Path to the NCHLT pronunciation dictionary."
    )

    parser.add_argument(
        "--street-data",
        required=True,
        help=(
            "CSV containing the experimental street names."
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for Script 4 results."
    )

    args = parser.parse_args()

    dictionary_path = Path(args.dictionary)
    street_data_path = Path(args.street_data)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not dictionary_path.exists():
        raise FileNotFoundError(
            f"Dictionary not found: {dictionary_path}"
        )

    if not street_data_path.exists():
        raise FileNotFoundError(
            f"Street-name CSV not found: {street_data_path}"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)
    print(
        "NCHLT Pronunciation Coverage Analysis "
        "for Experimental Street Names"
    )
    print("=" * 70)

    print(f"Dictionary:  {dictionary_path}")
    print(f"Street data: {street_data_path}")
    print(f"Output:      {output_dir}")
    print()

    # --------------------------------------------------------
    # Parse dictionary
    # --------------------------------------------------------

    print("Loading NCHLT pronunciation dictionary...")

    dictionary_df = parse_dictionary(dictionary_path)

    print(
        f"Dictionary entries loaded: "
        f"{len(dictionary_df):,}"
    )

    # --------------------------------------------------------
    # Handle duplicate dictionary words
    # --------------------------------------------------------

    duplicate_count = (
        dictionary_df["word"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate lexical entries: "
        f"{duplicate_count:,}"
    )

    # Keep first occurrence for simple lookup
    dictionary_df = (
        dictionary_df
        .drop_duplicates(
            subset=["word"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    dictionary_words = {
        row["word"]: row
        for _, row in dictionary_df.iterrows()
    }

    print(
        f"Unique dictionary words available: "
        f"{len(dictionary_words):,}"
    )

    print()

    # --------------------------------------------------------
    # Load street dataset
    # --------------------------------------------------------

    print("Loading experimental street-name dataset...")

    street_df = pd.read_csv(
        street_data_path
    )

    print(
        f"Rows in street dataset: "
        f"{len(street_df):,}"
    )

    print(
        "Columns detected:"
    )

    for column in street_df.columns:
        print(f"  - {column}")

    print()

    # --------------------------------------------------------
    # Identify street-name column
    # --------------------------------------------------------

    street_column = identify_street_column(
        street_df
    )

    if street_column is None:

        raise ValueError(
            "Could not automatically identify the "
            "street-name column. Please inspect the CSV "
            "column names and specify the appropriate "
            "column in the script."
        )

    print(
        f"Street-name column selected: "
        f"{street_column}"
    )

    print()

    # --------------------------------------------------------
    # Analyse each street
    # --------------------------------------------------------

    print("Analysing dictionary coverage...")

    results = []

    for index, row in street_df.iterrows():

        street_name = row[street_column]

        lookup = lookup_street_name(
            street_name,
            dictionary_words
        )

        result = {
            "street_index": index + 1,
            "street_name": street_name,
            **lookup
        }

        results.append(result)

    coverage_df = pd.DataFrame(results)

    print("Coverage analysis complete.")
    print()

    # --------------------------------------------------------
    # Coverage summary
    # --------------------------------------------------------

    total_streets = len(coverage_df)

    exact_matches = (
        coverage_df["match_status"]
        == "EXACT"
    ).sum()

    token_matches = (
        coverage_df["match_status"]
        == "TOKEN_MATCH"
    ).sum()

    no_matches = (
        coverage_df["match_status"]
        == "NO_MATCH"
    ).sum()

    empty_entries = (
        coverage_df["match_status"]
        == "EMPTY"
    ).sum()

    usable_matches = exact_matches + token_matches

    if total_streets > 0:

        exact_percentage = (
            exact_matches /
            total_streets *
            100
        )

        usable_percentage = (
            usable_matches /
            total_streets *
            100
        )

    else:

        exact_percentage = 0
        usable_percentage = 0

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    summary_df = pd.DataFrame([{

        "street_dataset":
            str(street_data_path),

        "dictionary_file":
            str(dictionary_path),

        "total_streets":
            total_streets,

        "exact_dictionary_matches":
            exact_matches,

        "token_level_matches":
            token_matches,

        "no_dictionary_match":
            no_matches,

        "empty_street_entries":
            empty_entries,

        "usable_dictionary_matches":
            usable_matches,

        "exact_match_percentage":
            exact_percentage,

        "usable_match_percentage":
            usable_percentage,

        "unique_dictionary_words":
            len(dictionary_words),

        "unique_phone_symbols":
            dictionary_df["phone_sequence"]
            .str.split()
            .explode()
            .nunique()

    }])

    # --------------------------------------------------------
    # Match-status summary
    # --------------------------------------------------------

    status_summary = (
        coverage_df
        .groupby("match_status")
        .size()
        .reset_index(
            name="street_count"
        )
    )

    status_summary[
        "percentage_of_streets"
    ] = (
        status_summary["street_count"]
        / total_streets
        * 100
        if total_streets > 0
        else 0
    )

    # --------------------------------------------------------
    # Candidate phone sequences
    # --------------------------------------------------------

    candidate_sequences = coverage_df[
        coverage_df["candidate_phone_sequences"]
        .astype(str)
        .str.strip()
        != ""
    ].copy()

    # --------------------------------------------------------
    # No-match list
    # --------------------------------------------------------

    no_match_df = coverage_df[
        coverage_df["match_status"]
        == "NO_MATCH"
    ].copy()

    # --------------------------------------------------------
    # Exact-match list
    # --------------------------------------------------------

    exact_match_df = coverage_df[
        coverage_df["match_status"]
        == "EXACT"
    ].copy()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    print("Writing Script 4 outputs...")

    coverage_df.to_csv(
        output_dir
        / "nchlt_script4_street_dictionary_coverage.csv",
        index=False,
        encoding="utf-8"
    )

    summary_df.to_csv(
        output_dir
        / "nchlt_script4_coverage_summary.csv",
        index=False,
        encoding="utf-8"
    )

    status_summary.to_csv(
        output_dir
        / "nchlt_script4_match_status_summary.csv",
        index=False,
        encoding="utf-8"
    )

    candidate_sequences.to_csv(
        output_dir
        / "nchlt_script4_candidate_phone_sequences.csv",
        index=False,
        encoding="utf-8"
    )

    exact_match_df.to_csv(
        output_dir
        / "nchlt_script4_exact_matches.csv",
        index=False,
        encoding="utf-8"
    )

    no_match_df.to_csv(
        output_dir
        / "nchlt_script4_no_dictionary_match.csv",
        index=False,
        encoding="utf-8"
    )

    print()
    print("Script 4 completed successfully.")
    print()
    print("Coverage summary:")
    print(
        f"  Total streets:          {total_streets}"
    )
    print(
        f"  Exact matches:          {exact_matches}"
    )
    print(
        f"  Token-level matches:    {token_matches}"
    )
    print(
        f"  No matches:             {no_matches}"
    )
    print(
        f"  Exact match percentage: "
        f"{exact_percentage:.2f}%"
    )
    print(
        f"  Usable match percentage:"
        f" {usable_percentage:.2f}%"
    )

    print()
    print("Outputs:")

    for file in sorted(
        output_dir.glob("*.csv")
    ):
        print(f"  {file.name}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()