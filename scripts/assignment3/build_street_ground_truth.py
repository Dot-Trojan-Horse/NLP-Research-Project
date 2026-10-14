import argparse
import csv
import re
from pathlib import Path


# ============================================================
# Utility functions
# ============================================================

def normalize_text(text):
    """
    Normalize text for lexical matching.
    Keeps the original form elsewhere.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_street_name(street):
    """
    Convert a street name into lexical tokens.

    Examples:
        'Mabele_Street' -> ['mabele']
        'Baphelabantu_Street' -> ['baphelabantu']
    """
    if not street:
        return []

    text = str(street).strip().lower()

    # Remove common street-type suffixes because the
    # experimental street-name pronunciation target is
    # the lexical street name itself.
    suffixes = [
        "_street",
        " street",
        "_road",
        " road",
        "_avenue",
        " avenue",
        "_drive",
        " drive",
        "_crescent",
        " crescent",
        "_close",
        " close",
        "_lane",
        " lane",
        "_place",
        " place",
    ]

    for suffix in suffixes:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break

    # Convert separators into spaces.
    text = re.sub(r"[_\-]+", " ", text)

    return [token for token in text.split() if token]


# ============================================================
# NCHLT dictionary
# ============================================================

def load_dictionary(dictionary_path):
    """
    Load the NCHLT pronunciation dictionary.

    Expected format:

        word    phone phone phone ...

    The script preserves duplicate lexical entries.
    """

    entries = {}

    malformed = 0
    duplicate_entries = 0

    with open(dictionary_path, "r", encoding="utf-8", errors="replace") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                malformed += 1
                continue

            word = normalize_text(parts[0])
            phones = parts[1:]

            if not word or not phones:
                malformed += 1
                continue

            if word in entries:
                duplicate_entries += 1

            entries.setdefault(word, []).append(
                " ".join(phones)
            )

    return entries, malformed, duplicate_entries


# ============================================================
# Street dataset
# ============================================================

def load_street_dataset(path):
    """
    Load the Layer 5 street evidence CSV.
    """

    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("Street dataset has no header row.")

        if "street" not in reader.fieldnames:
            raise ValueError(
                "Expected a 'street' column in the experimental dataset."
            )

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# Ground-truth candidate construction
# ============================================================

def analyse_street(street_row, dictionary):

    street = street_row.get("street", "").strip()

    normalized_street = normalize_text(street)

    tokens = tokenize_street_name(street)

    exact_matches = dictionary.get(normalized_street, [])

    token_results = []

    for token in tokens:

        matches = dictionary.get(token, [])

        token_results.append(
            {
                "token": token,
                "dictionary_match": bool(matches),
                "candidate_phones": " | ".join(matches)
                if matches
                else ""
            }
        )

    matched_tokens = [
        item for item in token_results
        if item["dictionary_match"]
    ]

    if exact_matches:

        match_status = "candidate_dictionary_match"

        candidate_sequence = " | ".join(exact_matches)

    elif matched_tokens:

        match_status = "partial_dictionary_match"

        candidate_sequence = " | ".join(
            f'{item["token"]}: {item["candidate_phones"]}'
            for item in matched_tokens
        )

    else:

        match_status = "no_dictionary_match"

        candidate_sequence = ""

    # Native evidence comes from the experimental dataset.
    # These fields deliberately remain evidence indicators,
    # not automatically accepted ground truth.
    native_available = (
        "yes"
        if street_row.get("native_duration_s", "").strip()
        else "no"
    )

    if native_available == "yes":
        validation_status = "requires_native_validation"
    else:
        validation_status = "pending"

    return {
        "street": street,
        "normalized_street": normalized_street,
        "lexical_tokens": " | ".join(tokens),
        "dictionary_exact_match": (
            "yes" if exact_matches else "no"
        ),
        "dictionary_token_matches": len(matched_tokens),
        "dictionary_token_count": len(tokens),
        "candidate_dictionary_phone_sequence": candidate_sequence,
        "dictionary_match_status": match_status,
        "native_recording_available": native_available,
        "native_recording_count": "",
        "primary_native_evidence": "",
        "supporting_native_evidence": "",
        "researcher_validation": "",
        "ground_truth_phone_sequence": "",
        "validation_status": validation_status,
        "notes": ""
    }


# ============================================================
# Output helpers
# ============================================================

def write_csv(path, rows, fieldnames):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main analysis
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build a structured ground-truth validation "
            "framework for the 33 experimental isiZulu street names."
        )
    )

    parser.add_argument(
        "--dictionary",
        required=True,
        help="Path to NCHLT pronunciation dictionary."
    )

    parser.add_argument(
        "--street-data",
        required=True,
        help="Path to Layer 5 experimental street evidence CSV."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory."
    )

    args = parser.parse_args()

    dictionary_path = Path(args.dictionary)
    street_data_path = Path(args.street_data)
    output_dir = Path(args.output)

    print("=" * 70)
    print("Assignment 3 - Experimental Street Ground-Truth Framework")
    print("=" * 70)

    print(f"Dictionary:  {dictionary_path}")
    print(f"Street data: {street_data_path}")
    print(f"Output:      {output_dir}")
    print()

    # --------------------------------------------------------
    # Load dictionary
    # --------------------------------------------------------

    print("Loading NCHLT pronunciation dictionary...")

    dictionary, malformed, duplicate_entries = load_dictionary(
        dictionary_path
    )

    print(
        f"Unique dictionary words available: {len(dictionary):,}"
    )

    print(
        f"Malformed dictionary lines: {malformed:,}"
    )

    print(
        f"Duplicate lexical entries: {duplicate_entries:,}"
    )

    print()

    # --------------------------------------------------------
    # Load experimental dataset
    # --------------------------------------------------------

    print("Loading experimental street-name dataset...")

    street_rows = load_street_dataset(
        street_data_path
    )

    print(
        f"Experimental street rows: {len(street_rows):,}"
    )

    print()

    # --------------------------------------------------------
    # Analyse
    # --------------------------------------------------------

    print("Building ground-truth validation framework...")

    master_rows = []

    for row in street_rows:

        result = analyse_street(
            row,
            dictionary
        )

        master_rows.append(result)

    # --------------------------------------------------------
    # Summary calculations
    # --------------------------------------------------------

    total = len(master_rows)

    exact_matches = sum(
        row["dictionary_exact_match"] == "yes"
        for row in master_rows
    )

    partial_matches = sum(
        row["dictionary_match_status"]
        == "partial_dictionary_match"
        for row in master_rows
    )

    no_matches = sum(
        row["dictionary_match_status"]
        == "no_dictionary_match"
        for row in master_rows
    )

    native_available = sum(
        row["native_recording_available"] == "yes"
        for row in master_rows
    )

    requires_validation = sum(
        row["validation_status"]
        == "requires_native_validation"
        for row in master_rows
    )

    # --------------------------------------------------------
    # Master table
    # --------------------------------------------------------

    master_fields = [
        "street",
        "normalized_street",
        "lexical_tokens",
        "dictionary_exact_match",
        "dictionary_token_matches",
        "dictionary_token_count",
        "candidate_dictionary_phone_sequence",
        "dictionary_match_status",
        "native_recording_available",
        "native_recording_count",
        "primary_native_evidence",
        "supporting_native_evidence",
        "researcher_validation",
        "ground_truth_phone_sequence",
        "validation_status",
        "notes"
    ]

    master_path = (
        output_dir
        / "assignment3_ground_truth_master.csv"
    )

    write_csv(
        master_path,
        master_rows,
        master_fields
    )

    # --------------------------------------------------------
    # Validation-status table
    # --------------------------------------------------------

    validation_rows = []

    for row in master_rows:

        validation_rows.append(
            {
                "street": row["street"],
                "dictionary_match_status":
                    row["dictionary_match_status"],
                "native_recording_available":
                    row["native_recording_available"],
                "primary_native_evidence":
                    row["primary_native_evidence"],
                "supporting_native_evidence":
                    row["supporting_native_evidence"],
                "researcher_validation":
                    row["researcher_validation"],
                "ground_truth_phone_sequence":
                    row["ground_truth_phone_sequence"],
                "validation_status":
                    row["validation_status"],
                "notes":
                    row["notes"]
            }
        )

    validation_path = (
        output_dir
        / "assignment3_ground_truth_validation_status.csv"
    )

    write_csv(
        validation_path,
        validation_rows,
        [
            "street",
            "dictionary_match_status",
            "native_recording_available",
            "primary_native_evidence",
            "supporting_native_evidence",
            "researcher_validation",
            "ground_truth_phone_sequence",
            "validation_status",
            "notes"
        ]
    )

    # --------------------------------------------------------
    # Phone-sequence table
    # --------------------------------------------------------

    phone_rows = []

    for row in master_rows:

        phone_rows.append(
            {
                "street": row["street"],
                "candidate_dictionary_phone_sequence":
                    row["candidate_dictionary_phone_sequence"],
                "ground_truth_phone_sequence":
                    row["ground_truth_phone_sequence"],
                "validation_status":
                    row["validation_status"]
            }
        )

    phone_path = (
        output_dir
        / "assignment3_ground_truth_phone_sequences.csv"
    )

    write_csv(
        phone_path,
        phone_rows,
        [
            "street",
            "candidate_dictionary_phone_sequence",
            "ground_truth_phone_sequence",
            "validation_status"
        ]
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    summary_rows = [
        {
            "metric": "total_experimental_streets",
            "value": total
        },
        {
            "metric": "exact_dictionary_matches",
            "value": exact_matches
        },
        {
            "metric": "partial_dictionary_matches",
            "value": partial_matches
        },
        {
            "metric": "no_dictionary_matches",
            "value": no_matches
        },
        {
            "metric": "native_recordings_available",
            "value": native_available
        },
        {
            "metric": "requiring_native_validation",
            "value": requires_validation
        }
    ]

    summary_path = (
        output_dir
        / "assignment3_ground_truth_summary.csv"
    )

    write_csv(
        summary_path,
        summary_rows,
        ["metric", "value"]
    )

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print()
    print("Ground-truth framework completed.")
    print()
    print("Summary:")
    print(f"  Experimental streets:       {total}")
    print(f"  Exact dictionary matches:   {exact_matches}")
    print(f"  Partial matches:            {partial_matches}")
    print(f"  No dictionary matches:      {no_matches}")
    print(f"  Native recordings available:{native_available}")
    print(f"  Requiring validation:       {requires_validation}")
    print()
    print("Outputs:")
    print(f"  {master_path.name}")
    print(f"  {validation_path.name}")
    print(f"  {phone_path.name}")
    print(f"  {summary_path.name}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()