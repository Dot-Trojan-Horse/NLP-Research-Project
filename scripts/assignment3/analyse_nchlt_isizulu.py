from pathlib import Path
import csv
import argparse
import statistics
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse the prepared NCHLT isiZulu corpus for Assignment 3."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to nchlt_audio_transcription_matches.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for Script 2 outputs.",
    )

    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values, percentile):
    """Calculate a simple percentile using linear interpolation."""

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = lower + 1

    if upper >= len(values):
        return values[-1]

    fraction = position - lower

    return values[lower] + (
        values[upper] - values[lower]
    ) * fraction


def median(values):
    if not values:
        return None

    return statistics.median(values)


def mean(values):
    if not values:
        return None

    return statistics.mean(values)


def analyse_partition(records):
    counts = Counter(
        row.get("partition", "").strip().lower()
        for row in records
    )

    rows = []

    for partition, count in sorted(counts.items()):
        rows.append(
            {
                "partition": partition,
                "record_count": count,
                "percentage": round(
                    (count / len(records)) * 100,
                    2,
                )
                if records
                else 0,
            }
        )

    return rows


def analyse_speakers(records):
    speaker_counts = Counter(
        row.get("speaker_id", "").strip()
        for row in records
        if row.get("speaker_id", "").strip()
    )

    rows = []

    for speaker_id, count in sorted(
        speaker_counts.items(),
        key=lambda item: int(item[0])
        if item[0].isdigit()
        else item[0],
    ):
        rows.append(
            {
                "speaker_id": speaker_id,
                "recording_count": count,
            }
        )

    return rows


def analyse_gender(records):
    counts = Counter()

    for row in records:
        gender = row.get("gender", "").strip().lower()

        if gender:
            counts[gender] += 1
        else:
            counts["missing"] += 1

    rows = []

    total = len(records)

    for gender, count in sorted(counts.items()):
        rows.append(
            {
                "gender": gender,
                "recording_count": count,
                "percentage": round(
                    (count / total) * 100,
                    2,
                )
                if total
                else 0,
            }
        )

    return rows


def analyse_age(records):
    age_values = []

    for row in records:
        age = safe_int(row.get("age", ""))

        if age is not None:
            age_values.append(age)

    if not age_values:
        return []

    return [
        {
            "measure": "count",
            "value": len(age_values),
        },
        {
            "measure": "minimum",
            "value": min(age_values),
        },
        {
            "measure": "maximum",
            "value": max(age_values),
        },
        {
            "measure": "mean",
            "value": round(mean(age_values), 2),
        },
        {
            "measure": "median",
            "value": median(age_values),
        },
        {
            "measure": "p25",
            "value": round(percentile(age_values, 0.25), 2),
        },
        {
            "measure": "p75",
            "value": round(percentile(age_values, 0.75), 2),
        },
    ]


def analyse_duration(records):
    duration_values = []

    for row in records:
        duration = safe_float(row.get("duration", ""))

        if duration is not None and duration >= 0:
            duration_values.append(duration)

    if not duration_values:
        return []

    return [
        {
            "measure": "count",
            "value": len(duration_values),
        },
        {
            "measure": "minimum_seconds",
            "value": round(min(duration_values), 4),
        },
        {
            "measure": "maximum_seconds",
            "value": round(max(duration_values), 4),
        },
        {
            "measure": "mean_seconds",
            "value": round(mean(duration_values), 4),
        },
        {
            "measure": "median_seconds",
            "value": round(median(duration_values), 4),
        },
        {
            "measure": "p25_seconds",
            "value": round(
                percentile(duration_values, 0.25),
                4,
            ),
        },
        {
            "measure": "p75_seconds",
            "value": round(
                percentile(duration_values, 0.75),
                4,
            ),
        },
    ]


def analyse_pdp_score(records):
    score_values = []

    for row in records:
        score = safe_float(row.get("pdp_score", ""))

        if score is not None:
            score_values.append(score)

    if not score_values:
        return []

    return [
        {
            "measure": "count",
            "value": len(score_values),
        },
        {
            "measure": "minimum",
            "value": round(min(score_values), 6),
        },
        {
            "measure": "maximum",
            "value": round(max(score_values), 6),
        },
        {
            "measure": "mean",
            "value": round(mean(score_values), 6),
        },
        {
            "measure": "median",
            "value": round(median(score_values), 6),
        },
        {
            "measure": "p25",
            "value": round(percentile(score_values, 0.25), 6),
        },
        {
            "measure": "p75",
            "value": round(percentile(score_values, 0.75), 6),
        },
    ]


def analyse_transcriptions(records):
    """
    Analyse the orthographic transcription associated with each
    recording.

    This is descriptive corpus analysis only.
    """

    transcription_lengths = []
    word_counts = []
    character_counts = []

    missing = 0

    all_words = Counter()
    transcription_counter = Counter()

    for row in records:

        transcription = row.get(
            "transcription",
            "",
        ).strip()

        if not transcription:
            missing += 1
            continue

        words = transcription.split()

        word_count = len(words)
        character_count = len(transcription)
        length_without_spaces = len(
            transcription.replace(" ", "")
        )

        transcription_lengths.append(
            length_without_spaces
        )

        word_counts.append(word_count)
        character_counts.append(character_count)

        all_words.update(
            word.lower()
            for word in words
        )

        transcription_counter[
            transcription.lower()
        ] += 1

    return (
        transcription_lengths,
        word_counts,
        character_counts,
        missing,
        all_words,
        transcription_counter,
    )


def descriptive_stats(values, prefix):
    if not values:
        return []

    return [
        {
            "measure": "count",
            "value": len(values),
        },
        {
            "measure": f"{prefix}_minimum",
            "value": min(values),
        },
        {
            "measure": f"{prefix}_maximum",
            "value": max(values),
        },
        {
            "measure": f"{prefix}_mean",
            "value": round(mean(values), 2),
        },
        {
            "measure": f"{prefix}_median",
            "value": median(values),
        },
        {
            "measure": f"{prefix}_p25",
            "value": round(
                percentile(values, 0.25),
                2,
            ),
        },
        {
            "measure": f"{prefix}_p75",
            "value": round(
                percentile(values, 0.75),
                2,
            ),
        },
    ]

def analyse_recording_distribution(records):
    """
    Count how many recordings belong to each speaker.
    """

    speaker_counts = Counter(
        row.get("speaker_id", "").strip()
        for row in records
        if row.get("speaker_id", "").strip()
    )

    values = list(speaker_counts.values())

    if not values:
        return []

    return descriptive_stats(
        values,
        "recordings_per_speaker",
    )


def main():

    args = parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    print("=" * 70)
    print("NCHLT isiZulu CORPUS ANALYSIS - SCRIPT 2")
    print("=" * 70)

    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =============================================================
    # STEP 1
    # =============================================================

    print("Reading prepared NCHLT dataset...")

    records = read_csv(input_path)

    print(
        f"Records loaded: {len(records)}"
    )
    print()

    if not records:
        raise ValueError(
            "The input CSV contains no records."
        )

    # =============================================================
    # STEP 2
    # Basic coverage
    # =============================================================

    matched = sum(
        1
        for row in records
        if row.get("match_status", "").strip().lower()
        == "matched"
    )

    unmatched = len(records) - matched

    coverage_rows = [
        {
            "metric": "total_records",
            "value": len(records),
        },
        {
            "metric": "matched_records",
            "value": matched,
        },
        {
            "metric": "unmatched_records",
            "value": unmatched,
        },
        {
            "metric": "match_percentage",
            "value": round(
                (matched / len(records)) * 100,
                2,
            ),
        },
    ]

    write_csv(
        output_dir / "nchlt_script2_coverage.csv",
        coverage_rows,
        ["metric", "value"],
    )

    # =============================================================
    # STEP 3
    # Partition analysis
    # =============================================================

    print("Analysing partitions...")

    partition_rows = analyse_partition(records)

    write_csv(
        output_dir / "nchlt_script2_partition_summary.csv",
        partition_rows,
        [
            "partition",
            "record_count",
            "percentage",
        ],
    )

    # =============================================================
    # STEP 4
    # Speaker analysis
    # =============================================================

    print("Analysing speakers...")

    speaker_rows = analyse_speakers(records)

    write_csv(
        output_dir / "nchlt_script2_speaker_summary.csv",
        speaker_rows,
        [
            "speaker_id",
            "recording_count",
        ],
    )

    speaker_distribution = analyse_recording_distribution(
        records
    )

    write_csv(
        output_dir
        / "nchlt_script2_recordings_per_speaker_stats.csv",
        speaker_distribution,
        [
            "measure",
            "value",
        ],
    )

    # =============================================================
    # STEP 5
    # Gender analysis
    # =============================================================

    print("Analysing speaker gender...")

    gender_rows = analyse_gender(records)

    write_csv(
        output_dir / "nchlt_script2_gender_summary.csv",
        gender_rows,
        [
            "gender",
            "recording_count",
            "percentage",
        ],
    )

    # =============================================================
    # STEP 6
    # Age analysis
    # =============================================================

    print("Analysing speaker ages...")

    age_rows = analyse_age(records)

    write_csv(
        output_dir / "nchlt_script2_age_statistics.csv",
        age_rows,
        [
            "measure",
            "value",
        ],
    )

    # =============================================================
    # STEP 7
    # Duration analysis
    # =============================================================

    print("Analysing recording durations...")

    duration_rows = analyse_duration(records)

    write_csv(
        output_dir
        / "nchlt_script2_duration_statistics.csv",
        duration_rows,
        [
            "measure",
            "value",
        ],
    )

    # =============================================================
    # STEP 8
    # PDP score analysis
    # =============================================================

    print("Analysing PDP scores...")

    pdp_rows = analyse_pdp_score(records)

    write_csv(
        output_dir
        / "nchlt_script2_pdp_score_statistics.csv",
        pdp_rows,
        [
            "measure",
            "value",
        ],
    )

    # =============================================================
    # STEP 9
    # Transcription analysis
    # =============================================================

    print("Analysing orthographic transcriptions...")

    (
        transcription_lengths,
        word_counts,
        character_counts,
        missing_transcriptions,
        word_counter,
        transcription_counter,
    ) = analyse_transcriptions(records)

    transcription_summary = [
        {
            "metric": "records",
            "value": len(records),
        },
        {
            "metric": "non_empty_transcriptions",
            "value": len(records)
            - missing_transcriptions,
        },
        {
            "metric": "missing_transcriptions",
            "value": missing_transcriptions,
        },
        {
            "metric": "unique_transcriptions",
            "value": len(transcription_counter),
        },
        {
            "metric": "unique_words",
            "value": len(word_counter),
        },
        {
            "metric": "total_word_tokens",
            "value": sum(word_counter.values()),
        },
    ]

    write_csv(
        output_dir
        / "nchlt_script2_transcription_summary.csv",
        transcription_summary,
        [
            "metric",
            "value",
        ],
    )

    # =============================================================
    # STEP 10
    # Transcription length statistics
    # =============================================================

    transcription_length_rows = descriptive_stats(
        transcription_lengths,
        "transcription_characters_no_spaces",
    )

    write_csv(
        output_dir
        / "nchlt_script2_transcription_length_statistics.csv",
        transcription_length_rows,
        [
            "measure",
            "value",
        ],
    )

    # =============================================================
    # STEP 11
    # Word-count statistics
    # =============================================================

    word_count_rows = descriptive_stats(
        word_counts,
        "words_per_transcription",
    )

    write_csv(
        output_dir
        / "nchlt_script2_word_count_statistics.csv",
        word_count_rows,
        [
            "measure",
            "value",
        ],
    )

    # =============================================================
    # STEP 12
    # Character-count statistics
    # =============================================================

    character_count_rows = descriptive_stats(
        character_counts,
        "characters_per_transcription",
    )

    write_csv(
        output_dir
        / "nchlt_script2_character_count_statistics.csv",
        character_count_rows,
        [
            "measure",
            "value",
        ],
    )

    # =============================================================
    # STEP 13
    # Most frequent words
    # =============================================================

    print("Calculating word frequencies...")

    word_frequency_rows = []

    for word, count in word_counter.most_common(100):

        word_frequency_rows.append(
            {
                "rank": len(word_frequency_rows) + 1,
                "word": word,
                "frequency": count,
            }
        )

    write_csv(
        output_dir
        / "nchlt_script2_top_100_words.csv",
        word_frequency_rows,
        [
            "rank",
            "word",
            "frequency",
        ],
    )

    # =============================================================
    # STEP 14
    # Most repeated transcriptions
    # =============================================================

    repeated_transcription_rows = []

    for transcription, count in (
        transcription_counter.most_common(100)
    ):

        repeated_transcription_rows.append(
            {
                "rank": len(repeated_transcription_rows) + 1,
                "transcription": transcription,
                "frequency": count,
            }
        )

    write_csv(
        output_dir
        / "nchlt_script2_top_100_transcriptions.csv",
        repeated_transcription_rows,
        [
            "rank",
            "transcription",
            "frequency",
        ],
    )

    # =============================================================
    # STEP 15
    # Master summary
    # =============================================================

    unique_speakers = len(
        {
            row.get("speaker_id", "").strip()
            for row in records
            if row.get("speaker_id", "").strip()
        }
    )

    master_summary = [
        {
            "metric": "total_recordings",
            "value": len(records),
        },
        {
            "metric": "unique_speakers",
            "value": unique_speakers,
        },
        {
            "metric": "matched_recordings",
            "value": matched,
        },
        {
            "metric": "unmatched_recordings",
            "value": unmatched,
        },
        {
            "metric": "match_percentage",
            "value": round(
                (matched / len(records)) * 100,
                2,
            ),
        },
        {
            "metric": "missing_transcriptions",
            "value": missing_transcriptions,
        },
        {
            "metric": "unique_transcriptions",
            "value": len(transcription_counter),
        },
        {
            "metric": "unique_word_types",
            "value": len(word_counter),
        },
        {
            "metric": "total_word_tokens",
            "value": sum(word_counter.values()),
        },
    ]

    write_csv(
        output_dir
        / "nchlt_script2_master_summary.csv",
        master_summary,
        [
            "metric",
            "value",
        ],
    )

    # =============================================================
    # FINAL REPORT
    # =============================================================

    print()
    print("=" * 70)
    print("SCRIPT 2 COMPLETE")
    print("=" * 70)

    print(
        f"Total recordings       : {len(records)}"
    )

    print(
        f"Unique speakers        : {unique_speakers}"
    )

    print(
        f"Matched recordings     : {matched}"
    )

    print(
        f"Unmatched recordings   : {unmatched}"
    )

    print(
        f"Match percentage       : "
        f"{(matched / len(records)) * 100:.2f}%"
    )

    print(
        f"Missing transcriptions : "
        f"{missing_transcriptions}"
    )

    print(
        f"Unique word types      : "
        f"{len(word_counter)}"
    )

    print(
        f"Total word tokens      : "
        f"{sum(word_counter.values())}"
    )

    print()
    print("All Script 2 outputs were written to:")
    print(output_dir)

    print()
    print(
        "IMPORTANT: This analysis is descriptive corpus "
        "analysis. It does not establish street-name ground "
        "truth and does not calculate PER or WER."
    )

if __name__ == "__main__":
    main()