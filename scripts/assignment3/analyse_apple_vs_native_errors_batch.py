from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# LAYER 2
# APPLE MAPS vs NATIVE SPEAKER ERROR ANALYSIS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Sam\Documents\NLP_Projects"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "batch_apple_vs_native"
)

INPUT_FILE = (
    RESULTS_DIR
    / "apple_vs_native_batch_summary.csv"
)

ERROR_FILE = (
    RESULTS_DIR
    / "apple_vs_native_error_analysis_layer2.csv"
)

STATS_FILE = (
    RESULTS_DIR
    / "apple_vs_native_error_statistics_layer2.csv"
)

REVIEW_FILE = (
    RESULTS_DIR
    / "apple_vs_native_manual_review_priority.csv"
)


# ============================================================
# THRESHOLDS
# ============================================================

VERY_SHORT_RATIO = 0.50
SHORT_RATIO = 0.75
COMPARABLE_HIGH = 1.25

LOW_SIMILARITY = 0.010
VERY_LOW_SIMILARITY = 0.008


# ============================================================
# LOAD
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}\n\n"
        "Run analyse_apple_vs_native_batch.py first."
    )


df = pd.read_csv(INPUT_FILE)


print("=" * 80)
print("LAYER 2: APPLE MAPS vs NATIVE ERROR ANALYSIS")
print("=" * 80)

print()
print(f"Input: {INPUT_FILE}")
print()
print(f"Streets loaded: {len(df)}")


# ============================================================
# VERIFY ACTUAL BATCH SCHEMA
# ============================================================

required_columns = [
    "street",
    "system_recording",
    "native_reference_recording",
    "native_recording_count",
    "system_duration_s",
    "native_median_duration_s",
    "duration_difference_s",
    "duration_ratio",
    "duration_category",
    "normalized_dtw",
    "similarity",
    "acoustic_difference_category",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:
    raise ValueError(
        "\nThe batch summary is missing required columns:\n"
        + "\n".join(f"  - {x}" for x in missing)
    )


# ============================================================
# DURATION CLASS
# ============================================================

def classify_duration(ratio):

    if pd.isna(ratio):
        return "unknown"

    if ratio < VERY_SHORT_RATIO:
        return "very_short"

    if ratio < SHORT_RATIO:
        return "shorter"

    if ratio <= COMPARABLE_HIGH:
        return "comparable"

    return "longer"


df["duration_class_layer2"] = (
    df["duration_ratio"]
    .apply(classify_duration)
)


# ============================================================
# ACOUSTIC CLASS
# ============================================================

def classify_acoustic(similarity):

    if pd.isna(similarity):
        return "unknown"

    if similarity < VERY_LOW_SIMILARITY:
        return "very_low_similarity"

    if similarity < LOW_SIMILARITY:
        return "low_similarity"

    return "higher_similarity"


df["acoustic_class_layer2"] = (
    df["similarity"]
    .apply(classify_acoustic)
)


# ============================================================
# CANDIDATE ERROR CLASSIFICATION
# ============================================================

def classify_error(row):

    ratio = row["duration_ratio"]
    similarity = row["similarity"]

    if pd.isna(ratio) or pd.isna(similarity):
        return "insufficient_evidence"

    # Strongest candidate:
    # very short system recording + very low similarity
    if (
        ratio < VERY_SHORT_RATIO
        and similarity < VERY_LOW_SIMILARITY
    ):
        return "strong_candidate_pronunciation_difference"

    # Low similarity regardless of timing
    if similarity < LOW_SIMILARITY:
        return "candidate_acoustic_pronunciation_difference"

    # Extremely compressed speech
    if ratio < VERY_SHORT_RATIO:
        return "timing_or_rate_difference"

    # Noticeably shorter
    if ratio < SHORT_RATIO:
        return "possible_timing_difference"

    # Similar duration but acoustic difference
    if (
        ratio <= COMPARABLE_HIGH
        and similarity < LOW_SIMILARITY
    ):
        return "possible_pronunciation_difference"

    return "no_strong_candidate"


df["candidate_error"] = (
    df.apply(
        classify_error,
        axis=1
    )
)


# ============================================================
# SEVERITY
# ============================================================

def classify_severity(error):

    if error == "strong_candidate_pronunciation_difference":
        return "high"

    if error == "candidate_acoustic_pronunciation_difference":
        return "high"

    if error == "possible_pronunciation_difference":
        return "medium"

    if error == "timing_or_rate_difference":
        return "medium"

    if error == "possible_timing_difference":
        return "low"

    if error == "no_strong_candidate":
        return "low"

    return "unknown"


df["severity"] = (
    df["candidate_error"]
    .apply(classify_severity)
)


# ============================================================
# EVIDENCE DESCRIPTION
# ============================================================

def create_evidence(row):

    ratio = row["duration_ratio"]
    similarity = row["similarity"]
    dtw = row["normalized_dtw"]

    evidence = []

    if ratio < VERY_SHORT_RATIO:
        evidence.append(
            "system recording is less than half "
            "the native median duration"
        )

    elif ratio < SHORT_RATIO:
        evidence.append(
            "system recording is noticeably shorter "
            "than the native median"
        )

    elif ratio > COMPARABLE_HIGH:
        evidence.append(
            "system recording is longer "
            "than the native median"
        )

    else:
        evidence.append(
            "system and native durations are broadly comparable"
        )

    if similarity < VERY_LOW_SIMILARITY:
        evidence.append(
            "very low acoustic similarity"
        )

    elif similarity < LOW_SIMILARITY:
        evidence.append(
            "low acoustic similarity"
        )

    else:
        evidence.append(
            "higher acoustic similarity relative "
            "to this dataset"
        )

    evidence.append(
        f"normalized DTW={dtw:.4f}"
    )

    return "; ".join(evidence)


df["evidence"] = (
    df.apply(
        create_evidence,
        axis=1
    )
)


# ============================================================
# EVIDENCE CATEGORY
# ============================================================

def evidence_category(error):

    if error in [
        "strong_candidate_pronunciation_difference",
        "candidate_acoustic_pronunciation_difference",
        "possible_pronunciation_difference",
    ]:
        return "acoustic_pronunciation_candidate"

    if error in [
        "timing_or_rate_difference",
        "possible_timing_difference",
    ]:
        return "timing_candidate"

    return "no_strong_candidate"


df["evidence_category"] = (
    df["candidate_error"]
    .apply(evidence_category)
)


# ============================================================
# REVIEW PRIORITY
# ============================================================

df["duration_deviation"] = (
    abs(
        1.0
        - df["duration_ratio"]
    )
)


# Rank strongest evidence first.

df["dtw_rank"] = (
    df["normalized_dtw"]
    .rank(
        ascending=False,
        method="min"
    )
)


df["similarity_rank"] = (
    df["similarity"]
    .rank(
        ascending=True,
        method="min"
    )
)


df["duration_rank"] = (
    df["duration_deviation"]
    .rank(
        ascending=False,
        method="min"
    )
)


df["review_priority_score"] = (
    df["dtw_rank"]
    + df["similarity_rank"]
    + df["duration_rank"]
)


# ============================================================
# PRIORITY
# ============================================================

q33 = df["review_priority_score"].quantile(
    0.33
)

q66 = df["review_priority_score"].quantile(
    0.66
)


def priority(score):

    if score <= q33:
        return "high"

    if score <= q66:
        return "medium"

    return "low"


df["review_priority"] = (
    df["review_priority_score"]
    .apply(priority)
)


# ============================================================
# MAIN OUTPUT
# ============================================================

error_output = df[
    [
        "street",
        "system_recording",
        "native_reference_recording",
        "native_recording_count",
        "system_duration_s",
        "native_median_duration_s",
        "duration_difference_s",
        "duration_ratio",
        "duration_category",
        "duration_class_layer2",
        "normalized_dtw",
        "similarity",
        "acoustic_difference_category",
        "acoustic_class_layer2",
        "candidate_error",
        "evidence_category",
        "severity",
        "review_priority",
        "review_priority_score",
        "evidence",
    ]
].copy()


error_output.to_csv(
    ERROR_FILE,
    index=False
)


# ============================================================
# STATISTICS
# ============================================================

counts = (
    error_output[
        "candidate_error"
    ]
    .value_counts()
    .rename_axis(
        "candidate_error"
    )
    .reset_index(
        name="street_count"
    )
)


counts["percentage"] = (
    counts["street_count"]
    / len(error_output)
    * 100
)


counts.to_csv(
    STATS_FILE,
    index=False
)


# ============================================================
# MANUAL REVIEW LIST
# ============================================================

review = (
    error_output
    .sort_values(
        [
            "review_priority_score",
            "normalized_dtw",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


review = review[
    [
        "street",
        "system_recording",
        "native_reference_recording",
        "candidate_error",
        "severity",
        "review_priority",
        "system_duration_s",
        "native_median_duration_s",
        "duration_ratio",
        "normalized_dtw",
        "similarity",
        "evidence",
    ]
]


review.to_csv(
    REVIEW_FILE,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("LAYER 2 RESULTS")
print("=" * 80)

print()
print(
    f"Total streets analysed: "
    f"{len(error_output)}"
)


print()
print("CANDIDATE ERROR CATEGORIES")
print("-" * 80)

for _, row in counts.iterrows():

    print(
        f"{row['candidate_error']:55} "
        f"{int(row['street_count']):3} "
        f"({row['percentage']:.1f}%)"
    )


print()
print("TOP MANUAL REVIEW CANDIDATES")
print("-" * 80)

for _, row in review.head(10).iterrows():

    print(
        f"{row['street']:30} "
        f"{row['candidate_error']:45} "
        f"priority={row['review_priority']}"
    )


print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print()
print(ERROR_FILE)
print(STATS_FILE)
print(REVIEW_FILE)

print()
print("=" * 80)
print("LAYER 2 COMPLETE")
print("=" * 80)
