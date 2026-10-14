from pathlib import Path
import sys
import csv
import json
import math
from collections import Counter


# =============================================================================
# MABELE ALLOSAURUS PHONE RECOGNITION + PER
# =============================================================================
#
# PURPOSE
# -------
# Experimental independent phone recognition for Mabele Road.
#
# Native recordings and the Apple Maps recording are passed through the SAME
# phone recogniser. The resulting phone sequences can then be compared using
# Levenshtein distance and Phone Error Rate (PER).
#
# IMPORTANT
# ---------
# This script does NOT assume that Allosaurus output is ground truth.
# The recognised sequences must be inspected before the PER result is used
# as final evidence in the research report.
#
# First run:
#   The script checks whether Allosaurus is installed.
#
# If it is not installed, the script stops and prints the installation command.
#
# After installation:
#   Run this same script again.
#
# =============================================================================


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\Sam\Documents\NLP_Projects")

RESULTS_ROOT = PROJECT_ROOT / "results"

# Existing Mabele analysis location.
MABELE_RESULTS_DIR = RESULTS_ROOT / "mabele_alignment"

# New output directory for this experiment.
OUTPUT_DIR = RESULTS_ROOT / "mabele_allosaurus_per"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# POSSIBLE AUDIO LOCATIONS
# =============================================================================

# We deliberately search several likely locations because the existing project
# has been built through multiple analysis layers.

SEARCH_ROOTS = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "datasets",
    PROJECT_ROOT / "recordings",
    PROJECT_ROOT / "audio",
    PROJECT_ROOT / "native_recordings",
    PROJECT_ROOT / "system_recordings",
    PROJECT_ROOT / "results",
]


# =============================================================================
# CONSOLE HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def clean_phone_sequence(sequence):
    """
    Convert recogniser output into a simple list of phone tokens.

    Allosaurus normally returns phone tokens separated by spaces.
    Timestamped output contains additional columns, so this function is
    deliberately conservative.
    """

    if sequence is None:
        return []

    text = str(sequence).strip()

    if not text:
        return []

    phones = []

    for token in text.split():

        token = token.strip()

        if not token:
            continue

        # Skip obvious timestamp fields.
        try:
            float(token)
            continue
        except ValueError:
            pass

        # Skip obvious punctuation-only fields.
        if token in {"|", ",", ";"}:
            continue

        phones.append(token)

    return phones


def sequence_to_string(sequence):
    return " ".join(sequence)


# =============================================================================
# ALLOSAURUS IMPORT CHECK
# =============================================================================

section("CHECKING ALLOSAURUS INSTALLATION")

try:
    from allosaurus.app import read_recognizer
except ImportError:
    print()
    print("ALLOSAURUS IS NOT INSTALLED IN THE CURRENT ENVIRONMENT.")
    print()
    print("You are currently using the nlp_env environment.")
    print()
    print("Run this command in PowerShell:")
    print()
    print(
        r"C:\Users\Sam\anaconda3\envs\nlp_env\python.exe -m pip install allosaurus"
    )
    print()
    print("Then verify the installation with:")
    print()
    print(
        r"C:\Users\Sam\anaconda3\envs\nlp_env\python.exe -c "
        r"\"from allosaurus.app import read_recognizer; print('Allosaurus OK')\""
    )
    print()
    print("After that, run this script again.")
    print()
    sys.exit(0)


print("Allosaurus import: OK")


# =============================================================================
# AUDIO FILE DISCOVERY
# =============================================================================

def find_wav_files():
    """
    Find WAV files throughout the project.

    Returns a list of unique paths.
    """

    found = set()

    for root in SEARCH_ROOTS:

        if not root.exists():
            continue

        try:
            for wav in root.rglob("*.wav"):
                found.add(wav.resolve())
        except (PermissionError, OSError):
            continue

    return sorted(found)


section("SEARCHING FOR MABELE AUDIO")

all_wavs = find_wav_files()

print(f"WAV files discovered: {len(all_wavs)}")


# =============================================================================
# IDENTIFY MABELE RECORDINGS
# =============================================================================

mabele_wavs = []

for wav in all_wavs:

    name = wav.name.lower()
    parent_text = str(wav.parent).lower()

    if "mabele" in name or "mabele" in parent_text:
        mabele_wavs.append(wav)


print(f"Mabele-related WAV files: {len(mabele_wavs)}")

for wav in mabele_wavs:
    print(f"  {wav}")


# =============================================================================
# CLASSIFY NATIVE VS APPLE RECORDING
# =============================================================================

def classify_audio(path):
    """
    Classify a Mabele WAV using filename/path information.

    This is intentionally conservative.

    Native recordings are identified using common native/reference naming
    patterns. Apple/system recordings are identified using Apple/system
    naming patterns.

    Anything uncertain is returned as 'unknown'.
    """

    text = str(path).lower()
    name = path.name.lower()

    apple_terms = [
        "apple",
        "system",
        "maps",
        "iphone",
        "ios",
        "mabele_road.wav",
        "mabele.wav",
    ]

    native_terms = [
        "native",
        "speaker",
        "participant",
        "reference",
        "ground_truth",
        "gt",
    ]

    if any(term in text for term in native_terms):
        return "native"

    if any(term in text for term in apple_terms):
        return "apple"

    return "unknown"


classified = []

for wav in mabele_wavs:

    category = classify_audio(wav)

    classified.append(
        {
            "path": wav,
            "category": category,
        }
    )


native_files = [
    item["path"]
    for item in classified
    if item["category"] == "native"
]

apple_files = [
    item["path"]
    for item in classified
    if item["category"] == "apple"
]


section("MABELE AUDIO CLASSIFICATION")

print(f"Native candidate files: {len(native_files)}")

for wav in native_files:
    print(f"  NATIVE: {wav}")

print()

print(f"Apple candidate files: {len(apple_files)}")

for wav in apple_files:
    print(f"  APPLE:  {wav}")


# =============================================================================
# SAFETY CHECK
# =============================================================================

if len(native_files) == 0:
    print()
    print("WARNING:")
    print("No native Mabele WAV files were automatically identified.")
    print()
    print(
        "The script will not guess which recordings are native recordings."
    )
    print()
    print(
        "Please check the paths/naming in the project before continuing."
    )

if len(apple_files) == 0:
    print()
    print("WARNING:")
    print("No Apple Mabele WAV file was automatically identified.")
    print()
    print(
        "The script will not guess which recording is the Apple recording."
    )


if len(native_files) == 0 or len(apple_files) == 0:

    section("AUDIO IDENTIFICATION INCOMPLETE")

    print(
        "No PER calculation has been performed because the required audio "
        "groups could not be identified safely."
    )

    print()
    print(
        "This is intentional. We must not accidentally compare the wrong "
        "recordings."
    )

    sys.exit(0)


# =============================================================================
# LOAD ALLOSAURUS
# =============================================================================

section("LOADING ALLOSAURUS")

try:
    recognizer = read_recognizer()
except Exception as exc:

    print()
    print("Allosaurus could not initialise.")
    print()
    print(f"Error: {exc}")
    print()
    print(
        "The package may be installed but its model may need to be "
        "downloaded/initialised."
    )

    sys.exit(1)


print("Allosaurus recogniser loaded successfully.")


# =============================================================================
# RECOGNITION
# =============================================================================

def recognise_file(wav_path):
    """
    Run Allosaurus on one WAV.

    We first request timestamp output because this gives us more information
    for inspection.

    If timestamp output fails, we fall back to ordinary phone recognition.
    """

    result = None
    timestamped = False

    # ------------------------------------------------------------
    # Attempt timestamped recognition
    # ------------------------------------------------------------

    try:

        result = recognizer.recognize(
            str(wav_path),
            lang_id="ipa",
            timestamp=True,
        )

        timestamped = True

    except Exception as first_error:

        print(
            f"  Timestamp recognition failed for {wav_path.name}: "
            f"{first_error}"
        )

        try:

            result = recognizer.recognize(
                str(wav_path),
                lang_id="ipa",
            )

            timestamped = False

        except Exception as second_error:

            return {
                "success": False,
                "raw_output": "",
                "phones": [],
                "timestamped": False,
                "error": str(second_error),
            }

    phones = clean_phone_sequence(result)

    return {
        "success": True,
        "raw_output": str(result),
        "phones": phones,
        "timestamped": timestamped,
        "error": "",
    }


# =============================================================================
# RUN RECOGNITION
# =============================================================================

section("RUNNING PHONE RECOGNITION")

recognition_results = []


for wav in native_files + apple_files:

    category = classify_audio(wav)

    print()
    print(f"Processing: {wav.name}")
    print(f"Category:   {category}")

    result = recognise_file(wav)

    recognition_results.append(
        {
            "file": str(wav),
            "filename": wav.name,
            "category": category,
            "success": result["success"],
            "timestamped": result["timestamped"],
            "phones": sequence_to_string(result["phones"]),
            "phone_count": len(result["phones"]),
            "raw_output": result["raw_output"],
            "error": result["error"],
        }
    )

    if result["success"]:

        print(
            "Phones:"
        )

        print(
            sequence_to_string(result["phones"])
        )

        print(
            f"Phone count: {len(result['phones'])}"
        )

    else:

        print(
            f"FAILED: {result['error']}"
        )


# =============================================================================
# SAVE RAW RECOGNITION RESULTS
# =============================================================================

raw_output_file = OUTPUT_DIR / "mabele_allosaurus_raw_recognition.csv"

with raw_output_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "filename",
            "category",
            "success",
            "timestamped",
            "phones",
            "phone_count",
            "raw_output",
            "error",
        ],
    )

    writer.writeheader()

    writer.writerows(recognition_results)


print()
print("Raw recognition results saved:")
print(f"  {raw_output_file}")


# =============================================================================
# SAVE PHONE SEQUENCES AS TXT
# =============================================================================

sequence_file = OUTPUT_DIR / "mabele_allosaurus_phone_sequences.txt"

with sequence_file.open(
    "w",
    encoding="utf-8",
) as f:

    for row in recognition_results:

        f.write("=" * 80 + "\n")
        f.write(f"FILE: {row['filename']}\n")
        f.write(f"CATEGORY: {row['category']}\n")
        f.write(f"SUCCESS: {row['success']}\n")
        f.write("\n")

        f.write("PHONE SEQUENCE:\n")
        f.write(row["phones"])
        f.write("\n\n")

        f.write("RAW ALLOSAURUS OUTPUT:\n")
        f.write(row["raw_output"])
        f.write("\n\n")


print("Phone sequences saved:")
print(f"  {sequence_file}")


# =============================================================================
# GROUP NATIVE RECORDINGS
# =============================================================================

native_results = [
    row
    for row in recognition_results
    if row["category"] == "native"
    and row["success"]
]

apple_results = [
    row
    for row in recognition_results
    if row["category"] == "apple"
    and row["success"]
]


# =============================================================================
# CREATE NATIVE CONSENSUS
# =============================================================================

def most_common_phone_sequence(results):
    """
    Select the most common recognised phone sequence.

    This is NOT a phonological ground truth.

    It is only a provisional recogniser-derived native reference for this
    experimental PER calculation.
    """

    if not results:
        return []

    sequences = [
        tuple(row["phones"].split())
        for row in results
        if row["phones"].strip()
    ]

    if not sequences:
        return []

    counts = Counter(sequences)

    return list(counts.most_common(1)[0][0])


native_consensus = most_common_phone_sequence(native_results)


# =============================================================================
# LEVENSHTEIN IMPLEMENTATION
# =============================================================================

def levenshtein_alignment(reference, hypothesis):
    """
    Calculate Levenshtein edit distance and the corresponding operations.

    Operations:
        C = correct
        S = substitution
        D = deletion
        I = insertion
    """

    n = len(reference)
    m = len(hypothesis)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    operations = [
        [""] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n + 1):

        dp[i][0] = i

        if i > 0:
            operations[i][0] = "D"

    for j in range(m + 1):

        dp[0][j] = j

        if j > 0:
            operations[0][j] = "I"

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if reference[i - 1] == hypothesis[j - 1]:

                cost = dp[i - 1][j - 1]

                dp[i][j] = cost

                operations[i][j] = "C"

            else:

                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1

                best = min(
                    substitution,
                    deletion,
                    insertion,
                )

                dp[i][j] = best

                if best == substitution:
                    operations[i][j] = "S"

                elif best == deletion:
                    operations[i][j] = "D"

                else:
                    operations[i][j] = "I"

    # ------------------------------------------------------------
    # Backtrace
    # ------------------------------------------------------------

    i = n
    j = m

    aligned_reference = []
    aligned_hypothesis = []
    aligned_operations = []

    while i > 0 or j > 0:

        operation = operations[i][j]

        if operation == "C" or operation == "S":

            aligned_reference.append(reference[i - 1])
            aligned_hypothesis.append(hypothesis[j - 1])
            aligned_operations.append(operation)

            i -= 1
            j -= 1

        elif operation == "D":

            aligned_reference.append(reference[i - 1])
            aligned_hypothesis.append("<DEL>")
            aligned_operations.append("D")

            i -= 1

        elif operation == "I":

            aligned_reference.append("<INS>")
            aligned_hypothesis.append(hypothesis[j - 1])
            aligned_operations.append("I")

            j -= 1

        else:
            break

    aligned_reference.reverse()
    aligned_hypothesis.reverse()
    aligned_operations.reverse()

    substitutions = aligned_operations.count("S")
    deletions = aligned_operations.count("D")
    insertions = aligned_operations.count("I")
    correct = aligned_operations.count("C")

    distance = substitutions + deletions + insertions

    return {
        "distance": distance,
        "correct": correct,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "aligned_reference": aligned_reference,
        "aligned_hypothesis": aligned_hypothesis,
        "aligned_operations": aligned_operations,
    }


# =============================================================================
# PER CALCULATION
# =============================================================================

section("CALCULATING EXPERIMENTAL PER")

if not native_consensus:

    print()
    print("No usable native phone consensus was produced.")

    per_results = []

else:

    print()
    print("PROVISIONAL NATIVE REFERENCE:")
    print(sequence_to_string(native_consensus))

    print()
    print(
        "IMPORTANT: this reference is recogniser-derived and is NOT "
        "human-validated phoneme ground truth."
    )

    per_results = []

    for row in apple_results:

        hypothesis = row["phones"].split()

        alignment = levenshtein_alignment(
            native_consensus,
            hypothesis,
        )

        reference_count = len(native_consensus)

        if reference_count > 0:

            per = (
                alignment["distance"]
                / reference_count
            )

        else:

            per = math.nan

        per_results.append(
            {
                "reference_file": "native_consensus",
                "apple_file": row["filename"],
                "reference_phone_count": reference_count,
                "apple_phone_count": len(hypothesis),
                "correct": alignment["correct"],
                "substitutions": alignment["substitutions"],
                "deletions": alignment["deletions"],
                "insertions": alignment["insertions"],
                "levenshtein_distance": alignment["distance"],
                "PER": per,
                "PER_percent": per * 100
                if not math.isnan(per)
                else math.nan,
            }
        )


# =============================================================================
# SAVE PER RESULTS
# =============================================================================

per_output_file = OUTPUT_DIR / "mabele_allosaurus_per_results.csv"

with per_output_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "reference_file",
        "apple_file",
        "reference_phone_count",
        "apple_phone_count",
        "correct",
        "substitutions",
        "deletions",
        "insertions",
        "levenshtein_distance",
        "PER",
        "PER_percent",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(per_results)


print()
print("PER results saved:")
print(f"  {per_output_file}")


# =============================================================================
# SAVE ALIGNMENT
# =============================================================================

alignment_output_file = OUTPUT_DIR / "mabele_allosaurus_phone_alignment.csv"

with alignment_output_file.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "reference_phone",
            "apple_phone",
            "operation",
        ],
    )

    writer.writeheader()

    if native_consensus and apple_results:

        hypothesis = apple_results[0]["phones"].split()

        alignment = levenshtein_alignment(
            native_consensus,
            hypothesis,
        )

        for ref_phone, hyp_phone, operation in zip(
            alignment["aligned_reference"],
            alignment["aligned_hypothesis"],
            alignment["aligned_operations"],
        ):

            writer.writerow(
                {
                    "reference_phone": ref_phone,
                    "apple_phone": hyp_phone,
                    "operation": operation,
                }
            )


print()
print("Alignment saved:")
print(f"  {alignment_output_file}")


# =============================================================================
# SAVE METADATA / METHODOLOGICAL WARNING
# =============================================================================

metadata = {
    "experiment": "Mabele Allosaurus phone recognition and experimental PER",
    "language": "isiZulu",
    "street": "Mabele Road",
    "recogniser": "Allosaurus",
    "recogniser_output": "phone sequence",
    "native_recordings": len(native_results),
    "apple_recordings": len(apple_results),
    "native_reference_type": (
        "provisional recogniser-derived consensus"
    ),
    "ground_truth_status": (
        "NOT human-validated phoneme transcription"
    ),
    "per_status": (
        "experimental until recogniser output is manually validated"
    ),
    "wer_status": (
        "not calculated because the experiment concerns pronunciation "
        "phone sequences rather than word transcription"
    ),
    "levenshtein_status": (
        "calculated between provisional native phone sequence and "
        "Apple phone sequence when valid outputs are available"
    ),
}


metadata_file = OUTPUT_DIR / "mabele_allosaurus_per_metadata.json"

with metadata_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False,
    )


# =============================================================================
# FINAL CONSOLE REPORT
# =============================================================================

section("ALLOSAURUS EXPERIMENT COMPLETE")

print(f"Native successful recognitions: {len(native_results)}")
print(f"Apple successful recognitions:  {len(apple_results)}")

print()

print("OUTPUT FILES")
print("-" * 80)

print(f"Raw recognition:")
print(f"  {raw_output_file}")

print()

print(f"Phone sequences:")
print(f"  {sequence_file}")

print()

print(f"PER:")
print(f"  {per_output_file}")

print()

print(f"Phone alignment:")
print(f"  {alignment_output_file}")

print()

print(f"Metadata:")
print(f"  {metadata_file}")

print()

if per_results:

    print("EXPERIMENTAL PER RESULTS")
    print("-" * 80)

    for result in per_results:

        print(
            f"Apple file: {result['apple_file']}"
        )

        print(
            f"  Levenshtein distance: "
            f"{result['levenshtein_distance']}"
        )

        print(
            f"  Substitutions: "
            f"{result['substitutions']}"
        )

        print(
            f"  Deletions: "
            f"{result['deletions']}"
        )

        print(
            f"  Insertions: "
            f"{result['insertions']}"
        )

        print(
            f"  PER: "
            f"{result['PER_percent']:.2f}%"
        )

        print()

else:

    print(
        "No PER value was produced."
    )

print("=" * 80)

print(
    "IMPORTANT: Do NOT place the PER result into the final research "
    "paper yet."
)

print(
    "First inspect the Allosaurus phone sequences and determine whether "
    "the recognised isiZulu phones are linguistically plausible."
)

print(
    "If the output is unreliable, we will report PER as experimental/"
    "not validated rather than presenting a misleading numerical result."
)

print("=" * 80)