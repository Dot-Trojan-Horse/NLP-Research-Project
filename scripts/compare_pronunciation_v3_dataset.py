from pathlib import Path
import sys
import pandas as pd

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# IMPORT MODULES
# =========================================================

from modules.audio_loader import (
    load_audio,
    load_ground_truth_recordings,
    find_ground_truth_recordings
)

from modules.feature_extraction import (
    extract_features
)

from modules.dtw_analysis import (
    compute_dtw
)

# =========================================================
# PATHS
# =========================================================

SYSTEM_FOLDER = Path("recordings/wav_system")
GROUND_TRUTH_FOLDER = Path("recordings/ground_truth")
RESULTS_FOLDER = Path("results")

RESULTS_FOLDER.mkdir(exist_ok=True)

MAX_STREETS = 20

# =========================================================
# GET SYSTEM RECORDINGS
# =========================================================

system_recordings = sorted(
    SYSTEM_FOLDER.glob("*.wav")
)

system_recordings = system_recordings[:MAX_STREETS]

# =========================================================
# START
# =========================================================

print()
print("="*70)
print("PRONUNCIATION COMPARISON")
print("="*70)

master_results = []

# =========================================================
# PROCESS EACH STREET
# =========================================================

for system_file in system_recordings:

    street_name = system_file.stem.replace(" ", "_")

    try:

        find_ground_truth_recordings(
            GROUND_TRUTH_FOLDER,
            street_name
        )

    except FileNotFoundError:

        print(f"\n{street_name} -> No ground truth. Skipping.")

        continue

    print()
    print("="*70)
    print(f"Processing : {street_name}")
    print("="*70)

    # -----------------------------------------------------

    print("Loading system recording...")

    system_audio, system_sr = load_audio(system_file)

    print("Done.")

    # -----------------------------------------------------

    print("Extracting system features...")

    system_features = extract_features(
        system_audio,
        system_sr
    )

    print("Done.")

    # -----------------------------------------------------

    print("Loading ground truth recordings...")

    ground_truth_recordings = load_ground_truth_recordings(
        GROUND_TRUTH_FOLDER,
        street_name
    )

    print(f"Loaded {len(ground_truth_recordings)} recordings.")

    # -----------------------------------------------------

    print("Comparing pronunciations...")

    street_results = []

    for recording_name, gt_audio, gt_sr in ground_truth_recordings:

        gt_features = extract_features(
            gt_audio,
            gt_sr
        )

        dtw = compute_dtw(
            system_features["mfcc"],
            gt_features["mfcc"]
        )

        street_results.append({

            "recording": recording_name,

            "dtw": dtw,

            "gt_features": gt_features

        })

    print("Done.")

    # -----------------------------------------------------

    print()
    print("-"*60)

    for result in street_results:

        print(

            f"{result['recording']:<10}"

            f" DTW: {result['dtw']['distance']:.2f}"

            f" | Similarity: {result['dtw']['similarity']:.4f}"

        )

        gt = result["gt_features"]

        master_results.append({

            "Street": street_name,

            "Ground Truth": result["recording"],

            "DTW Distance": result["dtw"]["distance"],

            "Normalized DTW": result["dtw"]["normalized_distance"],

            "Similarity": result["dtw"]["similarity"],

            "System Pitch": system_features["pitch"],

            "Ground Truth Pitch": gt["pitch"],

            "System Duration": system_features["duration"],

            "Ground Truth Duration": gt["duration"],

            "System Energy": system_features["energy"],

            "Ground Truth Energy": gt["energy"]

        })

# =========================================================
# SAVE MASTER CSV
# =========================================================

master_df = pd.DataFrame(master_results)

master_csv = RESULTS_FOLDER / "master_results.csv"

master_df.to_csv(
    master_csv,
    index=False
)


print()
print("="*70)
print("PROCESS COMPLETE")
print("="*70)

print()

print(f"Compared streets : {master_df['Street'].nunique()}")

print(f"Total comparisons : {len(master_df)}")

print()

print(f"Results saved to:\n{master_csv}")

