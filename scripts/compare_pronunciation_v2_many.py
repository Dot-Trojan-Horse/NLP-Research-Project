"""
compare_pronunciation_v2_many.py

Batch pronunciation comparison for isiZulu street names.

Expected structure:
NLP_Projects/
├── recordings/
│   ├── wav_system/
│   │   ├── Mgaga_Road.wav
│   │   └── ...
│   └── ground_truth/
│       ├── Mgaga_Road/
│       │   ├── gt1.wav
│       │   └── ...
│       └── ...
├── output/
└── figures/

Run examples:
    python scripts/compare_pronunciation_v2_many.py
    python scripts/compare_pronunciation_v2_many.py --limit 10
    python scripts/compare_pronunciation_v2_many.py --street Mgaga_Road
    python scripts/compare_pronunciation_v2_many.py --system-dir recordings/wav_system --ground-truth-dir recordings/ground_truth
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from librosa.sequence import dtw


SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.05


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name == "scripts" else THIS_FILE.parent
DEFAULT_SYSTEM_DIR = PROJECT_ROOT / "recordings" / "wav_system"
DEFAULT_GROUND_TRUTH_DIR = PROJECT_ROOT / "recordings" / "ground_truth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures"


# -----------------------------------------------------------------------------
# Audio loading and feature extraction
# -----------------------------------------------------------------------------

def load_and_clean(audio_path: Path) -> Tuple[np.ndarray, int]:
    """Load audio, trim silence, normalize volume, and return mono audio + sample rate."""
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    if y is None or len(y) == 0:
        raise ValueError("Audio file loaded, but it contains no samples.")

    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    if len(y_trimmed) == 0:
        raise ValueError("Audio became empty after silence trimming.")

    y_trimmed = librosa.util.normalize(y_trimmed)

    duration = librosa.get_duration(y=y_trimmed, sr=sr)
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError(f"Audio is too short after trimming: {duration:.3f}s.")

    return y_trimmed, sr


def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)


def extract_pitch(y: np.ndarray, sr: int) -> float:
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=500)
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return 0.0
    return float(np.mean(f0))


def extract_energy(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y)
    return float(np.mean(rms))


def get_duration(y: np.ndarray, sr: int) -> float:
    return float(librosa.get_duration(y=y, sr=sr))


def calculate_dtw(system_mfcc: np.ndarray, gt_mfcc: np.ndarray) -> float:
    D, wp = dtw(system_mfcc, gt_mfcc, metric="euclidean")
    if len(wp) == 0:
        raise ValueError("DTW failed because the warping path is empty.")
    return float(D[-1, -1] / len(wp))


# -----------------------------------------------------------------------------
# Scoring and error classification
# -----------------------------------------------------------------------------

def calculate_pas(
    dtw_score: float,
    pitch_diff: float,
    duration_diff: float,
    energy_diff: float,
) -> float:
    """Pronunciation Accuracy Score. Higher is better."""
    dtw_component = max(0, 100 - (dtw_score * 5))
    pitch_component = max(0, 100 - (pitch_diff / 5))
    duration_component = max(0, 100 - (duration_diff * 20))
    energy_component = max(0, 100 - (energy_diff * 100))

    pas = (
        dtw_component * 0.40
        + pitch_component * 0.30
        + duration_component * 0.20
        + energy_component * 0.10
    )
    return round(float(pas), 2)


def classify_error(dtw_score: float, pitch_diff: float, duration_diff: float) -> str:
    """
    Rule-based error taxonomy.

    These thresholds are starting points. After inspecting about 10 streets manually,
    update the cut-offs so that the automatic labels match what you hear.
    """
    errors: List[str] = []

    if dtw_score > 20:
        errors.append("Possible Phoneme Substitution")

    if pitch_diff > 50:
        errors.append("Stress/Tone Error")

    if duration_diff > 1.5:
        errors.append("Possible Agglutination Error")

    if 10 <= dtw_score <= 20 and duration_diff > 0.75:
        errors.append("Possible Consonant Cluster Error")

    if not errors:
        errors.append("No Major Error Detected")

    return "; ".join(errors)


def classify_quality(pas: Optional[float]) -> str:
    if pas is None or pd.isna(pas):
        return "Unavailable"
    if pas >= 80:
        return "Good"
    if pas >= 60:
        return "Moderate"
    return "Poor"


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def error_row(street: str, error_type: str, details: str, contributor: str = "") -> Dict[str, Any]:
    return {
        "Street": street,
        "Contributor": contributor,
        "Status": "ERROR",
        "Error_Type": error_type,
        "Details": details,
        "DTW": np.nan,
        "Duration_Diff": np.nan,
        "Pitch_Diff": np.nan,
        "Energy_Diff": np.nan,
        "PAS": np.nan,
        "Quality": "Unavailable",
    }


def extract_features(audio_path: Path) -> Dict[str, Any]:
    y, sr = load_and_clean(audio_path)
    return {
        "y": y,
        "sr": sr,
        "mfcc": extract_mfcc(y, sr),
        "pitch": extract_pitch(y, sr),
        "energy": extract_energy(y),
        "duration": get_duration(y, sr),
    }


def analyse_street(street: str, system_dir: Path, ground_truth_dir: Path) -> List[Dict[str, Any]]:
    system_file = system_dir / f"{street}.wav"
    gt_folder = ground_truth_dir / street

    if not system_file.exists():
        return [error_row(street, "Missing system audio", f"No file found at: {system_file}")]

    if not gt_folder.exists():
        return [error_row(street, "Missing ground truth folder", f"No folder found at: {gt_folder}")]

    gt_files = sorted(gt_folder.glob("*.wav"))
    if not gt_files:
        return [error_row(street, "No ground truth recordings", f"No .wav files found in: {gt_folder}")]

    try:
        system_features = extract_features(system_file)
    except Exception as exc:
        return [
            error_row(
                street,
                "System audio processing failed",
                f"{type(exc).__name__}: {exc}",
            )
        ]

    rows: List[Dict[str, Any]] = []

    for gt_file in gt_files:
        try:
            gt_features = extract_features(gt_file)

            dtw_score = calculate_dtw(system_features["mfcc"], gt_features["mfcc"])
            pitch_diff = abs(system_features["pitch"] - gt_features["pitch"])
            duration_diff = abs(system_features["duration"] - gt_features["duration"])
            energy_diff = abs(system_features["energy"] - gt_features["energy"])
            pas = calculate_pas(dtw_score, pitch_diff, duration_diff, energy_diff)
            error_type = classify_error(dtw_score, pitch_diff, duration_diff)

            rows.append(
                {
                    "Street": street,
                    "Contributor": gt_file.name,
                    "Status": "OK",
                    "Error_Type": error_type,
                    "Details": "",
                    "DTW": round(dtw_score, 2),
                    "Duration_Diff": round(duration_diff, 2),
                    "Pitch_Diff": round(pitch_diff, 2),
                    "Energy_Diff": round(energy_diff, 4),
                    "PAS": pas,
                    "Quality": classify_quality(pas),
                }
            )

        except Exception as exc:
            rows.append(
                error_row(
                    street,
                    "Ground truth processing failed",
                    f"{type(exc).__name__}: {exc}",
                    contributor=gt_file.name,
                )
            )

    return rows


def discover_streets(system_dir: Path, ground_truth_dir: Path) -> List[str]:
    """Use the union of system audio stems and ground-truth folder names."""
    system_streets = {p.stem for p in system_dir.glob("*.wav")} if system_dir.exists() else set()
    gt_streets = {p.name for p in ground_truth_dir.iterdir() if p.is_dir()} if ground_truth_dir.exists() else set()
    return sorted(system_streets | gt_streets)


def create_street_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["Status"] == "OK"].copy()

    if ok.empty:
        return pd.DataFrame(
            columns=[
                "Street",
                "N_Comparisons",
                "Mean_DTW",
                "Mean_Duration_Diff",
                "Mean_Pitch_Diff",
                "Mean_Energy_Diff",
                "Mean_PAS",
                "Worst_PAS",
                "Primary_Error_Type",
                "Quality",
            ]
        )

    summary = (
        ok.groupby("Street")
        .agg(
            N_Comparisons=("Contributor", "count"),
            Mean_DTW=("DTW", "mean"),
            Mean_Duration_Diff=("Duration_Diff", "mean"),
            Mean_Pitch_Diff=("Pitch_Diff", "mean"),
            Mean_Energy_Diff=("Energy_Diff", "mean"),
            Mean_PAS=("PAS", "mean"),
            Worst_PAS=("PAS", "min"),
        )
        .reset_index()
    )

    primary_error = (
        ok.groupby("Street")["Error_Type"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index(name="Primary_Error_Type")
    )

    summary = summary.merge(primary_error, on="Street", how="left")
    summary["Quality"] = summary["Mean_PAS"].apply(classify_quality)

    for col in [
        "Mean_DTW",
        "Mean_Duration_Diff",
        "Mean_Pitch_Diff",
        "Mean_Energy_Diff",
        "Mean_PAS",
        "Worst_PAS",
    ]:
        summary[col] = summary[col].round(2)

    return summary.sort_values("Mean_PAS", ascending=True)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    series.plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def create_plots(comparisons_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    ok = comparisons_df[comparisons_df["Status"] == "OK"].copy()

    # 1. Error frequency across all comparisons
    if not ok.empty:
        error_counts = ok["Error_Type"].value_counts().sort_values(ascending=True)
        save_bar(
            error_counts,
            "Error category frequency across all comparisons",
            "Number of comparisons",
            "Error type",
            figures_dir / "01_error_frequency.png",
        )

    # 2. Worst 20 streets by mean PAS
    if not summary_df.empty:
        worst20 = summary_df.nsmallest(20, "Mean_PAS").set_index("Street")["Mean_PAS"].sort_values(ascending=True)
        save_bar(
            worst20,
            "Worst 20 streets by mean pronunciation accuracy score",
            "Mean PAS",
            "Street",
            figures_dir / "02_worst_20_streets_by_pas.png",
        )

    # 3. Distribution of PAS for all streets
    if not summary_df.empty:
        plt.figure(figsize=(10, 6))
        summary_df["Mean_PAS"].dropna().plot(kind="hist", bins=30)
        plt.title("Distribution of mean PAS across streets")
        plt.xlabel("Mean PAS")
        plt.ylabel("Number of streets")
        plt.tight_layout()
        plt.savefig(figures_dir / "03_pas_distribution.png", dpi=200)
        plt.close()

    # 4. DTW vs pitch difference: pronunciation-shape vs tone/stress relationship
    if not ok.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(ok["DTW"], ok["Pitch_Diff"], alpha=0.6)
        plt.title("DTW vs pitch difference across comparisons")
        plt.xlabel("DTW distance")
        plt.ylabel("Pitch difference")
        plt.tight_layout()
        plt.savefig(figures_dir / "04_dtw_vs_pitch_diff.png", dpi=200)
        plt.close()

    # 5. Duration difference distribution
    if not ok.empty:
        plt.figure(figsize=(10, 6))
        ok["Duration_Diff"].dropna().plot(kind="hist", bins=30)
        plt.title("Distribution of duration differences")
        plt.xlabel("Duration difference in seconds")
        plt.ylabel("Number of comparisons")
        plt.tight_layout()
        plt.savefig(figures_dir / "05_duration_difference_distribution.png", dpi=200)
        plt.close()

    # 6. Processing status, useful when scaling to around 1000 streets
    status_counts = comparisons_df["Status"].value_counts().sort_values(ascending=True)
    save_bar(
        status_counts,
        "Processing status across all comparisons",
        "Number of rows",
        "Status",
        figures_dir / "06_processing_status.png",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch pronunciation comparison for street names.")
    parser.add_argument("--system-dir", type=Path, default=DEFAULT_SYSTEM_DIR)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--street", type=str, default=None, help="Analyse one specific street, e.g. Mgaga_Road")
    parser.add_argument("--limit", type=int, default=None, help="Analyse only the first N discovered streets")
    parser.add_argument("--debug", action="store_true", help="Print full stack traces for unexpected errors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    system_dir = args.system_dir if args.system_dir.is_absolute() else PROJECT_ROOT / args.system_dir
    ground_truth_dir = args.ground_truth_dir if args.ground_truth_dir.is_absolute() else PROJECT_ROOT / args.ground_truth_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    figures_dir = args.figures_dir if args.figures_dir.is_absolute() else PROJECT_ROOT / args.figures_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.street:
        streets = [args.street]
    else:
        streets = discover_streets(system_dir, ground_truth_dir)

    if args.limit is not None:
        streets = streets[: args.limit]

    if not streets:
        raise FileNotFoundError(
            f"No streets found. Check system dir '{system_dir}' and ground-truth dir '{ground_truth_dir}'."
        )

    all_rows: List[Dict[str, Any]] = []

    print(f"Project root: {PROJECT_ROOT}")
    print(f"System dir: {system_dir}")
    print(f"Ground-truth dir: {ground_truth_dir}")
    print(f"Analysing {len(streets)} street(s)...\n")

    for i, street in enumerate(streets, start=1):
        print(f"[{i}/{len(streets)}] {street}")
        try:
            all_rows.extend(analyse_street(street, system_dir, ground_truth_dir))
        except Exception as exc:
            if args.debug:
                traceback.print_exc()
            all_rows.append(
                error_row(
                    street,
                    "Unexpected street-level failure",
                    f"{type(exc).__name__}: {exc}",
                )
            )

    comparisons_df = pd.DataFrame(all_rows)
    summary_df = create_street_summary(comparisons_df)

    comparisons_csv = output_dir / "Pronunciation_Comparisons_All_Streets.csv"
    summary_csv = output_dir / "Pronunciation_Summary_By_Street.csv"
    excel_path = output_dir / "Pronunciation_Error_Report_All_Streets.xlsx"

    comparisons_df.to_csv(comparisons_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(excel_path) as writer:
        comparisons_df.to_excel(writer, sheet_name="All Comparisons", index=False)
        summary_df.to_excel(writer, sheet_name="Street Summary", index=False)

    create_plots(comparisons_df, summary_df, figures_dir)

    print("\nDone.")
    print(f"Saved detailed comparisons to: {comparisons_csv}")
    print(f"Saved street summary to: {summary_csv}")
    print(f"Saved Excel report to: {excel_path}")
    print(f"Saved figures to: {figures_dir}")

    print("\nProcessing status:")
    print(comparisons_df["Status"].value_counts(dropna=False))

    if not summary_df.empty:
        print("\nLowest mean PAS streets:")
        print(summary_df[["Street", "Mean_PAS", "Primary_Error_Type", "Quality"]].head(10))


if __name__ == "__main__":
    main()
