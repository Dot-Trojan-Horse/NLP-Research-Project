# =========================================================
# CASE STUDY FIGURES
# Spectrogram Comparisons
# =========================================================

from pathlib import Path
import sys

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# IMPORTS
# =========================================================

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from modules.audio_loader import (
    load_audio,
    load_ground_truth_recordings
)

from modules.feature_extraction import (
    extract_features
)

from modules.dtw_analysis import (
    compute_dtw
)

# =========================================================
# DATASET PATHS
# =========================================================

SYSTEM_FOLDER = Path("recordings/wav_system")

GROUND_TRUTH_FOLDER = Path("recordings/ground_truth")

FIGURES_FOLDER = Path("results/figures")

FIGURES_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# CASE STUDIES
# =========================================================

CASE_STUDIES = [

    "Mabele_Road",

    "Mgaga_Road"

]

# =========================================================
# LOAD CASE STUDY
# Automatically selects the best native recording
# =========================================================

def load_case_study(street_name):

    print()

    print("=" * 60)

    print(f"Loading Case Study : {street_name}")

    print("=" * 60)

    # -----------------------------------------------------
    # Apple Maps recording
    # -----------------------------------------------------

    system_file = SYSTEM_FOLDER / f"{street_name}.wav"

    system_audio, system_sr = load_audio(system_file)

    system_features = extract_features(

        system_audio,

        system_sr

    )

    # -----------------------------------------------------
    # Native recordings
    # -----------------------------------------------------

    ground_truth_recordings = load_ground_truth_recordings(

        GROUND_TRUTH_FOLDER,

        street_name

    )

    best_distance = float("inf")

    best_audio = None

    best_sr = None

    best_name = None

    # -----------------------------------------------------
    # Find lowest DTW
    # -----------------------------------------------------

    for recording_name, gt_audio, gt_sr in ground_truth_recordings:

        gt_features = extract_features(

            gt_audio,

            gt_sr

        )

        dtw_result = compute_dtw(

            system_features["mfcc"],

            gt_features["mfcc"]

        )

        if dtw_result["distance"] < best_distance:

            best_distance = dtw_result["distance"]

            best_audio = gt_audio

            best_sr = gt_sr

            best_name = recording_name

    print()

    print("Best Native Recording")

    print("----------------------")

    print(best_name)

    print(f"DTW Distance : {best_distance:.2f}")

    return {

        "street": street_name,

        "system_audio": system_audio,

        "system_sr": system_sr,

        "native_audio": best_audio,

        "native_sr": best_sr,

        "native_name": best_name,

        "dtw": best_distance

    }
# =========================================================
# PLOT SPECTROGRAM
# =========================================================

def plot_spectrogram(

    ax,

    audio,

    sr,

    title

):

    # -----------------------------------------------------
    # Short-Time Fourier Transform
    # -----------------------------------------------------

    stft = librosa.stft(

        audio,

        n_fft=2048,

        hop_length=512

    )

    # -----------------------------------------------------
    # Convert to Decibels
    # -----------------------------------------------------

    spectrogram = librosa.amplitude_to_db(

        np.abs(stft),

        ref=np.max

    )

    # -----------------------------------------------------
    # Draw Spectrogram
    # -----------------------------------------------------

    img = librosa.display.specshow(

        spectrogram,

        sr=sr,

        hop_length=512,

        x_axis="time",

        y_axis="hz",

        cmap="magma",

        ax=ax

    )

    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    ax.set_title(

        title,

        fontsize=11,

        fontweight="bold"

    )

    ax.set_xlabel(

        "Time (s)",

        fontsize=9

    )

    ax.set_ylabel(

        "Frequency (Hz)",

        fontsize=9

    )

    return img
# =========================================================
# PLOT MEL SPECTROGRAM
# =========================================================

def plot_mel_spectrogram(

    ax,

    audio,

    sr,

    title

):

    # -----------------------------------------------------
    # Mel Spectrogram
    # -----------------------------------------------------

    mel = librosa.feature.melspectrogram(

        y=audio,

        sr=sr,

        n_fft=2048,

        hop_length=512,

        n_mels=128

    )

    # -----------------------------------------------------
    # Convert to dB
    # -----------------------------------------------------

    mel_db = librosa.power_to_db(

        mel,

        ref=np.max

    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    img = librosa.display.specshow(

        mel_db,

        sr=sr,

        hop_length=512,

        x_axis="time",

        y_axis="mel",

        cmap="magma",

        ax=ax

    )

    ax.set_title(

        title,

        fontsize=11,

        fontweight="bold"

    )

    ax.set_xlabel(

        "Time (s)",

        fontsize=9

    )

    ax.set_ylabel(

        "Mel Frequency",

        fontsize=9

    )

    return img
# =========================================================
# FIGURE 8
# FOUR-PANEL SPECTROGRAM COMPARISON
# =========================================================

def create_figure8(mabele, mgaga):

    print()

    print("=" * 60)

    print("Generating Figure 8")

    print("=" * 60)

    # -----------------------------------------------------
    # Create Figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(

        2,

        2,

        figsize=(14,10),

        constrained_layout=True

    )

    # -----------------------------------------------------
    # Poor pronunciation
    # -----------------------------------------------------

    img = plot_spectrogram(

        axes[0,0],

        mabele["system_audio"],

        mabele["system_sr"],

        "(A) Apple Maps\nMabele Road"

    )

    plot_spectrogram(

        axes[0,1],

        mabele["native_audio"],

        mabele["native_sr"],

        "(B) Native Speaker\nMabele Road"

    )

    # -----------------------------------------------------
    # Good pronunciation
    # -----------------------------------------------------

    plot_spectrogram(

        axes[1,0],

        mgaga["system_audio"],

        mgaga["system_sr"],

        "(C) Apple Maps\nMgaga Road"

    )

    plot_spectrogram(

        axes[1,1],

        mgaga["native_audio"],

        mgaga["native_sr"],

        "(D) Native Speaker\nMgaga Road"

    )

    # -----------------------------------------------------
    # Shared Colour Bar
    # -----------------------------------------------------

    cbar = fig.colorbar(

        img,

        ax=axes,

        shrink=0.90,

        pad=0.02

    )

    cbar.set_label(

        "Amplitude (dB)",

        fontsize=10,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Main Title
    # -----------------------------------------------------

    fig.suptitle(

        "Figure 8. Spectrogram Comparison Between Apple Maps and Native Speakers",

        fontsize=15,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    figure_file = FIGURES_FOLDER / "Figure08_Spectrogram_Comparison.png"

    plt.savefig(

        figure_file,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print()

    print("✓ Figure 8 saved")

    print(figure_file)

# =========================================================
# FIGURE 9
# MEL SPECTROGRAM
# =========================================================

def create_figure9(mabele, mgaga):

    print()

    print("=" * 60)

    print("Generating Figure 9")

    print("=" * 60)

    # -----------------------------------------------------
    # Create Figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(

        2,

        2,

        figsize=(14,10),

        constrained_layout=True

    )

    # -----------------------------------------------------
    # Poor pronunciation
    # -----------------------------------------------------

    img = plot_mel_spectrogram(

        axes[0,0],

        mabele["system_audio"],

        mabele["system_sr"],

        "(A) Apple Maps\nMabele Road"

    )

    plot_mel_spectrogram(

        axes[0,1],

        mabele["native_audio"],

        mabele["native_sr"],

        "(B) Native Speaker\nMabele Road"

    )

    # -----------------------------------------------------
    # Good pronunciation
    # -----------------------------------------------------

    plot_mel_spectrogram(

        axes[1,0],

        mgaga["system_audio"],

        mgaga["system_sr"],

        "(C) Apple Maps\nMgaga Road"

    )

    plot_mel_spectrogram(

        axes[1,1],

        mgaga["native_audio"],

        mgaga["native_sr"],

        "(D) Native Speaker\nMgaga Road"

    )

    # -----------------------------------------------------
    # Shared Colour Bar
    # -----------------------------------------------------

    cbar = fig.colorbar(

        img,

        ax=axes,

        shrink=0.90,

        pad=0.02

    )

    cbar.set_label(

        "Amplitude (dB)",

        fontsize=10,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Main Title
    # -----------------------------------------------------

    fig.suptitle(

        "Figure 9. Mel Spectrogram Comparison Between Apple Maps and Native Speakers",

        fontsize=15,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    figure_file = FIGURES_FOLDER / "Figure09_MelSpectrogram_Comparison.png"

    plt.savefig(

        figure_file,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print()

    print("✓ Figure 9 saved")

    print(figure_file)

# =========================================================
# PLOT MFCC HEATMAP
# =========================================================

def plot_mfcc(

    ax,

    audio,

    sr,

    title

):

    # -----------------------------------------------------
    # Compute MFCCs
    # -----------------------------------------------------

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sr,

        n_mfcc=13

    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    img = librosa.display.specshow(

        mfcc,

        x_axis="time",

        cmap="viridis",

        ax=ax

    )

    ax.set_title(

        title,

        fontsize=11,

        fontweight="bold"

    )

    ax.set_xlabel(

        "Time (s)",

        fontsize=9

    )

    ax.set_ylabel(

        "MFCC Coefficient",

        fontsize=9

    )

    return img

# =========================================================
# FIGURE 10
# MFCC HEATMAP
# =========================================================

def create_figure10(mabele, mgaga):

    print()

    print("=" * 60)

    print("Generating Figure 10")

    print("=" * 60)

    # -----------------------------------------------------
    # Create Figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(

        2,

        2,

        figsize=(14,10),

        constrained_layout=True

    )

    # -----------------------------------------------------
    # Poor pronunciation
    # -----------------------------------------------------

    img = plot_mfcc(

        axes[0,0],

        mabele["system_audio"],

        mabele["system_sr"],

        "(A) Apple Maps\nMabele Road"

    )

    plot_mfcc(

        axes[0,1],

        mabele["native_audio"],

        mabele["native_sr"],

        "(B) Native Speaker\nMabele Road"

    )

    # -----------------------------------------------------
    # Good pronunciation
    # -----------------------------------------------------

    plot_mfcc(

        axes[1,0],

        mgaga["system_audio"],

        mgaga["system_sr"],

        "(C) Apple Maps\nMgaga Road"

    )

    plot_mfcc(

        axes[1,1],

        mgaga["native_audio"],

        mgaga["native_sr"],

        "(D) Native Speaker\nMgaga Road"

    )

    # -----------------------------------------------------
    # Shared Colour Bar
    # -----------------------------------------------------

    cbar = fig.colorbar(

        img,

        ax=axes,

        shrink=0.90,

        pad=0.02

    )

    cbar.set_label(

        "Amplitude (dB)",

        fontsize=10,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Main Title
    # -----------------------------------------------------

    fig.suptitle(

        "Figure 10. MFCC Heatmap Comparison Between Apple Maps and Native Speakers",

        fontsize=15,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    figure_file = FIGURES_FOLDER / "Figure10_MFCC_Comparison.png"

    plt.savefig(

        figure_file,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print()

    print("✓ Figure 10 saved")

    print(figure_file)

# =========================================================
# PLOT DTW ALIGNMENT
# =========================================================

def plot_dtw_alignment(

    ax,

    audio1,

    sr1,

    audio2,

    sr2,

    title

):

    # ----------------------------------------
    # MFCC Features
    # ----------------------------------------

    mfcc1 = librosa.feature.mfcc(

        y=audio1,

        sr=sr1,

        n_mfcc=13

    )

    mfcc2 = librosa.feature.mfcc(

        y=audio2,

        sr=sr2,

        n_mfcc=13

    )

    # ----------------------------------------
    # Cost Matrix
    # ----------------------------------------

    from librosa.sequence import dtw

    D, wp = dtw(

        X=mfcc1,

        Y=mfcc2,

        metric="euclidean"

    )

    wp = np.array(wp)

    # ----------------------------------------
    # Plot Cost Matrix
    # ----------------------------------------

    img = ax.imshow(

        D,

        origin="lower",

        aspect="auto",

        cmap="viridis"

    )

    # ----------------------------------------
    # DTW Path
    # ----------------------------------------

    ax.plot(

        wp[:,1],

        wp[:,0],

        color="red",

        linewidth=2

    )

    ax.set_title(

        title,

        fontsize=11,

        fontweight="bold"

    )

    ax.set_xlabel(

        "Native Frames",

        fontsize=9

    )

    ax.set_ylabel(

        "Apple Maps Frames",

        fontsize=9

    )

    return img

# =========================================================
# FIGURE 11
# DTW ALIGNMENT
# =========================================================

def create_figure11(mabele, mgaga):

    print()

    print("=" * 60)

    print("Generating Figure 11")

    print("=" * 60)

    # -----------------------------------------------------
    # Create Figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(

        2,

        2,

        figsize=(14,10),

        constrained_layout=True

    )

    # -----------------------------------------------------
    # Poor pronunciation
    # -----------------------------------------------------

    img = plot_dtw_alignment(

    axes[0,0],

    mabele["system_audio"],

    mabele["system_sr"],

    mabele["native_audio"],

    mabele["native_sr"],

    "(A) Apple Maps\nMabele Road"

)

    plot_dtw_alignment(

        axes[0,1],

         mabele["system_audio"],

        mabele["system_sr"],

        mabele["native_audio"],

        mabele["native_sr"],

        "(B) Native Speaker\nMabele Road"

    )

    # -----------------------------------------------------
    # Good pronunciation
    # -----------------------------------------------------

    plot_dtw_alignment(

        axes[1,0],

         mabele["system_audio"],

        mabele["system_sr"],

        mabele["native_audio"],

        mabele["native_sr"],

        "(C) Apple Maps\nMgaga Road"

    )

    plot_dtw_alignment(

        axes[1,1],

         mabele["system_audio"],

        mabele["system_sr"],

        mabele["native_audio"],

        mabele["native_sr"],

        "(D) Native Speaker\nMgaga Road"

    )

    # -----------------------------------------------------
    # Shared Colour Bar
    # -----------------------------------------------------

    cbar = fig.colorbar(

        img,

        ax=axes,

        shrink=0.90,

        pad=0.02

    )

    cbar.set_label(

        "Amplitude (dB)",

        fontsize=10,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Main Title
    # -----------------------------------------------------

    fig.suptitle(

        "Figure 11. Dynamic Time Warping (DTW) Alignment Between Apple Maps and Native Speaker Pronunciations",

        fontsize=15,

        fontweight="bold"

    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    figure_file = FIGURES_FOLDER / "Figure11_DTW_Alignment.png"

    plt.savefig(

        figure_file,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print()

    print("✓ Figure 11 saved")

    print(figure_file)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # ----------------------------------------
    # Load case studies
    # ----------------------------------------

    mabele = load_case_study(
        "Mabele_Road"
    )

    mgaga = load_case_study(
        "Mgaga_Road"
    )

    # ----------------------------------------
    # Generate Figure 8
    # ----------------------------------------

    create_figure8(
        mabele,
        mgaga
    )
    # ----------------------------------------
    # Generate Figure 9
    # ----------------------------------------

    create_figure9(
        mabele,
        mgaga
    )
    # ----------------------------------------
    # Generate Figure 10
    # ----------------------------------------

    create_figure10(
        mabele,
        mgaga
    )
      # ----------------------------------------
    # Generate Figure 11
    # ----------------------------------------

    create_figure11(
        mabele,
        mgaga
    )

