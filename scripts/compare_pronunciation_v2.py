import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from librosa.sequence import dtw

#---------------------------------------------------
#Load and Clean Audio

#---------------------------------------------------

def load_and_clean(audio_path):

    y, sr = librosa.load(audio_path, sr=16000)

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(
        y,
        top_db=20
    )

    # Normalize volume
    y_trimmed = librosa.util.normalize(y_trimmed)

    return y_trimmed, sr

#---------------------------------------------------
#MFCC Extraction

#---------------------------------------------------

def extract_mfcc(y, sr):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    return mfcc

#---------------------------------------------------
#Pitch Extration

#---------------------------------------------------

def extract_pitch(y, sr):

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=75,
        fmax=500
    )

    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        return 0

    return np.mean(f0)

#---------------------------------------------------
# Energy Extraction
#---------------------------------------------------

def extract_energy(y):

    rms = librosa.feature.rms(y=y)

    return np.mean(rms)

#---------------------------------------------------
#Duration

#---------------------------------------------------

def get_duration(y, sr):

    return librosa.get_duration(
        y=y,
        sr=sr
    )

#---------------------------------------------------
#DTW Comparison

#---------------------------------------------------

def calculate_dtw(system_mfcc, gt_mfcc):

    D, wp = dtw(
        system_mfcc,
        gt_mfcc,
        metric="euclidean"
    )

    normalized_dtw = D[-1,-1] / len(wp)

    return normalized_dtw

#---------------------------------------------------
# PAS Calculation
#---------------------------------------------------

def calculate_pas(
        dtw_score,
        pitch_diff,
        duration_diff,
        energy_diff):

    dtw_component = max(
        0,
        100 - (dtw_score * 5)
    )

    pitch_component = max(
        0,
        100 - (pitch_diff / 5)
    )

    duration_component = max(
        0,
        100 - (duration_diff * 20)
    )

    energy_component = max(
        0,
        100 - (energy_diff * 100)
    )

    pas = (
        dtw_component * 0.4 +
        pitch_component * 0.3 +
        duration_component * 0.2 +
        energy_component * 0.1
    )

    return round(pas, 2)

#---------------------------------------------------
#Waveform Figure

#---------------------------------------------------

def save_waveform(y, sr, filename):

    plt.figure(figsize=(10,4))

    librosa.display.waveshow(
        y,
        sr=sr
    )

    plt.title(filename)

    plt.tight_layout()

    plt.savefig(
        f"figures/{filename}_waveform.png"
    )

    plt.close()

#---------------------------------------------------
#Spectrogram Figure

#---------------------------------------------------
    
def save_spectrogram(y, sr, filename):

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y)),
        ref=np.max
    )

    plt.figure(figsize=(10,4))

    librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar()

    plt.title(
        f"Spectrogram - {filename}"
    )

    plt.tight_layout()

    plt.savefig(
        f"figures/{filename}_spectrogram.png"
    )

    plt.close()

#---------------------------------------------------
#MFCC Figure

#---------------------------------------------------

def save_mfcc(mfcc, filename):

    plt.figure(figsize=(10,4))

    librosa.display.specshow(
        mfcc,
        x_axis="time"
    )

    plt.colorbar()

    plt.title(
        f"MFCC - {filename}"
    )

    plt.tight_layout()

    plt.savefig(
        f"figures/{filename}_mfcc.png"
    )

    plt.close()

#---------------------------------------------------
#Error Classification

#---------------------------------------------------

def classify_error(
        dtw_score,
        pitch_diff,
        duration_diff):

    errors = []

    if dtw_score > 20:
        errors.append(
            "Possible Phoneme Substitution"
        )

    if pitch_diff > 50:
        errors.append(
            "Stress/Tone Error"
        )

    if duration_diff > 1.5:
        errors.append(
            "Possible Agglutination Error"
        )

    if len(errors) == 0:
        errors.append(
            "No Major Error Detected"
        )

    return "; ".join(errors)

#---------------------------------------------------
#Main Analysis

#---------------------------------------------------

system_file = (
    "recordings/wav_system/Mgaga_Road.wav"
)

ground_truth_folder = (
    "recordings/ground_truth/Mgaga_Road"
)

#---------------------------------------------------
#Load System Audio

#---------------------------------------------------

system_y, sr = load_and_clean(
    system_file
)

system_mfcc = extract_mfcc(
    system_y,
    sr
)

system_pitch = extract_pitch(
    system_y,
    sr
)

system_energy = extract_energy(
    system_y
)

system_duration = get_duration(
    system_y,
    sr
)

#---------------------------------------------------
#Generate Figures

#---------------------------------------------------

save_waveform(
    system_y,
    sr,
    "AppleMaps_Mgaga"
)

save_spectrogram(
    system_y,
    sr,
    "AppleMaps_Mgaga"
)

save_mfcc(
    system_mfcc,
    "AppleMaps_Mgaga"
)

#---------------------------------------------------
#Compare Against Contributors

#---------------------------------------------------

results = []

for file in os.listdir(
        ground_truth_folder):

    if not file.endswith(".wav"):
        continue

    gt_path = os.path.join(
        ground_truth_folder,
        file
    )

    gt_y, gt_sr = load_and_clean(
        gt_path
    )

    gt_mfcc = extract_mfcc(
        gt_y,
        gt_sr
    )

    gt_pitch = extract_pitch(
        gt_y,
        gt_sr
    )

    gt_energy = extract_energy(
        gt_y
    )

    gt_duration = get_duration(
        gt_y,
        gt_sr
    )

    dtw_score = calculate_dtw(
        system_mfcc,
        gt_mfcc
    )

    pitch_diff = abs(
        system_pitch - gt_pitch
    )

    duration_diff = abs(
        system_duration - gt_duration
    )

    energy_diff = abs(
        system_energy - gt_energy
    )

    pas = calculate_pas(
        dtw_score,
        pitch_diff,
        duration_diff,
        energy_diff
    )

    error_type = classify_error(
        dtw_score,
        pitch_diff,
        duration_diff
    )

    results.append([
        file,
        round(dtw_score,2),
        round(duration_diff,2),
        round(pitch_diff,2),
        round(energy_diff,4),
        pas,
        error_type
    ])

#---------------------------------------------------
#Save Report

#---------------------------------------------------

columns=[
    "Contributor",
    "DTW",
    "Duration_Diff",
    "Pitch_Diff",
    "Energy_Diff",
    "PAS",
    "Error_Type"
]

df = pd.DataFrame(results, columns=columns)

# Make sure the 'output' directory exists so the file can save properly
os.makedirs("output", exist_ok=True)

df.to_excel(
    "output/Mgaga_Road_Report_V2.xlsx",
    index=False
)

print(df)

print("\nSUMMARY")

print(
    "Average DTW:",
    round(df["DTW"].mean(),2)
)

print(
    "Average Pitch Diff:",
    round(df["Pitch_Diff"].mean(),2)
)

print(
    "Average Duration Diff:",
    round(df["Duration_Diff"].mean(),2)
)

print(
    "Average Energy Diff:",
    round(df["Energy_Diff"].mean(),4)
)

print(
    "Average PAS:",
    round(df["PAS"].mean(),2)
)

print("\nERROR FREQUENCY")

print(
    df["Error_Type"].value_counts()
)

#print("\nAverage PAS")
#print(df["PAS"].mean())
