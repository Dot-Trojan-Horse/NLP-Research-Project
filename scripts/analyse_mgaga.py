from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------

project = Path(__file__).resolve().parent.parent

system_audio = project / "recordings" / "wav_system" / "Mgaga_Road.wav"

ground_truth = project / "recordings" / "ground_truth" / "Mgaga_Road"

figures = project / "figures"

reports = project / "reports"

figures.mkdir(exist_ok=True)

reports.mkdir(exist_ok=True)

# ------------------------------------------------
# LOAD SYSTEM AUDIO
# ------------------------------------------------

system_y, sr = librosa.load(system_audio, sr=16000)

system_mfcc = librosa.feature.mfcc(
    y=system_y,
    sr=sr,
    n_mfcc=13
)

system_pitch = librosa.yin(
    system_y,
    fmin=50,
    fmax=500
)

results = []

# ------------------------------------------------
# LOOP THROUGH CONTRIBUTORS
# ------------------------------------------------

for wav in ground_truth.glob("*.wav"):

    gt_y, sr = librosa.load(wav, sr=16000)

    gt_mfcc = librosa.feature.mfcc(
        y=gt_y,
        sr=sr,
        n_mfcc=13
    )

    # DTW

    D, wp = librosa.sequence.dtw(
        X=system_mfcc,
        Y=gt_mfcc,
        metric="cosine"
    )

    dtw_distance = D[-1, -1]

    duration_system = librosa.get_duration(
        y=system_y,
        sr=sr
    )

    duration_gt = librosa.get_duration(
        y=gt_y,
        sr=sr
    )

    duration_difference = abs(
        duration_system-duration_gt
    )

    pitch_gt = librosa.yin(
        gt_y,
        fmin=50,
        fmax=500
    )

    pitch_difference = abs(

        np.nanmean(system_pitch)

        -

        np.nanmean(pitch_gt)

    )

    # Pronunciation Accuracy Score

    PAS = max(

        0,

        100

        -

        dtw_distance

        -

        duration_difference*20

        -

        pitch_difference/5

    )

    results.append({

        "Contributor": wav.name,

        "DTW": round(dtw_distance,2),

        "Duration_Difference": round(duration_difference,2),

        "Pitch_Difference": round(pitch_difference,2),

        "PAS": round(PAS,2)

    })

# ------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------

df = pd.DataFrame(results)

df.to_excel(

    reports/"Mgaga_Road_Report.xlsx",

    index=False

)

print(df)

print()

print("Average PAS")

print(df["PAS"].mean())

# ------------------------------------------------
# WAVEFORM
# ------------------------------------------------

plt.figure(figsize=(12,4))

librosa.display.waveshow(

    system_y,

    sr=sr

)

plt.title("Apple Maps : Mgaga Road")

plt.tight_layout()

plt.savefig(

    figures/"Apple_Waveform.png"

)

plt.close()

# ------------------------------------------------
# MFCC

# ------------------------------------------------

plt.figure(figsize=(10,4))

librosa.display.specshow(

    system_mfcc,

    x_axis="time"

)

plt.colorbar()

plt.title("MFCC")

plt.tight_layout()

plt.savefig(

    figures/"Apple_MFCC.png"

)

plt.close()

print()

print("Analysis Complete")