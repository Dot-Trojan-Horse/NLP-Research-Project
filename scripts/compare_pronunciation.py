import librosa
import numpy as np
from pathlib import Path

project = Path(__file__).resolve().parent.parent

system_file = project / "recordings" / "wav_system" / "Mgaga_Road.wav"

ground_truth_folder = project / "recordings" / "ground_truth" / "Mgaga_Road"

# ------------------------
# Load system recording
# ------------------------

system_audio, sr = librosa.load(system_file, sr=16000)

system_mfcc = librosa.feature.mfcc(
    y=system_audio,
    sr=sr,
    n_mfcc=13
)

print()

print("SYSTEM DURATION")

print(librosa.get_duration(y=system_audio, sr=sr))

print()

results = []

# ------------------------
# Compare every contributor
# ------------------------

for wav in ground_truth_folder.glob("*.wav"):

    gt_audio, sr = librosa.load(
        wav,
        sr=16000
    )

    gt_mfcc = librosa.feature.mfcc(
        y=gt_audio,
        sr=sr,
        n_mfcc=13
    )

    # Dynamic Time Warping

    D, wp = librosa.sequence.dtw(
        X=system_mfcc,
        Y=gt_mfcc,
        metric="cosine"
    )

    distance = D[-1, -1]

    duration_difference = abs(

        librosa.get_duration(y=system_audio, sr=sr)

        -

        librosa.get_duration(y=gt_audio, sr=sr)

    )

    results.append({

        "file": wav.name,

        "dtw_distance": distance,

        "duration_difference": duration_difference

    })

# ------------------------
# Results
# ------------------------

print("----------------------------------")

for r in results:

    print(r)

print("----------------------------------")

mean_distance = np.mean(

    [r["dtw_distance"] for r in results]

)

print()

print("Average DTW Distance")

print(mean_distance)