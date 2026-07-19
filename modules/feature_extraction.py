from modules.audio_loader import load_audio
from pathlib import Path

import librosa

import numpy as np

# =========================================================
# MODULE 1
# Extract MFCC
# =========================================================

def extract_mfcc(

    audio,

    sample_rate,

    n_mfcc=13

):

    """
    Extracts MFCC coefficients.

    Returns
    -------
    numpy.ndarray
    """

    mfcc = librosa.feature.mfcc(

        y=audio,

        sr=sample_rate,

        n_mfcc=n_mfcc

    )

    return mfcc

# =========================================================
# MODULE 2
# Extract Pitch
# =========================================================

def extract_pitch(audio, sample_rate):

    """
    Estimates the average fundamental frequency (F0)
    using the YIN algorithm.
    """

    f0 = librosa.yin(

        audio,

        fmin=75,

        fmax=400,

        sr=sample_rate

    )

    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:

        return 0.0

    return float(np.mean(f0))

# =========================================================
# MODULE 3
# Extract Duration
# =========================================================

def extract_duration(

    audio,

    sample_rate

):

    """
    Returns duration in seconds.
    """

    return librosa.get_duration(

        y=audio,

        sr=sample_rate

    )

# =========================================================
# MODULE 4
# Extract RMS Energy
# =========================================================

def extract_energy(

    audio

):

    """
    Computes RMS energy.
    """

    rms = librosa.feature.rms(

        y=audio

    )

    return float(np.mean(rms))

# =========================================================
# MODULE 5
# Extract All Features
# =========================================================

def extract_features(

    audio,

    sample_rate

):

    """
    Extracts all acoustic features
    used in the dissertation.
    """

    features = {

        "mfcc": extract_mfcc(

            audio,

            sample_rate

        ),

        "pitch": extract_pitch(

            audio,

            sample_rate

        ),

        "duration": extract_duration(

            audio,

            sample_rate

        ),

        "energy": extract_energy(

            audio

        )

    }

    return features

# =========================================================
# TEMPORARY TEST (COMMENTED OUT)
# Uncomment for module testing only.
# =========================================================

"""
if __name__ == "__main__":

    test_file = Path(
        "recordings/wav_system/Sigonyela_Road.wav"
    )

    print("\nLoading recording...")

    audio, sample_rate = load_audio(test_file)

    features = extract_features(audio, sample_rate)

    print()

    print("Results")
    print("---------------------------")
    print(f"MFCC Shape : {features['mfcc'].shape}")
    print(f"Pitch      : {features['pitch']:.2f} Hz")
    print(f"Duration   : {features['duration']:.2f} sec")
    print(f"Energy     : {features['energy']:.4f}")
"""