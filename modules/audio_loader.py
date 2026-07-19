from pathlib import Path

import librosa
import numpy as np

# =========================================================
# MODULE 1
# Load Audio
# =========================================================

def load_audio(file_path: Path):

    """
    Loads a WAV recording.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    audio : numpy.ndarray

    sample_rate : int
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    return audio, sample_rate

# =========================================================
# MODULE 2
# Find Ground Truth Recordings
# =========================================================

def find_ground_truth_recordings(
    ground_truth_root: Path,
    street_name: str
):

    """
    Returns every native-speaker recording
    for one street.

    Parameters
    ----------
    ground_truth_root : Path

    street_name : str

    Returns
    -------
    list[Path]
    """

    street_folder = ground_truth_root / street_name

    if not street_folder.exists():

        raise FileNotFoundError(

            f"Ground truth folder not found: {street_folder}"

        )

    recordings = sorted(

        street_folder.glob("*.wav")

    )

    return recordings

# =========================================================
# MODULE 3
# Load All Ground Truth Recordings
# =========================================================

def load_ground_truth_recordings(
    ground_truth_root: Path,
    street_name: str
):

    """
    Loads every native-speaker recording
    for one street.

    Returns
    -------
    list
        List of tuples

        (filename, audio, sample_rate)
    """

    recordings = []

    wav_files = find_ground_truth_recordings(

        ground_truth_root,

        street_name

    )

    for wav_file in wav_files:

        audio, sample_rate = load_audio(wav_file)

        recordings.append(

            (

                wav_file.name,

                audio,

                sample_rate

            )

        )

    return recordings

"""
# =========================================================
# TEMPORARY TEST
# =========================================================

if __name__ == "__main__":

    GROUND_TRUTH = Path("recordings/ground_truth")

    STREET = "Sigonyela_Road"

    print(f"\nSearching for: {STREET}")

    recordings = find_ground_truth_recordings(
        GROUND_TRUTH,
        STREET
    )

    print(f"\nFound {len(recordings)} recording(s):\n")

    for recording in recordings:

        audio, sample_rate = load_audio(recording)

        duration = librosa.get_duration(
            y=audio,
            sr=sample_rate
        )

        print(
            f"{recording.name:<10}"
            f" Sample Rate: {sample_rate} Hz"
            f" | Duration: {duration:.2f} sec"
        )
        """
