"""
===========================================================
preprocess_system_recordings.py

Prepares Apple Maps system recordings for pronunciation
analysis by converting MP4 recordings into clean WAV files.

Processing Pipeline

1. Locate MP4 recordings
2. Extract audio
3. Convert to 16 kHz mono WAV
4. Trim leading and trailing silence
5. Normalise amplitude
6. Verify processed audio
7. Save processed recording
8. Generate processing report

Author: Sam Khumalo
Project:
Leveraging Machine and Deep Learning Methods for Developing
NLP Applications and Language Models for Resource-Scarce
South African Languages
===========================================================
"""

from pathlib import Path
import os
import csv

import librosa
import soundfile as sf
import numpy as np

from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import which

# =========================================================
# CONFIGURATION
# =========================================================

# Input folder containing Apple Maps MP4 recordings
INPUT_FOLDER = Path("recordings/apple_maps")

# Output folder for processed WAV files
OUTPUT_FOLDER = Path("recordings/wav_system")

# Processing report
REPORT_FILE = OUTPUT_FOLDER / "dataset_preparation_report.csv"

# Audio settings
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Silence trimming threshold
TRIM_DB = 25

# Minimum acceptable duration after trimming (seconds)
MIN_DURATION = 0.40

# Create output folder if necessary
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================================================
# FFMPEG CONFIGURATION
# =========================================================

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

if AudioSegment.converter is None:
    raise RuntimeError("FFmpeg could not be found.")

if AudioSegment.ffprobe is None:
    raise RuntimeError("FFprobe could not be found.")

# =========================================================
# PROCESSING REPORT
# =========================================================

processing_log = []

# =========================================================
# MODULE 2
# Convert MP4 to WAV
# =========================================================

def convert_mp4_to_wav(mp4_file: Path) -> Path:
    """
    Converts an MP4 recording into a temporary
    16 kHz mono WAV file.

    Parameters
    ----------
    mp4_file : Path
        Path to the Apple Maps MP4 recording.

    Returns
    -------
    Path
        Path to the temporary WAV file.
    """

    # Temporary WAV filename
    temp_wav = OUTPUT_FOLDER / f"{mp4_file.stem}_temp.wav"

    # Load MP4
    audio = AudioSegment.from_file(mp4_file)

    # Convert to mono
    audio = audio.set_channels(TARGET_CHANNELS)

    # Convert to target sample rate
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

    # Export as WAV
    audio.export(
        temp_wav,
        format="wav"
    )

    return temp_wav



# =========================================================
# MODULE 3
# Trim Leading and Trailing Silence
# =========================================================

def trim_silence(wav_file: Path):
    """
    Removes leading and trailing silence from an audio file.

    Parameters
    ----------
    wav_file : Path
        Temporary WAV file.

    Returns
    -------
    tuple
        (
            trimmed_audio,
            duration_before,
            duration_after,
            silence_removed
        )
    """
    

    # Load audio
    y, sr = librosa.load(
        wav_file,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    # Duration before trimming
    duration_before = librosa.get_duration(y=y, sr=sr)

    # Conservative silence trimming
    trimmed_audio, _ = librosa.effects.trim(
        y,
        top_db=TRIM_DB
    )

    # Duration after trimming
    duration_after = librosa.get_duration(
        y=trimmed_audio,
        sr=sr
    )

    silence_removed = duration_before - duration_after

    return (
        trimmed_audio,
        duration_before,
        duration_after,
        silence_removed
    )


# =========================================================
# MODULE 4
# Audio Normalisation
# =========================================================

def normalize_audio(audio):

    """
    Normalises an audio signal so that its
    maximum absolute amplitude equals 1.

    Parameters
    ----------
    audio : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """

    max_value = np.max(np.abs(audio))

    # Prevent division by zero
    if max_value == 0:
        return audio

    normalized_audio = audio / max_value

    return normalized_audio

# =========================================================
# MODULE 5
# Save Processed Recording
# =========================================================

def save_processed_audio(audio, original_file: Path):

    """
    Saves the processed audio as the final WAV recording.

    Parameters
    ----------
    audio : numpy.ndarray
        Processed audio signal.

    original_file : Path
        Original MP4 filename.

    Returns
    -------
    Path
        Path to the saved WAV file.
    """

    output_file = OUTPUT_FOLDER / f"{original_file.stem}.wav"

    sf.write(
        output_file,
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )

    return output_file
# =========================================================
# MODULE 6
# Verify Processed Recording
# =========================================================

def verify_audio(wav_file: Path):

    """
    Verifies that the processed WAV file satisfies
    the project requirements.

    Returns
    -------
    dict
        Verification information.
    """

    y, sr = librosa.load(
        wav_file,
        sr=None,
        mono=False
    )

    duration = librosa.get_duration(y=y, sr=sr)

    status = "OK"

    if duration < MIN_DURATION:
        status = "TOO SHORT"

    return {

        "sample_rate": sr,

        "duration": round(duration, 3),

        "status": status

    }

# =========================================================
# MODULE 7
# Store Processing Information
# =========================================================

def log_processing(
    street_name,
    duration_before,
    duration_after,
    silence_removed,
    verification
):

    processing_log.append({

        "Street Name": street_name,

        "Duration Before (s)": round(duration_before,3),

        "Duration After (s)": round(duration_after,3),

        "Silence Removed (s)": round(silence_removed,3),

        "Sample Rate": verification["sample_rate"],

        "Status": verification["status"]

    })

    # =========================================================
# MODULE 7.1
# Save Processing Report
# =========================================================

def save_processing_report():

    if len(processing_log) == 0:
        return

    with open(

        REPORT_FILE,

        "w",

        newline="",

        encoding="utf-8"

    ) as csvfile:

        writer = csv.DictWriter(

            csvfile,

            fieldnames=processing_log[0].keys()

        )

        writer.writeheader()

        writer.writerows(processing_log)

    print()

    print(f"Processing report saved to")

    print(REPORT_FILE)

  # =========================================================
# MODULE 8
# Process One Recording
# =========================================================

def process_recording(mp4_file: Path):

    try:

        print(f"\nProcessing: {mp4_file.stem}")

        # Convert MP4 to WAV
        temp_wav = convert_mp4_to_wav(mp4_file)

        # Trim silence
        (
            trimmed_audio,
            before,
            after,
            removed
        ) = trim_silence(temp_wav)

        # Normalise
        normalized_audio = normalize_audio(trimmed_audio)

        # Save processed recording
        saved_file = save_processed_audio(
            normalized_audio,
            mp4_file
        )

        # Verify
        verification = verify_audio(saved_file)

        # Log successful processing
        log_processing(
            mp4_file.stem,
            before,
            after,
            removed,
            verification
        )

        # Remove temporary file
        if temp_wav.exists():
            temp_wav.unlink()

        print(f"✓ Saved : {saved_file.name}")
        print(f"✓ Status: {verification['status']}")

    except Exception as e:

        print(f"✗ ERROR processing {mp4_file.stem}")
        print(e)

        processing_log.append({

            "Street Name": mp4_file.stem,

            "Duration Before (s)": "",

            "Duration After (s)": "",

            "Silence Removed (s)": "",

            "Sample Rate": "",

            "Status": f"FAILED: {e}"

        })

# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("Preparing Apple Maps dataset...")
    print("========================================")

    mp4_files = sorted(INPUT_FOLDER.glob("*.mp4"))

    print(f"\nFiles found: {len(mp4_files)}\n")

    for mp4_file in tqdm(mp4_files):

        process_recording(mp4_file)

    save_processing_report()

    print("\n========================================")
    print("Dataset preparation complete.")
    print(f"Files processed : {len(mp4_files)}")
    print(f"Log saved to    : {REPORT_FILE}")
    print("========================================")
    