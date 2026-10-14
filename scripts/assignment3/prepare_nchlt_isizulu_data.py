from pathlib import Path
import csv
import argparse
import xml.etree.ElementTree as ET


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
TRANSCRIPTION_EXTENSIONS = {".xml"}

OUTPUT_FILES = {
    "audio_inventory": "nchlt_audio_inventory.csv",
    "transcription_inventory": "nchlt_transcription_inventory.csv",
    "likely_audio": "nchlt_likely_isizulu_audio.csv",
    "likely_transcriptions": "nchlt_likely_isizulu_transcriptions.csv",
    "matches": "nchlt_audio_transcription_matches.csv",
    "summary": "nchlt_preparation_summary.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and match NCHLT isiZulu audio and XML transcriptions."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to extracted NCHLT corpus directory.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory where preparation CSV files will be written.",
    )

    return parser.parse_args()


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def discover_audio_files(input_dir):
    """Find all supported audio files recursively."""

    audio_files = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(path)

    return sorted(audio_files)


def discover_xml_files(input_dir):
    """Find NCHLT XML transcription files recursively."""

    xml_files = []

    for path in input_dir.rglob("*.xml"):
        if path.is_file():
            xml_files.append(path)

    return sorted(xml_files)


def build_audio_inventory(audio_files, input_dir):
    """Build inventory of all discovered audio files."""

    rows = []

    for path in audio_files:
        try:
            relative_path = path.relative_to(input_dir)
        except ValueError:
            relative_path = path

        rows.append(
            {
                "audio_path": str(path),
                "relative_path": str(relative_path),
                "audio_filename": path.name,
                "extension": path.suffix.lower(),
                "parent_folder": path.parent.name,
            }
        )

    return rows


def normalize_path(value):
    """
    Normalize a path stored in XML so it can be compared with
    local Windows paths.

    Example:
        nchlt_zul/audio/001/file.wav
    becomes:
        audio/001/file.wav
    """

    if not value:
        return ""

    value = value.strip().replace("\\", "/")

    # Remove the corpus root prefix used inside NCHLT XML.
    prefixes = [
        "nchlt_zul/",
        "./nchlt_zul/",
        "/nchlt_zul/",
    ]

    for prefix in prefixes:
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix):]
            break

    return value.strip("/").lower()


def extract_partition(xml_path):
    """
    Determine whether the XML belongs to the training or test
    partition based on its filename.
    """

    name = xml_path.name.lower()

    if ".trn." in name:
        return "trn"

    if ".tst." in name:
        return "tst"

    if ".dev." in name:
        return "dev"

    return "unknown"


def parse_nchlt_xml(xml_path):
    """
    Parse one NCHLT XML file.

    Expected structure:

        <speaker id="001" age="22" gender="male">
            <recording
                audio="nchlt_zul/audio/001/nchlt_zul_001m_0001.wav"
                duration="3.12"
                pdp_score="-0.6875">
                <orth>ulwazi oluthile mayelana</orth>
            </recording>
        </speaker>
    """

    records = []

    partition = extract_partition(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        print(f"WARNING: Could not parse XML: {xml_path}")
        print(f"         {exc}")
        return records

    # Search recursively for speaker elements.
    for speaker in root.findall(".//speaker"):

        speaker_id = speaker.get("id", "")
        age = speaker.get("age", "")
        gender = speaker.get("gender", "")

        # Search recordings belonging to this speaker.
        for recording in speaker.findall("./recording"):

            audio_attr = recording.get("audio", "")
            duration = recording.get("duration", "")
            pdp_score = recording.get("pdp_score", "")

            orth = recording.findtext("orth", default="")

            audio_filename = Path(audio_attr).name

            records.append(
                {
                    "transcription_file": str(xml_path),
                    "partition": partition,
                    "speaker_id": speaker_id,
                    "age": age,
                    "gender": gender,
                    "audio_xml_path": audio_attr,
                    "audio_relative_path": normalize_path(audio_attr),
                    "audio_filename": audio_filename,
                    "transcription": orth.strip(),
                    "duration": duration,
                    "pdp_score": pdp_score,
                }
            )

    return records


def parse_all_xml(xml_files):
    """Parse all XML transcription files."""

    all_records = []

    for xml_file in xml_files:
        print(f"Parsing XML: {xml_file.name}")

        records = parse_nchlt_xml(xml_file)

        print(f"  Records extracted: {len(records)}")

        all_records.extend(records)

    return all_records


def build_audio_indexes(audio_files, input_dir):
    """
    Build indexes for matching XML references to local WAV files.

    Two indexes are created:

    1. Relative path:
       audio/001/file.wav

    2. Filename:
       file.wav

    The relative path is preferred because it is more precise.
    """

    relative_index = {}
    filename_index = {}

    for path in audio_files:

        try:
            relative_path = path.relative_to(input_dir)
        except ValueError:
            continue

        normalized_relative = normalize_path(str(relative_path))

        filename = path.name.lower()

        relative_index.setdefault(normalized_relative, []).append(path)
        filename_index.setdefault(filename, []).append(path)

    return relative_index, filename_index


def match_transcriptions_to_audio(
    transcription_records,
    relative_index,
    filename_index,
):
    """
    Match XML transcription records to local audio files.

    Matching priority:

        1. Normalized relative path
        2. Filename fallback

    The filename fallback is used only when the relative path
    cannot be matched.
    """

    matches = []

    matched_count = 0
    unmatched_count = 0
    relative_path_matches = 0
    filename_matches = 0
    ambiguous_filename_matches = 0

    for record in transcription_records:

        xml_relative = record["audio_relative_path"]
        filename = record["audio_filename"].lower()

        local_path = None
        match_method = "unmatched"
        match_status = "unmatched"

        # ---------------------------------------------------------
        # 1. Try exact normalized relative path
        # ---------------------------------------------------------

        candidates = relative_index.get(xml_relative, [])

        if len(candidates) == 1:

            local_path = candidates[0]
            match_method = "relative_path"
            match_status = "matched"
            relative_path_matches += 1

        # ---------------------------------------------------------
        # 2. Fallback to filename
        # ---------------------------------------------------------

        elif len(candidates) == 0:

            filename_candidates = filename_index.get(filename, [])

            if len(filename_candidates) == 1:

                local_path = filename_candidates[0]
                match_method = "filename"
                match_status = "matched"
                filename_matches += 1

            elif len(filename_candidates) > 1:

                match_status = "ambiguous_filename"
                match_method = "filename_ambiguous"
                ambiguous_filename_matches += 1

        # ---------------------------------------------------------
        # Build output record
        # ---------------------------------------------------------

        if local_path is not None:

            matched_count += 1

            try:
                local_relative = local_path.relative_to(
                    local_path.parents[2]
                )
            except ValueError:
                local_relative = local_path

            audio_path = str(local_path)
            local_relative_path = str(local_relative)

        else:

            unmatched_count += 1
            audio_path = ""
            local_relative_path = ""

        matches.append(
            {
                "audio_path": audio_path,
                "audio_filename": record["audio_filename"],
                "audio_relative_path": local_relative_path,
                "speaker_id": record["speaker_id"],
                "age": record["age"],
                "gender": record["gender"],
                "transcription": record["transcription"],
                "duration": record["duration"],
                "pdp_score": record["pdp_score"],
                "transcription_file": record["transcription_file"],
                "partition": record["partition"],
                "audio_xml_path": record["audio_xml_path"],
                "match_status": match_status,
                "match_method": match_method,
            }
        )

    statistics = {
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "relative_path_matches": relative_path_matches,
        "filename_matches": filename_matches,
        "ambiguous_filename_matches": ambiguous_filename_matches,
    }

    return matches, statistics


def find_duplicate_xml_audio_references(transcription_records):
    """Find audio filenames referenced more than once in XML."""

    counts = {}

    for record in transcription_records:

        key = record["audio_filename"].lower()

        if key:
            counts[key] = counts.get(key, 0) + 1

    duplicates = {
        filename: count
        for filename, count in counts.items()
        if count > 1
    }

    return duplicates


def main():

    args = parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    print("=" * 70)
    print("NCHLT isiZulu DATA PREPARATION")
    print("=" * 70)

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================
    # STEP 1: Discover audio
    # =============================================================

    print("Discovering audio files...")

    audio_files = discover_audio_files(input_dir)

    print(f"Audio files discovered: {len(audio_files)}")
    print()

    # =============================================================
    # STEP 2: Discover XML
    # =============================================================

    print("Discovering XML transcription files...")

    xml_files = discover_xml_files(input_dir)

    print(f"XML files discovered: {len(xml_files)}")
    print()

    # =============================================================
    # STEP 3: Audio inventory
    # =============================================================

    print("Building audio inventory...")

    audio_inventory = build_audio_inventory(
        audio_files,
        input_dir,
    )

    write_csv(
        output_dir / OUTPUT_FILES["audio_inventory"],
        audio_inventory,
        [
            "audio_path",
            "relative_path",
            "audio_filename",
            "extension",
            "parent_folder",
        ],
    )

    print(
        f"Audio inventory records: {len(audio_inventory)}"
    )
    print()

    # =============================================================
    # STEP 4: Parse XML
    # =============================================================

    print("Parsing NCHLT XML files...")

    transcription_records = parse_all_xml(xml_files)

    print()
    print(
        f"Total XML transcription records: "
        f"{len(transcription_records)}"
    )
    print()

    # =============================================================
    # STEP 5: XML transcription inventory
    # =============================================================

    transcription_inventory = []

    for xml_file in xml_files:

        file_records = [
            r
            for r in transcription_records
            if Path(r["transcription_file"]).resolve()
            == xml_file.resolve()
        ]

        transcription_inventory.append(
            {
                "transcription_file": str(xml_file),
                "partition": extract_partition(xml_file),
                "record_count": len(file_records),
            }
        )

    write_csv(
        output_dir / OUTPUT_FILES["transcription_inventory"],
        transcription_inventory,
        [
            "transcription_file",
            "partition",
            "record_count",
        ],
    )

    # =============================================================
    # STEP 6: Identify likely isiZulu resources
    # =============================================================

    # These files are already inside the NCHLT isiZulu corpus,
    # so the entire discovered audio inventory is treated as
    # likely isiZulu audio for preparation purposes.
    #
    # This does NOT establish ground truth for street names.

    likely_audio = []

    for row in audio_inventory:

        likely_audio.append(
            {
                **row,
                "language_status": "likely_isiZulu",
            }
        )

    write_csv(
        output_dir / OUTPUT_FILES["likely_audio"],
        likely_audio,
        [
            "audio_path",
            "relative_path",
            "audio_filename",
            "extension",
            "parent_folder",
            "language_status",
        ],
    )

    likely_transcriptions = []

    for record in transcription_records:

        likely_transcriptions.append(
            {
                **record,
                "language_status": "likely_isiZulu",
            }
        )

    write_csv(
        output_dir / OUTPUT_FILES["likely_transcriptions"],
        likely_transcriptions,
        [
            "transcription_file",
            "partition",
            "speaker_id",
            "age",
            "gender",
            "audio_xml_path",
            "audio_relative_path",
            "audio_filename",
            "transcription",
            "duration",
            "pdp_score",
            "language_status",
        ],
    )

    # =============================================================
    # STEP 7: Build matching indexes
    # =============================================================

    print("Building audio matching indexes...")

    relative_index, filename_index = build_audio_indexes(
        audio_files,
        input_dir,
    )

    # =============================================================
    # STEP 8: Match XML records to audio
    # =============================================================

    print("Matching XML transcription records to audio...")

    matches, match_statistics = match_transcriptions_to_audio(
        transcription_records,
        relative_index,
        filename_index,
    )

    write_csv(
        output_dir / OUTPUT_FILES["matches"],
        matches,
        [
            "audio_path",
            "audio_filename",
            "audio_relative_path",
            "speaker_id",
            "age",
            "gender",
            "transcription",
            "duration",
            "pdp_score",
            "transcription_file",
            "partition",
            "audio_xml_path",
            "match_status",
            "match_method",
        ],
    )

    # =============================================================
    # STEP 9: Duplicate XML references
    # =============================================================

    duplicate_xml_refs = find_duplicate_xml_audio_references(
        transcription_records
    )

    # =============================================================
    # STEP 10: Summary
    # =============================================================

    matched = match_statistics["matched_count"]
    unmatched = match_statistics["unmatched_count"]

    audio_count = len(audio_files)
    xml_record_count = len(transcription_records)

    match_percentage = (
        (matched / xml_record_count) * 100
        if xml_record_count
        else 0
    )

    audio_coverage_percentage = (
        (matched / audio_count) * 100
        if audio_count
        else 0
    )

    summary_rows = [
        {
            "metric": "audio_files_discovered",
            "value": audio_count,
        },
        {
            "metric": "xml_files_discovered",
            "value": len(xml_files),
        },
        {
            "metric": "xml_transcription_records",
            "value": xml_record_count,
        },
        {
            "metric": "matched_records",
            "value": matched,
        },
        {
            "metric": "unmatched_records",
            "value": unmatched,
        },
        {
            "metric": "relative_path_matches",
            "value": match_statistics[
                "relative_path_matches"
            ],
        },
        {
            "metric": "filename_fallback_matches",
            "value": match_statistics[
                "filename_matches"
            ],
        },
        {
            "metric": "ambiguous_filename_matches",
            "value": match_statistics[
                "ambiguous_filename_matches"
            ],
        },
        {
            "metric": "duplicate_xml_audio_references",
            "value": len(duplicate_xml_refs),
        },
        {
            "metric": "xml_match_percentage",
            "value": round(match_percentage, 2),
        },
        {
            "metric": "audio_coverage_percentage",
            "value": round(audio_coverage_percentage, 2),
        },
    ]

    write_csv(
        output_dir / OUTPUT_FILES["summary"],
        summary_rows,
        [
            "metric",
            "value",
        ],
    )

    # =============================================================
    # Final report
    # =============================================================

    print()
    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print(f"Audio files discovered       : {audio_count}")
    print(f"XML files discovered         : {len(xml_files)}")
    print(f"XML transcription records    : {xml_record_count}")
    print(f"Matched records              : {matched}")
    print(f"Unmatched records            : {unmatched}")
    print(
        f"Relative-path matches        : "
        f"{match_statistics['relative_path_matches']}"
    )
    print(
        f"Filename fallback matches    : "
        f"{match_statistics['filename_matches']}"
    )
    print(
        f"Ambiguous filename matches   : "
        f"{match_statistics['ambiguous_filename_matches']}"
    )
    print(
        f"Duplicate XML audio refs     : "
        f"{len(duplicate_xml_refs)}"
    )
    print(
        f"XML match percentage         : "
        f"{match_percentage:.2f}%"
    )
    print(
        f"Audio coverage percentage    : "
        f"{audio_coverage_percentage:.2f}%"
    )

    print()
    print("Output files:")

    for filename in OUTPUT_FILES.values():
        print(f"  {output_dir / filename}")

    print()
    print(
        "NOTE: This script only prepares and matches the NCHLT "
        "resources. It does not create ground truth, train a model, "
        "or calculate PER/WER."
    )


if __name__ == "__main__":
    main()