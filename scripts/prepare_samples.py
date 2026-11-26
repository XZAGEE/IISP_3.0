import hashlib
import json
import random
import shutil
from pathlib import Path

CLASSES = ["yes", "no", "up", "down"]

FULL_DATASET = Path("../IISP_2/dataset")
CI_DATASET = Path("ci_dataset")
SAMPLES_DIR = Path("samples")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def speaker_id(path):
    return path.name.split("_nohash_")[0]


def main():
    random.seed(42)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    expected = {}
    sources = {}

    ci_files = list(CI_DATASET.rglob("*.wav"))

    ci_hashes = {
        file_hash(path)
        for path in ci_files
    }

    ci_speakers = {
        speaker_id(path)
        for path in ci_files
    }

    for class_name in CLASSES:
        full_files = sorted(
            (FULL_DATASET / class_name).glob("*.wav")
        )

        ci_names = {
            path.name
            for path in (CI_DATASET / class_name).glob("*.wav")
        }

        candidates = [
            path
            for path in full_files
            if path.name not in ci_names
            and file_hash(path) not in ci_hashes
        ]

        independent_speakers = [
            path
            for path in candidates
            if speaker_id(path) not in ci_speakers
        ]

        if independent_speakers:
            candidates = independent_speakers

        if not candidates:
            raise RuntimeError(
                f"No independent samples for {class_name}"
            )

        selected = random.choice(candidates)

        destination = SAMPLES_DIR / f"{class_name}.wav"

        if destination.exists():
            raise FileExistsError(
                f"{destination} already exists"
            )

        shutil.copy2(selected, destination)

        expected[destination.name] = class_name

        sources[destination.name] = {
            "source": f"{class_name}/{selected.name}",
            "sha256": file_hash(selected),
            "speaker_in_ci": (
                speaker_id(selected) in ci_speakers
            ),
        }

        print(f"{selected.name} -> {destination}")

    (SAMPLES_DIR / "expected.json").write_text(
        json.dumps(expected, indent=4),
        encoding="utf-8",
    )

    (SAMPLES_DIR / "sources.json").write_text(
        json.dumps(sources, indent=4),
        encoding="utf-8",
    )

    print("Golden samples prepared successfully.")


if __name__ == "__main__":
    main()