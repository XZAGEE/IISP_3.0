import hashlib
import json
import random
import shutil
from pathlib import Path

CLASSES = ["yes", "no", "up", "down"]

FULL_DATASET = Path("../IISP_2/dataset")
CI_DATASET = Path("ci_dataset")
SAMPLES_DIR = Path("samples")

TARGET_PER_CLASS = 1000
SEED = 42


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def speaker_id(path):
    return Path(path).name.split("_nohash_")[0]


def main():
    rng = random.Random(SEED)

    sources = json.loads(
        (SAMPLES_DIR / "sources.json").read_text(
            encoding="utf-8"
        )
    )

    excluded_speakers = {
        speaker_id(item["source"])
        for item in sources.values()
    }

    excluded_hashes = {
        file_hash(SAMPLES_DIR / name)
        for name in sources
    }

    existing_hashes = {
        file_hash(path)
        for path in CI_DATASET.rglob("*.wav")
    }

    for class_name in CLASSES:
        source_dir = FULL_DATASET / class_name
        destination_dir = CI_DATASET / class_name

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing = list(destination_dir.glob("*.wav"))

        for path in existing:
            if (
                speaker_id(path) in excluded_speakers
                or file_hash(path) in excluded_hashes
            ):
                raise RuntimeError(
                    f"Golden sample leakage: {path}"
                )

        needed = TARGET_PER_CLASS - len(existing)

        if needed <= 0:
            print(
                f"{class_name}: {len(existing)} files"
            )
            continue

        candidates = sorted(source_dir.glob("*.wav"))
        rng.shuffle(candidates)

        added = 0

        for path in candidates:
            if added >= needed:
                break

            if speaker_id(path) in excluded_speakers:
                continue

            destination = destination_dir / path.name

            if destination.exists():
                continue

            digest = file_hash(path)

            if digest in excluded_hashes:
                continue

            if digest in existing_hashes:
                continue

            shutil.copy2(path, destination)

            existing_hashes.add(digest)
            added += 1

        total = len(list(destination_dir.glob("*.wav")))

        if total < TARGET_PER_CLASS:
            raise RuntimeError(
                f"Not enough samples for {class_name}"
            )

        print(
            f"{class_name}: added={added}, total={total}"
        )

    print("Dataset expanded successfully.")


if __name__ == "__main__":
    main()