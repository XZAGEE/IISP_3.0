import hashlib
import json
import random
import shutil
from pathlib import Path

FULL_DATASET = Path("../IISP_2/dataset/no")
CI_DATASET = Path("ci_dataset")
SAMPLES = Path("samples")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def speaker_id(path):
    return path.name.split("_nohash_")[0]


ci_files = list(CI_DATASET.rglob("*.wav"))

ci_hashes = {file_hash(path) for path in ci_files}
ci_speakers = {speaker_id(path) for path in ci_files}

old_hash = file_hash(SAMPLES / "no.wav")

candidates = [
    path
    for path in sorted(FULL_DATASET.glob("*.wav"))
    if speaker_id(path) not in ci_speakers
    and file_hash(path) not in ci_hashes
    and file_hash(path) != old_hash
]

random.Random(123).shuffle(candidates)

if not candidates:
    raise RuntimeError("No independent samples found")

selected = candidates[0]

backup = Path("results/invalid_no.wav")
backup.parent.mkdir(parents=True, exist_ok=True)

shutil.copy2(SAMPLES / "no.wav", backup)
shutil.copy2(selected, SAMPLES / "no.wav")

sources_path = SAMPLES / "sources.json"

sources = json.loads(
    sources_path.read_text(encoding="utf-8")
)

sources["no.wav"] = {
    "source": f"no/{selected.name}",
    "sha256": file_hash(selected),
    "speaker_in_ci": False,
    "replacement_reason": "Original recording was unintelligible",
}

sources_path.write_text(
    json.dumps(sources, indent=4),
    encoding="utf-8",
)

print(f"New sample: {selected.name}")