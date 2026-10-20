import json
from pathlib import Path


def load_dataset(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
