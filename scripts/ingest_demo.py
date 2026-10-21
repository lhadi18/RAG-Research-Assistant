"""Explicitly index the checked-in demonstration corpus through the running API."""
from pathlib import Path
import requests

base = "http://localhost:8000"
files = []
for path in sorted(Path("data/demo_corpus").glob("*.md")):
    files.append(("files", (path.name, path.read_bytes(), "text/markdown")))
if not files:
    raise SystemExit("No demo Markdown documents found.")
response = requests.post(base + "/documents/upload", files=files, timeout=600)
response.raise_for_status()
print(response.text)
