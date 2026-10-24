"""Run after experiments finish: fingerprint model files and package small artifacts.

Never archives model weights, virtual environments, credentials or unrelated files.
"""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "results/main/DONE").exists():
    raise RuntimeError("Main experiment is not complete")
manifest = {}
for p in sorted((ROOT / "models/Qwen3-1.7B").iterdir()):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    manifest[p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
(ROOT / "results/model-manifest.json").write_text(json.dumps(manifest, indent=2))
with zipfile.ZipFile(
    ROOT / "evidence.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6
) as archive:
    for directory in ["results", "logs"]:
        for p in sorted((ROOT / directory).rglob("*")):
            if not p.is_file() or p.name.endswith(".pid"):
                continue
            archive.write(p, p.relative_to(ROOT))
    archive.write(ROOT / "models/Qwen3-1.7B/config.json", "model-config.json")
print("Packaged", (ROOT / "evidence.zip").stat().st_size, "bytes", flush=True)
