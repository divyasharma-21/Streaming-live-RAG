"""The corpus is read-only: its bytes must match MANIFEST.json, and it must not carry
the watermark or footer personal information from the source PDF."""

import hashlib
import json
import re
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


def _manifest():
    return json.loads((CORPUS_DIR / "MANIFEST.json").read_text(encoding="utf-8"))


def test_manifest_lists_every_corpus_file():
    listed = {d["file"] for d in _manifest()["documents"]}
    on_disk = {p.name for p in CORPUS_DIR.glob("*.md")}
    assert listed == on_disk


def test_corpus_files_match_manifest_hashes():
    for doc in _manifest()["documents"]:
        data = (CORPUS_DIR / doc["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == doc["sha256"], doc["file"]


def test_corpus_excludes_watermark_and_personal_info():
    email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    for path in CORPUS_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "Samsung" not in text, f"watermark text found in {path.name}"
        assert not email.search(text), f"e-mail address found in {path.name}"
