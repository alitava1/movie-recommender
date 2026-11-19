"""JSON model artifacts with explicit version and kind validation."""

import json
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_VERSION = 1


def save(path, payload, kind):
    document = {"version": ARTIFACT_VERSION, "kind": kind, "created": datetime.now(timezone.utc).isoformat(), "payload": payload}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load(path, kind):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"{path}: unsupported artifact version {document.get('version')!r}; expected {ARTIFACT_VERSION}")
    if document.get("kind") != kind:
        raise ValueError(f"{path}: expected artifact kind {kind!r}, found {document.get('kind')!r}")
    return document["payload"]
