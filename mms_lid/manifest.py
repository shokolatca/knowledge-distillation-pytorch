"""Manifest loading utilities for clip-level audio datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ManifestEntry:
    clip_id: str
    audio_path: str
    split: Optional[str]
    label: Optional[int]


_REQUIRED_FIELDS = {"id", "audio_path"}


def _parse_label(raw_value: object) -> Optional[int]:
    if raw_value in (None, "", "null"):
        return None
    return int(raw_value)


def _validate_row(row: dict, manifest_path: Path) -> None:
    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"Manifest '{manifest_path}' row is missing required fields: {missing}"
        )


def _entry_from_row(row: dict) -> ManifestEntry:
    return ManifestEntry(
        clip_id=str(row["id"]),
        audio_path=str(row["audio_path"]),
        split=row.get("split"),
        label=_parse_label(row.get("label")),
    )


def load_manifest(manifest_path: str | Path, split: Optional[str] = None) -> List[ManifestEntry]:
    """Load entries from JSONL or CSV manifest.

    Expected fields:
      - id (string)
      - audio_path (string)
      - split (optional string)
      - label (optional int)
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    entries: List[ManifestEntry] = []

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                _validate_row(row, path)
                entry = _entry_from_row(row)
                if split is None or entry.split == split:
                    entries.append(entry)
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                _validate_row(row, path)
                entry = _entry_from_row(row)
                if split is None or entry.split == split:
                    entries.append(entry)
    else:
        raise ValueError("Manifest format must be .jsonl or .csv")

    if not entries:
        split_text = "all splits" if split is None else f"split='{split}'"
        raise ValueError(f"No entries found in {path} for {split_text}")

    return entries
