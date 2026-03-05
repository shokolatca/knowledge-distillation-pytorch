"""Small JSON-backed config helper for MMS LID scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class Params:
    """Class that loads hyperparameters from a JSON file."""

    def __init__(self, json_path: str | Path):
        self.update(json_path)

    def save(self, json_path: str | Path) -> None:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.__dict__, handle, indent=4, ensure_ascii=False)

    def update(self, json_path: str | Path) -> None:
        with open(json_path, "r", encoding="utf-8") as handle:
            params: Dict[str, Any] = json.load(handle)
        self.__dict__.update(params)

    @property
    def dict(self) -> Dict[str, Any]:
        return self.__dict__
