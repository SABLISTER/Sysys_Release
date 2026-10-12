"""Helpers for validating and saving hard-mode YAML text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def validate_hard_mode_yaml_text(text: str) -> dict[str, Any]:
    """Parse hard-mode YAML text and require a mapping at the top level."""
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError("Hard mode config must be a YAML mapping at the top level.")
    return raw


def write_text_atomically(path: Path, text: str) -> None:
    """Write text to disk atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
