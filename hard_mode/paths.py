"""Filesystem layout for hard-mode runs."""

import re
from pathlib import Path

from core.config import Config

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")


def sanitize_run_id(run_id: str) -> str:
    """Reject path separators and parent-directory segments in a run id."""
    text = str(run_id).strip()
    if not _SAFE_RUN_ID.fullmatch(text):
        raise ValueError(
            "run_id must be 1-81 characters of letters, digits, '_' or '-'; "
            f"got {run_id!r}"
        )
    return text


def hard_mode_base(config: Config) -> Path:
    override = getattr(config, "hard_mode_base_dir", None)
    if override:
        return Path(override)
    return config.data_dir / "hard_mode"


def hard_mode_run_dir(config: Config, run_id: str) -> Path:
    return hard_mode_base(config) / sanitize_run_id(run_id)


def ensure_run_dir(config: Config, run_id: str) -> Path:
    d = hard_mode_run_dir(config, run_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
