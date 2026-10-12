"""Path resolution utilities."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_list_len(path: Path) -> int | None:
    """Return the length of a JSON list file, or None if unreadable/not a list."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return len(data) if isinstance(data, list) else None


def resolve_claims_path(config, prefer_filtered: bool = True) -> Path:
    """Return claims_filtered.json if it exists, else claims.json.

    Parameters
    ----------
    config : Config
        Pipeline configuration (must have ``claims_dir`` attribute).
    prefer_filtered : bool
        When *True* (default), prefer the filtered variant produced by
        Stage 5.  Set to *False* to always return the unfiltered file.

    Returns
    -------
    Path
        Absolute path to the best available claims JSON file.
    """
    filtered = config.claims_dir / "claims_filtered.json"
    base = config.claims_dir / "claims.json"
    if prefer_filtered and filtered.exists():
        # An empty filtered file means Stage 5 found nothing to keep — using
        # it would silently give downstream stages zero papers.
        n_filtered = _load_list_len(filtered)
        n_base = _load_list_len(base) if base.exists() else None
        if n_filtered == 0 and n_base:
            logger.warning(
                "claims_filtered.json is empty but claims.json has %d papers — "
                "falling back to the unfiltered claims",
                n_base,
            )
            return base
        logger.debug("Using filtered claims: %s", filtered)
        return filtered
    logger.debug("Using base claims: %s", base)
    return base
