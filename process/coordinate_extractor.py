"""
Brain coordinate extraction from full-text articles.

Extracts MNI/Talairach coordinates using regex patterns, converts between
coordinate spaces, and exports in NiMARE, GingerALE/Sleuth, and CSV formats.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BrainCoordinate:
    """A single peak activation coordinate extracted from a paper."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    space: str = "MNI"             # "MNI" or "Talairach"
    statistic_type: str = ""       # "Z", "T", "F", ""
    statistic_value: float = 0.0
    cluster_size: int = 0          # voxels, if reported
    region_label: str = ""         # anatomical label from paper
    contrast: str = ""             # what comparison produced this peak
    article_id: str = ""
    n_subjects: int = 0


# ── Regex patterns ────────────────────────────────────────────────────

# Coordinate space detection
_MNI_PATTERN = re.compile(r'\bMNI\b', re.IGNORECASE)
_TAL_PATTERN = re.compile(r'\b(?:Talairach|TAL)\b', re.IGNORECASE)

# Coordinate table header: look for x, y, z columns
_TABLE_HEADER_PATTERN = re.compile(
    r'(?:^|\n)\s*'
    r'(?:Region|Area|Label|Structure|BA)?\s*'
    r'[xX]\s+[yY]\s+[zZ]',
    re.MULTILINE,
)

# Parenthesised triplet: (x, y, z) or [x, y, z]
_PAREN_TRIPLET = re.compile(
    r'[\(\[]\s*'
    r'(-?\d{1,3}(?:\.\d+)?)\s*[,;/\s]\s*'
    r'(-?\d{1,3}(?:\.\d+)?)\s*[,;/\s]\s*'
    r'(-?\d{1,3}(?:\.\d+)?)\s*'
    r'[\)\]]'
)

# Tab/space-separated row of 3 numbers (typical coordinate table row)
# Optionally preceded by a region label and/or followed by stat values
_ROW_PATTERN = re.compile(
    r'(?:^|\n)\s*'
    r'(?P<label>[A-Za-z][A-Za-z\s/\(\)]{0,60}?)?\s*'
    r'(?P<x>-?\d{1,3}(?:\.\d+)?)\s+'
    r'(?P<y>-?\d{1,3}(?:\.\d+)?)\s+'
    r'(?P<z>-?\d{1,3}(?:\.\d+)?)'
    r'(?:\s+(?P<stat_val>\d+\.?\d*))?',
    re.MULTILINE,
)

# Statistical value near coordinates
_STAT_PATTERN = re.compile(
    r'(?P<type>[ZTF])\s*[=:]\s*(?P<val>\d+\.?\d*)',
    re.IGNORECASE,
)

# Cluster size
_CLUSTER_PATTERN = re.compile(
    r'(?:k\s*[=:]\s*|(\d+)\s*voxels?)(\d+)?',
    re.IGNORECASE,
)

# Sample size from context
_SAMPLE_PATTERN = re.compile(
    r'[Nn]\s*[=:]\s*(\d+)',
)


# ── Coordinate range validation ───────────────────────────────────────

def _is_plausible_coord(x: float, y: float, z: float) -> bool:
    """Check that coordinates fall within plausible brain-space range."""
    return (-100 <= x <= 100) and (-130 <= y <= 100) and (-80 <= z <= 100)


# ── Extraction functions ──────────────────────────────────────────────

def _detect_space(text: str) -> str:
    """Detect coordinate space from surrounding text context."""
    mni_hits = len(_MNI_PATTERN.findall(text))
    tal_hits = len(_TAL_PATTERN.findall(text))
    if tal_hits > mni_hits:
        return "Talairach"
    return "MNI"


def _extract_stat_from_context(text: str, pos: int) -> tuple[str, float]:
    """Look for statistical values near a coordinate position."""
    window = text[max(0, pos - 80):pos + 80]
    m = _STAT_PATTERN.search(window)
    if m:
        return m.group("type").upper(), float(m.group("val"))
    return "", 0.0


def _extract_cluster_from_context(text: str, pos: int) -> int:
    """Look for cluster size near a coordinate position."""
    window = text[max(0, pos - 80):pos + 80]
    m = _CLUSTER_PATTERN.search(window)
    if m:
        val = m.group(1) or m.group(2)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _extract_sample_from_context(text: str) -> int:
    """Extract sample size from text context."""
    m = _SAMPLE_PATTERN.search(text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0


def extract_coordinates_from_text(
    text: str, article_id: str = ""
) -> list[BrainCoordinate]:
    """
    Extract brain coordinates from full-text using regex patterns.

    Looks for:
    - Parenthesised triplets: (x, y, z)
    - Table-like rows: region  x  y  z  [stat]
    - MNI vs Talairach detection from surrounding context
    - Statistical values (Z=, T=, F=) near coordinates
    - Cluster sizes (k=, voxels) near coordinates
    """
    if not text or not text.strip():
        return []

    coords: list[BrainCoordinate] = []
    space = _detect_space(text)
    n_subjects = _extract_sample_from_context(text)
    seen: set[tuple[float, float, float]] = set()

    # Strategy 1: Parenthesised triplets
    for m in _PAREN_TRIPLET.finditer(text):
        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if not _is_plausible_coord(x, y, z):
            continue
        key = (x, y, z)
        if key in seen:
            continue
        seen.add(key)
        stat_type, stat_val = _extract_stat_from_context(text, m.start())
        cluster = _extract_cluster_from_context(text, m.start())
        coords.append(BrainCoordinate(
            x=x, y=y, z=z, space=space,
            statistic_type=stat_type, statistic_value=stat_val,
            cluster_size=cluster, article_id=article_id,
            n_subjects=n_subjects,
        ))

    # Strategy 2: Table-style rows (only in sections with coordinate headers)
    for header_match in _TABLE_HEADER_PATTERN.finditer(text):
        # Search the block after the header (up to ~3000 chars)
        block_start = header_match.end()
        block = text[block_start:block_start + 3000]

        for m in _ROW_PATTERN.finditer(block):
            try:
                x = float(m.group("x"))
                y = float(m.group("y"))
                z = float(m.group("z"))
            except (ValueError, TypeError):
                continue
            if not _is_plausible_coord(x, y, z):
                continue
            key = (x, y, z)
            if key in seen:
                continue
            seen.add(key)

            label = (m.group("label") or "").strip()
            stat_type, stat_val = _extract_stat_from_context(
                text, block_start + m.start()
            )
            cluster = _extract_cluster_from_context(
                text, block_start + m.start()
            )
            stat_from_row = m.group("stat_val")
            if stat_from_row and not stat_val:
                stat_val = float(stat_from_row)

            coords.append(BrainCoordinate(
                x=x, y=y, z=z, space=space,
                statistic_type=stat_type, statistic_value=stat_val,
                cluster_size=cluster, region_label=label,
                article_id=article_id, n_subjects=n_subjects,
            ))

    return coords


# ── Talairach → MNI conversion ────────────────────────────────────────

def _invert_4x4(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Invert a 4x4 affine with Gauss-Jordan elimination."""
    size = 4
    aug = [list(matrix[row]) + [1.0 if col == row else 0.0 for col in range(size)] for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Lancaster transform matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][index] for index, value in enumerate(aug[row])]
    return tuple(tuple(row[size:]) for row in aug)


def _apply_affine(
    matrix: tuple[tuple[float, ...], ...],
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    point = (x, y, z, 1.0)
    transformed = [sum(matrix[row][col] * point[col] for col in range(4)) for row in range(3)]
    return transformed[0], transformed[1], transformed[2]


# Lancaster et al. (2007) icbm_spm (MNI/SPM -> Talairach), inverted once.
_ICBM_SPM_TO_TAL = (
    (0.9254, 0.0024, -0.0118, -1.0207),
    (-0.0048, 0.9316, -0.0871, -1.7667),
    (0.0152, 0.0883, 0.8924, 4.0926),
    (0.0, 0.0, 0.0, 1.0),
)
_TAL_TO_ICBM_SPM = _invert_4x4(_ICBM_SPM_TO_TAL)


def convert_tal_to_mni(coords: list[BrainCoordinate]) -> list[BrainCoordinate]:
    """
    Convert Talairach coordinates to MNI using the Lancaster et al. (2007)
    icbm_tal2mni transformation.

    Returns a new list; Talairach coords are converted, MNI coords pass through.
    """
    result: list[BrainCoordinate] = []
    for c in coords:
        if c.space != "Talairach":
            result.append(c)
            continue

        mx, my, mz = _apply_affine(_TAL_TO_ICBM_SPM, c.x, c.y, c.z)

        result.append(BrainCoordinate(
            x=round(mx, 1), y=round(my, 1), z=round(mz, 1),
            space="MNI",
            statistic_type=c.statistic_type,
            statistic_value=c.statistic_value,
            cluster_size=c.cluster_size,
            region_label=c.region_label,
            contrast=c.contrast,
            article_id=c.article_id,
            n_subjects=c.n_subjects,
        ))
    return result


# ── Corpus-level extraction ───────────────────────────────────────────

def extract_coordinates_from_corpus(config: Any) -> list[BrainCoordinate]:
    """
    Walk fulltext files in config.data_dir / "fulltext" and extract coordinates.
    Also checks data/fulltext_corpus/ for corpus-level fulltext.
    Returns all coordinates across all papers.
    """
    all_coords: list[BrainCoordinate] = []
    fulltext_dirs = [
        config.data_dir / "fulltext",
        config.data_dir / "fulltext_corpus",
    ]

    for ft_dir in fulltext_dirs:
        if not ft_dir.is_dir():
            logger.debug("Fulltext directory not found: %s", ft_dir)
            continue
        for fpath in sorted(ft_dir.iterdir()):
            if fpath.suffix not in (".txt", ".md", ".html", ".xml"):
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Could not read %s: %s", fpath, exc)
                continue
            article_id = fpath.stem
            coords = extract_coordinates_from_text(text, article_id=article_id)
            all_coords.extend(coords)

    logger.info(
        "Extracted %d coordinates from %d fulltext directories",
        len(all_coords),
        sum(1 for d in fulltext_dirs if d.is_dir()),
    )
    return all_coords


# ── Export: NiMARE ─────────────────────────────────────────────────────

def export_nimare_dataset(
    coordinates: list[BrainCoordinate], output_path: Path
) -> Path:
    """
    Export as NiMARE-compatible JSON dataset.

    Groups coordinates by article_id, then by contrast.
    """
    dataset: dict[str, Any] = {}

    for c in coordinates:
        study_id = c.article_id or "unknown"
        contrast_id = c.contrast or "contrast_1"

        if study_id not in dataset:
            dataset[study_id] = {
                "contrasts": {},
                "metadata": {"article_id": study_id},
            }

        contrasts = dataset[study_id]["contrasts"]
        if contrast_id not in contrasts:
            contrasts[contrast_id] = {
                "coords": {"x": [], "y": [], "z": []},
                "space": c.space,
                "n": c.n_subjects or 0,
            }

        entry = contrasts[contrast_id]
        entry["coords"]["x"].append(c.x)
        entry["coords"]["y"].append(c.y)
        entry["coords"]["z"].append(c.z)
        if c.n_subjects and c.n_subjects > entry["n"]:
            entry["n"] = c.n_subjects

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    logger.info("Exported NiMARE dataset to %s (%d studies)", output_path, len(dataset))
    return output_path


# ── Export: Sleuth / GingerALE ─────────────────────────────────────────

def export_sleuth_format(
    coordinates: list[BrainCoordinate], output_path: Path
) -> Path:
    """
    Export as Sleuth/GingerALE text format.

    // Reference=MNI
    // Study 1: article_id (N=30)
    x1  y1  z1
    ...
    """
    # Determine dominant space
    spaces = {c.space for c in coordinates}
    ref_space = "MNI" if "MNI" in spaces else ("Talairach" if spaces else "MNI")

    # Group by article_id
    by_study: dict[str, list[BrainCoordinate]] = {}
    for c in coordinates:
        by_study.setdefault(c.article_id or "unknown", []).append(c)

    lines: list[str] = [f"// Reference={ref_space}"]
    for study_id, study_coords in sorted(by_study.items()):
        n = max((c.n_subjects for c in study_coords), default=0)
        n_str = f" (N={n})" if n else ""
        lines.append(f"// {study_id}{n_str}")
        for c in study_coords:
            lines.append(f"{c.x}\t{c.y}\t{c.z}")
        lines.append("")  # blank line between studies

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Exported Sleuth format to %s (%d studies)", output_path, len(by_study))
    return output_path


# ── Export: CSV ────────────────────────────────────────────────────────

def export_csv(
    coordinates: list[BrainCoordinate], output_path: Path
) -> Path:
    """Export as CSV with all coordinate fields."""
    fieldnames = [
        "study_id", "x", "y", "z", "space", "stat_type", "stat_value",
        "cluster_size", "region", "contrast", "n_subjects",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in coordinates:
            writer.writerow({
                "study_id": c.article_id,
                "x": c.x,
                "y": c.y,
                "z": c.z,
                "space": c.space,
                "stat_type": c.statistic_type,
                "stat_value": c.statistic_value,
                "cluster_size": c.cluster_size,
                "region": c.region_label,
                "contrast": c.contrast,
                "n_subjects": c.n_subjects,
            })

    logger.info("Exported CSV to %s (%d rows)", output_path, len(coordinates))
    return output_path
