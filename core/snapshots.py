"""Run snapshot system for living reviews.

Provides save/load/list/diff functionality for pipeline run snapshots,
enabling longitudinal comparison of systematic review updates over time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.io import atomic_write_json

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """A frozen record of pipeline state at a point in time."""
    run_id: str
    timestamp: str                  # ISO format
    config_hash: str                # hash of research_config.yaml for comparison
    stage_data: dict                # {stage_name: {file_path: str, record_count: int, hash: str}}
    summary: dict                   # {n_corpus: int, n_relevant: int, n_claims: int, ...}


@dataclass
class SnapshotDiff:
    """Comparison between two snapshots."""
    old_run_id: str
    new_run_id: str
    new_articles: list[dict] = field(default_factory=list)
    removed_articles: list[dict] = field(default_factory=list)
    score_changes: list[dict] = field(default_factory=list)
    claim_changes: list[dict] = field(default_factory=list)
    grade_changes: dict = field(default_factory=dict)
    summary_text: str = ""


# ── File hashing ──────────────────────────────────────────────────────────


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return ""


def _count_records(path: Path) -> int:
    """Count records in a JSON file (list length or articles key)."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("articles", "records", "items", "claims", "results"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
            return len(data)
        return 0
    except (json.JSONDecodeError, OSError):
        return 0


def _load_json_safe(path: Path) -> Optional[list | dict]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ── Config hashing ────────────────────────────────────────────────────────


def _config_hash(config) -> str:
    """Hash the research config to detect configuration changes between runs."""
    config_str = json.dumps({
        "hypothesis": config.hypothesis_text,
        "provider": config.llm_provider,
        "model": config.llm_model,
        "threshold": config.relevance_threshold,
    }, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ── Snapshot save/load/list ───────────────────────────────────────────────


def save_snapshot(config, run_id: str = "") -> Path:
    """Save current pipeline state as a timestamped snapshot.

    1. Generate run_id from timestamp if not provided
    2. Walk all pipeline output files, compute record counts and file hashes
    3. Collect summary statistics
    4. Save to config.output_dir / "snapshots" / f"{run_id}.json"
    """
    now = datetime.now()
    if not run_id:
        run_id = now.strftime("run_%Y%m%d_%H%M%S")

    snapshot_dir = config.output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Collect stage data from pipeline output files
    stage_files = {
        "corpus": config.corpus_dir / "corpus.json",
        "relevant": config.output_dir / "relevant.json",
        "claims": config.output_dir / "claims.json",
        "claims_filtered": config.output_dir / "claims_filtered.json",
        "synthesis": config.output_dir / "synthesis.json",
        "robustness": config.output_dir / "robustness.json",
        "interrater": config.output_dir / "interrater.json",
        "grade": config.output_dir / "grade_results.json",
        "full_results": config.output_dir / "full_results.json",
    }

    stage_data = {}
    for stage_name, file_path in stage_files.items():
        if file_path.exists():
            stage_data[stage_name] = {
                "file_path": str(file_path),
                "record_count": _count_records(file_path),
                "hash": _file_hash(file_path),
            }

    # Summary statistics
    summary = {
        "n_corpus": stage_data.get("corpus", {}).get("record_count", 0),
        "n_relevant": stage_data.get("relevant", {}).get("record_count", 0),
        "n_claims": stage_data.get("claims", {}).get("record_count", 0),
        "n_claims_filtered": stage_data.get("claims_filtered", {}).get("record_count", 0),
        "stages_completed": len(stage_data),
        "total_files": len(stage_data),
    }

    snapshot = Snapshot(
        run_id=run_id,
        timestamp=now.isoformat(),
        config_hash=_config_hash(config),
        stage_data=stage_data,
        summary=summary,
    )

    out_path = snapshot_dir / f"{run_id}.json"
    atomic_write_json(out_path, asdict(snapshot), ensure_ascii=True)

    logger.info("Snapshot saved: %s (%d stages)", run_id, len(stage_data))
    return out_path


def load_snapshot(path: Path) -> Snapshot:
    """Load a snapshot from JSON."""
    with open(path) as f:
        data = json.load(f)
    return Snapshot(**data)


def list_snapshots(config) -> list[Snapshot]:
    """List all snapshots, sorted by timestamp (newest first)."""
    snapshot_dir = config.output_dir / "snapshots"
    if not snapshot_dir.exists():
        return []

    snapshots = []
    for path in sorted(snapshot_dir.glob("*.json"), reverse=True):
        try:
            snapshots.append(load_snapshot(path))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Failed to load snapshot %s: %s", path, exc)
    return snapshots


# ── Snapshot diff ─────────────────────────────────────────────────────────


def _article_key(article: dict) -> str:
    """Stable key for an article: DOI if available, else lowercase title."""
    doi = article.get("doi", "")
    if doi:
        return f"doi:{doi.lower()}"
    title = article.get("title", "")
    return f"title:{title.lower().strip()}"


def diff_snapshots(old: Snapshot, new: Snapshot, config=None) -> SnapshotDiff:
    """Compare two snapshots and identify changes.

    1. Load corpus from both snapshot paths
    2. Diff by DOI/title to find new/removed articles
    3. Compare relevance scores for shared articles
    4. Compare claim data for shared articles
    5. Compare GRADE results if available
    6. Generate summary text
    """
    diff = SnapshotDiff(old_run_id=old.run_id, new_run_id=new.run_id)

    # Load corpora
    old_corpus_path = old.stage_data.get("corpus", {}).get("file_path", "")
    new_corpus_path = new.stage_data.get("corpus", {}).get("file_path", "")

    old_articles = _load_json_safe(Path(old_corpus_path)) if old_corpus_path else None
    new_articles = _load_json_safe(Path(new_corpus_path)) if new_corpus_path else None

    if isinstance(old_articles, dict):
        old_articles = old_articles.get("articles", [])
    if isinstance(new_articles, dict):
        new_articles = new_articles.get("articles", [])

    old_articles = old_articles or []
    new_articles = new_articles or []

    # Index by key
    old_by_key = {_article_key(a): a for a in old_articles}
    new_by_key = {_article_key(a): a for a in new_articles}

    old_keys = set(old_by_key.keys())
    new_keys = set(new_by_key.keys())

    # New and removed articles
    for key in new_keys - old_keys:
        art = new_by_key[key]
        diff.new_articles.append({
            "title": art.get("title", ""),
            "doi": art.get("doi", ""),
            "year": art.get("year", ""),
        })

    for key in old_keys - new_keys:
        art = old_by_key[key]
        diff.removed_articles.append({
            "title": art.get("title", ""),
            "doi": art.get("doi", ""),
            "year": art.get("year", ""),
        })

    # Score changes for shared articles
    for key in old_keys & new_keys:
        old_art = old_by_key[key]
        new_art = new_by_key[key]
        old_score = old_art.get("relevance_score", old_art.get("score"))
        new_score = new_art.get("relevance_score", new_art.get("score"))
        if old_score is not None and new_score is not None:
            try:
                if abs(float(old_score) - float(new_score)) > 0.01:
                    diff.score_changes.append({
                        "title": new_art.get("title", ""),
                        "old_score": old_score,
                        "new_score": new_score,
                    })
            except (ValueError, TypeError):
                pass

    # Claim changes — compare by hash of stage data
    old_claims_hash = old.stage_data.get("claims", {}).get("hash", "")
    new_claims_hash = new.stage_data.get("claims", {}).get("hash", "")
    if old_claims_hash and new_claims_hash and old_claims_hash != new_claims_hash:
        old_n = old.stage_data.get("claims", {}).get("record_count", 0)
        new_n = new.stage_data.get("claims", {}).get("record_count", 0)
        diff.claim_changes.append({
            "type": "count_change",
            "old_count": old_n,
            "new_count": new_n,
        })

    # GRADE changes
    old_grade_hash = old.stage_data.get("grade", {}).get("hash", "")
    new_grade_hash = new.stage_data.get("grade", {}).get("hash", "")
    if old_grade_hash and new_grade_hash and old_grade_hash != new_grade_hash:
        diff.grade_changes["changed"] = True

    # Config change detection
    config_changed = old.config_hash != new.config_hash

    # Generate summary
    parts = [f"Comparison: {old.run_id} -> {new.run_id}"]
    if config_changed:
        parts.append("WARNING: Pipeline configuration changed between runs.")
    parts.append(f"New articles: {len(diff.new_articles)}")
    parts.append(f"Removed articles: {len(diff.removed_articles)}")
    parts.append(f"Score changes: {len(diff.score_changes)}")
    if diff.claim_changes:
        parts.append(f"Claim data changed (old: {diff.claim_changes[0].get('old_count', '?')}, new: {diff.claim_changes[0].get('new_count', '?')})")
    if diff.grade_changes:
        parts.append("GRADE assessment results changed.")

    # Stage completion comparison
    old_stages = set(old.stage_data.keys())
    new_stages = set(new.stage_data.keys())
    added_stages = new_stages - old_stages
    removed_stages = old_stages - new_stages
    if added_stages:
        parts.append(f"New stages completed: {', '.join(sorted(added_stages))}")
    if removed_stages:
        parts.append(f"Stages no longer present: {', '.join(sorted(removed_stages))}")

    diff.summary_text = "\n".join(parts)
    return diff


# ── Diff report ───────────────────────────────────────────────────────────


def generate_diff_report(diff: SnapshotDiff, output_path: Path) -> Path:
    """Save diff as a formatted markdown report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Snapshot Comparison Report",
        "",
        f"**Old run:** {diff.old_run_id}",
        f"**New run:** {diff.new_run_id}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        diff.summary_text,
        "",
    ]

    if diff.new_articles:
        lines.extend([
            "## New Articles",
            "",
            "| Title | DOI | Year |",
            "|-------|-----|------|",
        ])
        for art in diff.new_articles:
            lines.append(f"| {art.get('title', '')} | {art.get('doi', '')} | {art.get('year', '')} |")
        lines.append("")

    if diff.removed_articles:
        lines.extend([
            "## Removed Articles",
            "",
            "| Title | DOI | Year |",
            "|-------|-----|------|",
        ])
        for art in diff.removed_articles:
            lines.append(f"| {art.get('title', '')} | {art.get('doi', '')} | {art.get('year', '')} |")
        lines.append("")

    if diff.score_changes:
        lines.extend([
            "## Score Changes",
            "",
            "| Title | Old Score | New Score |",
            "|-------|-----------|-----------|",
        ])
        for sc in diff.score_changes:
            lines.append(f"| {sc.get('title', '')} | {sc.get('old_score', '')} | {sc.get('new_score', '')} |")
        lines.append("")

    if diff.claim_changes:
        lines.extend([
            "## Claim Changes",
            "",
        ])
        for cc in diff.claim_changes:
            lines.append(f"- {cc.get('type', '')}: old={cc.get('old_count', '?')}, new={cc.get('new_count', '?')}")
        lines.append("")

    if diff.grade_changes:
        lines.extend([
            "## GRADE Changes",
            "",
            "GRADE assessment results differ between the two runs.",
            "",
        ])

    lines.extend([
        "---",
        "*Generated by Sysys Living Review Snapshot System*",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Diff report saved to %s", output_path)
    return output_path
