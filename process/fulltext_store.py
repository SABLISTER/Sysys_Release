"""Storage helpers for downloaded full-text artifacts."""
from __future__ import annotations

import asyncio
import gzip
import logging
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from acquire.fulltext import write_json_atomic

logger = logging.getLogger(__name__)


def _is_cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    if hasattr(cancel_event, "is_set"):
        return cancel_event.is_set()
    return bool(cancel_event)


def _row_key(row: dict) -> str:
    return str(
        row.get("doi")
        or row.get("pmc")
        or row.get("key")
        or row.get("title")
        or row.get("path")
        or ""
    ).strip()


def _archive_one_source(row: dict, archive_dir: Path) -> dict:
    updated = dict(row)
    source_value = str(row.get("source_path") or "").strip()
    if not source_value:
        return updated
    source_path = Path(source_value)
    if not source_path.is_file() or source_path.suffix.lower() == ".txt":
        return updated

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{source_path.name}.gz"
    if not archive_path.exists():
        with open(source_path, "rb") as src, gzip.open(archive_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    source_path.unlink(missing_ok=True)
    updated["source_archive_path"] = str(archive_path)
    updated["source_archive_format"] = "gzip"
    updated["source_path"] = ""
    return updated


def archive_original_sources(
    rows: list[dict],
    archive_dir: Path,
    *,
    workers: int = 3,
) -> list[dict]:
    """Compress loose PDF/HTML originals and keep extracted text in place."""
    if not rows:
        return []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return list(pool.map(lambda row: _archive_one_source(row, archive_dir), rows))


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"indexed": []}
    try:
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("indexed", [])
            return data
    except Exception:
        pass
    return {"indexed": []}


async def _add_rows_to_paperqa(
    rows: list[dict],
    db_dir: Path,
    manifest: dict,
    *,
    progress_callback: Any = None,
    cancel_event: Any = None,
    batch_size: int = 100,
) -> dict:
    from paperqa import Docs, Settings

    docs_path = db_dir / "paperqa_docs.pkl"
    indexed = set(str(item) for item in manifest.get("indexed", []))
    if indexed and not docs_path.exists():
        logger.info(
            "PaperQA pickle missing; clearing stale manifest (%d entries)",
            len(indexed),
        )
        indexed = set()
        manifest["indexed"] = []
    # Skip the expensive pickle load entirely when there is no new work:
    # unpickling a multi-hundred-MB Docs object just to discover that
    # every row is already indexed wastes both RAM and time.
    pending_keys = {_row_key(row) for row in rows} - indexed
    pending_keys.discard("")
    if not pending_keys:
        return {
            "available": True,
            "docs_path": str(docs_path),
            "added": 0,
            "skipped": len(rows),
            "failed": [],
            "indexed_total": len(indexed),
            "cancelled": _is_cancelled(cancel_event),
        }
    if docs_path.exists():
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = Docs()
        manifest["indexed"] = []

    settings = Settings(embedding="st-multi-qa-MiniLM-L6-cos-v1")
    added = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    total = len(rows)

    def _emit(current: int) -> None:
        if progress_callback:
            progress_callback(f"PaperQA indexed {current}/{total} texts")

    for i, row in enumerate(rows):
        if _is_cancelled(cancel_event):
            logger.info("PaperQA indexing cancelled at %d/%d", i, total)
            break

        key = _row_key(row)
        text_value = str(row.get("path") or "").strip()
        if not text_value:
            skipped += 1
            continue
        text_path = Path(text_value)
        if not key or key in indexed or not text_path.is_file():
            skipped += 1
            continue
        try:
            await docs.aadd(
                str(text_path),
                citation=str(row.get("title") or key),
                docname=key,
                title=str(row.get("title") or ""),
                doi=str(row.get("doi") or ""),
                settings=settings,
            )
        except Exception as exc:
            failed.append({"key": key, "error": str(exc)})
            continue
        indexed.add(key)
        added += 1

        if (i + 1) % batch_size == 0 or (i + 1) == total:
            _emit(i + 1)
            # Checkpoint the pickle incrementally so a crash only loses the
            # current in-flight batch instead of the entire run.
            try:
                with open(docs_path, "wb") as f:
                    pickle.dump(docs, f)
                manifest["indexed"] = sorted(indexed)
            except Exception as exc:
                logger.warning("PaperQA incremental checkpoint failed: %s", exc)

    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    manifest["indexed"] = sorted(indexed)
    return {
        "available": True,
        "docs_path": str(docs_path),
        "added": added,
        "skipped": skipped,
        "failed": failed,
        "indexed_total": len(indexed),
        "cancelled": _is_cancelled(cancel_event),
    }


def update_fulltext_store(
    rows: list[dict],
    store_dir: Path,
    *,
    workers: int = 3,
    paperqa_enabled: bool = True,
    paperqa_batch_size: int = 100,
    progress_callback: Any = None,
    cancel_event: Any = None,
) -> tuple[list[dict], dict]:
    """Archive originals and optionally index extracted text with PaperQA."""
    store_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = store_dir / "source_archive"
    paperqa_dir = store_dir / "paperqa"
    manifest_path = paperqa_dir / "manifest.json"

    archived_rows = archive_original_sources(rows, archive_dir, workers=workers)
    archived_count = sum(1 for row in archived_rows if row.get("source_archive_path"))

    if _is_cancelled(cancel_event):
        return archived_rows, {
            "archive_dir": str(archive_dir),
            "archived_originals": archived_count,
            "paperqa": {"available": False, "error": "cancelled"},
        }

    paperqa_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)

    if not paperqa_enabled:
        write_json_atomic(manifest_path, manifest)
        return archived_rows, {
            "archive_dir": str(archive_dir),
            "archived_originals": archived_count,
            "paperqa": {
                "available": False,
                "skipped": True,
                "reason": "disabled",
                "indexed_total": len(manifest.get("indexed", [])),
            },
        }

    try:
        paperqa_result = asyncio.run(_add_rows_to_paperqa(
            archived_rows,
            paperqa_dir,
            manifest,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            batch_size=max(1, int(paperqa_batch_size)),
        ))
    except Exception as exc:
        paperqa_result = {
            "available": False,
            "error": str(exc),
            "docs_path": "",
            "added": 0,
            "skipped": 0,
            "failed": [],
            "indexed_total": 0,
        }

    write_json_atomic(manifest_path, manifest)
    metadata = {
        "archive_dir": str(archive_dir),
        "archived_originals": archived_count,
        "paperqa": paperqa_result,
    }
    return archived_rows, metadata
