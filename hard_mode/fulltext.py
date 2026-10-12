"""Full-text helpers for hard-mode runs."""

from __future__ import annotations

import json
from typing import Any

from acquire.fulltext import write_json_atomic
from core.config import Config
from hard_mode.load_config import FULL_TEXT_DEFAULTS
from hard_mode.paths import ensure_run_dir


def fulltext_cfg(hm: dict[str, Any]) -> dict[str, Any]:
    # Single-sourced from hard_mode.load_config.FULL_TEXT_DEFAULTS.
    cfg = dict(FULL_TEXT_DEFAULTS)
    cfg.update(hm.get("full_text") or {})
    return cfg


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def filter_to_fulltext_only(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [paper for paper in papers if _norm_text(paper.get("full_text") or "")]


def triage_text_bundle(paper: dict[str, Any], hm: dict[str, Any]) -> dict[str, Any]:
    cfg = fulltext_cfg(hm)
    title = _norm_text(paper.get("title") or "")
    abstract = _norm_text(paper.get("abstract") or "")
    use_full = bool(
        cfg.get("enabled")
        and cfg.get("prefer_for_cheap_triage")
        and _norm_text(paper.get("full_text") or "")
    )

    parts = [part for part in (title, abstract) if part]
    if use_full:
        limit = max(1000, int(cfg.get("triage_max_chars") or 20000))
        full_excerpt = _norm_text(paper.get("full_text") or "")[:limit]
        if full_excerpt:
            parts.append(full_excerpt)
        return {
            "text": "\n".join(parts).strip(),
            "text_source": "full_text",
            "body_chars": len(full_excerpt),
        }

    return {
        "text": "\n".join(parts).strip(),
        "text_source": "abstract",
        "body_chars": len(abstract),
    }


def score_text_bundle(
    config: Config,
    hm: dict[str, Any],
    paper: dict[str, Any],
    *,
    title_limit: int,
    abstract_limit: int,
    fulltext_limit: int | None = None,
) -> dict[str, Any]:
    cfg = fulltext_cfg(hm)
    title = _norm_text(paper.get("title") or "")[:title_limit]
    abstract = _norm_text(paper.get("abstract") or "")[:abstract_limit]
    full_text = _norm_text(paper.get("full_text") or "")
    use_full = bool(cfg.get("enabled") and cfg.get("prefer_for_pass3") and full_text)

    if use_full:
        if fulltext_limit is not None:
            limit = max(1000, int(fulltext_limit))
        else:
            raw_limit = cfg.get("score_max_chars")
            if raw_limit is None:
                limit = max(1000, int(getattr(config, "score_fulltext_chars", 20000)))
            else:
                limit = max(1000, int(raw_limit))
        full_excerpt = full_text[:limit]
        return {
            "workload": "fulltext",
            "text_source": "full_text",
            "title": title,
            "abstract": abstract,
            "body_label": "Full text excerpt",
            "body_text": full_excerpt,
            "body_chars": len(full_excerpt),
        }

    return {
        "workload": "abstract",
        "text_source": "abstract",
        "title": title,
        "abstract": abstract,
        "body_label": "Abstract",
        "body_text": abstract,
        "body_chars": len(abstract),
    }


def enrich_run_with_fulltext(
    config: Config,
    hm: dict[str, Any],
    run_id: str,
    *,
    cancel_event=None,
    progress_callback=None,
) -> dict[str, Any]:
    from stages import s2b_fulltext

    run_dir = ensure_run_dir(config, run_id)
    corpus_path = run_dir / "corpus_merged.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing {corpus_path}")
    output_path = run_dir / "corpus_fulltext.json"
    result = s2b_fulltext.run(
        config,
        corpus_path=corpus_path,
        output_path=output_path,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    cfg = fulltext_cfg(hm)
    if output_path.exists() and cfg.get("keep_only_full_text"):
        with open(output_path, encoding="utf-8") as f:
            papers = json.load(f)
        filtered = filter_to_fulltext_only(papers)
        write_json_atomic(output_path, filtered)
        result["paper_count_before_filter"] = len(papers)
        result["paper_count"] = len(filtered)
        result["fulltext_count"] = len(filtered)
        result["dropped_without_fulltext"] = max(0, len(papers) - len(filtered))
        result["kept_only_full_text"] = True
    return result


def export_run_to_regular_pipeline(config: Config, hm: dict[str, Any], run_id: str) -> dict[str, Any]:
    run_dir = ensure_run_dir(config, run_id)
    merged_path = run_dir / "corpus_merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing {merged_path}")
    cfg = fulltext_cfg(hm)
    keep_only = bool(cfg.get("keep_only_full_text"))

    with open(merged_path, encoding="utf-8") as f:
        merged = json.load(f)
    fulltext_path = run_dir / "corpus_fulltext.json"
    export_corpus = merged
    exported = {
        "corpus_path": str(config.corpus_dir / "corpus.json"),
        "paper_count": len(merged),
        "fulltext_path": "",
        "fulltext_count": 0,
        "kept_only_full_text": keep_only,
    }

    if keep_only and not fulltext_path.exists():
        raise FileNotFoundError(
            f"Full-text-only mode is enabled but missing {fulltext_path}. Run Pass 1b first."
        )

    if fulltext_path.exists():
        with open(fulltext_path, encoding="utf-8") as f:
            fulltext = json.load(f)
        if keep_only:
            export_corpus = filter_to_fulltext_only(fulltext)
        write_json_atomic(config.corpus_dir / "corpus_fulltext.json", fulltext)
        exported["fulltext_path"] = str(config.corpus_dir / "corpus_fulltext.json")
        exported["fulltext_count"] = sum(1 for paper in fulltext if paper.get("full_text"))

    write_json_atomic(config.corpus_dir / "corpus.json", export_corpus)
    exported["paper_count"] = len(export_corpus)

    manifest_path = run_dir / "regular_pipeline_export.json"
    write_json_atomic(manifest_path, exported)
    exported["manifest_path"] = str(manifest_path)
    return exported
