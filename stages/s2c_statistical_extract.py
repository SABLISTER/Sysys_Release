"""Stage 2c: deterministic statistical evidence extraction.

Runs after relevance scoring to surface statistical tests, subjects, numerical
results, source locations, and human-review cases before LLM claim extraction.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from core.io import atomic_write_json
from process.meta_analysis import STAGE2C_META_ANALYSIS_INPUT
from process.statistical_extractor import (
    apply_llm_context_reviews,
    collect_human_review,
    extract_statistical_evidence,
    metrics_for_meta_analysis,
    summarize,
)

logger = logging.getLogger(__name__)


def run(
    config=None,
    input_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    metrics_path: Optional[str | Path] = None,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> dict:
    """Extract statistical evidence from Stage 2 relevant papers."""
    if config is None:
        raise ValueError("config is required")

    input_path = Path(input_path) if input_path else config.claims_dir / "relevant.json"
    output_path = (
        Path(output_path)
        if output_path
        else config.claims_dir / "statistical_extractions.json"
    )
    review_path = output_path.with_name("statistical_review_required.json")
    # Default metrics mirror goes to this stage's own input file, not Stage 7's
    # auto_extracted_metrics.json (different schema; sharing the path made the
    # file's schema depend on which stage ran last).
    metrics_path = (
        Path(metrics_path)
        if metrics_path
        else config.claims_dir / STAGE2C_META_ANALYSIS_INPUT
    )
    meta_analysis_input_path = output_path.with_name(STAGE2C_META_ANALYSIS_INPUT)

    def _progress(message: str) -> None:
        logger.info(message)
        if progress_callback:
            progress_callback(message)

    _progress("Stage 2c: Statistical Evidence Extraction")
    _progress(f"  Input: {input_path}")
    _progress(f"  Output: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        papers = json.load(f)
    if not isinstance(papers, list):
        raise ValueError(f"Expected a list of papers in {input_path}")

    result = extract_statistical_evidence(
        papers,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    if result.get("cancelled"):
        return {
            "cancelled": True,
            "summary": result.get("summary", {}),
            "output_path": str(output_path),
        }

    records = result.get("records", [])
    if kwargs.get("use_llm_context_check"):
        from process.llm import set_config

        llm_model = str(kwargs.get("llm_model") or getattr(config, "llm_model", "") or "").strip()
        _progress(
            "Phase 2: LLM context check on local statistic windows "
            f"(model={llm_model or 'not configured'})"
        )
        set_config(config)
        llm_counts = apply_llm_context_reviews(
            records,
            model=llm_model,
            max_contexts=int(kwargs.get("max_llm_contexts", 80)),
            include_human_flags=True,
            use_thinking=bool(kwargs.get("llm_use_thinking", False)),
            max_tokens=int(kwargs.get("llm_max_tokens") or 900),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        _progress(
            "LLM context check complete: "
            f"{llm_counts.get('llm_supported', 0)} supported, "
            f"{llm_counts.get('llm_corrected', 0)} corrected, "
            f"{llm_counts.get('llm_rejected', 0)} rejected/unsafe"
        )

    human_review = collect_human_review(records)
    summary = summarize(records, human_review)
    metrics = metrics_for_meta_analysis(records)

    artifact = {
        "stage": "statistical_evidence_extraction",
        "input_path": str(input_path),
        "llm_context_check": {
            "enabled": bool(kwargs.get("use_llm_context_check")),
            "provider": getattr(config, "llm_provider", ""),
            "model": str(kwargs.get("llm_model") or getattr(config, "llm_model", "") or ""),
            "max_contexts": int(kwargs.get("max_llm_contexts", 80)),
        },
        "summary": summary,
        "records": records,
        "human_review": human_review,
    }
    atomic_write_json(output_path, artifact)
    atomic_write_json(review_path, human_review)
    atomic_write_json(meta_analysis_input_path, metrics)
    if metrics_path != meta_analysis_input_path:
        atomic_write_json(metrics_path, metrics)

    _progress(
        "Statistical extraction complete: "
        f"{summary.get('results', 0)} results, "
        f"{summary.get('human_review_items', 0)} review items"
    )
    _progress(f"  Review required: {review_path}")
    _progress(f"  Meta-analysis input: {meta_analysis_input_path}")
    if metrics_path != meta_analysis_input_path:
        _progress(f"  Legacy metrics mirror: {metrics_path}")

    return {
        "summary": summary,
        "records": records,
        "human_review": human_review,
        "output_path": str(output_path),
        "review_path": str(review_path),
        "meta_analysis_input_path": str(meta_analysis_input_path),
        "metrics_path": str(metrics_path),
    }


def check_output(config) -> dict:
    path = config.claims_dir / "statistical_extractions.json"
    review_path = config.claims_dir / "statistical_review_required.json"
    meta_analysis_input_path = config.claims_dir / STAGE2C_META_ANALYSIS_INPUT
    metrics_path = config.data_dir / "fulltext" / "auto_extracted_metrics.json"
    return {
        "exists": path.exists(),
        "path": str(path),
        "review_path": str(review_path),
        "review_exists": review_path.exists(),
        "meta_analysis_input_path": str(meta_analysis_input_path),
        "meta_analysis_input_exists": meta_analysis_input_path.exists(),
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
    }
