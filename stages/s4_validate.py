"""Stage 4: Method review & span validation.

Reads relevant.json (or claims.json) from earlier stages, classifies each
paper's study design via LLM, validates evidence spans against the abstract,
and outputs method_profiles_validated.json + method_review_manual.csv +
method_type_summary.json.
"""
import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from core.config import Config

from core.io import atomic_write_json

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/claims/relevant.json")

# Fallback confidence threshold used when research_config.yaml does not specify one.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Heuristic classification (keyword-based fallback)
# ---------------------------------------------------------------------------

from process.domain_patterns import contains_term, method_rules


def _heuristic_classify(paper: dict, config=None, *, rules=None) -> dict:
    """Apply ordered, configurable study-design rules to whole terms."""
    text = paper.get("abstract", "") + " " + paper.get("title", "")
    rules = method_rules(config) if rules is None else rules
    for rule in rules:
        any_terms, all_terms = rule["any_terms"], rule["all_terms"]
        if ((not any_terms or any(contains_term(text, t) for t in any_terms))
                and all(contains_term(text, t) for t in all_terms)):
            return {"study_method_type": rule["study_method_type"],
                    "confidence": rule["confidence"]}
    return {"study_method_type": "unknown", "confidence": 0.1}


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(
    config: "Optional[Config]" = None,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> dict:
    """Run Stage 4: method review & span validation.

    For each paper in relevant.json:
      1. Send abstract to LLM with method classification prompt
      2. Parse response
      3. Validate evidence spans against abstract text
      4. Save results

    Returns a summary dict.
    """
    from process.llm import set_config, chat as llm_chat
    from process.method_review import (
        extraction_prompt,
        parse_json_object,
        validate_method_profile,
        summarize_method_types,
    )

    rules = method_rules(config)  # Reject invalid rules before doing any work.

    if config:
        set_config(config)

    # Resolve paths
    input_path = kwargs.get("input_path") or (
        config.claims_dir / "relevant.json" if config else DEFAULT_INPUT)
    input_path = Path(input_path)

    out_dir = kwargs.get("output_dir") or (
        config.data_dir / "method_review" if config else Path("data/method_review"))
    out_dir = Path(out_dir)

    model = kwargs.get("model") or (
        config.llm_model if config else "")

    logger.info("Stage 4 — Method Review & Span Validation")
    logger.info("  Input:  %s", input_path)
    logger.info("  Output dir: %s", out_dir)
    logger.info("  Model: %s", model)

    # Load papers
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(input_path, encoding="utf-8") as f:
        papers = json.load(f)

    total = len(papers)
    logger.info("  Loaded %d papers", total)
    if progress_callback:
        progress_callback(f"Loaded {total} papers for method review")

    if total == 0:
        logger.warning("No papers to validate — writing empty outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "method_profiles_validated.json", [])
        atomic_write_json(out_dir / "method_type_summary.json",
                    {"total_papers": 0, "flagged_for_review": 0,
                     "flagged_pct": 0.0, "type_counts": {}})
        return {"total": 0, "validated": 0,
                "output_path": str(out_dir / "method_profiles_validated.json")}

    # Determine if LLM is available
    use_llm = bool(config and getattr(config, "llm_provider", "") and model)
    if use_llm:
        logger.info("  LLM classification: provider=%s, model=%s",
                    getattr(config, "llm_provider", ""), model)
    else:
        logger.info("  No LLM configured — heuristic-only mode")

    # Configured confidence threshold (loop-invariant)
    mr_cfg = (config.get_method_review_config() if config else None) or {}
    confidence_threshold = float(mr_cfg.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD))

    # Process each paper concurrently. run_sliding_window collects results
    # in input order, so validated_records is identical to a sequential run.
    # Set concurrent_workers=1 for local inference servers (Ollama / LM
    # Studio) to avoid model contention.
    from core.checkpoint import run_sliding_window

    _workers = kwargs.get("concurrent_workers")
    if _workers is None and config is not None:
        _workers = getattr(config, "concurrent_workers", None)
    _workers = max(1, int(_workers or 4))
    _stagger = max(0.0, float(kwargs.get("stagger_delay", 0.5)))
    if "stagger_delay" not in kwargs and config is not None:
        _stagger = max(0.0, float(getattr(config, "stagger_delay", _stagger)))
    _timeout = float(getattr(config, "worker_timeout_seconds", None) or 300.0)
    logger.info("Method review concurrency: workers=%d", _workers)

    def _classify_one(paper: dict) -> dict:
        title = paper.get("title", "?")[:80]
        abstract = paper.get("abstract", "")
        logger.info("Classifying: %s", title)

        parsed = None
        classification_method = "heuristic_only"

        # Try LLM classification
        if use_llm and abstract:
            try:
                prompt = extraction_prompt(title, abstract)
                response = llm_chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1500,
                )
                parsed = parse_json_object(response)
                if parsed:
                    classification_method = "llm"
                else:
                    logger.warning(
                        "  LLM returned unparseable response for '%s', "
                        "falling back to heuristic", title)
            except Exception as exc:
                logger.warning(
                    "  LLM call failed for '%s' (%s), "
                    "falling back to heuristic", title, exc)

        # Heuristic fallback
        if parsed is None:
            heuristic = _heuristic_classify(paper, config, rules=rules)
            parsed = {
                "study_method_type": heuristic["study_method_type"],
                "confidence": heuristic["confidence"],
                "sample_size": -1,
                "has_control_group": False,
                "is_longitudinal": False,
                "design_notes": "heuristic classification",
                "evidence_spans": [],
            }
            classification_method = "heuristic_fallback"

        # Validate and normalize — use configured confidence threshold
        profile = validate_method_profile(abstract, parsed,
                                          confidence_threshold=confidence_threshold)
        profile["classification_method"] = classification_method
        return {**paper, "method_profile": profile}

    def _build_result(idx: int, record: dict | None) -> dict:
        if record is None:
            logger.error("Worker timed out or failed for paper %d/%d, using heuristic fallback", idx + 1, total)
            record = {**papers[idx], "method_profile": _heuristic_classify(papers[idx], config, rules=rules)}
            record["method_profile"]["classification_method"] = "heuristic_fallback_timeout"
        title = record.get("title", "?")[:80]
        mp = record.get("method_profile", {})
        result_msg = (f"[{idx + 1}/{total}] {title} → "
                      f"{mp.get('study_method_type')} "
                      f"(conf={mp.get('confidence', 0.0):.2f}) "
                      f"[{mp.get('classification_method')}]")
        logger.info(result_msg)
        if progress_callback:
            progress_callback(result_msg)
        return record

    validated_records, was_cancelled = run_sliding_window(
        items=papers,
        worker_fn=_classify_one,
        result_fn=_build_result,
        workers=_workers,
        stagger=_stagger,
        cancel_event=cancel_event,
        future_timeout=_timeout,
    )
    if was_cancelled:
        return {"total": total, "validated": len(validated_records),
                "cancelled": True,
                "output_path": str(out_dir / "method_profiles_validated.json")}

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. method_profiles_validated.json
    validated_path = out_dir / "method_profiles_validated.json"
    atomic_write_json(validated_path, validated_records)
    logger.info("Saved %d validated profiles to %s",
                len(validated_records), validated_path)

    # 2. method_review_manual.csv — papers flagged for manual review
    manual_path = out_dir / "method_review_manual.csv"
    flagged = [r for r in validated_records
               if r.get("method_profile", {}).get("needs_manual_review", False)]
    _write_manual_csv(manual_path, flagged)
    logger.info("Flagged %d papers for manual review → %s",
                len(flagged), manual_path)

    # 3. method_type_summary.json
    summary_data = summarize_method_types(validated_records)
    summary_path = out_dir / "method_type_summary.json"
    atomic_write_json(summary_path, summary_data)
    logger.info("Method type summary → %s", summary_path)

    summary = {
        "total": total,
        "validated": len(validated_records),
        "flagged": len(flagged),
        "output_path": str(validated_path),
        "summary_path": str(summary_path),
        "manual_csv_path": str(manual_path),
    }
    logger.info("Stage 4 summary: %s", summary)
    if progress_callback:
        progress_callback(
            f"Method review complete: {len(validated_records)} papers, "
            f"{len(flagged)} flagged for manual review")

    return summary


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write_manual_csv(path: Path, records: list[dict]) -> None:
    """Write flagged papers to CSV for manual review."""
    fieldnames = [
        "paper_id", "title", "study_method_type", "confidence",
        "classification_method", "manual_review_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            mp = rec.get("method_profile", {})
            writer.writerow({
                "paper_id": rec.get("paper_id", rec.get("id", "")),
                "title": rec.get("title", ""),
                "study_method_type": mp.get("study_method_type", ""),
                "confidence": mp.get("confidence", 0.0),
                "classification_method": mp.get("classification_method", ""),
                "manual_review_reason": mp.get("manual_review_reason", ""),
            })


def check_output(config) -> dict:
    """Check if Stage 4 output exists and return summary."""
    path = config.data_dir / "method_review" / "method_profiles_validated.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"exists": True, "path": str(path), "count": len(data)}
