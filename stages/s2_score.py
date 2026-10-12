"""Stage 2: LLM relevance scoring.

Reads the corpus produced by Stage 1, scores each paper with an LLM,
and writes ``relevant.json`` containing papers that meet the threshold.
"""
import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def run(
    config,
    corpus: Optional[list[dict]] = None,
    workload: str = "abstract",
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Run Stage 2 scoring.

    Parameters
    ----------
    config : Config
        Pipeline configuration (must have ``corpus_dir``, ``claims_dir``,
        ``relevance_threshold``, ``llm_model``, etc.).
    corpus : list[dict], optional
        Pre-loaded corpus.  If *None*, loads ``corpus.json`` from
        ``config.corpus_dir``.
    workload : str
        ``"abstract"`` or ``"fulltext"`` — controls which text field is
        sent to the LLM.
    cancel_event : threading.Event, optional
        Set this to signal cancellation.
    progress_callback : callable, optional
        Called with progress strings for the GUI.

    Returns
    -------
    list[dict]
        Papers that scored >= threshold, enriched with scoring metadata.
    """
    from process.llm import set_config
    from process.relevance_filter import filter_corpus

    set_config(config)

    # ── Load corpus ──
    if corpus is None:
        corpus_path = config.corpus_dir / "corpus.json"
        logger.info("Loading corpus from %s", corpus_path)
        with open(corpus_path, encoding="utf-8") as f:
            loaded_corpus: list[dict] = json.load(f)
        corpus = loaded_corpus

    relevance_model = getattr(config, "relevance_model", "") or config.llm_model
    logger.info(
        "Stage 2: Scoring %d papers (threshold=%d, model=%s, workload=%s)",
        len(corpus), config.relevance_threshold, relevance_model, workload,
    )

    # ── Score ──
    nlq = ""
    if hasattr(config, "get_natural_language_question"):
        try:
            nlq = config.get_natural_language_question()
        except Exception:
            nlq = ""
    hypothesis = nlq or getattr(config, "hypothesis_text", "")
    relevant, _cancelled, scored_all = filter_corpus(
        corpus,
        relevance_model,
        threshold=config.relevance_threshold,
        checkpoint_dir=config.claims_dir,
        checkpoint_every=getattr(config, "checkpoint_every", 1),
        stagger_delay=getattr(config, "stagger_delay", None),
        concurrent_workers=getattr(config, "concurrent_workers", None),
        future_timeout=getattr(config, "worker_timeout_seconds", None),
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        hypothesis=hypothesis,
        use_thinking=getattr(config, "score_use_thinking", True),
        score_max_tokens=getattr(config, "score_max_tokens", 4096),
        score_batch_max_tokens=getattr(config, "score_batch_max_tokens", 12288),
        batch_size=getattr(config, "score_batch_size", 1),
        abstract_chars=getattr(config, "score_abstract_chars", 8000),
        fulltext_chars=(
            0 if str(workload).lower().strip() == "abstract"
            else getattr(config, "score_fulltext_chars", 18000)
        ),
        return_scored=True,
    )

    # ── Save relevance artifacts ──
    out_path = config.claims_dir / "relevant.json"
    scored_path = config.claims_dir / "scored_all.json"
    config.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relevant, f, indent=2, default=str)
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(scored_all, f, indent=2, default=str)
    logger.info(
        "Stage 2 complete: %d/%d papers relevant (saved to %s; all scores to %s)",
        len(relevant), len(corpus), out_path, scored_path,
    )

    return relevant


def check_output(config) -> dict:
    """Check whether Stage 2 output already exists."""
    path = config.claims_dir / "relevant.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {"exists": True, "paper_count": len(data), "path": str(path)}
    return {"exists": False}
