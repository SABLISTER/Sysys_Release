"""Stage runner — async wrappers for each pipeline stage.

All 11 stages are fully wired to their stage modules in stages/.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from core.config import (
    EMBEDDING_RATERS as EMBEDDING_SECOND_RATERS,
    PROVIDER_MODEL_FIELD,
    normalize_provider_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> list[dict]:
    """Load a JSON file and return its contents."""
    logger.info("Loading JSON: %s", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("  Loaded %d items from %s (%d bytes)", len(data), path, path.stat().st_size)
    return data


def _stage_banner(stage_num: int, stage_name: str) -> None:
    """Log a prominent stage boundary banner."""
    logger.info("═" * 60)
    logger.info("STAGE %d: %s", stage_num, stage_name.upper())
    logger.info("═" * 60)


# ---------------------------------------------------------------------------
# Stage 1: Search (delegates to search_runner)
# ---------------------------------------------------------------------------
async def run_search(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> list[dict]:
    """Stage 1: Literature search — delegates to search_runner."""
    from gui.search_runner import run_gui_search, ProviderChoice

    _stage_banner(1, "LITERATURE SEARCH")

    nl_question = kw.get("natural_language_question", "")
    semantic_terms = kw.get("semantic_terms", [nl_question] if nl_question else [])
    boolean_terms = kw.get("boolean_terms", [])
    provider = kw.get("provider", ProviderChoice.ALL)
    max_results = kw.get("max_results", 500)

    return await run_gui_search(
        natural_language_question=nl_question,
        semantic_terms=semantic_terms,
        boolean_terms=boolean_terms,
        provider=provider,
        config=config,
        max_results=max_results,
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# Stage 2: Score
# ---------------------------------------------------------------------------
async def run_score(
    config,
    threshold: int = 3,
    corpus_path: Optional[Path] = None,
    workload: str = "abstract",
    batch_size: Optional[int] = None,
    cancel_event: Optional[asyncio.Event] = None,
    output_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> list[dict]:
    """Stage 2: LLM relevance scoring."""
    import stages.s2_score as s2

    _stage_banner(2, "LLM RELEVANCE SCORING")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    logger.info("  Threshold: %d", threshold)
    logger.info("  Workload: %s", workload)
    logger.info("  Workers: %d", getattr(config, "concurrent_workers", 1))

    corpus = None
    if corpus_path:
        corpus = _load_json(corpus_path)

    config.relevance_threshold = threshold
    result = s2.run(
        config,
        corpus=corpus,
        workload=workload,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 3: Extract
# ---------------------------------------------------------------------------
async def run_extract(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 3: Claim extraction."""
    import stages.s3_extract as s3
    from process.llm import set_config

    _stage_banner(3, "CLAIM EXTRACTION")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    logger.info("  Max tokens: %s", kw.get("max_tokens", 4096))
    logger.info("  Use thinking: %s", kw.get("use_thinking", False))

    set_config(config)

    result = s3.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 4: Validate
# ---------------------------------------------------------------------------
async def run_validate(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 4: Method review & span validation."""
    import stages.s4_validate as s4
    from process.llm import set_config

    _stage_banner(4, "METHOD REVIEW & SPAN VALIDATION")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)

    set_config(config)

    result = s4.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 5: Audit
# ---------------------------------------------------------------------------
async def run_audit(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 5: Span audit & quality filtering."""
    import stages.s5_audit as s5

    _stage_banner(5, "SPAN AUDIT & QUALITY FILTERING")
    logger.info("  Claims dir: %s", config.claims_dir)
    logger.info("  Data dir:   %s", config.data_dir)

    result = s5.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 6: Synthesize
# ---------------------------------------------------------------------------
async def run_synthesize(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 6: Cluster-level synthesis — cluster, narrate, draft abstract."""
    import stages.s6_synthesize as s6
    from process.llm import set_config

    _stage_banner(6, "SYNTHESIS (CLUSTER · NARRATE · ABSTRACT)")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    set_config(config)

    result = s6.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 7: Fulltext retrieval
# ---------------------------------------------------------------------------
async def run_fulltext(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 7: Full-text retrieval & metric extraction."""
    import stages.s7_fulltext as s7

    _stage_banner(7, "FULL-TEXT RETRIEVAL & METRIC EXTRACTION")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s7.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 2b / 8: Corpus full-text re-score
# ---------------------------------------------------------------------------
async def run_corpus_fulltext(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    **kw,
) -> dict:
    """Stage 2b: Corpus full-text enrichment before scoring."""
    logger.info("═" * 60)
    logger.info("STAGE 2b: CORPUS FULL-TEXT ENRICHMENT")
    logger.info("═" * 60)
    try:
        import stages.s2b_fulltext as s2b
        result = await asyncio.to_thread(
            lambda: s2b.run(config=config, cancel_event=cancel_event, **kw)
        )
        return result
    except Exception as exc:
        logger.error("Corpus fulltext failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Stage 2c: Statistical extraction
# ---------------------------------------------------------------------------
async def run_statistical_extract(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 2c: deterministic statistical evidence extraction."""
    import stages.s2c_statistical_extract as s2c

    logger.info("═" * 60)
    logger.info("STAGE 2c: STATISTICAL EVIDENCE EXTRACTION")
    logger.info("═" * 60)
    result = await asyncio.to_thread(
        lambda: s2c.run(
            config=config,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            **kw,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Stage 8: Interrater
# ---------------------------------------------------------------------------
def resolve_second_rater(config, second_provider: str,
                         second_model: Optional[str]) -> tuple[Any, str, str]:
    """Build the second rater's config, provider and model.

    ``config`` must carry the PRIMARY rater's settings. Returns
    ``(second_cfg, provider, model)`` where ``model`` is always concrete —
    never ``None`` — so the caller and ``s8.run`` agree on what ran.

    Raises ValueError for an unsupported provider rather than letting
    ``llm.chat`` fall through to its Ollama default, which would silently
    rate with a model nobody asked for.
    """
    import copy

    second_cfg = copy.deepcopy(config)
    provider = normalize_provider_name(second_provider)
    model = (second_model or "").strip()

    # Cross-encoders bypass the LLM stack entirely: the model string is a
    # checkpoint path, so leave the deep-copied LLM settings alone.
    if provider in EMBEDDING_SECOND_RATERS or second_provider in EMBEDDING_SECOND_RATERS:
        return second_cfg, second_provider, model

    field = PROVIDER_MODEL_FIELD.get(provider)
    if field is None:
        raise ValueError(
            f"Unsupported second-rater provider: {second_provider!r}. "
            f"Supported: {sorted(PROVIDER_MODEL_FIELD)} "
            f"or a cross-encoder ({sorted(EMBEDDING_SECOND_RATERS)})."
        )

    second_cfg.llm_provider = provider
    if model:
        setattr(second_cfg, field, model)
    else:
        # Fall back to whatever that provider is configured to use, so the
        # resolved name is reported instead of a bare "default".
        model = getattr(second_cfg, field, "") or ""

    return second_cfg, provider, model


def resolve_primary_rater(config, primary_provider: Optional[str] = None,
                          primary_model: Optional[str] = None) -> tuple[str, str]:
    """Identify the rater that produced the scores already in the corpus.

    Callers that know it explicitly (the GUI reads it off the Stage 2
    widgets) should pass it; otherwise fall back to the config's LLM
    settings. A blank model is resolved to the provider's configured model
    so it compares equal to a second rater that names that model outright.
    """
    provider = (primary_provider or "").strip()
    model = (primary_model or "").strip()

    if not provider:
        provider = getattr(config, "llm_provider", "") or ""
        return provider, model or (getattr(config, "llm_model", "") or "")

    if provider in EMBEDDING_SECOND_RATERS:
        return provider, model

    provider = normalize_provider_name(provider)
    if not model:
        field = PROVIDER_MODEL_FIELD.get(provider)
        if field:
            model = getattr(config, field, "") or ""
    return provider, model


def _same_rater(primary: tuple[str, str], second: tuple[str, str]) -> bool:
    """True when the two raters are indistinguishable.

    Both sides arrive with their models already resolved to concrete names,
    so this is a plain comparison; an unknown side never claims a match.
    """
    p_provider, p_model = primary
    s_provider, s_model = second
    if not p_provider or not s_provider:
        return False
    return (p_provider, p_model or "") == (s_provider, s_model or "")


async def run_interrater(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    sample_size: int = 100,
    second_provider: str = "ollama",
    second_model: Optional[str] = None,
    primary_provider: Optional[str] = None,
    primary_model: Optional[str] = None,
    **kw,
) -> dict:
    """Stage 8: Inter-rater reliability — cross-model validation.

    The primary rater is whatever produced the scores already stored in the
    corpus; pass it as ``primary_provider``/``primary_model`` when known,
    otherwise ``config``'s LLM settings are used. ``second_provider``/
    ``second_model`` select an independent second rater. If the two resolve
    to the same provider and model the run measures one model's
    self-consistency, not inter-rater reliability, so that case is reported
    loudly instead of being passed off as agreement.
    """
    import stages.s8_interrater as s8
    from process.llm import set_config

    _stage_banner(8, "INTER-RATER RELIABILITY")

    def _emit(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    second_cfg, provider, model = resolve_second_rater(
        config, second_provider, second_model)
    primary = resolve_primary_rater(config, primary_provider, primary_model)
    same_rater = _same_rater(primary, (provider, model))

    logger.info("  Primary provider: %s / %s", primary[0], primary[1] or "default")
    logger.info("  Second provider: %s / %s", provider, model or "default")
    logger.info("  Sample size: %d", sample_size)

    if same_rater:
        _emit(
            f"WARNING: the second rater ({provider} / {model or 'default'}) is "
            f"identical to the primary rater. Agreement will reflect one "
            f"model's self-consistency, not inter-rater reliability. Pick a "
            f"different provider or model on the Interrater tab."
        )

    set_config(second_cfg)
    try:
        result = s8.run(
            config=config,
            sample_size=sample_size,
            second_provider=provider,
            second_model=model,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            **kw,
        )
    finally:
        # Always restore the primary config. The GUI process is long-lived, so
        # leaking the second rater's provider would silently redirect every
        # later stage to the wrong model.
        set_config(config)

    if isinstance(result, dict):
        result.setdefault("second_provider", provider)
        result.setdefault("second_model", model)
        result.setdefault("primary_provider", primary[0])
        result.setdefault("primary_model", primary[1])
        result.setdefault("same_rater", same_rater)
    return result


# ---------------------------------------------------------------------------
# Stage 9: Robustness
# ---------------------------------------------------------------------------
async def run_robustness(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 9: Robustness analyses — bootstrap, split-half, sensitivity,
    permutation test, condition–entity network analysis, and condition sensitivity."""
    import stages.s9_robustness as s9

    _stage_banner(9, "ROBUSTNESS ANALYSES")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s9.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 10: Verify
# ---------------------------------------------------------------------------
async def run_verify(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Stage 10: Abstract number verification — fact-check synthesised abstracts."""
    import stages.s10_verify as s10

    _stage_banner(10, "ABSTRACT NUMBER VERIFICATION")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s10.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 11: Compare Runs
# ---------------------------------------------------------------------------
async def run_compare_runs(
    config,
    cancel_event: Optional[asyncio.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    claims_paths: Optional[list] = None,
    comparison_name: str = "run_compare",
    **kw,
) -> dict:
    """Stage 11: Compare pipeline runs."""
    import stages.s11_compare_runs as s11

    _stage_banner(11, "COMPARE RUNS")
    logger.info("  Claims paths: %s", claims_paths)
    logger.info("  Comparison name: %s", comparison_name)

    result = s11.run(
        config=config,
        claims_paths=claims_paths,
        comparison_name=comparison_name,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage Registry
# ---------------------------------------------------------------------------
# Each entry: stage_name -> (async_func, is_async_flag)
_STAGE_REGISTRY: dict[str, tuple[Callable, bool]] = {}


def _register_stages() -> None:
    """Populate the global _STAGE_REGISTRY dict."""
    global _STAGE_REGISTRY
    _STAGE_REGISTRY = {}
    # Each entry: stage_name -> (async_func, is_async_flag). All stage
    # runners share the (config, cancel_event, **kw) signature.
    for _name, _func in (
        ("search", run_search),
        ("score", run_score),
        ("extract", run_extract),
        ("validate", run_validate),
        ("audit", run_audit),
        ("synthesize", run_synthesize),
        ("fulltext", run_fulltext),
        ("corpus_fulltext", run_corpus_fulltext),
        ("statistical_extract", run_statistical_extract),
        ("interrater", run_interrater),
        ("robustness", run_robustness),
        ("verify", run_verify),
        ("compare_runs", run_compare_runs),
    ):
        _STAGE_REGISTRY[_name] = (
            lambda config, cancel_event=None, _f=_func, **kw: _f(
                config, cancel_event=cancel_event, **kw
            ),
            True,
        )


# Auto-register on import
_register_stages()
