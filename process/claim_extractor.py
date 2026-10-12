"""LLM-based claim extraction for the Sysys pipeline."""
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from core.checkpoint import CheckpointManager
from core.claims import claim_list

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_EVERY = 1

EXTRACTION_PROMPT = """You are extracting structured research claims from an academic paper for a systematic evidence review.

Research question:
{research_question}

Paper title: {title}
Abstract: {abstract}

Extract the following structured information as a JSON object:
{schema}

Configured extraction guidance:
{extraction_guidance}

Use supports_primary_question for the configured research question above.
Respond with ONLY the JSON object, no other text.
"""

# Legacy default schema, used when no research_config (or no
# extraction.fields list) is supplied. Preserved for backward compatibility.
_LEGACY_FIELDS: list[tuple[str, str, str]] = [
    ("findings", "list", "list of key findings/results stated in the paper"),
    ("mechanisms", "list", "list of mechanisms, constructs, or explanatory factors discussed"),
    ("conditions", "list", "list of populations, conditions, constructs, or contextual factors mentioned"),
    ("key_entities", "list", "list of the specific entities, systems, variables, or constructs the findings concern"),
    ("methodology", "string", "brief description of study design"),
    ("supports_primary_question", "boolean", "whether the findings support the primary research question"),
    ("network_effects_described", "boolean", "whether network-level effects are described"),
    ("relationships", "string", "description of relationships among the primary topic, populations, constructs, or contextual factors"),
    ("key_quote", "string", "the single most relevant sentence from the abstract"),
]

_FIELD_TYPES = ("list", "string", "boolean")


def _field_specs(research_config: dict | None) -> list[tuple[str, str, str]]:
    """Return (name, type, description) specs for the extraction schema.

    When ``research_config`` defines ``extraction.fields``, the schema is
    built from exactly those fields, so any research question can define its
    own extraction targets. Otherwise the legacy default schema is used.
    """
    if isinstance(research_config, dict):
        extraction = research_config.get("extraction")
        if isinstance(extraction, dict):
            fields = extraction.get("fields")
            if isinstance(fields, list) and fields:
                specs: list[tuple[str, str, str]] = []
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    name = str(field.get("name") or "").strip()
                    ftype = str(field.get("type") or "string").strip().lower()
                    desc = str(field.get("description") or "").strip()
                    if not name or ftype not in _FIELD_TYPES:
                        continue
                    specs.append((name, ftype, desc or name))
                if specs:
                    return specs
    return list(_LEGACY_FIELDS)


def _schema_block(specs: list[tuple[str, str, str]]) -> str:
    """Render the JSON schema block for the extraction prompt."""
    lines = ["{{"]
    for i, (name, ftype, desc) in enumerate(specs):
        if ftype == "list":
            rendered = f'["{desc}"]'
        elif ftype == "boolean":
            rendered = "true/false"
        else:
            rendered = f'"{desc}"'
        comma = "," if i < len(specs) - 1 else ""
        lines.append(f'    "{name}": {rendered}{comma}')
    lines.append("}}")
    return "\n".join(lines)


def _empty_claims(specs: list[tuple[str, str, str]] | None = None) -> dict:
    """Type-appropriate empty claims dict for the given schema."""
    result: dict = {}
    for name, ftype, _ in specs or _LEGACY_FIELDS:
        result[name] = [] if ftype == "list" else (False if ftype == "boolean" else "")
    result["error"] = None
    return result


_EMPTY_CLAIMS = _empty_claims()

_CHECKPOINT_FILENAME = "extract_checkpoint.json"


def _claim_think_mode(model: str, enabled: bool = True) -> bool | str | None:
    """Prefer lightweight exposed reasoning for models that support it."""
    if not enabled:
        return None
    base = model.split(":", 1)[0].strip().lower()
    if base.startswith("gpt-oss"):
        return "low"
    return True


# ---------------------------------------------------------------------------
# Single-paper extraction
# ---------------------------------------------------------------------------

def _research_question_from_config(research_config: dict | None) -> str:
    if not isinstance(research_config, dict):
        return "What does the scholarly literature report for this research question?"
    relevance = research_config.get("relevance")
    if not isinstance(relevance, dict):
        relevance = {}
    for value in (
        research_config.get("natural_language_question"),
        relevance.get("research_question"),
        research_config.get("hypothesis_text"),
        research_config.get("hypothesis"),
        research_config.get("question"),
        research_config.get("description"),
    ):
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return "What does the scholarly literature report for this research question?"


def _format_extraction_guidance(research_config: dict | None) -> str:
    if not isinstance(research_config, dict):
        return "- Use the generic schema descriptions above."
    extraction = research_config.get("extraction")
    if not isinstance(extraction, dict):
        return "- Use the generic schema descriptions above."
    fields = extraction.get("fields")
    if not isinstance(fields, list):
        return "- Use the generic schema descriptions above."

    lines: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        description = str(field.get("description") or "").strip()
        if name and description:
            lines.append(f"- {name}: {description}")
    return "\n".join(lines) if lines else "- Use the generic schema descriptions above."


def build_extraction_prompt(
    title: str,
    abstract: str,
    research_config: dict | None = None,
) -> str:
    """Build the configured claim-extraction prompt for one paper."""
    from process.llm import compress_prompt_text

    specs = _field_specs(research_config)
    return EXTRACTION_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        title=title,
        abstract=compress_prompt_text(abstract[:8000]),
        schema=_schema_block(specs),
        extraction_guidance=_format_extraction_guidance(research_config),
    )


def extract_claims(model: str, paper: dict, think=None,
                   max_tokens: int = 4096,
                   research_config: dict | None = None) -> dict:
    """Extract structured claims from a single paper using the LLM."""
    from process.llm import chat as llm_chat

    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "")
    if not abstract.strip():
        logger.warning("[Extract] No abstract for '%s' — using heuristic fallback",
                       title[:60])
        return {**_EMPTY_CLAIMS, "error": "no_abstract"}

    prompt = build_extraction_prompt(title, abstract, research_config)
    specs = _field_specs(research_config)
    messages = [{"role": "user", "content": prompt}]

    logger.info("[Extract] Extracting claims: %s", title[:80])
    response = llm_chat(model, messages, temperature=0, max_tokens=max_tokens,
                        think=think)
    result = _parse_extraction_response(response, specs)

    n_findings = len(result.get("findings", []))
    n_mechanisms = len(result.get("mechanisms", []))
    n_entities = len(claim_list(result, "key_entities"))
    logger.info("[Extract] Result: %d findings, %d mechanisms, %d entities",
                n_findings, n_mechanisms, n_entities)
    return result


def _parse_extraction_response(
    text: str,
    specs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Parse LLM response into claims dict."""
    from process.llm import parse_llm_json

    specs = specs or _LEGACY_FIELDS
    data = parse_llm_json(text)
    if data is not None:
        return _normalize_claims(data, specs)

    # Last resort: try to extract individual fields with regex
    logger.warning("[Extract] Could not parse JSON from LLM response, "
                   "attempting field-level regex extraction")
    result = _empty_claims(specs)
    result["error"] = "parse_failed"

    # Extract list fields
    for name, ftype, _ in specs:
        if ftype != "list":
            continue
        pattern = rf'"{re.escape(name)}"\s*:\s*\[(.*?)\]'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            result[name] = items

    # Extract string fields
    for name, ftype, _ in specs:
        if ftype != "string":
            continue
        pattern = rf'"{re.escape(name)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            result[name] = m.group(1)

    # Extract boolean fields
    for name, ftype, _ in specs:
        if ftype != "boolean":
            continue
        pattern = rf'"{re.escape(name)}"\s*:\s*(true|false)'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[name] = m.group(1).lower() == "true"

    return result


def _normalize_claims(
    data: dict,
    specs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Ensure all expected fields exist with correct types."""
    specs = specs or _LEGACY_FIELDS
    result = _empty_claims(specs)
    for name, ftype, _ in specs:
        val = data.get(name)
        if ftype == "list" and isinstance(val, list):
            result[name] = [str(v) for v in val if v]
        elif ftype == "string" and isinstance(val, str):
            result[name] = val
        elif ftype == "boolean" and isinstance(val, bool):
            result[name] = val
    result["error"] = None
    return result


# ---------------------------------------------------------------------------
# Batch extraction with checkpoint/resume
# ---------------------------------------------------------------------------

def extract_all_claims(
    papers: list[dict],
    model: str,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_thinking: bool = False,
    max_tokens: int = 4096,
    research_config: dict | None = None,
    config: Any = None,
    concurrent_workers: int | None = None,
    **kw,
) -> tuple[list[dict], bool]:
    """Extract claims from all papers with checkpoint/resume.

    Papers are processed concurrently through
    ``core.checkpoint.run_sliding_window``; results are collected in input
    order, so output records are identical to a sequential run. Set
    ``concurrent_workers=1`` for local inference servers (Ollama / LM
    Studio) to avoid model contention.

    Returns (papers_with_claims, was_cancelled).
    """
    from core.checkpoint import run_sliding_window

    total = len(papers)
    logger.info("extract_all_claims: %d papers, model=%s", total, model)

    think = True if use_thinking else None

    # Resume from checkpoint
    ckpt = CheckpointManager(
        checkpoint_dir, _CHECKPOINT_FILENAME, checkpoint_every)
    extracted, start_idx = ckpt.load()
    if start_idx > 0 and start_idx <= total:
        logger.info("Resuming extraction from paper %d/%d", start_idx + 1, total)
        if progress_callback:
            progress_callback(
                f"Resuming extraction from paper {start_idx + 1}/{total}")

    # Spacing between submission of concurrent calls. The old hard-coded
    # 2.0s sleep after *every* call was pure overhead; real rate-limit
    # protection is the 429-with-backoff retry in process.llm.
    stagger_delay = float(kw.get("stagger_delay", 0.5))
    if config is not None and "stagger_delay" not in kw:
        stagger_delay = float(getattr(config, "stagger_delay", stagger_delay))
    if stagger_delay < 0:
        stagger_delay = 0.0

    _workers = concurrent_workers
    if _workers is None and config is not None:
        _workers = getattr(config, "concurrent_workers", None)
    if _workers is None:
        _workers = 4
    _workers = max(1, int(_workers))

    _timeout = getattr(config, "worker_timeout_seconds", None) or 300.0

    logger.info(
        "Extraction concurrency: workers=%d, stagger=%.1fs",
        _workers, stagger_delay,
    )

    new_records: list[dict] = []

    def _worker(paper: dict) -> tuple[dict, float]:
        title = paper.get("title", "?")[:80]
        logger.info("[Extract] Extracting claims: %s", title)
        t0 = time.time()
        try:
            claims = extract_claims(
                model,
                paper,
                think=think,
                max_tokens=max_tokens,
                research_config=research_config,
            )
        except Exception as exc:
            logger.error("[Extract] Error extracting '%s': %s", title, exc)
            claims = {**_EMPTY_CLAIMS, "error": str(exc)}
        return claims, time.time() - t0

    def _build_result(idx: int, worker_out: tuple[dict, float] | None) -> dict:
        if worker_out is None:
            logger.error("Worker timed out or failed for paper %d/%d, using empty claims", idx + 1, total)
            claims, elapsed = {**_EMPTY_CLAIMS, "error": "worker_timeout"}, 0.0
        else:
            claims, elapsed = worker_out
        record = {**papers[idx], "claims": claims}
        new_records.append(record)
        n_f = len(claims.get("findings", []))
        n_m = len(claims.get("mechanisms", []))
        n_r = len(claims.get("brain_regions", []))
        title = papers[idx].get("title", "?")[:80]
        result_msg = (
            f"[{idx + 1}/{total}] {title} → "
            f"{n_f} findings, {n_m} mechanisms, {n_r} regions ({elapsed:.1f}s)"
        )
        logger.info(result_msg)
        if progress_callback:
            progress_callback(result_msg)
        return record

    def _on_checkpoint() -> None:
        ckpt.save(extracted + new_records, total)

    new_results, was_cancelled = run_sliding_window(
        items=papers,
        worker_fn=_worker,
        result_fn=_build_result,
        start_index=start_idx,
        workers=_workers,
        stagger=stagger_delay,
        checkpoint_fn=_on_checkpoint,
        cancel_event=cancel_event,
        future_timeout=_timeout,
    )
    extracted.extend(new_results)

    # Final checkpoint if cancelled
    if was_cancelled:
        if not ckpt.force_save(extracted, total):
            logger.warning("Extraction cancelled but checkpoint save failed.")
        logger.info("Extraction cancelled: %d/%d papers processed",
                    len(extracted), total)
    else:
        # Clear checkpoint on successful completion
        ckpt.cleanup()

    logger.info("Extraction complete: %d/%d papers processed%s",
                len(extracted), total,
                " (cancelled)" if was_cancelled else "")
    return extracted, was_cancelled
