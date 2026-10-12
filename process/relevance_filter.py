"""LLM-based relevance filtering of paper abstracts/full text."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Callable, Literal, Optional, overload

from process.llm import chat as llm_chat
from process.llm import parse_llm_json
from core.checkpoint import CheckpointManager, run_sliding_window

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_EVERY = 1
DEFAULT_BATCH_SIZE = 1
# Fallback defaults if no config passed; prefer config.concurrent_workers
_DEFAULT_WORKERS = 2
_DEFAULT_STAGGER = 2.0
_MAX_BATCH_TOTAL_CHARS = 36000
_BERT_RELEVANCE_MODEL_ALIASES = {
    # Canonical name: BRT-BERT (BERT Relevance Trained)
    "brt-bert",
    "brt-bert-relevance",
    "brtbert",
    "brt",
    # Legacy aliases — kept so existing saved configs and project files keep working.
    "housecatbert-scibert-releevance",
    "housecatbert-scibert-relevance",
    "housecatbert-scibert",
    "housecatbert/scibert",
}
_BERT_RELEVANT_LABEL_HINTS = (
    "relevant",
    "relevance",
    "positive",
    "entailment",
    "yes",
    "true",
    "support",
)
_BERT_SCORER_CACHE: dict[str, tuple[Any, Any, Any]] = {}
_BERT_SCORER_LOAD_ERRORS: dict[str, str] = {}
_BERT_SCORER_LOCK = threading.Lock()
_LOCAL_BERT_RELEVANCE_DIR = (
    Path(__file__).resolve().parent.parent / "brt-bert-relevance"
)

# Legacy checkpoint filenames (for cleanup of old-format checkpoints)
_LEGACY_FILES = ("relevance_partial.json", "relevance_progress.json")

RELEVANCE_PROMPT = """You are a research assistant. Rate this paper's relevance to the following research question on a scale of 0-5:

RESEARCH QUESTION: {hypothesis}

SCORING:
0 = Completely irrelevant
1 = Tangentially related (shares topic area but does not answer the question)
2 = Somewhat related (touches one relevant concept but weakly)
3 = Moderately relevant (addresses at least one core concept from the question)
4 = Highly relevant (directly addresses multiple key concepts in the question)
5 = Essential (centrally and explicitly answers the research question)

PAPER TITLE: {title}

TEXT SOURCE: {text_source}

PAPER TEXT: {paper_text}

Respond with ONLY valid JSON:
{{
  "score": 0,
  "quoted_span": "exact verbatim quote copied from PAPER TEXT"
}}

Rules:
- score must be a single integer 0-5
- quoted_span must be copied exactly from PAPER TEXT
- quoted_span should be short (one or two sentences)
- if no supporting quote is available, quoted_span should be an empty string"""

_EMPTY_RELEVANCE = {
    "relevance_score": 0,
    "relevance_quote": "",
    "relevance_quote_source": "abstract",
    "relevance_quote_valid": False,
}

BATCH_RELEVANCE_PROMPT = """You are a research assistant. Rate each paper's relevance to the following research question on a scale of 0-5.

RESEARCH QUESTION: {hypothesis}

SCORING:
0 = Completely irrelevant
1 = Tangentially related (shares topic area but does not answer the question)
2 = Somewhat related (touches one relevant concept but weakly)
3 = Moderately relevant (addresses at least one core concept from the question)
4 = Highly relevant (directly addresses multiple key concepts in the question)
5 = Essential (centrally and explicitly answers the research question)

Return ONLY valid JSON:
{{
  "results": [
    {{
      "id": "paper_1",
      "score": 0,
      "quoted_span": "exact verbatim quote copied from that paper's PAPER TEXT"
    }}
  ]
}}

Rules:
- return exactly one result object per paper id
- score must be a single integer 0-5
- quoted_span must be copied exactly from the matching PAPER TEXT
- quoted_span should be short (one or two sentences)
- if no supporting quote is available, quoted_span should be an empty string
- judge each paper independently

PAPERS:
{paper_blocks}"""


from process.claim_extractor import _claim_think_mode as _stage2_think_mode


def _is_provider_request_validation_error(exc: Exception) -> bool:
    """Return True when a provider rejects one specific request payload/prompt."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)

    payload = {}
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                payload = parsed.get("error") or parsed
        except Exception:
            payload = {}

    haystack = " ".join(
        str(part or "")
        for part in (
            status,
            payload.get("code") if isinstance(payload, dict) else "",
            payload.get("type") if isinstance(payload, dict) else "",
            payload.get("message") if isinstance(payload, dict) else "",
            exc,
        )
    ).lower()
    return (
        status == 400
        or "invalid_parameter" in haystack
        or "invalid_request_error" in haystack
        or "invalid_request" in haystack
    )


def _is_bert_relevance_model(model: str) -> bool:
    """Return True when relevance scoring should use a local BERT classifier."""
    name = (model or "").strip().lower()
    if name in _BERT_RELEVANCE_MODEL_ALIASES:
        return True
    if not model:
        return False
    path = Path(model).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path.exists()


def _resolve_bert_model_name(model: str) -> str:
    """Map shorthand aliases to the local relevance model path."""
    name = (model or "").strip()
    lower = name.lower()
    if lower in _BERT_RELEVANCE_MODEL_ALIASES:
        return str(_resolve_local_bert_relevance_dir())
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if path.exists():
        return str(_resolve_local_bert_relevance_dir(path))
    return str(_resolve_local_bert_relevance_dir())


def _resolve_local_bert_relevance_dir(base_dir: Path = _LOCAL_BERT_RELEVANCE_DIR) -> Path:
    base_dir = base_dir.expanduser().resolve()
    if (base_dir / "config.json").exists():
        return base_dir

    candidates = []
    if base_dir.exists():
        for child in base_dir.iterdir():
            if (
                child.is_dir()
                and child.name.startswith("checkpoint-")
                and (child / "config.json").exists()
                and (
                    (child / "model.safetensors").exists()
                    or (child / "pytorch_model.bin").exists()
                )
            ):
                try:
                    step = int(child.name.rsplit("-", 1)[1])
                except Exception:
                    step = -1
                candidates.append((step, child))

    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    return base_dir


def _get_cached_bert_scorer(model: str) -> tuple[Any, Any, Any]:
    """Load and cache tokenizer/model/device for BERT relevance scoring."""
    model_name = _resolve_bert_model_name(model)
    with _BERT_SCORER_LOCK:
        prior_error = _BERT_SCORER_LOAD_ERRORS.get(model_name)
        if prior_error:
            raise RuntimeError(
                f"BERT relevance model '{model_name}' failed to load earlier: {prior_error}"
            )

        cached = _BERT_SCORER_CACHE.get(model_name)
        if cached is not None:
            return cached

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model_obj = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except Exception as exc:
            message = (
                f"Unable to load BERT relevance model '{model_name}'. "
                "Ensure 'torch' and 'transformers' are installed and the model id is valid."
            )
            _BERT_SCORER_LOAD_ERRORS[model_name] = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(message) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_obj.to(device)
        model_obj.eval()
        cached = (tokenizer, model_obj, device)
        _BERT_SCORER_CACHE[model_name] = cached
        _BERT_SCORER_LOAD_ERRORS.pop(model_name, None)
        logger.info("Loaded BERT relevance model: %s (%s)", model_name, device)
        return cached


def _bert_relevant_label_index(model_obj: Any) -> int:
    """Pick the classifier label index that most likely means 'relevant'."""
    num_labels = int(getattr(getattr(model_obj, "config", None), "num_labels", 0) or 0)
    if num_labels <= 1:
        return 0

    id2label = getattr(getattr(model_obj, "config", None), "id2label", {}) or {}
    for idx, raw_label in id2label.items():
        label = str(raw_label).strip().lower()
        if any(hint in label for hint in _BERT_RELEVANT_LABEL_HINTS):
            try:
                return int(idx)
            except Exception:
                pass
    # Safe default for binary heads where LABEL_1 is typically "positive/relevant".
    return 1 if num_labels > 1 else 0


def _score_relevance_with_bert(
    model: str,
    paper_text: str,
    *,
    hypothesis: str = "",
) -> dict:
    """Score relevance with a sequence-classification BERT model."""
    tokenizer, model_obj, device = _get_cached_bert_scorer(model)
    question = (hypothesis or "").strip() or "the research topic"

    try:
        import torch
    except Exception as exc:  # pragma: no cover - handled by scorer loader too
        raise RuntimeError("BERT relevance scoring requires 'torch'.") from exc

    inputs = tokenizer(
        question,
        paper_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits_tensor = model_obj(**inputs).logits[0].detach().cpu()

    num_labels = int(logits_tensor.shape[0])
    rel_idx = _bert_relevant_label_index(model_obj)
    rel_idx = max(0, min(rel_idx, num_labels - 1))

    if num_labels <= 1:
        relevance_logit = float(logits_tensor[0].item())
        relevance_probability = 1.0 / (1.0 + math.exp(-relevance_logit))
    else:
        probabilities = torch.softmax(logits_tensor, dim=0)
        relevance_logit = float(logits_tensor[rel_idx].item())
        relevance_probability = float(probabilities[rel_idx].item())

    relevance_score = max(0, min(5, int(round(relevance_probability * 5.0))))
    return {
        "relevance_score": relevance_score,
        "relevance_quote": "",
        "relevance_quote_source": "bert_logits",
        "relevance_quote_valid": False,
        "relevance_logit": relevance_logit,
        "relevance_confidence": relevance_probability,
    }


def _score_relevance_text(
    model: str,
    paper: dict,
    paper_text: str,
    text_source: str,
    *,
    use_thinking: bool = True,
    max_tokens: int = 4096,
    hypothesis: str = "",
) -> dict:
    """Score one paper against a specific text payload."""
    prompt = RELEVANCE_PROMPT.format(
        hypothesis=hypothesis or "the research topic",
        title=paper.get("title", ""),
        text_source=text_source,
        paper_text=paper_text,
    )

    text = llm_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max(1, int(max_tokens)),
        think=_stage2_think_mode(model, enabled=use_thinking),
    )

    parsed = _parse_relevance_response(text)
    if parsed["relevance_score"] is None:
        logger.warning(f"Could not parse relevance score from: {text[:200]}...")
        return {**_EMPTY_RELEVANCE, "relevance_quote_source": text_source}

    quote = parsed["relevance_quote"]
    quote_valid = bool(isinstance(quote, str) and quote and quote in paper_text)
    if quote and not quote_valid:
        logger.warning("Relevance quote was not an exact substring for '%s'", paper.get("title", "?")[:80])

    return {
        "relevance_score": parsed["relevance_score"],
        "relevance_quote": quote if quote_valid else "",
        "relevance_quote_source": text_source,
        "relevance_quote_valid": quote_valid,
    }


def score_relevance(
    model: str,
    paper: dict,
    *,
    use_thinking: bool = True,
    max_tokens: int = 4096,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
    hypothesis: str = "",
) -> dict:
    """Score a single paper's relevance and capture an evidence quote."""
    paper_text, text_source = _paper_text_for_scoring(
        paper,
        abstract_chars=abstract_chars,
        fulltext_chars=fulltext_chars,
    )
    if _is_bert_relevance_model(model):
        bert_result = _score_relevance_with_bert(
            model,
            paper_text,
            hypothesis=hypothesis,
        )
        # Keep source trace aligned with actual text source used for pair scoring.
        bert_result["relevance_quote_source"] = text_source
        return bert_result

    try:
        return _score_relevance_text(
            model,
            paper,
            paper_text,
            text_source,
            use_thinking=use_thinking,
            max_tokens=max_tokens,
            hypothesis=hypothesis,
        )
    except Exception as exc:
        abstract_text = (paper.get("abstract") or "").strip()[:max(1, int(abstract_chars))]
        if (
            text_source == "full_text"
            and abstract_text
            and _is_provider_request_validation_error(exc)
        ):
            logger.warning(
                "Provider rejected full-text relevance request for '%s'; retrying with abstract only: %s",
                paper.get("title", "?")[:80],
                exc,
            )
            try:
                return _score_relevance_text(
                    model,
                    paper,
                    abstract_text,
                    "abstract",
                    use_thinking=use_thinking,
                    max_tokens=max_tokens,
                    hypothesis=hypothesis,
                )
            except Exception as retry_exc:
                if _is_provider_request_validation_error(retry_exc):
                    logger.warning(
                        "Provider still rejected abstract fallback for '%s'; skipping paper: %s",
                        paper.get("title", "?")[:80],
                        retry_exc,
                    )
                    return {**_EMPTY_RELEVANCE, "relevance_quote_source": "abstract"}
                raise

        if _is_provider_request_validation_error(exc):
            logger.warning(
                "Provider rejected relevance request for '%s'; skipping paper: %s",
                paper.get("title", "?")[:80],
                exc,
            )
            return {**_EMPTY_RELEVANCE, "relevance_quote_source": text_source}
        raise


def _paper_text_for_scoring(
    paper: dict,
    *,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
) -> tuple[str, str]:
    """Prefer full text when available, otherwise fall back to abstract."""
    from process.llm import compress_prompt_text

    full_text = (paper.get("full_text") or "").strip()
    use_full_text = int(fulltext_chars) > 0
    if full_text and use_full_text:
        # Keep prompts bounded; the opening sections usually contain the abstract,
        # intro, and methods, which are enough for coarse relevance scoring.
        # Compress first so the char budget covers more substantive content.
        return compress_prompt_text(full_text)[:max(1, int(fulltext_chars))], "full_text"

    abstract = (paper.get("abstract") or "").strip()
    return abstract[:max(1, int(abstract_chars))], "abstract"


def _batch_max_tokens(batch_size: int, max_tokens: int = 12288) -> int:
    """Return the configured max output budget for batched scoring."""
    del batch_size
    return max(1, int(max_tokens))


def _parse_batched_relevance_response(text: str) -> dict[str, dict[str, int | str]]:
    """Parse batched JSON results keyed by paper id."""
    if not text or not text.strip():
        return {}

    # Shared helper handles fences/prose around a JSON object; the legacy
    # array-shaped payload needs its own scan (parse_llm_json is object-only).
    payload = parse_llm_json(text)
    if payload is None:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                payload = None

    if payload is None:
        return {}

    if isinstance(payload, dict):
        payload = payload.get("results")
    if not isinstance(payload, list):
        return {}

    parsed: dict[str, dict[str, int | str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("id", "")).strip()
        score = item.get("score")
        quote = item.get("quoted_span") or item.get("quote") or ""
        if paper_id and isinstance(score, int) and 0 <= score <= 5:
            parsed[paper_id] = {
                "relevance_score": score,
                "relevance_quote": quote.strip() if isinstance(quote, str) else "",
            }
    return parsed


def _score_batch_payload(
    papers: list[dict],
    *,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
) -> list[dict]:
    """Prepare stable ids and bounded text for a batch scoring request."""
    payload = []
    for idx, paper in enumerate(papers, start=1):
        paper_text, text_source = _paper_text_for_scoring(
            paper,
            abstract_chars=abstract_chars,
            fulltext_chars=fulltext_chars,
        )
        payload.append({
            "id": f"paper_{idx}",
            "title": paper.get("title", ""),
            "paper_text": paper_text,
            "text_source": text_source,
        })
    return payload


def score_relevance_batch(
    model: str,
    papers: list[dict],
    *,
    use_thinking: bool = True,
    max_tokens: int = 12288,
    single_max_tokens: int = 4096,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
    hypothesis: str = "",
) -> list[dict]:
    """Score a batch of papers in one prompt; fall back per-paper if needed."""
    if _is_bert_relevance_model(model):
        return [
            score_relevance(
                model,
                paper,
                use_thinking=use_thinking,
                max_tokens=single_max_tokens,
                abstract_chars=abstract_chars,
                fulltext_chars=fulltext_chars,
                hypothesis=hypothesis,
            )
            for paper in papers
        ]

    if len(papers) <= 1:
        return [
            score_relevance(
                model,
                papers[0],
                use_thinking=use_thinking,
                max_tokens=single_max_tokens,
                abstract_chars=abstract_chars,
                fulltext_chars=fulltext_chars,
                hypothesis=hypothesis,
            )
        ] if papers else []

    payload = _score_batch_payload(
        papers,
        abstract_chars=abstract_chars,
        fulltext_chars=fulltext_chars,
    )
    paper_blocks = []
    for item in payload:
        paper_blocks.append(
            "\n".join(
                [
                    f"ID: {item['id']}",
                    f"TITLE: {item['title']}",
                    f"TEXT SOURCE: {item['text_source']}",
                    f"PAPER TEXT: {item['paper_text']}",
                    "END PAPER",
                ]
            )
        )

    prompt = BATCH_RELEVANCE_PROMPT.format(
        hypothesis=hypothesis or "the research topic",
        paper_blocks="\n\n".join(paper_blocks),
    )

    try:
        text = llm_chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=_batch_max_tokens(len(papers), max_tokens=max_tokens),
            think=_stage2_think_mode(model, enabled=use_thinking),
        )
        parsed = _parse_batched_relevance_response(text)
    except Exception as e:
        logger.warning(
            "Batch relevance scoring failed for %d papers; retrying individually: %s",
            len(papers),
            e,
        )
        parsed = {}

    results: list[dict] = []
    for item, paper in zip(payload, papers):
        parsed_item = parsed.get(item["id"])
        if parsed_item is None:
            logger.warning(
                "Batch scoring returned no usable result for %s; retrying individually",
                paper.get("title", "?")[:80],
            )
            results.append(
                score_relevance(
                    model,
                    paper,
                    use_thinking=use_thinking,
                    max_tokens=single_max_tokens,
                    abstract_chars=abstract_chars,
                    fulltext_chars=fulltext_chars,
                    hypothesis=hypothesis,
                )
            )
            continue

        quote = parsed_item["relevance_quote"]
        quote_valid = bool(quote and quote in item["paper_text"])
        if quote and not quote_valid:
            logger.warning(
                "Batch relevance quote was not an exact substring for '%s'; retrying individually",
                paper.get("title", "?")[:80],
            )
            results.append(
                score_relevance(
                    model,
                    paper,
                    use_thinking=use_thinking,
                    max_tokens=single_max_tokens,
                    abstract_chars=abstract_chars,
                    fulltext_chars=fulltext_chars,
                    hypothesis=hypothesis,
                )
            )
            continue

        results.append({
            "relevance_score": parsed_item["relevance_score"],
            "relevance_quote": quote if quote_valid else "",
            "relevance_quote_source": item["text_source"],
            "relevance_quote_valid": quote_valid,
        })

    return results


def _build_relevance_batches(
    papers: list[dict],
    start_index: int,
    batch_size: int,
    max_chars: int = _MAX_BATCH_TOTAL_CHARS,
    *,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
) -> list[list[int]]:
    """Pack contiguous paper indices into context-bounded batches."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_chars = 0

    for idx in range(start_index, len(papers)):
        paper_text, _ = _paper_text_for_scoring(
            papers[idx],
            abstract_chars=abstract_chars,
            fulltext_chars=fulltext_chars,
        )
        paper_chars = len(paper_text)
        would_overflow = current and current_chars + paper_chars > max_chars
        would_hit_limit = current and len(current) >= batch_size
        if would_overflow or would_hit_limit:
            batches.append(current)
            current = []
            current_chars = 0

        current.append(idx)
        current_chars += paper_chars

    if current:
        batches.append(current)
    return batches


def _parse_relevance_response(text: str) -> dict[str, int | str | None]:
    """Extract score and quote from the model response."""
    if not text or not text.strip():
        return {"relevance_score": None, "relevance_quote": ""}

    json_data = parse_llm_json(text)
    if json_data is not None:
        score = json_data.get("score")
        quote = json_data.get("quoted_span") or json_data.get("quote") or ""
        if isinstance(score, int) and 0 <= score <= 5:
            return {
                "relevance_score": score,
                "relevance_quote": quote.strip() if isinstance(quote, str) else "",
            }

    # Fallback: prefer end of response (conclusion) over first digit.
    tail = text.strip()[-600:]
    for pattern in (
        r"(?:score|rating|relevance)\s*[:\s]+\s*([0-5])\b",
        r"\*\*score\*\*\s*[:\s]*\s*([0-5])\b",
        r"([0-5])\s*/\s*5",
        r"([0-5])\s+out\s+of\s+5",
        r"(?:rate|assign|give)\s+(?:this\s+)?(?:a\s+)?([0-5])\b",
        r"=\s*([0-5])\b",
    ):
        match = re.search(pattern, tail, re.IGNORECASE)
        if match:
            return {
                "relevance_score": int(match.group(1)),
                "relevance_quote": _parse_relevance_quote(text),
            }

    # Fallback: last digit 0-5 in the response (thinking models often end with the score)
    digits = [int(m.group()) for m in re.finditer(r"[0-5]", tail)]
    if digits:
        return {
            "relevance_score": digits[-1],
            "relevance_quote": _parse_relevance_quote(text),
        }

    # Last resort: first digit in response tail (avoid matching prompt scale text)
    match = re.search(r"[0-5]", tail)
    if match:
        return {
            "relevance_score": int(match.group()),
            "relevance_quote": _parse_relevance_quote(text),
        }
    return {"relevance_score": None, "relevance_quote": ""}


def _parse_relevance_quote(text: str) -> str:
    """Best-effort quote extraction when JSON parsing fails."""
    for pattern in (
        r'(?:quoted_span|quote)\s*[:=]\s*"([^"]+)"',
        r"(?:quoted_span|quote)\s*[:=]\s*'([^']+)'",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


@overload
def filter_corpus(
    papers: list[dict],
    model: str,
    threshold: int = 3,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    stagger_delay: Optional[float] = None,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    hypothesis: str = "",
    concurrent_workers: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    future_timeout: float | None = None,
    use_thinking: bool = True,
    score_max_tokens: int = 4096,
    score_batch_max_tokens: int = 12288,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
    *,
    return_scored: Literal[False] = False,
    **kwargs,
) -> tuple[list[dict], bool]: ...


@overload
def filter_corpus(
    papers: list[dict],
    model: str,
    threshold: int = 3,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    stagger_delay: Optional[float] = None,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    hypothesis: str = "",
    concurrent_workers: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    future_timeout: float | None = None,
    use_thinking: bool = True,
    score_max_tokens: int = 4096,
    score_batch_max_tokens: int = 12288,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
    *,
    return_scored: Literal[True],
    **kwargs,
) -> tuple[list[dict], bool, list[dict]]: ...


def filter_corpus(
    papers: list[dict],
    model: str,
    threshold: int = 3,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    stagger_delay: Optional[float] = None,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    hypothesis: str = "",
    concurrent_workers: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    future_timeout: float | None = None,
    use_thinking: bool = True,
    score_max_tokens: int = 4096,
    score_batch_max_tokens: int = 12288,
    abstract_chars: int = 8000,
    fulltext_chars: int = 18000,
    return_scored: bool = False,
    **kwargs,
) -> tuple[list[dict], bool] | tuple[list[dict], bool, list[dict]]:
    """Filter papers by relevance score. Saves progress to checkpoint_dir so you can resume."""

    ckpt = CheckpointManager(checkpoint_dir, "relevance_checkpoint.json", checkpoint_every)
    scored, start_index = ckpt.load()
    for record in scored:
        record.setdefault("relevance_quote", "")
        record.setdefault("relevance_quote_source", "abstract")
        record.setdefault("relevance_quote_valid", False)
    current_batch: list[dict] = []

    # Also clean up any legacy two-file checkpoints from older runs
    ckpt.cleanup_legacy(*_LEGACY_FILES)

    if start_index > 0 and start_index < len(papers):
        logger.info(
            "Resuming relevance filter: %d already scored, %d remaining",
            start_index, len(papers) - start_index,
        )

    _workers = concurrent_workers if concurrent_workers is not None else _DEFAULT_WORKERS
    _stagger = stagger_delay if stagger_delay is not None else _DEFAULT_STAGGER
    _batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    _future_timeout = max(1.0, float(future_timeout or 120.0))
    _score_max_tokens = max(1, int(score_max_tokens or 4096))
    _score_batch_max_tokens = max(1, int(score_batch_max_tokens or 12288))
    _abstract_chars = max(1, int(abstract_chars or 8000))
    _fulltext_chars = 18000 if fulltext_chars is None else max(0, int(fulltext_chars))
    _using_bert = _is_bert_relevance_model(model)

    if _using_bert:
        if use_thinking:
            logger.info("BERT relevance scoring ignores use_thinking setting.")
        if _score_max_tokens != 4096 or _score_batch_max_tokens != 12288:
            logger.info("BERT relevance scoring ignores score_max_tokens/score_batch_max_tokens settings.")
        if _workers > 1:
            logger.info(
                "BERT relevance scoring runs with concurrent_workers=1 to avoid model contention; "
                "ignoring concurrent_workers=%d",
                _workers,
            )
            _workers = 1
        if _batch_size > 1:
            logger.info(
                "BERT relevance scoring ignores batch prompts; falling back to per-paper scoring "
                "(requested batch_size=%d)",
                _batch_size,
            )
            _batch_size = 1

    if _batch_size > 1:
        batches = _build_relevance_batches(
            papers,
            start_index,
            _batch_size,
            abstract_chars=_abstract_chars,
            fulltext_chars=_fulltext_chars,
        )
        logger.info(
            "Batch scoring enabled: up to %d papers/request, %d batches, max ~%d chars/request",
            _batch_size,
            len(batches),
            _MAX_BATCH_TOTAL_CHARS,
        )
        if _workers > 1:
            logger.info(
                "Batch mode runs sequentially to reduce repeated model offload churn; "
                "ignoring concurrent_workers=%d",
                _workers,
            )

        total_done = start_index
        for batch_num, batch_indices in enumerate(batches, start=1):
            if cancel_event and cancel_event.is_set():
                logger.info(
                    "Relevance filter: stop requested after %d/%d papers",
                    total_done,
                    len(papers),
                )
                ckpt.force_save(scored + current_batch, len(papers))
                all_scored = scored + current_batch
                filtered = [p for p in all_scored if p.get("relevance_score", 0) >= threshold]
                if return_scored:
                    return filtered, True, all_scored
                return filtered, True
            batch_papers = [papers[idx] for idx in batch_indices]
            batch_results = score_relevance_batch(
                model,
                batch_papers,
                use_thinking=use_thinking,
                max_tokens=_score_batch_max_tokens,
                single_max_tokens=_score_max_tokens,
                abstract_chars=_abstract_chars,
                fulltext_chars=_fulltext_chars,
                hypothesis=hypothesis,
            )

            for idx, result in zip(batch_indices, batch_results):
                if result is None:
                    result = dict(_EMPTY_RELEVANCE)
                record = {**papers[idx], **result}
                current_batch.append(record)
                total_done += 1
                ckpt.save(scored + current_batch, len(papers))
                logger.info(
                    "Scored %d/%d  (batch %d/%d, score=%s)",
                    total_done,
                    len(papers),
                    batch_num,
                    len(batches),
                    record.get("relevance_score", "?"),
                )

        scored.extend(current_batch)
        filtered = [p for p in scored if p.get("relevance_score", 0) >= threshold]
        logger.info(f"Relevance filter: {len(papers)} -> {len(filtered)} papers (threshold={threshold})")
        ckpt.cleanup()
        if return_scored:
            return filtered, False, scored
        return filtered, False

    def _worker(paper):
        return score_relevance(
            model,
            paper,
            use_thinking=use_thinking,
            max_tokens=_score_max_tokens,
            abstract_chars=_abstract_chars,
            fulltext_chars=_fulltext_chars,
            hypothesis=hypothesis,
        )

    def _build_result(idx, result):
        if result is None:
            result = dict(_EMPTY_RELEVANCE)
        record = {**papers[idx], **result}
        current_batch.append(record)
        return record

    def _on_checkpoint():
        ckpt.save(scored + current_batch, len(papers))

    def _on_progress(done, total):
        score = current_batch[-1].get("relevance_score", "?") if current_batch else "?"
        logger.info("Scored %d/%d  (score=%s)", done + start_index, total, score)

    new_results, cancelled = run_sliding_window(
        items=papers,
        worker_fn=_worker,
        result_fn=_build_result,
        start_index=start_index,
        workers=_workers,
        stagger=_stagger,
        checkpoint_fn=_on_checkpoint,
        log_fn=_on_progress,
        cancel_event=cancel_event,
        future_timeout=_future_timeout,
    )
    scored.extend(new_results)

    filtered = [p for p in scored if p.get("relevance_score", 0) >= threshold]
    logger.info(f"Relevance filter: {len(papers)} -> {len(filtered)} papers (threshold={threshold})")

    if cancelled:
        ckpt.force_save(scored, len(papers))
        logger.info(
            "Relevance filter stopped early: %d/%d papers processed",
            len(scored),
            len(papers),
        )
        if return_scored:
            return filtered, True, scored
        return filtered, True

    ckpt.cleanup()
    if return_scored:
        return filtered, False, scored
    return filtered, False
