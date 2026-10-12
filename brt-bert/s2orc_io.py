"""Shared S2ORC parsing helpers for BRT-BERT training pipelines."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator


_SECTION_PRIORITY = (
    "abstract",
    "introduction",
    "background",
    "methods",
    "results",
    "discussion",
    "conclusion",
)


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2}


def title_jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_to_query(title: str) -> str:
    """Lightweight title → research question for synthetic triplets."""
    title = normalize_whitespace(title).rstrip(".")
    if not title:
        return ""
    if title.endswith("?"):
        return title
    lower = title.lower()
    if lower.startswith(("does ", "do ", "is ", "are ", "can ", "how ", "what ", "why ")):
        return f"{title}?"
    return f"What is the evidence regarding {title}?"


@dataclass(frozen=True)
class S2orcPassage:
    paper_id: str
    title: str
    abstract: str
    passage: str
    venue: str = ""
    year: str = ""
    source: str = ""

    def to_json(self) -> dict[str, str]:
        return asdict(self)


def _pick_paper_id(record: dict[str, Any]) -> str:
    for key in ("corpusid", "corpus_id", "paper_id", "id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    external = record.get("external_ids") or {}
    if isinstance(external, dict):
        for key in ("DOI", "doi", "PubMed", "pubmed", "MAG", "CorpusId"):
            value = external.get(key)
            if value:
                return f"{key}:{value}"
    return ""


def _extract_abstract(record: dict[str, Any]) -> str:
    for key in ("abstract", "paperAbstract", "paper_abstract"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value)
    return ""


def _section_text(section: dict[str, Any]) -> str:
    if not isinstance(section, dict):
        return ""
    parts: list[str] = []
    for key in ("text", "content", "body"):
        value = section.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(normalize_whitespace(value))
    spans = section.get("spans") or section.get("paragraphs") or []
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict):
                text = span.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(normalize_whitespace(text))
            elif isinstance(span, str) and span.strip():
                parts.append(normalize_whitespace(span))
    return "\n\n".join(parts)


def _extract_body_sections(record: dict[str, Any]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    raw_sections = record.get("sections") or record.get("body_text") or []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            name = normalize_whitespace(str(item.get("section") or item.get("heading") or ""))
            text = _section_text(item)
            if text:
                sections.append((name.lower(), text))
    body = record.get("text")
    if isinstance(body, str) and body.strip() and not sections:
        sections.append(("body", normalize_whitespace(body)))
    return sections


def _select_intro_text(sections: list[tuple[str, str]], *, max_chars: int = 4000) -> str:
    if not sections:
        return ""
    for priority in _SECTION_PRIORITY:
        for name, text in sections:
            if priority in name:
                return text[:max_chars]
    return sections[0][1][:max_chars]


def build_passage(
    record: dict[str, Any],
    *,
    include_intro: bool = False,
    intro_max_chars: int = 4000,
    source: str = "",
) -> S2orcPassage | None:
    title = normalize_whitespace(str(record.get("title") or ""))
    abstract = _extract_abstract(record)
    sections = _extract_body_sections(record)
    intro = _select_intro_text(sections, max_chars=intro_max_chars) if include_intro else ""

    parts = [p for p in (title, abstract, intro) if p]
    passage = "\n\n".join(parts).strip()
    if len(passage) < 80:
        return None

    paper_id = _pick_paper_id(record) or f"title:{hash(title) & 0xFFFFFFFF:08x}"
    venue = normalize_whitespace(str(record.get("venue") or record.get("journal") or ""))
    year = normalize_whitespace(str(record.get("year") or ""))
    return S2orcPassage(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        passage=passage,
        venue=venue,
        year=year,
        source=source,
    )


def load_record(raw: str) -> dict[str, Any]:
    return json.loads(raw)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = open
    if str(path).endswith(".gz"):
        import gzip

        opener = gzip.open  # type: ignore[assignment]
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[call-overload]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield load_record(line)


def iter_hf_s2orc(
    dataset_name: str = "AlgorithmicResearchGroup/s2orc_full",
    *,
    split: str = "train",
    streaming: bool = True,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "HuggingFace datasets is required to stream S2ORC. "
            "Install with: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    for row in ds:
        yield dict(row)


def write_passages_jsonl(path: Path, passages: Iterable[S2orcPassage]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(passage.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count
