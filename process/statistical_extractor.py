"""Programmatic statistical-result extraction for post-score review.

This module deliberately stays deterministic. It extracts high-confidence
statistical tests and nearby meta-analysis fields, and marks incomplete cases
for human review with the source paragraph preserved.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


NUMBER_RE = r"[-+\u2212]?(?:\d+\s*\.\s*\d+|\.\s*\d+|\d+)"


@dataclass(frozen=True)
class TextBlock:
    text: str
    source: str
    paragraph_index: int
    char_start: int
    section: str

    @property
    def char_end(self) -> int:
        return self.char_start + len(self.text)


TEST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "t_test",
        re.compile(
            rf"\bt\s*\(\s*(?P<df>{NUMBER_RE})\s*\)\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "f_test",
        re.compile(
            rf"\bF\s*\(\s*(?P<df1>{NUMBER_RE})\s*,\s*(?P<df2>{NUMBER_RE})\s*\)"
            rf"\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "chi_square",
        re.compile(
            f"(?:chi[- ]?square|x\\^?2|\u03c7\\s*(?:2|\\^2|\u00b2))"
            rf"\s*\(\s*(?P<df>{NUMBER_RE})\s*\)\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "z_test",
        re.compile(rf"\bz\s*=\s*(?P<stat>{NUMBER_RE})", re.IGNORECASE),
    ),
    (
        "correlation",
        re.compile(rf"\b(?:r|rho|rs)\s*=\s*(?P<stat>{NUMBER_RE})", re.IGNORECASE),
    ),
    (
        "regression_coefficient",
        re.compile(
            rf"\b(?:beta|b|coef(?:ficient)?)\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "effect_size",
        re.compile(
            rf"\b(?:Cohen['']?s\s+)?(?P<effect_type>d|g)\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "odds_or_risk_ratio",
        re.compile(rf"\b(?P<effect_type>OR|RR|HR)\s*=\s*(?P<stat>{NUMBER_RE})"),
    ),
)

P_VALUE_RE = re.compile(
    rf"\bp\s*(?P<operator><=|>=|[<>=])\s*(?P<value>{NUMBER_RE})(?:\b|(?=\s))",
    re.IGNORECASE,
)
CI_RE = re.compile(
    rf"(?P<level>9[059])%?\s*CI\s*[:=]?\s*[\[(]?"
    rf"\s*(?P<lower>{NUMBER_RE})\s*(?:,|;|\bto\b|-)\s*(?P<upper>{NUMBER_RE})",
    re.IGNORECASE,
)
SAMPLE_RE = re.compile(
    r"(?:\b(?:total\s+of|included|enrolled|sample(?:\s+of)?)\s+)?"
    r"(?:\b[Nn]\s*=\s*)?(?P<n>\d{1,6})\s+"
    r"(?P<label>participants|patients|subjects|children|adults|adolescents|"
    r"students|controls|cases|mice|rats|animals|dyads|families|respondents|"
    r"households|schools|firms|plots|sites|countries|teachers|employees)",
    re.IGNORECASE,
)
N_EQUALS_RE = re.compile(r"\b[Nn]\s*=\s*(?P<n>\d{1,6})\b")
AGE_RE = re.compile(
    rf"\bage\s*(?:mean|M)?\s*[=:]?\s*(?P<age>{NUMBER_RE})"
    rf"(?:\s*(?:years|yrs?))?(?:\s*,?\s*SD\s*=\s*(?P<sd>{NUMBER_RE}))?",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(
    r"\b(?P<a>[A-Za-z][A-Za-z0-9 /-]{2,40})\s+"
    r"(?:vs\.?|versus|compared with|compared to)\s+"
    r"(?P<b>[A-Za-z][A-Za-z0-9 /-]{2,40})",
    re.IGNORECASE,
)

SUBJECT_WORDS = (
    "participants", "patients", "subjects", "children", "adults", "adolescents",
    "students", "controls", "cases", "mice", "rats", "animals", "caregivers",
    "families", "dyads", "cohort", "sample", "respondents", "households",
    "schools", "firms", "plots", "sites", "countries", "teachers", "employees",
)


def article_id(paper: dict[str, Any], index: int = 0) -> str:
    """Return a stable best-effort article id."""
    for key in ("doi", "pmid", "s2_paper_id", "paperId", "openalex_id", "id"):
        value = str(paper.get(key, "") or "").strip()
        if value:
            return value
    title = str(paper.get("title", "") or "").strip()
    return title[:80] if title else f"paper_{index + 1}"


def text_candidates(paper: dict[str, Any]) -> list[tuple[str, str]]:
    """Return text fields in preferred extraction order."""
    candidates: list[tuple[str, str]] = []
    fields = (
        ("full_text", "full_text"),
        ("fulltext", "full_text"),
        ("extracted_text", "full_text"),
        ("text", "text"),
        ("abstract", "abstract"),
    )
    for key, label in fields:
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append((label, value.strip()))

    path_value = str(paper.get("full_text_path") or paper.get("text_path") or "").strip()
    if path_value:
        path = Path(path_value)
        if path.is_file():
            try:
                candidates.insert(0, ("full_text_file", path.read_text(encoding="utf-8")))
            except OSError:
                pass

    return candidates


def paragraph_blocks(text: str, source: str) -> list[TextBlock]:
    """Split text into reviewable paragraphs while keeping offsets."""
    blocks: list[TextBlock] = []
    if not text:
        return blocks

    matches = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S))
    if not matches:
        matches = [re.match(r".+", text, re.S)]  # type: ignore[list-item]

    section = ""
    for idx, match in enumerate(m for m in matches if m is not None):
        raw = match.group(0).strip()
        if not raw:
            continue
        if len(raw) <= 80 and re.match(r"^[A-Z][A-Za-z /-]{2,}:?$", raw):
            section = raw.rstrip(":")
            continue
        blocks.append(
            TextBlock(
                text=re.sub(r"\s+", " ", raw).strip(),
                source=source,
                paragraph_index=len(blocks),
                char_start=match.start(),
                section=section,
            )
        )

    if len(blocks) == 1 and len(blocks[0].text) > 1200:
        return sentence_window_blocks(blocks[0])
    return blocks


def sentence_window_blocks(block: TextBlock, max_sentences: int = 4) -> list[TextBlock]:
    """Break long abstracts into smaller review windows."""
    sentence_matches = list(re.finditer(r"[^.!?]+[.!?]?", block.text))
    windows: list[TextBlock] = []
    for i in range(0, len(sentence_matches), max_sentences):
        chunk = sentence_matches[i:i + max_sentences]
        if not chunk:
            continue
        text = " ".join(m.group(0).strip() for m in chunk).strip()
        if not text:
            continue
        windows.append(
            TextBlock(
                text=text,
                source=block.source,
                paragraph_index=len(windows),
                char_start=block.char_start + chunk[0].start(),
                section=block.section,
            )
        )
    return windows or [block]


def normalize_number(raw: str) -> float | None:
    raw = str(raw or "").strip()
    raw = raw.replace("\u2212", "-")
    raw = re.sub(r"\s+", "", raw)
    if raw.startswith("."):
        raw = "0" + raw
    if raw.startswith("-."):
        raw = raw.replace("-.", "-0.", 1)
    if raw.startswith("+."):
        raw = raw.replace("+.", "+0.", 1)
    try:
        return float(raw)
    except ValueError:
        return None


def _has_spaced_decimal(text: str) -> bool:
    return bool(re.search(r"[-+\u2212]?(?:\d+\s*\.\s+\d+|\.\s+\d+)", text))


def _closest_context_match(matches: list[re.Match[str]], context: str) -> re.Match[str] | None:
    """Pick the likely source match in an old centered context window."""
    if not matches:
        return None
    center = len(context) / 2
    return min(matches, key=lambda match: abs(((match.start() + match.end()) / 2) - center))


def repair_decimal_spaced_result_numbers(result: dict[str, Any]) -> dict[str, Any]:
    """Repair legacy rows where PDF spacing truncated decimals to integers.

    Earlier extraction patterns saw text such as ``p = 0. 001`` as ``p = 0``.
    The original context is preserved in the artifact, so the GUI can use this
    best-effort repair when loading older review files.
    """
    if not isinstance(result, dict):
        return result
    context = str(result.get("context") or "")
    if not context or not _has_spaced_decimal(context):
        return result

    repaired = dict(result)
    test_type = str(result.get("test_type") or "")
    selected_match = None
    for candidate_type, pattern in TEST_PATTERNS:
        if candidate_type != test_type:
            continue
        matches = list(pattern.finditer(context))
        match = _closest_context_match(matches, context)
        if match is None:
            break
        selected_match = match
        if _is_incomplete_decimal_tail(context, match.end(), match.groupdict().get("stat")):
            if result.get("statistic") == 0.0:
                repaired["statistic"] = ""
            break
        stat_value = normalize_number(match.groupdict().get("stat", ""))
        if stat_value is not None:
            repaired["statistic"] = stat_value
            repaired["test_text"] = match.group(0)
        break

    p_values = (
        parse_p_values_near(context, selected_match.start(), selected_match.end())
        if selected_match is not None
        else parse_p_values(context)
    )
    if p_values:
        repaired["p_values"] = p_values
    elif any(
        isinstance(item, dict) and item.get("value") == 0.0
        for item in (result.get("p_values") or [])
    ):
        repaired["p_values"] = []
    return repaired


def _is_incomplete_decimal_tail(text: str, end: int, raw_value: str | None = None) -> bool:
    if raw_value and "." in raw_value:
        return False
    tail = text[end:]
    return bool(re.match(r"\s*\.(?:\s*)?$", tail)) or bool(re.match(r"\s*\.\s+($|[^\d])", tail))


def _p_value_matches(text: str) -> list[dict[str, Any]]:
    values = []
    for match in P_VALUE_RE.finditer(text):
        if _is_incomplete_decimal_tail(text, match.end(), match.group("value")):
            continue
        value = normalize_number(match.group("value"))
        if value is None or not 0 <= value <= 1:
            continue
        values.append({
            "operator": match.group("operator"),
            "value": value,
            "text": match.group(0),
            "_span": match.span(),
        })
    return values


def _without_internal_spans(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "_span"} for item in values]


def parse_p_values(text: str) -> list[dict[str, Any]]:
    return _without_internal_spans(_p_value_matches(text))


def parse_p_values_near(
    text: str,
    anchor_start: int,
    anchor_end: int,
    *,
    max_distance: int = 180,
) -> list[dict[str, Any]]:
    """Return p-values nearest a parsed statistic span."""
    values = _p_value_matches(text)
    if not values:
        return []

    def distance(item: dict[str, Any]) -> int:
        p_start, p_end = item["_span"]
        if p_start <= anchor_end and anchor_start <= p_end:
            return 0
        return min(abs(p_start - anchor_end), abs(anchor_start - p_end))

    ordered = sorted(values, key=distance)
    nearby = [item for item in ordered if distance(item) <= max_distance]
    if nearby:
        return _without_internal_spans(nearby)
    return _without_internal_spans(ordered[:1])


def parse_confidence_intervals(text: str) -> list[dict[str, Any]]:
    intervals = []
    for match in CI_RE.finditer(text):
        lower = normalize_number(match.group("lower"))
        upper = normalize_number(match.group("upper"))
        if lower is None or upper is None or lower >= upper:
            continue
        intervals.append({
            "level": int(match.group("level")),
            "lower": lower,
            "upper": upper,
            "text": match.group(0),
        })
    return intervals


def parse_effect_sizes(text: str) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    patterns = (
        re.compile(
            rf"\b(?:Cohen['']?s\s+)?(?P<type>d|g)\s*=\s*(?P<value>{NUMBER_RE})",
            re.IGNORECASE,
        ),
        re.compile(rf"\b(?P<type>OR|RR|HR)\s*=\s*(?P<value>{NUMBER_RE})"),
        re.compile(rf"\b(?P<type>r|rho|rs)\s*=\s*(?P<value>{NUMBER_RE})", re.IGNORECASE),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = normalize_number(match.group("value"))
            if value is None:
                continue
            effect_type = match.group("type").lower()
            key = (effect_type, value)
            if key in seen:
                continue
            seen.add(key)
            effects.append({
                "type": effect_type,
                "value": value,
                "text": match.group(0),
            })
    return effects


def parse_subjects(text: str) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for match in SAMPLE_RE.finditer(text):
        n = int(match.group("n"))
        label = match.group("label").lower()
        if not valid_sample_size(n):
            continue
        key = (n, label)
        if key in seen:
            continue
        seen.add(key)
        subjects.append({"n": n, "label": label, "text": match.group(0)})

    for match in N_EQUALS_RE.finditer(text):
        n = int(match.group("n"))
        if not valid_sample_size(n):
            continue
        key = (n, "sample")
        if key in seen:
            continue
        seen.add(key)
        subjects.append({"n": n, "label": "sample", "text": match.group(0)})

    age_match = AGE_RE.search(text)
    if age_match:
        age = normalize_number(age_match.group("age"))
        sd = normalize_number(age_match.group("sd") or "")
        if age is not None:
            subjects.append({"age_mean": age, "age_sd": sd, "text": age_match.group(0)})

    return subjects


def valid_sample_size(n: int) -> bool:
    return 2 <= n <= 999999 and not (1900 <= n <= 2099)


def parse_comparisons(text: str) -> list[str]:
    comparisons = []
    for match in COMPARISON_RE.finditer(text):
        value = f"{match.group('a').strip()} vs {match.group('b').strip()}"
        if len(value) <= 100:
            comparisons.append(value)
    return comparisons[:3]


def infer_subject_terms(text: str) -> list[str]:
    lower = text.lower()
    return [word for word in SUBJECT_WORDS if word in lower]


def nearby_text(block: TextBlock, start: int, end: int, radius: int = 280) -> str:
    left = max(0, start - radius)
    right = min(len(block.text), end + radius)
    return block.text[left:right].strip()


def result_missing_fields(result: dict[str, Any], block_subjects: list[dict[str, Any]]) -> list[str]:
    missing = []
    if not result.get("p_values") and not result.get("confidence_intervals"):
        missing.append("p_value_or_ci")
    if not result.get("effect_sizes") and result.get("test_type") not in {"effect_size", "odds_or_risk_ratio", "correlation"}:
        missing.append("effect_size")
    if not block_subjects:
        missing.append("subjects")
    if not result.get("comparisons"):
        missing.append("comparison")
    return missing


def extract_from_block(block: TextBlock, article: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract result rows and human-review flags from one paragraph/window."""
    results: list[dict[str, Any]] = []
    review_flags: list[dict[str, Any]] = []
    block_subjects = parse_subjects(block.text)
    subject_terms = infer_subject_terms(block.text)
    comparisons = parse_comparisons(block.text)

    occupied: list[tuple[int, int]] = []
    for test_type, pattern in TEST_PATTERNS:
        for match in pattern.finditer(block.text):
            span = match.span()
            if _is_incomplete_decimal_tail(block.text, span[1], match.groupdict().get("stat")):
                continue
            if any(max(span[0], lo) < min(span[1], hi) for lo, hi in occupied):
                continue
            occupied.append(span)
            context = nearby_text(block, span[0], span[1])
            p_values = parse_p_values_near(block.text, span[0], span[1])
            intervals = parse_confidence_intervals(context)
            local_subjects = parse_subjects(context) or block_subjects
            stat_value = normalize_number(match.groupdict().get("stat", ""))
            effect_type = match.groupdict().get("effect_type") or ""
            effect_sizes = parse_effect_sizes(context)
            if test_type in {"effect_size", "odds_or_risk_ratio", "correlation"} and stat_value is not None:
                current_effect = {
                    "type": effect_type.lower() or ("r" if test_type == "correlation" else test_type),
                    "value": stat_value,
                    "text": match.group(0),
                }
                if not any(
                    effect.get("type") == current_effect["type"]
                    and effect.get("value") == current_effect["value"]
                    for effect in effect_sizes
                ):
                    effect_sizes.append(current_effect)

            dfs = {
                key: normalize_number(value)
                for key, value in match.groupdict().items()
                if key.startswith("df") and value is not None
            }
            result = {
                "test_type": test_type,
                "test_text": match.group(0),
                "statistic": stat_value,
                "degrees_of_freedom": dfs,
                "effect_sizes": effect_sizes,
                "p_values": p_values,
                "confidence_intervals": intervals,
                "sample_sizes": local_subjects,
                "subject_terms": subject_terms,
                "comparisons": parse_comparisons(context) or comparisons,
                "context": context,
                "location": {
                    "text_source": block.source,
                    "paragraph_index": block.paragraph_index,
                    "section": block.section,
                    "char_start": block.char_start + span[0],
                    "char_end": block.char_start + span[1],
                },
            }
            missing = result_missing_fields(result, local_subjects)
            result["needs_human_review"] = bool(missing)
            result["missing_fields"] = missing
            results.append(result)
            if missing:
                review_flags.append({
                    "reason": "partial_statistical_result",
                    "missing_fields": missing,
                    "test_text": result["test_text"],
                    "context": block.text,
                    "location": result["location"],
                })

    if not results and likely_result_context(block.text):
        review_flags.append({
            "reason": "result_context_without_parseable_statistic",
            "missing_fields": ["statistical_test", "effect_size", "p_value_or_ci"],
            "test_text": "",
            "context": block.text,
            "location": {
                "text_source": block.source,
                "paragraph_index": block.paragraph_index,
                "section": block.section,
                "char_start": block.char_start,
                "char_end": block.char_end,
            },
        })

    return results, review_flags


def likely_result_context(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in (
            "significant", "non-significant", "nonsignificant", "associated with",
            "correlated with", "increased", "decreased", "no difference",
            "main effect", "interaction", "predicted", "odds", "risk",
        )
    )


LLM_CONTEXT_PROMPT = """You are validating one local evidence window from a scientific paper.

You must decide whether the extracted statistic represents actual study data that can be used in evidence synthesis or meta-analysis. You are not seeing the whole paper by design. Use only the local context provided.

Paper title:
{title}

Deterministic extraction candidate:
{candidate_json}

Local context window:
{context}

Return ONLY a JSON object with these fields:
{{
  "represents_study_data": true/false,
  "usable_for_meta_analysis": true/false,
  "data_role": "primary_result|secondary_result|descriptive_sample|model_statistic|background_or_citation|not_result|unclear",
  "population": "brief population/subject description or empty string",
  "outcome": "brief outcome/measure description or empty string",
  "comparison": "groups/conditions being compared or empty string",
  "effect_direction": "positive|negative|mixed|none|unclear",
  "corrected_fields": {{
    "test_type": "corrected statistical test type or empty string",
    "test_text": "corrected source text span or empty string",
    "statistic": null,
    "effect_sizes": [],
    "sample_sizes": [],
    "confidence_intervals": [],
    "p_values": []
  }},
  "confidence": 0.0,
  "reason": "one concise sentence explaining the judgment",
  "needs_human_review": true/false
}}

If the deterministic candidate is wrong but the local context contains the correct study-data statistic, fill corrected_fields with the updated test/statistic/p/effect/N/CI values and set usable_for_meta_analysis=true when those corrected values are synthesis-ready.
If the statistic is from background literature, a citation being discussed, a model-fit-only number, or context is insufficient and no corrected study-data statistic is available in the local window, set represents_study_data=false, usable_for_meta_analysis=false, needs_human_review=true, and leave corrected_fields empty.
"""


def _jsonish(value: Any, limit: int = 1800) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def build_llm_context_prompt(title: str, candidate: dict[str, Any], context: str) -> str:
    """Build a privacy/cost-bounded prompt around one local statistic window."""
    compact_candidate = {
        "test_type": candidate.get("test_type") or candidate.get("reason"),
        "test_text": candidate.get("test_text", ""),
        "statistic": candidate.get("statistic"),
        "effect_sizes": candidate.get("effect_sizes", []),
        "p_values": candidate.get("p_values", []),
        "confidence_intervals": candidate.get("confidence_intervals", []),
        "sample_sizes": candidate.get("sample_sizes", []),
        "missing_fields": candidate.get("missing_fields", []),
        "location": candidate.get("location", {}),
    }
    return LLM_CONTEXT_PROMPT.format(
        title=title or "",
        candidate_json=_jsonish(compact_candidate, limit=2000),
        context=(context or "")[:2200],
    )


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return normalize_number(value)
    return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _normalize_corrected_effect_sizes(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        effect_type = ""
        text = "LLM corrected"
        if isinstance(item, dict):
            value = _coerce_float(item.get("value"))
            effect_type = str(item.get("type") or item.get("effect_type") or "")
            text = str(item.get("text") or item.get("raw") or text)
        else:
            value = _coerce_float(item)
        if value is not None:
            normalized.append({"type": effect_type, "value": value, "text": text})
    return normalized


def _normalize_corrected_sample_sizes(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        label = "sample"
        text = "LLM corrected"
        if isinstance(item, dict):
            n = _coerce_int(item.get("n", item.get("value", item.get("sample_size"))))
            label = str(item.get("label") or item.get("group") or label)
            text = str(item.get("text") or item.get("raw") or text)
        else:
            n = _coerce_int(item)
        if n is not None and valid_sample_size(n):
            normalized.append({"n": n, "label": label, "text": text})
    return normalized


def _normalize_corrected_confidence_intervals(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        level = 95
        text = "LLM corrected"
        if isinstance(item, dict):
            lower = _coerce_float(item.get("lower"))
            upper = _coerce_float(item.get("upper"))
            level = _coerce_int(item.get("level")) or level
            text = str(item.get("text") or item.get("raw") or text)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lower = _coerce_float(item[0])
            upper = _coerce_float(item[1])
        else:
            lower = upper = None
        if lower is not None and upper is not None and lower < upper:
            normalized.append({"level": level, "lower": lower, "upper": upper, "text": text})
    return normalized


def _normalize_corrected_p_values(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        operator = "="
        text = "LLM corrected"
        if isinstance(item, dict):
            value = _coerce_float(item.get("value", item.get("p")))
            operator = str(item.get("operator") or item.get("op") or operator)
            text = str(item.get("text") or item.get("raw") or text)
        else:
            value = _coerce_float(item)
        if operator not in {"=", "<", ">", "<=", ">="}:
            operator = "="
        if value is not None and 0 <= value <= 1:
            normalized.append({"operator": operator, "value": value, "text": text})
    return normalized


def normalize_llm_corrected_fields(corrected: dict[str, Any]) -> dict[str, Any]:
    """Normalize model-proposed corrections into the deterministic row shape."""
    statistic = _coerce_float(corrected.get("statistic", corrected.get("statistic_value")))
    return {
        "test_type": str(corrected.get("test_type") or ""),
        "test_text": str(corrected.get("test_text") or corrected.get("text") or ""),
        "statistic": statistic,
        "effect_sizes": _normalize_corrected_effect_sizes(corrected.get("effect_sizes")),
        "sample_sizes": _normalize_corrected_sample_sizes(corrected.get("sample_sizes")),
        "confidence_intervals": _normalize_corrected_confidence_intervals(
            corrected.get("confidence_intervals")
        ),
        "p_values": _normalize_corrected_p_values(corrected.get("p_values")),
    }


def has_llm_corrected_fields(review: Any) -> bool:
    if not isinstance(review, dict):
        return False
    corrected = review.get("corrected_fields")
    if not isinstance(corrected, dict):
        return False
    return (
        corrected.get("statistic") is not None
        or bool(corrected.get("test_text"))
        or any(
            corrected.get(key)
            for key in ("effect_sizes", "sample_sizes", "confidence_intervals", "p_values")
        )
    )


def apply_llm_corrected_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an extracted result with LLM-proposed fields applied."""
    repaired = repair_decimal_spaced_result_numbers(result)
    review = repaired.get("llm_context_review")
    if not has_llm_corrected_fields(review):
        return repaired

    corrected_fields = review.get("corrected_fields", {}) if isinstance(review, dict) else {}
    corrected_fields = (
        normalize_llm_corrected_fields(corrected_fields)
        if isinstance(corrected_fields, dict)
        else normalize_llm_corrected_fields({})
    )
    corrected = dict(repaired)
    if corrected_fields.get("test_type"):
        corrected["test_type"] = corrected_fields["test_type"]
    if corrected_fields.get("test_text"):
        corrected["test_text"] = corrected_fields["test_text"]
    if corrected_fields.get("statistic") is not None:
        corrected["statistic"] = corrected_fields["statistic"]
    for key in ("effect_sizes", "sample_sizes", "confidence_intervals", "p_values"):
        if corrected_fields.get(key):
            corrected[key] = corrected_fields[key]
    corrected["llm_correction_applied"] = True
    return corrected


def parse_llm_context_response(text: str) -> dict[str, Any]:
    """Parse and normalize the context-validation JSON returned by an LLM."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    data: dict[str, Any] = {}
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = {}

    corrected = data.get("corrected_fields")
    if not isinstance(corrected, dict):
        corrected = {}
    corrected = normalize_llm_corrected_fields(corrected)

    confidence = data.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    represents = bool(data.get("represents_study_data", False))
    usable = bool(data.get("usable_for_meta_analysis", False))
    needs_review = bool(data.get("needs_human_review", not (represents and usable)))

    return {
        "represents_study_data": represents,
        "usable_for_meta_analysis": usable,
        "data_role": str(data.get("data_role") or "unclear"),
        "population": str(data.get("population") or ""),
        "outcome": str(data.get("outcome") or ""),
        "comparison": str(data.get("comparison") or ""),
        "effect_direction": str(data.get("effect_direction") or "unclear"),
        "corrected_fields": corrected,
        "confidence": confidence,
        "reason": str(data.get("reason") or "No parseable LLM reason returned."),
        "needs_human_review": needs_review,
    }


def _review_targets(records: list[dict[str, Any]], include_human_flags: bool) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    targets: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for record in records:
        for result in record.get("results", []) or []:
            context = str(result.get("context") or "")
            if context.strip():
                targets.append((record, result, context))
        if include_human_flags:
            for flag in record.get("human_review", []) or []:
                context = str(flag.get("context") or "")
                if context.strip():
                    targets.append((record, flag, context))
    return targets


def apply_llm_context_reviews(
    records: list[dict[str, Any]],
    *,
    model: str,
    max_contexts: int = 80,
    include_human_flags: bool = True,
    use_thinking: bool = False,
    max_tokens: int = 900,
    progress_callback=None,
    cancel_event=None,
    chat_fn=None,
) -> dict[str, int]:
    """Ask an LLM to validate local context windows around extracted statistics."""
    if not model:
        return {"llm_reviewed": 0, "llm_supported": 0, "llm_rejected": 0, "llm_unclear": 0}

    if chat_fn is None:
        from process.llm import chat as chat_fn

    targets = _review_targets(records, include_human_flags=include_human_flags)
    if max_contexts > 0:
        targets = targets[:max_contexts]

    counts = {
        "llm_reviewed": 0,
        "llm_supported": 0,
        "llm_rejected": 0,
        "llm_unclear": 0,
        "llm_corrected": 0,
    }
    for index, (record, candidate, context) in enumerate(targets, start=1):
        if cancel_event is not None:
            is_set = cancel_event.is_set() if hasattr(cancel_event, "is_set") else bool(cancel_event)
            if is_set:
                break
        if progress_callback:
            progress_callback(
                f"[{index}/{len(targets)}] LLM context check: "
                f"{str(record.get('title') or record.get('article_id') or '')[:80]}"
            )
        prompt = build_llm_context_prompt(str(record.get("title") or ""), candidate, context)
        try:
            response = chat_fn(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                think=bool(use_thinking) or None,
            )
            review = parse_llm_context_response(response)
        except Exception as exc:
            review = {
                "represents_study_data": False,
                "usable_for_meta_analysis": False,
                "data_role": "unclear",
                "population": "",
                "outcome": "",
                "comparison": "",
                "effect_direction": "unclear",
                "corrected_fields": {
                    "test_type": "",
                    "test_text": "",
                    "statistic": None,
                    "effect_sizes": [],
                    "sample_sizes": [],
                    "confidence_intervals": [],
                    "p_values": [],
                },
                "confidence": 0.0,
                "reason": f"LLM context check failed: {exc}",
                "needs_human_review": True,
                "error": str(exc),
            }
        candidate["llm_context_review"] = review
        counts["llm_reviewed"] += 1
        if has_llm_corrected_fields(review):
            counts["llm_corrected"] = counts.get("llm_corrected", 0) + 1
        if review.get("represents_study_data") and review.get("usable_for_meta_analysis"):
            counts["llm_supported"] += 1
        elif review.get("represents_study_data") is False:
            counts["llm_rejected"] += 1
        else:
            counts["llm_unclear"] += 1

    return counts


def extract_article(paper: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Extract statistical evidence for one article."""
    aid = article_id(paper, index)
    candidates = text_candidates(paper)
    selected_source = candidates[0][0] if candidates else "none"
    selected_text = candidates[0][1] if candidates else ""
    blocks = paragraph_blocks(selected_text, selected_source)

    results: list[dict[str, Any]] = []
    review_flags: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []
    for block in blocks:
        block_results, block_flags = extract_from_block(block, paper)
        results.extend(block_results)
        review_flags.extend(block_flags)
        subjects.extend(parse_subjects(block.text))

    if not candidates:
        review_flags.append({
            "reason": "no_text_available",
            "missing_fields": ["text", "statistical_test", "subjects", "effect_size"],
            "test_text": "",
            "context": "",
            "location": {
                "text_source": "none",
                "paragraph_index": -1,
                "section": "",
                "char_start": -1,
                "char_end": -1,
            },
        })
    elif not results:
        context = blocks[0].text if blocks else selected_text[:1000]
        review_flags.append({
            "reason": "no_statistical_result_found",
            "missing_fields": ["statistical_test", "effect_size", "p_value_or_ci"],
            "test_text": "",
            "context": context,
            "location": {
                "text_source": selected_source,
                "paragraph_index": 0 if blocks else -1,
                "section": blocks[0].section if blocks else "",
                "char_start": blocks[0].char_start if blocks else 0,
                "char_end": blocks[0].char_end if blocks else min(len(selected_text), 1000),
            },
        })

    return {
        "article_id": aid,
        "title": paper.get("title", ""),
        "doi": paper.get("doi", ""),
        "pmid": paper.get("pmid", ""),
        "year": paper.get("year", ""),
        "text_source": selected_source,
        "text_length": len(selected_text),
        "subject_summary": dedupe_subjects(subjects),
        "results": results,
        "human_review": review_flags,
        "needs_human_review": bool(review_flags),
    }


def dedupe_subjects(subjects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    out = []
    for item in subjects:
        key = (item.get("n"), item.get("label"), item.get("age_mean"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def extract_statistical_evidence(
    papers: list[dict[str, Any]],
    *,
    progress_callback=None,
    cancel_event=None,
) -> dict[str, Any]:
    """Extract statistical evidence from a scored/relevant-paper list."""
    records = []
    human_review = []
    for index, paper in enumerate(papers):
        if cancel_event is not None:
            is_set = cancel_event.is_set() if hasattr(cancel_event, "is_set") else bool(cancel_event)
            if is_set:
                return {
                    "cancelled": True,
                    "records": records,
                    "human_review": human_review,
                    "summary": summarize(records, human_review),
                }
        if progress_callback:
            title = str(paper.get("title", "") or f"paper {index + 1}")[:80]
            progress_callback(f"[{index + 1}/{len(papers)}] Extracting statistics: {title}")
        record = extract_article(paper, index)
        records.append(record)
        for flag in record["human_review"]:
            human_review.append({
                "article_id": record["article_id"],
                "title": record["title"],
                **flag,
            })

    return {
        "cancelled": False,
        "records": records,
        "human_review": collect_human_review(records),
        "summary": summarize(records, human_review),
    }


def collect_human_review(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    human_review = []
    for record in records:
        for flag in record.get("human_review", []) or []:
            human_review.append({
                "article_id": record.get("article_id", ""),
                "title": record.get("title", ""),
                **flag,
            })
        for result in record.get("results", []) or []:
            review = result.get("llm_context_review")
            if isinstance(review, dict) and review.get("needs_human_review"):
                human_review.append({
                    "article_id": record.get("article_id", ""),
                    "title": record.get("title", ""),
                    "reason": "llm_context_review",
                    "missing_fields": result.get("missing_fields", []),
                    "test_text": result.get("test_text", ""),
                    "context": result.get("context", ""),
                    "location": result.get("location", {}),
                    "llm_context_review": review,
                })
    return human_review


def summarize(records: list[dict[str, Any]], human_review: list[dict[str, Any]]) -> dict[str, int]:
    result_count = sum(len(record.get("results", [])) for record in records)
    complete_count = sum(
        1
        for record in records
        for result in record.get("results", [])
        if not result.get("needs_human_review")
    )
    llm_reviews = [
        result.get("llm_context_review")
        for record in records
        for result in record.get("results", [])
        if isinstance(result.get("llm_context_review"), dict)
    ]
    return {
        "papers": len(records),
        "papers_with_results": sum(1 for record in records if record.get("results")),
        "results": result_count,
        "complete_results": complete_count,
        "human_review_items": len(human_review),
        "llm_reviewed": len(llm_reviews),
        "llm_supported": sum(
            1 for review in llm_reviews
            if review.get("represents_study_data") and review.get("usable_for_meta_analysis")
        ),
        "llm_rejected": sum(1 for review in llm_reviews if review.get("represents_study_data") is False),
        "llm_corrected": sum(1 for review in llm_reviews if has_llm_corrected_fields(review)),
    }


def metrics_for_meta_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten extracted rows into the shape consumed by meta-analysis loaders."""
    metrics = []
    for record in records:
        effect_sizes: list[float] = []
        sample_sizes: list[int] = []
        confidence_intervals: list[list[float]] = []
        p_values: list[float] = []
        tests: list[dict[str, Any]] = []

        for result in record.get("results", []):
            metric_result = repair_decimal_spaced_result_numbers(result)
            review = metric_result.get("llm_context_review")
            if isinstance(review, dict) and not review.get("usable_for_meta_analysis", False):
                continue
            if isinstance(review, dict) and has_llm_corrected_fields(review):
                metric_result = apply_llm_corrected_fields(metric_result)
            tests.append({
                "test_type": metric_result.get("test_type"),
                "test_text": metric_result.get("test_text"),
                "statistic": metric_result.get("statistic"),
                "location": metric_result.get("location"),
                "needs_human_review": metric_result.get("needs_human_review", False),
                "llm_correction_applied": bool(metric_result.get("llm_correction_applied")),
                "llm_context_review": review if isinstance(review, dict) else None,
            })
            for effect in metric_result.get("effect_sizes", []):
                value = effect.get("value")
                if isinstance(value, (int, float)):
                    effect_sizes.append(float(value))
            for sample in metric_result.get("sample_sizes", []):
                value = sample.get("n")
                if isinstance(value, int):
                    sample_sizes.append(value)
            for ci in metric_result.get("confidence_intervals", []):
                lower = ci.get("lower")
                upper = ci.get("upper")
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                    confidence_intervals.append([float(lower), float(upper)])
            for pval in metric_result.get("p_values", []):
                value = pval.get("value")
                if isinstance(value, (int, float)):
                    p_values.append(float(value))

        if effect_sizes or sample_sizes or confidence_intervals or p_values:
            metrics.append({
                "article_id": record.get("article_id", ""),
                "doi": record.get("doi", ""),
                "pmid": record.get("pmid", ""),
                "title": record.get("title", ""),
                "effect_sizes": effect_sizes,
                "sample_sizes": sorted(set(sample_sizes), reverse=True),
                "confidence_intervals": confidence_intervals,
                "p_values": p_values,
                "statistical_tests": tests,
                "needs_human_review": record.get("needs_human_review", False),
            })
    return metrics
