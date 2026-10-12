"""Deduplicate paper records while keeping the richest available copy."""
import logging

logger = logging.getLogger(__name__)

TITLE_SIMILARITY_THRESHOLD = 90

IDENTIFIER_FIELDS = (
    "doi",
    "pmid",
    "pmc",
    "pmcid",
    "s2_id",
    "paperId",
    "oa_id",
    "openalex_id",
    "wos_id",
    "scopus_id",
    "elicit_id",
)

RICH_METADATA_FIELDS = (
    "doi",
    "pmid",
    "pmc",
    "pmcid",
    "s2_id",
    "oa_id",
    "wos_id",
    "scopus_id",
    "elicit_id",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "journal",
    "citation_count",
    "tldr",
    "fields_of_study",
    "urls",
    "url",
)

FULL_TEXT_FIELDS = (
    "full_text",
    "fulltext",
    "full_text_path",
    "full_text_source_path",
    "full_text_source_archive_path",
)

FULL_TEXT_AVAILABILITY_FIELDS = (
    "full_text_url",
    "full_text_source_url",
    "s2_open_access_pdf_url",
    "openalex_oa_url",
    "openalex_best_oa_url",
    "openalex_best_oa_pdf_url",
    "openalex_primary_url",
    "openalex_primary_pdf_url",
    "openalex_content_url",
    "oa_url",
)

FULL_TEXT_FLAGS = (
    "has_full_text",
    "open_access",
    "openalex_is_oa",
    "openalex_any_repository_has_fulltext",
    "openalex_has_pdf_content",
    "openalex_has_grobid_xml",
)


def _normalize_doi(doi: str | None) -> str | None:
    """Lowercase, strip whitespace and common URL prefixes."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi if doi else None


def _clean_identifier(value) -> str | None:
    """Normalize generic identifier-like values for exact duplicate keys."""
    if value in (None, "", [], {}):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("https://") or text.startswith("http://"):
        text = text.rstrip("/").split("/")[-1]
    return text.lower()


def _dedupe_keys(paper: dict) -> list[str]:
    """Return exact-match keys that should collapse records into one paper."""
    keys: list[str] = []
    doi = _normalize_doi(paper.get("doi"))
    if doi:
        keys.append(f"doi:{doi}")

    canonical_fields = {
        "pmcid": "pmc",
        "paperId": "s2_id",
        "oa_id": "openalex",
        "openalex_id": "openalex",
    }
    for field in IDENTIFIER_FIELDS:
        if field == "doi":
            continue
        ident = _clean_identifier(paper.get(field))
        if ident:
            keys.append(f"{canonical_fields.get(field, field)}:{ident}")
    return keys


def _has_value(value) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _source_count(value) -> int:
    if not value:
        return 0
    if isinstance(value, list):
        return len([item for item in value if _has_value(item)])
    return len([part for part in str(value).split("+") if part.strip()])


def _full_text_rank(paper: dict) -> int:
    """Rank full-text strength: embedded text/file first, then OA availability."""
    if paper.get("text_source") == "full_text":
        return 3
    if any(_has_value(paper.get(field)) for field in FULL_TEXT_FIELDS):
        return 3
    if int(paper.get("full_text_chars") or 0) > 0:
        return 3
    if any(_has_value(paper.get(field)) for field in FULL_TEXT_AVAILABILITY_FIELDS):
        return 2
    if any(bool(paper.get(field)) for field in FULL_TEXT_FLAGS):
        return 2
    return 0


def _metadata_value_count(paper: dict) -> int:
    return sum(1 for field in RICH_METADATA_FIELDS if _has_value(paper.get(field)))


def _identifier_count(paper: dict) -> int:
    return len(_dedupe_keys(paper))


def _paper_quality_key(paper: dict) -> tuple:
    """Sort key for choosing the canonical copy of a duplicate group.

    Priority is:
    1. actual/available full text
    2. connected identifiers such as DOI, PMID, PMCID, and provider IDs
    3. richer bibliographic metadata
    4. longer abstract/full text payloads
    5. citation count and number of contributing sources
    """
    abstract_len = len(str(paper.get("abstract") or ""))
    full_text_len = len(str(paper.get("full_text") or paper.get("fulltext") or ""))
    try:
        citations = int(paper.get("citation_count") or 0)
    except (TypeError, ValueError):
        citations = 0
    return (
        _full_text_rank(paper),
        _identifier_count(paper),
        _metadata_value_count(paper),
        full_text_len,
        abstract_len,
        citations,
        _source_count(paper.get("source")),
    )


def _merge_unique_lists(existing: list, new: list) -> list:
    merged = list(existing)
    seen = {str(item).strip().lower() for item in merged if _has_value(item)}
    for item in new:
        marker = str(item).strip().lower()
        if marker and marker not in seen:
            merged.append(item)
            seen.add(marker)
    return merged


def _merge_papers(existing: dict, new: dict) -> dict:
    """Merge duplicate records, keeping the highest-quality copy as canonical."""
    if _paper_quality_key(new) > _paper_quality_key(existing):
        preferred, secondary = new, existing
    else:
        preferred, secondary = existing, new

    merged = dict(preferred)
    for key, val in secondary.items():
        if key == "source":
            old_src = merged.get("source", "")
            if not val:
                continue
            old_parts = [part for part in str(old_src).split("+") if part]
            new_parts = [part for part in str(val).split("+") if part]
            for part in new_parts:
                if part not in old_parts:
                    old_parts.append(part)
            merged["source"] = "+".join(old_parts)
        elif isinstance(merged.get(key), list) and isinstance(val, list):
            merged[key] = _merge_unique_lists(merged[key], val)
        elif isinstance(merged.get(key), dict) and isinstance(val, dict):
            merged[key] = {**val, **merged[key]}
        elif not _has_value(merged.get(key)) and _has_value(val):
            merged[key] = val
    return merged


def _title_similarity(a: str, b: str) -> float:
    """Jaccard word-set similarity x 100 (to compare against threshold)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return (len(intersection) / len(union)) * 100


_TITLE_LEADING_ARTICLES = frozenset({"the", "a", "an"})


def _block_tokens(paper: dict) -> tuple[str, str, str]:
    """Year, first-author token, and title prefix used to block fuzzy compares."""
    year = str(paper.get("year") or "").strip()
    authors = paper.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    author = ""
    if authors:
        first = str(authors[0] or "").strip().lower()
        if "," in first:
            token = first.split(",", 1)[0]
        else:
            parts = first.split()
            if len(parts) >= 2 and len(parts[-1]) <= 2:
                token = parts[0]
            else:
                token = parts[-1] if parts else ""
        author = "".join(ch for ch in token if ch.isalnum())[:12]
    title = (paper.get("title") or "").strip().lower()
    words = [w for w in title.split() if len(w) > 2 and w not in _TITLE_LEADING_ARTICLES]
    prefix = words[0][:6] if words else ""
    return year, author, prefix


def _block_keys(year: str, author: str, prefix: str) -> list[tuple[str, str, str]]:
    keys: list[tuple[str, str, str]] = []
    if year and author:
        keys.append(("ya", year, author))
    elif year:
        keys.append(("y", year, ""))
    elif author:
        keys.append(("a", "", author))
    if prefix:
        keys.append(("p", prefix, ""))
    if not keys:
        keys.append(("u", "", ""))
    return keys


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate by exact identifiers, then by fuzzy title match."""
    before = len(papers)

    # Phase 1: exact identifier clustering. This catches DOI, PMID, PMCID, and
    # provider IDs, including bridge records that connect two partial records.
    parent = list(range(before))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    seen_keys: dict[str, int] = {}
    for index, paper in enumerate(papers):
        for key in _dedupe_keys(paper):
            if key in seen_keys:
                union(index, seen_keys[key])
            else:
                seen_keys[key] = index

    grouped: dict[int, list[dict]] = {}
    group_order: list[int] = []
    for index, paper in enumerate(papers):
        root = find(index)
        if root not in grouped:
            grouped[root] = []
            group_order.append(root)
        grouped[root].append(paper)

    unique: list[dict] = []
    for root in group_order:
        group = grouped[root]
        merged = group[0]
        for paper in group[1:]:
            merged = _merge_papers(merged, paper)
        unique.append(merged)

    after_exact = len(unique)

    # Phase 2: Fuzzy title dedup, blocked by year, first author, and title prefix
    # so each paper is compared only with plausible neighbors.
    final: list[dict] = []
    seen_titles: list[tuple[str, int]] = []
    blocks: dict[tuple[str, str, str], list[int]] = {}

    for p in unique:
        title = (p.get("title") or "").strip().lower()
        if not title:
            final.append(p)
            seen_titles.append(("", len(final) - 1))
            continue

        year, author, prefix = _block_tokens(p)
        candidate_positions: list[int] = []
        seen_positions: set[int] = set()
        for key in _block_keys(year, author, prefix):
            for pos in blocks.get(key, []):
                if pos not in seen_positions:
                    seen_positions.add(pos)
                    candidate_positions.append(pos)

        match_position = None
        for pos in candidate_positions:
            seen, _idx = seen_titles[pos]
            if seen and _title_similarity(title, seen) >= TITLE_SIMILARITY_THRESHOLD:
                match_position = pos
                break
        if match_position is None:
            pos = len(final)
            seen_titles.append((title, pos))
            final.append(p)
            for key in _block_keys(year, author, prefix):
                blocks.setdefault(key, []).append(pos)
        else:
            final_idx = seen_titles[match_position][1]
            final[final_idx] = _merge_papers(final[final_idx], p)
            updated_title = (final[final_idx].get("title") or "").strip().lower()
            if updated_title:
                seen_titles[match_position] = (updated_title, final_idx)

    removed = before - len(final)
    logger.info(
        "Deduplication: %d -> %d papers (%d removed: %d by exact ID, %d by title)",
        before, len(final), removed,
        before - after_exact, after_exact - len(final),
    )
    return final
