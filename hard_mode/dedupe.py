"""Aggressive deduplication with structured provenance merge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from acquire.deduplicator import _merge_papers, _normalize_doi
from hard_mode.records import ensure_spine, merge_hits, paper_uid

logger = logging.getLogger(__name__)

TITLE_FUZZ_THRESHOLD = 90
TITLE_AUTHOR_YEAR_FUZZ = 85


def _norm_pmid(pmid: str | None) -> str | None:
    if pmid is None:
        return None
    s = str(pmid).strip()
    return s if s else None


def _first_author(authors: list[str] | None) -> str:
    if not authors:
        return ""
    return (authors[0] or "").strip().lower()


def _deep_union(preferred: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Union nested dicts. Scalar conflicts keep the preferred record's value."""
    merged: dict[str, Any] = dict(secondary)
    for key, value in preferred.items():
        other = merged.get(key)
        if isinstance(value, dict) and isinstance(other, dict):
            merged[key] = _deep_union(value, other)
        elif isinstance(value, list) and isinstance(other, list):
            combined = list(other)
            seen = {
                json.dumps(item, sort_keys=True, default=str) if isinstance(item, (dict, list)) else str(item)
                for item in combined
            }
            for item in value:
                marker = (
                    json.dumps(item, sort_keys=True, default=str)
                    if isinstance(item, (dict, list))
                    else str(item)
                )
                if marker not in seen:
                    combined.append(item)
                    seen.add(marker)
            merged[key] = combined
        else:
            merged[key] = value
    return merged


def _merge_hard_mode(into: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    merged = _merge_papers(into, other)
    merge_hits(merged, other)
    for key in ("cheap", "machine", "human", "derived"):
        preferred = into.get(key) if isinstance(into.get(key), dict) else {}
        secondary = other.get(key) if isinstance(other.get(key), dict) else {}
        if preferred or secondary:
            merged[key] = _deep_union(preferred, secondary)
    merged["hard_mode_uid"] = paper_uid(merged)
    return merged


def dedupe_hard_mode(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DOI -> PMID -> fuzzy title -> first author + year + fuzzy title."""
    papers = [ensure_spine(dict(p)) for p in papers]
    doi_map: dict[str, int] = {}
    pmid_map: dict[str, int] = {}
    result: list[dict[str, Any]] = []

    for paper in papers:
        doi = _normalize_doi(paper.get("doi"))
        if doi and doi in doi_map:
            idx = doi_map[doi]
            result[idx] = _merge_hard_mode(result[idx], paper)
            continue
        pmid = _norm_pmid(paper.get("pmid"))
        if pmid and pmid in pmid_map:
            idx = pmid_map[pmid]
            result[idx] = _merge_hard_mode(result[idx], paper)
            continue
        if doi:
            doi_map[doi] = len(result)
        if pmid:
            pmid_map[pmid] = len(result)
        result.append(paper)

    # Fuzzy title among remainder (no doi match in this pass — already merged by doi)
    final: list[dict[str, Any]] = []
    title_index: list[tuple[str, str, str]] = []  # lower title, first_author, year
    blocks: dict[tuple[str, str], list[int]] = {}

    def _fuzzy_keys(title_lower: str, author: str, year: str) -> list[tuple[str, str]]:
        words = title_lower.split()
        prefix = words[0][:6] if words else ""
        keys: list[tuple[str, str]] = []
        if year and author:
            keys.append(("ya", f"{year}|{author[:12]}"))
        elif year:
            keys.append(("y", year))
        elif author:
            keys.append(("a", author[:12]))
        if prefix:
            keys.append(("p", prefix))
        if not keys:
            keys.append(("u", ""))
        return keys

    for paper in result:
        title = (paper.get("title") or "").strip()
        if not title:
            final.append(paper)
            title_index.append(("", "", ""))
            continue
        tl = title.lower()
        fa = _first_author(paper.get("authors"))
        yr = str(paper.get("year") or "")
        keys = _fuzzy_keys(tl, fa, yr)

        matched = False
        seen_indexes: set[int] = set()
        for key in keys:
            for i in blocks.get(key, []):
                if i in seen_indexes:
                    continue
                seen_indexes.add(i)
                et, efa, eyr = title_index[i]
                if not et:
                    continue
                ratio = fuzz.ratio(tl, et)
                if ratio >= TITLE_FUZZ_THRESHOLD or (
                    fa and efa and yr and eyr and yr == eyr and ratio >= TITLE_AUTHOR_YEAR_FUZZ
                ):
                    final[i] = _merge_hard_mode(final[i], paper)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            index = len(final)
            title_index.append((tl, fa, yr))
            final.append(paper)
            for key in keys:
                blocks.setdefault(key, []).append(index)

    logger.info("Hard-mode dedup: %d -> %d papers", len(papers), len(final))
    return final


def load_json_paper_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def merge_and_dedupe_files(paths: list[Path]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for p in paths:
        combined.extend(load_json_paper_list(p))
    return dedupe_hard_mode(combined)
