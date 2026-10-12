"""Canonical paper spine + retrieval hit provenance."""

from __future__ import annotations

import hashlib
from typing import Any


def paper_uid(paper: dict[str, Any]) -> str:
    """Stable id from DOI, PMID, or title+year."""
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        for p in ("https://doi.org/", "http://doi.org/", "doi:"):
            if doi.startswith(p):
                doi = doi[len(p) :]
        return hashlib.sha256(f"doi:{doi}".encode()).hexdigest()[:32]
    pmid = str(paper.get("pmid") or "").strip()
    if pmid:
        return hashlib.sha256(f"pmid:{pmid}".encode()).hexdigest()[:32]
    title = (paper.get("title") or "").strip().lower()
    year = str(paper.get("year") or "")
    return hashlib.sha256(f"{title}|{year}".encode()).hexdigest()[:32]


def ensure_spine(paper: dict[str, Any]) -> dict[str, Any]:
    """Ensure core keys and retrieval_hits list exist."""
    p = dict(paper)
    p.setdefault("retrieval_hits", [])
    p.setdefault("hard_mode_uid", paper_uid(p))
    p.setdefault("cheap", {})
    p.setdefault("machine", {})
    p.setdefault("human", {})
    p.setdefault("derived", {})
    return p


def append_hit(
    paper: dict[str, Any],
    *,
    provider: str,
    query_family_id: str,
    query_text: str,
    rank: int | None = None,
) -> None:
    hit: dict[str, Any] = {
        "provider": provider,
        "query_family_id": query_family_id,
        "query_text": query_text,
    }
    if rank is not None:
        hit["rank"] = rank
    paper.setdefault("retrieval_hits", [])
    # Dedup identical hits
    key = (provider, query_family_id, query_text)
    for h in paper["retrieval_hits"]:
        if (h.get("provider"), h.get("query_family_id"), h.get("query_text")) == key:
            return
    paper["retrieval_hits"].append(hit)


def merge_hits(into: dict[str, Any], other: dict[str, Any]) -> None:
    for h in other.get("retrieval_hits") or []:
        append_hit(
            into,
            provider=str(h.get("provider", "")),
            query_family_id=str(h.get("query_family_id", "")),
            query_text=str(h.get("query_text", "")),
            rank=h.get("rank"),
        )
