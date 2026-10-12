"""Lightweight co-occurrence hints from reviewed papers (Phase 3)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


_STOP = frozenset(
    """
    the a an and or for of in on at to from with by as is are was were be been
    being this that these those it its we our their they them not no yes all
    any some more most other such into than then so if how when what which
    who will can may could should would about after also before between both
    each few further during same so than very just where while study studies
    patients results using used use new findings based data analysis model
    """.split(),
)


def _tokens(text: str) -> list[str]:
    return [
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        if t.lower() not in _STOP
    ]


def build_keyword_suggestions(
    papers: list[dict[str, Any]],
    hm: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank tokens from human-relevant papers; suggest underused terms."""
    min_n = int((hm.get("graph") or {}).get("min_reviewed_for_graph", 5))
    top_k = int((hm.get("graph") or {}).get("top_keyword_suggestions", 12))

    rel = [
        p
        for p in papers
        if (p.get("human") or {}).get("reviewer_label") == "relevant"
    ]
    if len(rel) < min_n:
        return []

    counter: Counter[str] = Counter()
    for p in rel:
        blob = f"{p.get('title','')} {p.get('abstract','')}"
        for tok in set(_tokens(blob)):
            counter[tok] += 1

    # Downweight tokens that already appear in every query family label
    family_text = " ".join(
        str(f.get("label", "")) + " " + " ".join(f.get("semantic_terms") or [])
        for f in (hm.get("query_families") or [])
    ).lower()
    scored = []
    # An optional config-provided qualifier keeps suggestions in-domain for the
    # configured review (e.g. "neuroscience" for the shipped ASD config) without
    # hardcoding any domain vocabulary here.
    suffix = str((hm.get("graph") or {}).get("query_suffix", "") or "").strip()
    for term, c in counter.most_common(200):
        if term in family_text.split():
            continue
        suggestion = f'"{term}" {suffix}'.strip() if suffix else f'"{term}"'
        scored.append({"term": term, "count": c, "suggestion": suggestion})
    return scored[:top_k]


def graph_summary_text(papers: list[dict[str, Any]], hm: dict[str, Any]) -> str:
    sug = build_keyword_suggestions(papers, hm)
    if not sug:
        return (
            "Not enough human-reviewed relevant papers for graph hints "
            f"(need {(hm.get('graph') or {}).get('min_reviewed_for_graph', 5)})."
        )
    lines = ["Suggested expansion terms (from reviewed relevant set):"]
    for s in sug:
        lines.append(f"  - {s['term']} (n={s['count']}) -> query: {s['suggestion']}")
    return "\n".join(lines)
