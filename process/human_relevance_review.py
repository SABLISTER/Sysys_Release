"""Human calibration queue for relevance-screening agreement checks."""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any


ARTICLE_ID_KEYS = (
    "doi",
    "pmid",
    "s2_id",
    "s2_paper_id",
    "paperId",
    "oa_id",
    "openalex_id",
    "id",
)


def article_id(paper: dict[str, Any], index: int = 0) -> str:
    for key in ARTICLE_ID_KEYS:
        value = str(paper.get(key, "") or "").strip()
        if value:
            return value
    title = str(paper.get("title", "") or "").strip()
    return title[:120] if title else f"paper_{index + 1}"


def recommended_sample_size(corpus_size: int) -> int:
    """Pick a small human-review queue size that scales from 20 to 50."""
    if corpus_size <= 0:
        return 0
    if corpus_size < 20:
        return corpus_size
    if corpus_size <= 150:
        return 20
    if corpus_size <= 500:
        return 30
    if corpus_size <= 1500:
        return 40
    return 50


def _abstract_snippet(paper: dict[str, Any], limit: int = 420) -> str:
    text = str(
        paper.get("abstract")
        or paper.get("tldr")
        or paper.get("full_text")
        or paper.get("text")
        or ""
    ).strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _score_bin(paper: dict[str, Any]) -> str:
    score = paper.get("relevance_score")
    if not isinstance(score, (int, float)):
        return "unscored"
    if score <= 2:
        return "low"
    if score <= 4:
        return "mid"
    return "high"


def _stratified_sample(papers: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n >= len(papers):
        return list(papers)

    rng = random.Random(seed)
    bins: dict[str, list[dict[str, Any]]] = {"low": [], "mid": [], "high": [], "unscored": []}
    for paper in papers:
        bins[_score_bin(paper)].append(paper)

    non_empty = {name: rows for name, rows in bins.items() if rows}
    if not non_empty:
        return []

    selected: list[dict[str, Any]] = []
    remaining = n
    for rows in non_empty.values():
        alloc = max(1, round(len(rows) / len(papers) * n))
        alloc = min(alloc, len(rows), remaining)
        selected.extend(rng.sample(rows, alloc))
        remaining -= alloc
        if remaining <= 0:
            break

    if remaining > 0:
        selected_ids = {id(paper) for paper in selected}
        pool = [paper for paper in papers if id(paper) not in selected_ids]
        selected.extend(rng.sample(pool, min(remaining, len(pool))))
    return selected


def build_review_queue(
    papers: list[dict[str, Any]],
    *,
    sample_size: int | None = None,
    seed: int = 42,
    source_path: str = "",
) -> dict[str, Any]:
    size = recommended_sample_size(len(papers)) if not sample_size else int(sample_size)
    size = max(0, min(size, len(papers)))
    sample = _stratified_sample(papers, size, seed)

    records = []
    for index, paper in enumerate(sample):
        records.append({
            "review_id": f"human_{index + 1:03d}",
            "article_id": article_id(paper, index),
            "title": str(paper.get("title") or ""),
            "year": paper.get("year"),
            "doi": paper.get("doi"),
            "pmid": paper.get("pmid"),
            "source": paper.get("source"),
            "abstract": _abstract_snippet(paper),
            "model_score": paper.get("relevance_score"),
            "model_quote": paper.get("relevance_quote", ""),
            "human_score": paper.get("human_score"),
            "human_verified": bool(paper.get("human_verified", False)),
            "human_notes": paper.get("human_notes", ""),
        })

    return {
        "stage": "human_relevance_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": source_path,
        "corpus_size": len(papers),
        "sample_size": len(records),
        "seed": seed,
        "score_scale": "0=not relevant, 5=highly relevant",
        "records": records,
    }


def _score_from_row(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and 0 <= int(value) <= 5:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
        if 0 <= score <= 5:
            return score
    return None


def _weighted_kappa(scores_a: list[int], scores_b: list[int], max_score: int = 5) -> float:
    if not scores_a or len(scores_a) != len(scores_b):
        return 0.0
    n = len(scores_a)
    observed = sum(abs(a - b) / max_score for a, b in zip(scores_a, scores_b)) / n
    counts_a = [scores_a.count(i) for i in range(max_score + 1)]
    counts_b = [scores_b.count(i) for i in range(max_score + 1)]
    expected = 0.0
    for i, count_a in enumerate(counts_a):
        for j, count_b in enumerate(counts_b):
            expected += (count_a / n) * (count_b / n) * (abs(i - j) / max_score)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return round(1.0 - (observed / expected), 4)


def _cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    labels = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum((labels_a.count(label) / n) * (labels_b.count(label) / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 4)


def _index_scored_papers(scored_papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, paper in enumerate(scored_papers):
        key = article_id(paper, index)
        indexed[key] = paper
        title = str(paper.get("title") or "").strip()
        if title:
            indexed.setdefault(title[:120], paper)
    return indexed


def compare_human_to_model(
    review_records: list[dict[str, Any]],
    scored_papers: list[dict[str, Any]],
    *,
    threshold: int = 3,
) -> dict[str, Any]:
    scored_index = _index_scored_papers(scored_papers)
    comparisons: list[dict[str, Any]] = []

    for row in review_records:
        human_score = _score_from_row(row, "human_score")
        if human_score is None:
            continue
        key = str(row.get("article_id") or "").strip()
        matched = scored_index.get(key)
        if matched is None:
            title = str(row.get("title") or "").strip()
            matched = scored_index.get(title[:120])
        score_source = "model_output"
        if row.get("model_score_human_corrected"):
            model_score = _score_from_row(row, "model_score")
            score_source = "human_correction"
        else:
            model_score = _score_from_row(matched or row, "relevance_score")
            if model_score is None:
                model_score = _score_from_row(row, "model_score")
        if model_score is None:
            comparisons.append({
                "article_id": key,
                "title": row.get("title", ""),
                "human_score": human_score,
                "model_score": None,
                "model_score_source": score_source,
                "human_verified": bool(row.get("human_verified", False)),
                "matched": False,
            })
            continue
        comparisons.append({
            "article_id": key,
            "title": row.get("title", ""),
            "human_score": human_score,
            "model_score": model_score,
            "model_score_source": score_source,
            "human_verified": bool(row.get("human_verified", False)),
            "human_relevant": human_score >= threshold,
            "model_relevant": model_score >= threshold,
            "matched": True,
        })

    matched = [row for row in comparisons if row.get("matched")]
    human_scores = [row["human_score"] for row in matched]
    model_scores = [row["model_score"] for row in matched]
    exact = sum(1 for h, m in zip(human_scores, model_scores) if h == m)
    threshold_matches = sum(
        1
        for row in matched
        if row.get("human_relevant") == row.get("model_relevant")
    )
    human_labels = ["relevant" if score >= threshold else "not_relevant" for score in human_scores]
    model_labels = ["relevant" if score >= threshold else "not_relevant" for score in model_scores]

    return {
        "stage": "human_relevance_agreement",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "reviewed": len([row for row in review_records if _score_from_row(row, "human_score") is not None]),
        "human_verified": len(
            [row for row in review_records if bool(row.get("human_verified", False))]
        ),
        "matched": len(matched),
        "unmatched": len(comparisons) - len(matched),
        "exact_agreement_pct": round(100 * exact / len(matched), 2) if matched else 0.0,
        "threshold_agreement_pct": round(100 * threshold_matches / len(matched), 2) if matched else 0.0,
        "score_weighted_kappa": _weighted_kappa(human_scores, model_scores),
        "threshold_kappa": _cohens_kappa(human_labels, model_labels),
        "comparisons": comparisons,
    }
