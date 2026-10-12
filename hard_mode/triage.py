"""Cheap triage for hard-mode retrieval before expensive LLM scoring."""

from __future__ import annotations

import copy
import math
import logging
import random
import re
from collections import Counter
from typing import Any

from core.config import Config
from hard_mode.load_config import CHEAP_TRIAGE_DEFAULTS
from hard_mode.fulltext import triage_text_bundle
from hard_mode.records import ensure_spine

logger = logging.getLogger(__name__)

_SEMANTIC_MODEL_CACHE: dict[str, Any] = {}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_/-]+", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
}


def _cheap_triage_cfg(hm: dict[str, Any]) -> dict[str, Any]:
    # Single-sourced from hard_mode.load_config.CHEAP_TRIAGE_DEFAULTS.
    cfg = copy.deepcopy(CHEAP_TRIAGE_DEFAULTS)
    cfg.update(hm.get("cheap_triage") or {})
    return cfg


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_score(value: Any) -> float:
    return round(max(0.0, min(1.0, _safe_float(value))), 4)


def _family_specs(hm: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for idx, family in enumerate(hm.get("query_families") or []):
        fid = str(family.get("id") or f"family_{idx}")
        label = str(family.get("label") or fid)
        terms = [str(term).strip() for term in (family.get("semantic_terms") or []) if str(term).strip()]
        query_text = " ".join(
            part
            for part in (
                label,
                " ".join(terms),
                str(family.get("pubmed_query") or "").strip(),
                str(family.get("boolean_pubmed") or "").strip(),
            )
            if part
        ).strip()
        specs.append({"id": fid, "label": label, "query_text": query_text or label})

    if specs:
        return specs

    fallback = (
        str(hm.get("natural_language_question") or "").strip()
        or str((hm.get("relevance") or {}).get("research_question") or "").strip()
        or str(hm.get("description") or "").strip()
    )
    if fallback:
        return [{"id": "global_query", "label": "Global query", "query_text": fallback}]
    return [{"id": "global_query", "label": "Global query", "query_text": "systematic review query"}]


def _resolve_embedding_model(
    config: Config,
    hm: dict[str, Any],
    override: str | None = None,
) -> str:
    if isinstance(override, str) and override.strip():
        return override.strip()
    cfg = _cheap_triage_cfg(hm)
    model_name = str(cfg.get("embedding_model") or "").strip()
    if model_name:
        return model_name
    return str(getattr(config, "embedding_model", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2")


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= 3 and token.lower() not in _STOP_WORDS
    }


def _tfidf_terms(text: str) -> list[str]:
    base = [token for token in _TOKEN_RE.findall(text.lower()) if len(token) >= 3 and token.lower() not in _STOP_WORDS]
    bigrams = [f"{base[i]} {base[i + 1]}" for i in range(len(base) - 1)]
    return base + bigrams


def _cosine_sparse(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    numer = sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())
    denom_a = math.sqrt(sum(weight * weight for weight in vec_a.values()))
    denom_b = math.sqrt(sum(weight * weight for weight in vec_b.values()))
    if denom_a == 0.0 or denom_b == 0.0:
        return 0.0
    return numer / (denom_a * denom_b)


def _token_overlap_matrix(doc_texts: list[str], family_specs: list[dict[str, str]]) -> list[list[float]]:
    doc_tokens = [_tokenize(text) for text in doc_texts]
    family_tokens = [_tokenize(spec["query_text"]) for spec in family_specs]
    matrix = [[0.0 for _ in family_specs] for _ in doc_texts]
    for i, dset in enumerate(doc_tokens):
        for j, fset in enumerate(family_tokens):
            if not dset or not fset:
                continue
            inter = len(dset & fset)
            if inter == 0:
                continue
            recall = inter / len(fset)
            jaccard = inter / len(dset | fset)
            matrix[i][j] = (0.7 * recall) + (0.3 * jaccard)
    return matrix


def _tfidf_similarity_matrix(doc_texts: list[str], family_specs: list[dict[str, str]]) -> list[list[float]]:
    all_terms = [_tfidf_terms(text) for text in [*doc_texts, *[spec["query_text"] for spec in family_specs]]]
    total_docs = len(all_terms)
    doc_freq: Counter[str] = Counter()
    for terms in all_terms:
        doc_freq.update(set(terms))

    idf = {
        term: math.log((1.0 + total_docs) / (1.0 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }

    def _vectorize(terms: list[str]) -> dict[str, float]:
        counts = Counter(terms)
        return {
            term: (1.0 + math.log(count)) * idf.get(term, 1.0)
            for term, count in counts.items()
            if count > 0
        }

    doc_vecs = [_vectorize(_tfidf_terms(text)) for text in doc_texts]
    fam_vecs = [_vectorize(_tfidf_terms(spec["query_text"])) for spec in family_specs]
    return [
        [max(0.0, min(1.0, _cosine_sparse(doc_vec, fam_vec))) for fam_vec in fam_vecs]
        for doc_vec in doc_vecs
    ]


def _semantic_similarity_matrix(
    config: Config,
    hm: dict[str, Any],
    doc_texts: list[str],
    family_specs: list[dict[str, str]],
    *,
    embedding_model_override: str | None = None,
) -> tuple[list[list[float]], bool, str]:
    cfg = _cheap_triage_cfg(hm)
    model_name = _resolve_embedding_model(config, hm, override=embedding_model_override)
    if not cfg.get("use_sentence_transformers", True):
        return [[0.0 for _ in family_specs] for _ in doc_texts], False, model_name
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        logger.info("Cheap triage: sentence-transformers unavailable, skipping semantic similarity: %s", exc)
        return [[0.0 for _ in family_specs] for _ in doc_texts], False, model_name

    try:
        model = _SEMANTIC_MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _SEMANTIC_MODEL_CACHE[model_name] = model
        doc_emb = model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        fam_emb = model.encode(
            [spec["query_text"] for spec in family_specs],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.info("Cheap triage: sentence-transformers encode failed, skipping semantic similarity: %s", exc)
        return [[0.0 for _ in family_specs] for _ in doc_texts], False, model_name

    sims: list[list[float]] = []
    for doc_vec in doc_emb:
        row: list[float] = []
        for fam_vec in fam_emb:
            score = sum(float(a) * float(b) for a, b in zip(doc_vec, fam_vec))
            row.append(max(0.0, min(1.0, score)))
        sims.append(row)
    return sims, True, model_name


def _retrieval_hit_matrix(papers: list[dict[str, Any]], family_specs: list[dict[str, str]]) -> list[list[float]]:
    family_index = {spec["id"]: idx for idx, spec in enumerate(family_specs)}
    matrix = [[0.0 for _ in family_specs] for _ in papers]
    for i, paper in enumerate(papers):
        for hit in paper.get("retrieval_hits") or []:
            idx = family_index.get(str(hit.get("query_family_id") or ""))
            if idx is not None:
                matrix[i][idx] = 1.0
    return matrix


def cheap_score_corpus(
    config: Config,
    hm: dict[str, Any],
    papers: list[dict[str, Any]],
    *,
    embedding_model_override: str | None = None,
) -> list[dict[str, Any]]:
    """Run cheap triage across the merged corpus."""
    cfg = _cheap_triage_cfg(hm)
    out = [ensure_spine(dict(paper)) for paper in papers]
    if not out:
        return []
    if not bool(cfg.get("enabled", True)):
        return out

    family_specs = _family_specs(hm)
    text_bundles = [triage_text_bundle(paper, hm) for paper in out]
    doc_texts = [bundle["text"] for bundle in text_bundles]
    tfidf_scores = _tfidf_similarity_matrix(doc_texts, family_specs)
    token_scores = _token_overlap_matrix(doc_texts, family_specs)
    semantic_scores, used_semantic, semantic_model = _semantic_similarity_matrix(
        config,
        hm,
        doc_texts,
        family_specs,
        embedding_model_override=embedding_model_override,
    )
    hit_scores = _retrieval_hit_matrix(out, family_specs)

    base_weights = {
        "semantic": _safe_float(cfg.get("semantic_weight"), 0.45),
        "tfidf": _safe_float(cfg.get("tfidf_weight"), 0.35),
        "token_overlap": _safe_float(cfg.get("token_overlap_weight"), 0.15),
        "retrieval_hit": _safe_float(cfg.get("retrieval_hit_weight"), 0.05),
    }
    if not used_semantic:
        base_weights["semantic"] = 0.0
    weight_total = sum(base_weights.values()) or 1.0

    high_thr = _safe_float(cfg.get("high_threshold"), 0.5)
    med_thr = _safe_float(cfg.get("medium_threshold"), 0.28)

    for i, paper in enumerate(out):
        family_scores: dict[str, dict[str, Any]] = {}
        best_family: dict[str, Any] | None = None
        best_composite = -1.0

        for j, family in enumerate(family_specs):
            sem = _round_score(semantic_scores[i][j])
            tfidf = _round_score(tfidf_scores[i][j])
            token = _round_score(token_scores[i][j])
            hit = _round_score(hit_scores[i][j])
            composite = _round_score(
                (
                    (base_weights["semantic"] * sem)
                    + (base_weights["tfidf"] * tfidf)
                    + (base_weights["token_overlap"] * token)
                    + (base_weights["retrieval_hit"] * hit)
                )
                / weight_total
            )
            row = {
                "label": family["label"],
                "semantic_similarity": sem,
                "tfidf_similarity": tfidf,
                "token_overlap": token,
                "retrieval_hit": hit,
                "composite": composite,
            }
            family_scores[family["id"]] = row
            if composite > best_composite:
                best_composite = composite
                best_family = {"id": family["id"], **row}

        best_family = best_family or {
            "id": family_specs[0]["id"],
            "label": family_specs[0]["label"],
            "semantic_similarity": 0.0,
            "tfidf_similarity": 0.0,
            "token_overlap": 0.0,
            "retrieval_hit": 0.0,
            "composite": 0.0,
        }

        if best_family["composite"] >= high_thr:
            band = "high"
            tier = 1
        elif best_family["composite"] >= med_thr:
            band = "medium"
            tier = 2
        else:
            band = "low"
            tier = 3

        paper["cheap"] = {
            "queue_band": band,
            "priority_tier": tier,
            "text_source": text_bundles[i]["text_source"],
            "body_chars_used": text_bundles[i]["body_chars"],
            "used_sentence_transformers": used_semantic,
            "semantic_embedding_model": semantic_model if used_semantic else None,
            "family_scores": family_scores,
            "top_family_id": best_family["id"],
            "top_family_label": best_family["label"],
            "semantic_similarity": best_family["semantic_similarity"],
            "tfidf_similarity": best_family["tfidf_similarity"],
            "token_overlap": best_family["token_overlap"],
            "retrieval_hit": best_family["retrieval_hit"],
            "composite_score": best_family["composite"],
        }
        paper.setdefault("derived", {})
        paper["derived"]["cheap_queue_band"] = band
        paper["derived"]["cheap_priority_tier"] = tier
        paper["derived"]["cheap_top_query_family_id"] = best_family["id"]
        paper["derived"]["cheap_top_query_family_label"] = best_family["label"]
        paper["derived"].setdefault("priority_tier", tier)

    return out


def compare_embedding_models(
    config: Config,
    hm: dict[str, Any],
    papers: list[dict[str, Any]],
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    sample_size: int | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Compare cheap-triage outputs for two embedding models on a sampled subset."""
    cfg = _cheap_triage_cfg(hm)
    if not bool(cfg.get("use_sentence_transformers", True)):
        raise ValueError("Cheap triage sentence-transformers is disabled in hard mode config.")

    resolved_a = _resolve_embedding_model(config, hm, override=model_a)
    resolved_b = str(model_b or cfg.get("compare_embedding_model") or "").strip()
    if not resolved_b:
        raise ValueError("Set a compare embedding model before running the subset comparison.")
    if resolved_a == resolved_b:
        raise ValueError("Primary and comparison embedding models must be different.")

    all_papers = [ensure_spine(dict(paper)) for paper in papers]
    if not all_papers:
        return {
            "sample_size": 0,
            "corpus_size": 0,
            "model_a": resolved_a,
            "model_b": resolved_b,
            "rows": [],
        }

    target = max(1, int(sample_size or cfg.get("compare_subset_size") or 100))
    seed = int(random_seed or cfg.get("compare_random_seed") or 13)
    sample = list(all_papers)
    if len(sample) > target:
        sample = random.Random(seed).sample(sample, k=target)

    triaged_a = cheap_score_corpus(
        config,
        hm,
        copy.deepcopy(sample),
        embedding_model_override=resolved_a,
    )
    triaged_b = cheap_score_corpus(
        config,
        hm,
        copy.deepcopy(sample),
        embedding_model_override=resolved_b,
    )

    used_a = any(bool((paper.get("cheap") or {}).get("used_sentence_transformers")) for paper in triaged_a)
    used_b = any(bool((paper.get("cheap") or {}).get("used_sentence_transformers")) for paper in triaged_b)
    if not used_a or not used_b:
        raise ValueError(
            "Embedding comparison requires sentence-transformers to load and encode successfully for both models."
        )

    rows: list[dict[str, Any]] = []
    band_counts_a: dict[str, int] = {}
    band_counts_b: dict[str, int] = {}
    band_matches = 0
    family_matches = 0
    abs_delta_total = 0.0

    for paper_a, paper_b in zip(triaged_a, triaged_b):
        cheap_a = paper_a.get("cheap") or {}
        cheap_b = paper_b.get("cheap") or {}
        band_a = str(cheap_a.get("queue_band") or "")
        band_b = str(cheap_b.get("queue_band") or "")
        family_a = str(cheap_a.get("top_family_id") or "")
        family_b = str(cheap_b.get("top_family_id") or "")
        score_a = _safe_float(cheap_a.get("composite_score"), 0.0)
        score_b = _safe_float(cheap_b.get("composite_score"), 0.0)
        if band_a:
            band_counts_a[band_a] = band_counts_a.get(band_a, 0) + 1
        if band_b:
            band_counts_b[band_b] = band_counts_b.get(band_b, 0) + 1
        if band_a == band_b:
            band_matches += 1
        if family_a == family_b:
            family_matches += 1
        abs_delta_total += abs(score_a - score_b)
        rows.append(
            {
                "hard_mode_uid": paper_a.get("hard_mode_uid"),
                "title": paper_a.get("title", ""),
                "year": paper_a.get("year"),
                "text_source": cheap_a.get("text_source") or cheap_b.get("text_source") or "",
                "model_a": resolved_a,
                "model_b": resolved_b,
                "band_a": band_a,
                "band_b": band_b,
                "same_band": band_a == band_b,
                "top_family_a": family_a,
                "top_family_b": family_b,
                "same_top_family": family_a == family_b,
                "score_a": _round_score(score_a),
                "score_b": _round_score(score_b),
                "score_delta": round(score_b - score_a, 4),
                "absolute_score_delta": round(abs(score_b - score_a), 4),
            }
        )

    n = len(rows) or 1
    return {
        "model_a": resolved_a,
        "model_b": resolved_b,
        "corpus_size": len(all_papers),
        "sample_size": len(rows),
        "random_seed": seed,
        "band_counts_a": band_counts_a,
        "band_counts_b": band_counts_b,
        "band_agreement": round(band_matches / n, 4),
        "top_family_agreement": round(family_matches / n, 4),
        "mean_absolute_score_delta": round(abs_delta_total / n, 4),
        "rows": rows,
    }


def _label_subset_metrics(papers: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [
        paper
        for paper in papers
        if str(((paper.get("human") or {}).get("reviewer_label") or "")).strip().lower()
        in {"relevant", "maybe", "irrelevant"}
    ]
    relevant = [
        paper
        for paper in reviewed
        if str(((paper.get("human") or {}).get("reviewer_label") or "")).strip().lower() == "relevant"
    ]
    maybe = [
        paper
        for paper in reviewed
        if str(((paper.get("human") or {}).get("reviewer_label") or "")).strip().lower() == "maybe"
    ]
    irrelevant = [
        paper
        for paper in reviewed
        if str(((paper.get("human") or {}).get("reviewer_label") or "")).strip().lower() == "irrelevant"
    ]

    def _count_band(items: list[dict[str, Any]], allowed: set[str]) -> int:
        return sum(
            1
            for paper in items
            if str(((paper.get("cheap") or {}).get("queue_band") or "")).strip().lower() in allowed
        )

    def _count_tier(items: list[dict[str, Any]], max_tier: int) -> int:
        return sum(
            1
            for paper in items
            if int(((paper.get("cheap") or {}).get("priority_tier") or 99)) <= max_tier
        )

    def _precision(items: list[dict[str, Any]], predicate) -> float | None:
        selected = [paper for paper in reviewed if predicate(paper)]
        if not selected:
            return None
        rel = sum(
            1
            for paper in selected
            if str(((paper.get("human") or {}).get("reviewer_label") or "")).strip().lower() == "relevant"
        )
        return round(rel / len(selected), 4)

    def _mean_score(items: list[dict[str, Any]]) -> float | None:
        if not items:
            return None
        total = sum(_safe_float((paper.get("cheap") or {}).get("composite_score"), 0.0) for paper in items)
        return round(total / len(items), 4)

    rel_total = len(relevant)
    return {
        "reviewed_count": len(reviewed),
        "relevant_count": rel_total,
        "maybe_count": len(maybe),
        "irrelevant_count": len(irrelevant),
        "relevant_in_high": _count_band(relevant, {"high"}),
        "relevant_in_high_or_medium": _count_band(relevant, {"high", "medium"}),
        "relevant_in_tier1": _count_tier(relevant, 1),
        "relevant_recall_high": round(_count_band(relevant, {"high"}) / rel_total, 4) if rel_total else None,
        "relevant_recall_high_or_medium": (
            round(_count_band(relevant, {"high", "medium"}) / rel_total, 4) if rel_total else None
        ),
        "relevant_recall_tier1": round(_count_tier(relevant, 1) / rel_total, 4) if rel_total else None,
        "precision_high": _precision(
            reviewed,
            lambda paper: str(((paper.get("cheap") or {}).get("queue_band") or "")).strip().lower() == "high",
        ),
        "precision_high_or_medium": _precision(
            reviewed,
            lambda paper: str(((paper.get("cheap") or {}).get("queue_band") or "")).strip().lower()
            in {"high", "medium"},
        ),
        "precision_tier1": _precision(
            reviewed,
            lambda paper: int(((paper.get("cheap") or {}).get("priority_tier") or 99)) <= 1,
        ),
        "mean_score_relevant": _mean_score(relevant),
        "mean_score_maybe": _mean_score(maybe),
        "mean_score_irrelevant": _mean_score(irrelevant),
    }


def benchmark_embedding_models_against_reviewed(
    config: Config,
    hm: dict[str, Any],
    reviewed_papers: list[dict[str, Any]],
    *,
    model_a: str | None = None,
    model_b: str | None = None,
) -> dict[str, Any]:
    """Benchmark two embedding models against saved human review labels."""
    cfg = _cheap_triage_cfg(hm)
    if not bool(cfg.get("use_sentence_transformers", True)):
        raise ValueError("Cheap triage sentence-transformers is disabled in hard mode config.")

    resolved_a = _resolve_embedding_model(config, hm, override=model_a)
    resolved_b = str(model_b or cfg.get("compare_embedding_model") or "").strip()
    if not resolved_b:
        raise ValueError("Set a compare embedding model before running the reviewed benchmark.")
    if resolved_a == resolved_b:
        raise ValueError("Primary and comparison embedding models must be different.")

    labeled = [
        ensure_spine(dict(paper))
        for paper in reviewed_papers
        if str(((paper.get("human") or {}).get("reviewer_label") or "")).strip().lower()
        in {"relevant", "maybe", "irrelevant"}
    ]
    if not labeled:
        raise ValueError("No reviewed papers with relevant/maybe/irrelevant labels were found.")

    triaged_a = cheap_score_corpus(
        config,
        hm,
        copy.deepcopy(labeled),
        embedding_model_override=resolved_a,
    )
    triaged_b = cheap_score_corpus(
        config,
        hm,
        copy.deepcopy(labeled),
        embedding_model_override=resolved_b,
    )

    used_a = any(bool((paper.get("cheap") or {}).get("used_sentence_transformers")) for paper in triaged_a)
    used_b = any(bool((paper.get("cheap") or {}).get("used_sentence_transformers")) for paper in triaged_b)
    if not used_a or not used_b:
        raise ValueError(
            "Reviewed benchmark requires sentence-transformers to load and encode successfully for both models."
        )

    summary_a = _label_subset_metrics(triaged_a)
    summary_b = _label_subset_metrics(triaged_b)

    rows: list[dict[str, Any]] = []
    for paper_a, paper_b in zip(triaged_a, triaged_b):
        human = paper_a.get("human") or {}
        cheap_a = paper_a.get("cheap") or {}
        cheap_b = paper_b.get("cheap") or {}
        rows.append(
            {
                "hard_mode_uid": paper_a.get("hard_mode_uid"),
                "title": paper_a.get("title", ""),
                "reviewer_label": human.get("reviewer_label", ""),
                "text_source": cheap_a.get("text_source") or cheap_b.get("text_source") or "",
                "band_a": str(cheap_a.get("queue_band") or ""),
                "band_b": str(cheap_b.get("queue_band") or ""),
                "same_band": str(cheap_a.get("queue_band") or "") == str(cheap_b.get("queue_band") or ""),
                "tier_a": cheap_a.get("priority_tier"),
                "tier_b": cheap_b.get("priority_tier"),
                "top_family_a": str(cheap_a.get("top_family_id") or ""),
                "top_family_b": str(cheap_b.get("top_family_id") or ""),
                "score_a": _round_score(cheap_a.get("composite_score")),
                "score_b": _round_score(cheap_b.get("composite_score")),
                "score_delta": round(
                    _safe_float(cheap_b.get("composite_score"), 0.0)
                    - _safe_float(cheap_a.get("composite_score"), 0.0),
                    4,
                ),
            }
        )

    relevant_band_improved = sum(
        1
        for row in rows
        if row["reviewer_label"] == "relevant"
        and row["band_a"] != "high"
        and row["band_b"] == "high"
    )
    relevant_band_worsened = sum(
        1
        for row in rows
        if row["reviewer_label"] == "relevant"
        and row["band_a"] == "high"
        and row["band_b"] != "high"
    )

    return {
        "model_a": resolved_a,
        "model_b": resolved_b,
        "reviewed_count": len(rows),
        "summary_a": summary_a,
        "summary_b": summary_b,
        "relevant_promoted_to_high_in_b": relevant_band_improved,
        "relevant_demoted_from_high_in_b": relevant_band_worsened,
        "rows": rows,
    }


def select_for_pass3(hm: dict[str, Any], papers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select Pass 3 candidates from cheap-triaged papers."""
    cfg = _cheap_triage_cfg(hm)
    out = [ensure_spine(dict(paper)) for paper in papers]
    if not out:
        return [], {"selected": 0, "total": 0, "by_band": {}, "fallback_all": False}
    if not bool(cfg.get("enabled", True)):
        for paper in out:
            paper.setdefault("derived", {})
            paper["derived"]["pass3_selected"] = True
            paper["derived"]["pass3_selection_reason"] = "cheap_triage_disabled"
        return out, {
            "total": len(out),
            "selected": len(out),
            "selected_ratio": 1.0,
            "by_band": {},
            "fallback_all": True,
        }

    include_bands = {str(band).strip().lower() for band in (cfg.get("pass3_include_bands") or ["high", "medium"])}
    rng = random.Random(int(cfg.get("pass3_random_seed", 13)))

    if any(not str(((paper.get("cheap") or {}).get("queue_band") or "")).strip() for paper in out):
        for paper in out:
            paper.setdefault("derived", {})
            paper["derived"]["pass3_selected"] = True
            paper["derived"]["pass3_selection_reason"] = "cheap_triage_missing"
        return out, {
            "total": len(out),
            "selected": len(out),
            "selected_ratio": 1.0,
            "by_band": {"missing": len(out)},
            "fallback_all": True,
        }

    low_candidates: list[dict[str, Any]] = []
    by_band: dict[str, int] = {}

    for paper in out:
        band = str(((paper.get("cheap") or {}).get("queue_band") or "")).strip().lower()
        by_band[band] = by_band.get(band, 0) + 1
        paper.setdefault("derived", {})
        if band in include_bands:
            paper["derived"]["pass3_selected"] = True
            paper["derived"]["pass3_selection_reason"] = f"cheap_triage_{band}"
        else:
            paper["derived"]["pass3_selected"] = False
            paper["derived"]["pass3_selection_reason"] = f"cheap_triage_skip_{band}"
            low_candidates.append(paper)

    if low_candidates:
        sample_min = max(0, int(cfg.get("pass3_low_band_min", 5)))
        sample_max = max(0, int(cfg.get("pass3_low_band_max", 25)))
        sample_frac = max(0.0, min(1.0, _safe_float(cfg.get("pass3_low_band_fraction"), 0.1)))
        target = int(round(len(low_candidates) * sample_frac))
        target = max(sample_min if len(low_candidates) > 0 else 0, target)
        target = min(len(low_candidates), sample_max if sample_max > 0 else len(low_candidates), target)
        if target > 0:
            chosen = {paper["hard_mode_uid"] for paper in rng.sample(low_candidates, k=target)}
            for paper in low_candidates:
                if paper["hard_mode_uid"] in chosen:
                    paper["derived"]["pass3_selected"] = True
                    paper["derived"]["pass3_selection_reason"] = "cheap_triage_low_sample"

    selected = sum(1 for paper in out if (paper.get("derived") or {}).get("pass3_selected"))
    stats = {
        "total": len(out),
        "selected": selected,
        "selected_ratio": round(selected / len(out), 4) if out else 0.0,
        "by_band": by_band,
        "fallback_all": False,
    }
    return out, stats


def merge_scored_subset(
    triaged_papers: list[dict[str, Any]],
    scored_subset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge Pass 3 scored papers back into the triaged corpus."""
    by_uid = {
        str((paper.get("hard_mode_uid") or "")): paper
        for paper in scored_subset
        if paper.get("hard_mode_uid")
    }
    merged: list[dict[str, Any]] = []
    for paper in triaged_papers:
        uid = str(paper.get("hard_mode_uid") or "")
        merged.append(by_uid.get(uid, paper))
    return merged
