"""Stage 5: Span audit & quality filtering.

Cross-validates evidence spans against abstracts using similarity metrics,
then filters out papers with ungrounded claims.  No LLM calls — pure
similarity computation.

Optional packages (graceful degradation):
  - sentence_transformers  → semantic similarity   (else 0)
  - sklearn                → TF-IDF cosine         (else token Jaccard)
  - rapidfuzz              → deep fuzzy analysis    (else skip Phase 2)
"""
import csv
import json
import logging
import math
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from core.io import atomic_write_json
from core.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Optional dependency probes ────────────────────────────────────────────
# These probes deliberately catch ``Exception`` rather than ``ImportError``.
# A *broken* optional dependency (wrong version, missing distribution
# metadata, failed native extension) raises something other than ImportError
# during import, and that would otherwise propagate and make this module —
# and therefore the whole ``stages`` package — unimportable, defeating the
# graceful degradation promised in the module docstring.
_HAS_SENTENCE_TRANSFORMERS = False
SentenceTransformer = None
st_util = None
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception as _exc:  # pragma: no cover - depends on local environment
    logger.warning(
        "sentence_transformers unavailable (%s: %s) — semantic similarity disabled",
        type(_exc).__name__, _exc,
    )

_HAS_RAPIDFUZZ = False
rfuzz = None
try:
    from rapidfuzz import fuzz as rfuzz
    _HAS_RAPIDFUZZ = True
except Exception as _exc:  # pragma: no cover - depends on local environment
    logger.warning(
        "rapidfuzz unavailable (%s: %s) — Phase 2 deep analysis will be skipped",
        type(_exc).__name__, _exc,
    )

_HAS_SKLEARN = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _HAS_SKLEARN = True
except Exception as _exc:  # pragma: no cover - depends on local environment
    logger.warning(
        "sklearn unavailable (%s: %s)", type(_exc).__name__, _exc,
    )


# ── Utility helpers ───────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens (alpha-only, len >= 2)."""
    return {w for w in re.findall(r"[a-z]{2,}", text.lower())}


def _token_similarity_sets(ta: set[str], tb: set[str]) -> float:
    """Jaccard similarity between two pre-tokenized sets."""
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# TF-IDF term pattern — matches scikit-learn's default ``token_pattern``.
_TFIDF_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Inverse document frequency over a two-document collection with
# ``smooth_idf=True``: idf(t) = ln((1 + n) / (1 + df)) + 1 for n = 2.
_IDF_IN_BOTH = 1.0                       # df = 2 → ln(3/3) + 1
_IDF_IN_ONE = math.log(3.0 / 2.0) + 1.0  # df = 1 → ln(3/2) + 1


def _tfidf_counts(text: str) -> Counter:
    """Term-frequency counts using scikit-learn's default tokenization."""
    return Counter(_TFIDF_TOKEN_RE.findall(text.lower()))


def _tfidf_similarity_counts(ca: Counter, cb: Counter) -> float:
    """Exact TF-IDF cosine between two documents, from precomputed counts.

    Reproduces ``TfidfVectorizer().fit_transform([a, b])`` followed by cosine
    similarity, but without constructing a vectorizer per call. Because the
    collection is exactly the two input documents, every term's document
    frequency is either 1 or 2, so the idf values collapse to two constants
    and the whole score can be computed in a single pass over the smaller
    counter.

    This also removes the previous dependency on scikit-learn for this metric,
    so composite audit scores no longer change based on whether sklearn
    happens to be installed.
    """
    if not ca or not cb:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for term, tf in ca.items():
        weight = tf * (_IDF_IN_BOTH if term in cb else _IDF_IN_ONE)
        norm_a += weight * weight
        if term in cb:
            dot += weight * (cb[term] * _IDF_IN_BOTH)
    for term, tf in cb.items():
        weight = tf * (_IDF_IN_BOTH if term in ca else _IDF_IN_ONE)
        norm_b += weight * weight
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _tfidf_similarity(a: str, b: str) -> float:
    """TF-IDF cosine similarity between two strings."""
    if not a.strip() or not b.strip():
        return 0.0
    return _tfidf_similarity_counts(_tfidf_counts(a), _tfidf_counts(b))


def _word_error_rate(ref: list[str], hyp: list[str]) -> float:
    """Levenshtein-based word error rate (WER)."""
    if not ref:
        return 1.0 if hyp else 0.0
    n = len(ref)
    m = len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)
    return d[n][m] / n


def _write_audit_csv(path: Path, rows: list[dict]) -> None:
    """Write Phase 1 span similarity audit CSV."""
    fieldnames = [
        "paper_id", "title", "span_field", "semantic_sim", "token_sim",
        "tfidf_sim", "composite", "priority",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_deep_csv(path: Path, rows: list[dict]) -> None:
    """Write Phase 2 deep analysis CSV."""
    fieldnames = [
        "paper_id", "title", "span_field", "composite", "priority",
        "fuzzy_partial", "fuzzy_sort", "wer", "reclassified",
        "reclassification_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_updated_csv(path: Path, rows: list[dict]) -> None:
    """Write Phase 3 merged/updated audit CSV."""
    fieldnames = [
        "paper_id", "title", "span_field", "semantic_sim", "token_sim",
        "tfidf_sim", "composite", "priority", "reclassified",
        "reclassification_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ── Paper identity ────────────────────────────────────────────────────────

def _derive_paper_id(paper: dict, index: int) -> str:
    """Build a stable identifier for a paper that lacks a usable one.

    Prefers real bibliographic identifiers so the same paper keeps the same
    id across runs, and only falls back to positional numbering when nothing
    identifying is present.
    """
    for key in ("doi", "pmid", "pmcid"):
        value = str(paper.get(key) or "").strip()
        if value:
            return f"{key}:{value.lower()}"
    title = str(paper.get("title") or "").strip().lower()
    if title:
        year = str(paper.get("year") or paper.get("publication_year") or "")
        digest = zlib.crc32(f"{title}|{year}".encode("utf-8"))
        return f"title:{digest:08x}"
    return f"index:{index}"


def _normalize_paper_ids(claims: list[dict], profiles: list[dict]) -> int:
    """Ensure every paper has a unique, non-empty ``paper_id``, in place.

    Stage 5 keys audit rows, per-paper score aggregation, and the bottom-third
    filter on ``paper_id``. Blank or duplicated ids therefore cause one paper's
    audit rows to overwrite another's, so the filter can remove or keep the
    wrong records. Normalizing once here means Phase 1 and Phase 4 cannot
    disagree about identity.

    Returns the number of records whose id had to be repaired.
    """
    seen: set[str] = set()
    repaired = 0
    id_remap: dict[int, str] = {}

    for index, paper in enumerate(claims):
        original = str(paper.get("paper_id") or paper.get("id") or "").strip()
        pid = original or _derive_paper_id(paper, index)
        if pid in seen:
            suffix = 2
            while f"{pid}#{suffix}" in seen:
                suffix += 1
            pid = f"{pid}#{suffix}"
        if pid != original:
            repaired += 1
        seen.add(pid)
        if original:
            id_remap.setdefault(id(paper), pid)
        paper["paper_id"] = pid

    # Profiles are looked up by the same key; keep any that already agree and
    # only fill in blanks so we do not break an existing valid association.
    for index, profile in enumerate(profiles):
        original = str(profile.get("paper_id") or profile.get("id") or "").strip()
        if not original:
            profile["paper_id"] = _derive_paper_id(profile, index)

    if repaired:
        logger.warning(
            "  Repaired %d paper_id value(s) that were missing or duplicated; "
            "audit aggregation and the bottom-third filter would otherwise have "
            "merged distinct papers", repaired,
        )
    return repaired


# ── Phase 1: Span similarity scoring ─────────────────────────────────────

_SPAN_MODEL_NAME = DEFAULT_EMBEDDING_MODEL
# Sentinel distinguishes "never attempted" from "attempted and failed", so a
# failed load is not retried on every call.
_span_model_cache: list = [None, False]  # [model, attempted]


def _get_span_model():
    """Return the cached span-similarity model, loading it at most once.

    The model was previously constructed on every ``_compute_span_similarities``
    call. Loading ``all-MiniLM-L6-v2`` costs seconds and hundreds of megabytes,
    so repeated Stage 5 runs in a single GUI session paid that cost each time.
    """
    model, attempted = _span_model_cache
    if attempted:
        return model
    _span_model_cache[1] = True
    if not _HAS_SENTENCE_TRANSFORMERS:
        return None
    assert SentenceTransformer is not None  # guaranteed by the _HAS_SENTENCE_TRANSFORMERS check above
    try:
        model = SentenceTransformer(_SPAN_MODEL_NAME)
        _span_model_cache[0] = model
        logger.info("  Loaded sentence-transformers model: %s", _SPAN_MODEL_NAME)
        return model
    except Exception as exc:
        logger.warning("  Failed to load sentence-transformers model: %s", exc)
        return None


def _compute_span_similarities(
    claims: list[dict],
    profiles: list[dict],
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Any = None,
) -> list[dict]:
    """Compute similarity between evidence spans and abstract text.

    For each paper, compare key_quote and relevance_quote against abstract.
    Returns a list of audit rows (one per span per paper).
    """
    # Build profile lookup
    profile_map: dict[str, dict] = {}
    for p in profiles:
        pid = p.get("paper_id", p.get("id", ""))
        if pid:
            profile_map[pid] = p

    # Load sentence-transformers model if available (process-wide cache)
    st_model = _get_span_model()

    audit_rows: list[dict] = []
    total = len(claims)
    span_fields = ["key_quote", "relevance_quote"]

    for idx, paper in enumerate(claims):
        if cancel_event is not None:
            is_set = (cancel_event.is_set()
                      if hasattr(cancel_event, "is_set") else bool(cancel_event))
            if is_set:
                logger.info("Phase 1 cancelled at %d/%d", idx + 1, total)
                break

        pid = paper.get("paper_id", paper.get("id", ""))
        title = paper.get("title", "?")[:80]
        abstract = paper.get("abstract", "")
        if not abstract:
            continue

        # Gather spans to check. Stage 3 nests extracted fields under
        # paper["claims"], so fall back to it when the top level is blank.
        nested = paper.get("claims")
        nested = nested if isinstance(nested, dict) else {}
        spans_to_check: list[tuple[str, str]] = []
        for field in span_fields:
            val = paper.get(field) or nested.get(field) or ""
            if val:
                spans_to_check.append((field, val))
        # Also check method_profile evidence_spans from profiles
        prof = profile_map.get(pid, {})
        mp = prof.get("method_profile", {})
        for es in mp.get("evidence_spans", []):
            if isinstance(es, dict):
                txt = es.get("text", es.get("quote", ""))
            else:
                txt = str(es)
            if txt:
                spans_to_check.append(("evidence_span", txt))

        if not spans_to_check:
            continue

        # Precompute the abstract side of every metric once per paper. All
        # spans of a paper are scored against the same abstract, so tokenizing
        # and embedding it per span repeated identical work N_spans times.
        abstract_tokens = _tokenize(abstract)
        abstract_counts = _tfidf_counts(abstract)
        span_embeddings = None
        abstract_embedding = None
        if st_model is not None:
            try:
                # One batched encode per paper instead of one 2-item encode
                # per span; batching is also much faster on GPU/MPS.
                encoded = st_model.encode(
                    [t for _, t in spans_to_check] + [abstract],
                    convert_to_tensor=True,
                )
                span_embeddings = encoded[:-1]
                abstract_embedding = encoded[-1]
            except Exception:
                span_embeddings = None
                abstract_embedding = None

        for span_idx, (span_field, span_text) in enumerate(spans_to_check):
            # 1. Semantic similarity
            sem_sim = 0.0
            if span_embeddings is not None and abstract_embedding is not None:
                assert st_util is not None  # embeddings only get set when sentence_transformers loaded
                try:
                    sem_sim = float(
                        st_util.cos_sim(
                            span_embeddings[span_idx], abstract_embedding
                        ).item()
                    )
                except Exception:
                    sem_sim = 0.0

            # 2. Token Jaccard
            tok_sim = _token_similarity_sets(_tokenize(span_text), abstract_tokens)

            # 3. TF-IDF cosine
            tfidf_sim = _tfidf_similarity_counts(
                _tfidf_counts(span_text), abstract_counts
            )

            # Composite — adjust weights when semantic unavailable
            if st_model is not None:
                composite = 0.4 * sem_sim + 0.3 * tok_sim + 0.3 * tfidf_sim
            else:
                # No semantic: reweight to 0.5 token + 0.5 tfidf
                composite = 0.5 * tok_sim + 0.5 * tfidf_sim

            # Priority assignment
            if (sem_sim < 0.5 and tok_sim < 0.3) or composite < 0.35:
                priority = "P1"
            elif composite < 0.55:
                priority = "P2"
            else:
                priority = "P3"

            audit_rows.append({
                "paper_id": pid,
                "title": title,
                "span_field": span_field,
                "span_text": span_text,
                "abstract": abstract,
                "semantic_sim": round(sem_sim, 4),
                "token_sim": round(tok_sim, 4),
                "tfidf_sim": round(tfidf_sim, 4),
                "composite": round(composite, 4),
                "priority": priority,
            })

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            msg = f"Phase 1: Scored {idx + 1}/{total} papers"
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

    # Log distribution summary
    p_counts = {"P1": 0, "P2": 0, "P3": 0}
    composites = []
    for r in audit_rows:
        p_counts[r["priority"]] = p_counts.get(r["priority"], 0) + 1
        composites.append(r["composite"])

    if composites:
        composites.sort()
        med = composites[len(composites) // 2]
        logger.info("  Similarity distribution: min=%.4f median=%.4f max=%.4f",
                    composites[0], med, composites[-1])
    logger.info("  Priority counts: P1=%d, P2=%d, P3=%d",
                p_counts["P1"], p_counts["P2"], p_counts["P3"])

    return audit_rows


# ── Phase 2: Deep fuzzy analysis ─────────────────────────────────────────

def _deep_analysis(
    audit_rows: list[dict],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Run deeper analysis on P1/P2 flagged spans.

    Uses rapidfuzz (partial_ratio, token_sort_ratio) + WER.
    Reclassifies spans that pass thresholds as likely_valid.
    """
    flagged = [r for r in audit_rows if r["priority"] in ("P1", "P2")]
    if not flagged:
        logger.info("Phase 2: No P1/P2 spans to analyse")
        return []

    logger.info("Phase 2: Deep analysis on %d flagged spans "
                "(rapidfuzz=%s)", len(flagged), _HAS_RAPIDFUZZ)

    deep_rows: list[dict] = []

    for idx, row in enumerate(flagged):
        span_text = row.get("span_text", "")
        abstract = row.get("abstract", "")

        fuzzy_partial = 0.0
        fuzzy_sort = 0.0
        if _HAS_RAPIDFUZZ and span_text and abstract:
            assert rfuzz is not None  # guaranteed by the _HAS_RAPIDFUZZ check above
            try:
                fuzzy_partial = rfuzz.partial_ratio(span_text, abstract)
                fuzzy_sort = rfuzz.token_sort_ratio(span_text, abstract)
            except Exception:
                pass

        # Word error rate
        ref_tokens = re.findall(r"\w+", span_text.lower())
        hyp_tokens = re.findall(r"\w+", abstract.lower())
        wer = _word_error_rate(ref_tokens, hyp_tokens) if ref_tokens else 1.0

        # Reclassification logic
        reclassified = False
        reason = ""

        if _HAS_RAPIDFUZZ and fuzzy_partial >= 95:
            reclassified = True
            reason = "near_substring"
        elif _HAS_RAPIDFUZZ and fuzzy_sort >= 85 and wer < 0.3:
            reclassified = True
            reason = "high_fuzzy_low_wer"
        elif row.get("semantic_sim", 0) >= 0.85:
            reclassified = True
            reason = "high_semantic_sim"

        deep_rows.append({
            "paper_id": row["paper_id"],
            "title": row["title"],
            "span_field": row["span_field"],
            "composite": row["composite"],
            "priority": row["priority"],
            "fuzzy_partial": round(fuzzy_partial, 2),
            "fuzzy_sort": round(fuzzy_sort, 2),
            "wer": round(wer, 4),
            "reclassified": reclassified,
            "reclassification_reason": reason,
        })

        if (idx + 1) % 50 == 0 or idx + 1 == len(flagged):
            msg = f"Phase 2: Analysed {idx + 1}/{len(flagged)} flagged spans"
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

    rescued = sum(1 for d in deep_rows if d["reclassified"])
    logger.info("  Phase 2 complete: %d/%d flagged spans reclassified as likely_valid",
                rescued, len(deep_rows))

    return deep_rows


# ── Phase 3: Merge reclassifications ──────────────────────────────────────

def _merge_reclassifications(
    audit_rows: list[dict],
    deep_rows: list[dict],
) -> list[dict]:
    """Update audit rows with deep analysis reclassifications."""
    # Build lookup: (paper_id, span_field) → deep row
    deep_map: dict[tuple[str, str], dict] = {}
    for dr in deep_rows:
        key = (dr["paper_id"], dr["span_field"])
        deep_map[key] = dr

    updated: list[dict] = []
    for row in audit_rows:
        merged = dict(row)
        key = (row["paper_id"], row["span_field"])
        dr = deep_map.get(key)
        if dr and dr.get("reclassified"):
            merged["reclassified"] = True
            merged["reclassification_reason"] = dr["reclassification_reason"]
        else:
            merged["reclassified"] = False
            merged["reclassification_reason"] = ""
        updated.append(merged)

    reclassified_count = sum(1 for u in updated if u["reclassified"])
    logger.info("Phase 3: Merged reclassifications — %d spans marked likely_valid",
                reclassified_count)

    return updated


# ── Phase 4: Filter ───────────────────────────────────────────────────────

def _filter_claims(
    claims: list[dict],
    audit_rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Remove bottom third of papers by average composite similarity.

    Papers with reclassification notes (likely_valid) are protected.
    Returns (filtered_claims, removed_papers_with_reasons).
    """
    # Aggregate per paper: average composite + reclassified flag
    paper_scores: dict[str, dict] = {}
    for row in audit_rows:
        pid = row["paper_id"]
        if pid not in paper_scores:
            paper_scores[pid] = {
                "composites": [],
                "has_reclassified": False,
            }
        paper_scores[pid]["composites"].append(row["composite"])
        if row.get("reclassified"):
            paper_scores[pid]["has_reclassified"] = True

    if not paper_scores:
        # Zero audit rows means there was nothing to score — not evidence
        # that every paper failed. Filtering here would wipe the corpus.
        logger.warning(
            "Phase 4: no audit rows produced — keeping all %d papers. "
            "Check that claims carry key_quote/evidence spans if filtering "
            "was expected.",
            len(claims),
        )
        return list(claims), []

    # Compute averages
    paper_avgs: list[tuple[str, float, bool]] = []
    for pid, info in paper_scores.items():
        avg = sum(info["composites"]) / len(info["composites"])
        paper_avgs.append((pid, avg, info["has_reclassified"]))

    # Sort by average composite ascending
    paper_avgs.sort(key=lambda x: x[1])

    # Bottom third threshold
    n_total = len(paper_avgs)
    cutoff_idx = n_total // 3
    bottom_third_pids = set()
    for pid, avg, protected in paper_avgs[:cutoff_idx]:
        if not protected:
            bottom_third_pids.add(pid)

    # Build removal log
    removed: list[dict] = []
    for pid, avg, protected in paper_avgs[:cutoff_idx]:
        entry = {
            "paper_id": pid,
            "avg_composite": round(avg, 4),
            "protected": protected,
            "removed": pid in bottom_third_pids,
            "reason": ("protected_by_reclassification"
                       if protected else "bottom_third_composite"),
        }
        removed.append(entry)

    # Papers with no audit rows should not silently pass through the filter.
    # Log them as removed so the removal log matches the filtered count.
    audited_pids = set(paper_scores.keys())
    unfiltered_unaudited_pids = set()
    for c in claims:
        pid = c.get("paper_id", c.get("id", ""))
        if pid not in audited_pids and pid not in bottom_third_pids:
            unfiltered_unaudited_pids.add(pid)

    for pid in unfiltered_unaudited_pids:
        removed.append({
            "paper_id": pid,
            "avg_composite": None,
            "protected": False,
            "removed": True,
            "reason": "no_audit_rows",
        })

    filtered = [c for c in claims
                if c.get("paper_id", c.get("id", "")) not in bottom_third_pids
                and c.get("paper_id", c.get("id", "")) not in unfiltered_unaudited_pids]

    n_removed = len(removed)
    n_protected = sum(1 for r in removed if r.get("protected"))
    n_unaudited = len(unfiltered_unaudited_pids)
    logger.info("Phase 4: %d papers in bottom third, %d protected, %d unaudited removed, %d total removed",
                cutoff_idx, n_protected, n_unaudited, n_removed)

    return filtered, removed


# ── Main entry point ──────────────────────────────────────────────────────

def run(
    config=None,
    claims_path: Optional[str] = None,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Run 4-phase span audit.

    Phase 1: Compute span similarities (semantic + token Jaccard + TF-IDF)
    Phase 2: Deep fuzzy analysis on P1/P2 rows (rapidfuzz if available)
    Phase 3: Merge reclassifications (rescue likely_valid spans)
    Phase 4: Filter — remove bottom-third papers by composite similarity

    Input : claims.json + method_profiles_validated.json
    Output: claims_filtered.json + removal_log.json + 3 audit CSVs
    """
    logger.info("═" * 60)
    logger.info("STAGE 5: SPAN AUDIT & QUALITY FILTERING")
    logger.info("═" * 60)
    logger.info("  Optional deps: sentence_transformers=%s, rapidfuzz=%s, sklearn=%s",
                _HAS_SENTENCE_TRANSFORMERS, _HAS_RAPIDFUZZ, _HAS_SKLEARN)

    # ── Resolve paths ─────────────────────────────────────────────────
    claims_file = Path(claims_path) if claims_path else (
        config.claims_dir / "claims.json" if config else Path("data/claims/claims.json"))

    profiles_dir = (config.data_dir / "method_review" if config
                    else Path("data/method_review"))
    profiles_file = profiles_dir / "method_profiles_validated.json"

    review_dir = profiles_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    claims_out_dir = config.claims_dir if config else Path("data/claims")
    claims_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Claims input:  %s", claims_file)
    logger.info("  Profiles input: %s", profiles_file)
    logger.info("  Review dir:    %s", review_dir)

    # ── Load data ─────────────────────────────────────────────────────
    if not claims_file.exists():
        msg = f"Claims file not found: {claims_file}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(claims_file, encoding="utf-8") as f:
        claims = json.load(f)
    logger.info("  Loaded %d claims", len(claims))

    profiles: list[dict] = []
    if profiles_file.exists():
        with open(profiles_file, encoding="utf-8") as f:
            profiles = json.load(f)
        logger.info("  Loaded %d method profiles", len(profiles))
    else:
        logger.warning("  Method profiles not found at %s — proceeding without",
                       profiles_file)

    if not claims:
        logger.warning("No claims to audit — writing empty outputs")
        atomic_write_json(claims_out_dir / "claims_filtered.json", [])
        atomic_write_json(claims_out_dir / "removal_log.json", [])
        return {"total": 0, "filtered": 0, "removed": 0,
                "output_path": str(claims_out_dir / "claims_filtered.json")}

    if progress_callback:
        progress_callback(f"Stage 5: Auditing {len(claims)} papers")

    # Guarantee unique paper ids before anything keys off them.
    _normalize_paper_ids(claims, profiles)

    # ── Phase 1 ───────────────────────────────────────────────────────
    logger.info("Phase 1: Computing similarities for %d papers...", len(claims))
    if progress_callback:
        progress_callback(f"Phase 1: Computing similarities for {len(claims)} papers...")

    audit_rows = _compute_span_similarities(
        claims, profiles,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    logger.info("Phase 1 complete: %d span audit rows", len(audit_rows))

    # Check cancellation
    if cancel_event is not None:
        is_set = (cancel_event.is_set()
                  if hasattr(cancel_event, "is_set") else bool(cancel_event))
        if is_set:
            return {"total": len(claims), "cancelled": True}

    # Write Phase 1 CSV
    audit_csv_path = review_dir / "span_similarity_audit.csv"
    _write_audit_csv(audit_csv_path, audit_rows)
    logger.info("  Wrote %s (%d rows)", audit_csv_path, len(audit_rows))

    # ── Phase 2 ───────────────────────────────────────────────────────
    logger.info("Phase 2: Deep fuzzy analysis on flagged spans...")
    if progress_callback:
        progress_callback("Phase 2: Deep fuzzy analysis on flagged spans...")

    deep_rows = _deep_analysis(audit_rows, progress_callback=progress_callback)

    # Write Phase 2 CSV
    deep_csv_path = review_dir / "p2_deep_analysis.csv"
    _write_deep_csv(deep_csv_path, deep_rows)
    logger.info("  Wrote %s (%d rows)", deep_csv_path, len(deep_rows))

    # ── Phase 3 ───────────────────────────────────────────────────────
    logger.info("Phase 3: Merging reclassifications...")
    if progress_callback:
        progress_callback("Phase 3: Merging reclassifications...")

    updated_rows = _merge_reclassifications(audit_rows, deep_rows)

    # Write Phase 3 CSV
    updated_csv_path = review_dir / "span_similarity_audit_updated.csv"
    _write_updated_csv(updated_csv_path, updated_rows)
    logger.info("  Wrote %s (%d rows)", updated_csv_path, len(updated_rows))

    # ── Phase 4 ───────────────────────────────────────────────────────
    logger.info("Phase 4: Filtering bottom third by composite score...")
    if progress_callback:
        progress_callback("Phase 4: Filtering bottom third by composite score...")

    filtered_claims, removal_log = _filter_claims(claims, updated_rows)

    # Write outputs
    filtered_path = claims_out_dir / "claims_filtered.json"
    atomic_write_json(filtered_path, filtered_claims)
    logger.info("  Wrote %s (%d papers)", filtered_path, len(filtered_claims))

    removal_path = claims_out_dir / "removal_log.json"
    atomic_write_json(removal_path, removal_log)
    logger.info("  Wrote %s (%d entries)", removal_path, len(removal_log))

    # ── Summary ───────────────────────────────────────────────────────
    n_removed = len(claims) - len(filtered_claims)
    summary = {
        "total": len(claims),
        "filtered": len(filtered_claims),
        "removed": n_removed,
        "audit_spans": len(audit_rows),
        "deep_analysed": len(deep_rows),
        "reclassified": sum(1 for r in updated_rows if r.get("reclassified")),
        "output_path": str(filtered_path),
        "removal_log_path": str(removal_path),
        "audit_csv_path": str(audit_csv_path),
        "deep_csv_path": str(deep_csv_path),
        "updated_csv_path": str(updated_csv_path),
    }
    logger.info("Stage 5 summary: %s", summary)
    if progress_callback:
        progress_callback(
            f"Stage 5 complete: {len(filtered_claims)} papers kept, "
            f"{n_removed} removed, {summary['reclassified']} rescued")

    return summary


def check_output(config) -> dict:
    """Check if Stage 5 output exists and return summary."""
    path = config.claims_dir / "claims_filtered.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"exists": True, "path": str(path), "count": len(data)}
