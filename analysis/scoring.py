"""Keyword-based scoring of articles against hypothesis components.

Scores each article against each hypothesis component using keyword frequency,
co-occurrence, and directional language analysis.  No LLM calls — pure text.
"""
import logging
import re
from typing import Optional

from core.models import (
    Article,
    HypothesisComponent,
    EvidenceLink,
    EvidenceDirection,
    AlignmentResult,
    TrendPoint,
    AnalysisSession,
)

logger = logging.getLogger(__name__)

# Words that do not carry useful search meaning by themselves.
QUERY_STOPWORDS = {
    "a",
    "an",
    "about",
    "affect",
    "affected",
    "affecting",
    "affects",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "current",
    "does",
    "evidence",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "question",
    "research",
    "say",
    "scholarly",
    "that",
    "this",
    "the",
    "their",
    "to",
    "through",
    "user",
    "what",
    "vs",
    "with",
}

# ---------------------------------------------------------------------------
# Language patterns
# ---------------------------------------------------------------------------
NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno\b", r"\bnor\b", r"\bnever\b", r"\bneither\b",
    r"\bwithout\b", r"\black\b", r"\bfail\w*\b", r"\binsufficient\b",
    r"\bunrelated\b", r"\bcontrary\b", r"\bunlike\b", r"\babsen\w*\b",
]

SUPPORT_PATTERNS = [
    r"\bsupport\w*\b", r"\bconfirm\w*\b", r"\bconsistent\b", r"\bdemonstrat\w*\b",
    r"\bcorrelat\w*\b", r"\bassociat\w*\b", r"\bsignificant\w*\b",
    r"\bincreas\w*\b", r"\benhance\w*\b", r"\bimplicat\w*\b", r"\bcontribut\w*\b",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _coerce_year(value) -> Optional[int]:
    """Coerce *value* to an integer year, handling date strings like '2024-01-15'."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return int(text) if text.isdigit() else None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _count_keyword_hits(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    text_lower = _normalize_text(text)
    matched = []
    total = 0
    for kw in keywords:
        kw_lower = kw.lower().strip()
        if not kw_lower:
            continue
        escaped = re.escape(kw_lower)
        # If the keyword contains word characters, use word boundaries to avoid
        # matching inside other words (e.g. "cat" in "caterpillar").
        if re.search(r"\w", kw_lower):
            count = len(re.findall(rf"\b{escaped}\b", text_lower))
        else:
            count = len(re.findall(escaped, text_lower))
        if count:
            matched.append(kw)
            total += count
    return total, matched


def _keyword_frequency_score(text: str, keywords: list[str]) -> float:
    """0-1 score based on how many keywords appear and how often."""
    if not keywords:
        return 0.0
    hits, matched = _count_keyword_hits(text, keywords)
    coverage = len(matched) / len(keywords)
    density = min(hits / max(len(text.split()), 1) * 50, 1.0)
    return 0.6 * coverage + 0.4 * density


def _cooccurrence_score(text: str, keywords_a: list[str], keywords_b: list[str]) -> float:
    """Score based on co-occurrence of two keyword sets in the same text."""
    _, matched_a = _count_keyword_hits(text, keywords_a)
    _, matched_b = _count_keyword_hits(text, keywords_b)
    if not matched_a or not matched_b:
        return 0.0
    return min(len(matched_a), len(matched_b)) / max(len(keywords_a), len(keywords_b), 1)


def _directional_score(text: str) -> float:
    """Return -1 to +1 based on supportive vs contradictory language."""
    text_lower = _normalize_text(text)
    support = sum(1 for p in SUPPORT_PATTERNS if re.search(p, text_lower))
    negate = sum(1 for p in NEGATION_PATTERNS if re.search(p, text_lower))
    total = support + negate
    if total == 0:
        return 0.0
    return (support - negate) / total


def _extract_key_excerpts(text: str, keywords: list[str], max_excerpts: int = 3) -> list[str]:
    """Find sentences containing the most keywords."""
    sentences = re.split(r"[.!?]+", text)
    scored = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        hits, _ = _count_keyword_hits(s, keywords)
        if hits:
            scored.append((hits, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_excerpts]]


def _search_term_id(term: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")
    return f"term_{slug or 'query'}"


def _component_id(prefix: str, label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{prefix}_{slug or 'component'}"


def _unique_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        cleaned = str(term).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9+\-/]*", query.lower()):
        token = raw.strip("-/")
        if len(token) < 3 or token in QUERY_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _query_token_segments(query: str) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9+\-/]*", query.lower()):
        token = raw.strip("-/")
        if len(token) < 3 or token in QUERY_STOPWORDS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _query_phrase_candidates(query: str, *, min_size: int = 2, max_size: int = 4) -> list[str]:
    phrases: list[str] = []
    for tokens in _query_token_segments(query):
        for size in range(min(max_size, len(tokens)), min_size - 1, -1):
            for idx in range(0, max(0, len(tokens) - size + 1)):
                phrase = " ".join(tokens[idx:idx + size])
                if phrase:
                    phrases.append(phrase)
    return _unique_terms(phrases)


def quote_boolean_term(term: str) -> str:
    cleaned = " ".join(str(term).split()).strip()
    if not cleaned:
        return ""
    if " " in cleaned or "/" in cleaned:
        return f'"{cleaned}"'
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_search_term_components(query: str) -> list[HypothesisComponent]:
    """Build weighted components from meaningful query phrases and words.

    The Search tab uses these components as the editable model for keyword
    analysis. Phrase components let the system learn that a compound concept
    retrieved useful papers, while word components keep the query tunable.
    """
    seen: set[str] = set()
    components: list[HypothesisComponent] = []

    for phrase in _query_phrase_candidates(query):
        if phrase in seen:
            continue
        seen.add(phrase)
        components.append(HypothesisComponent(
            id=_search_term_id(f"phrase_{phrase}"),
            label=phrase,
            description=f"Phrase from question: {phrase}",
            keywords=[phrase],
            weight=1.0,
        ))

    for term in _query_tokens(query):
        if term in seen:
            continue
        seen.add(term)
        components.append(HypothesisComponent(
            id=_search_term_id(term),
            label=term,
            description=f"Search term from question: {term}",
            keywords=[term],
            weight=1.0,
        ))
    return components


def build_weighted_boolean_query(
    components: list[HypothesisComponent],
    *,
    max_terms: int = 18,
) -> str:
    """Build a provider-neutral weighted Boolean query from component weights."""
    candidates: list[tuple[float, int, int, str]] = []
    for comp_idx, component in enumerate(components):
        weight = float(getattr(component, "weight", 1.0) or 0.0)
        if weight <= 0:
            continue
        terms = getattr(component, "keywords", []) or [component.label]
        for term_idx, term in enumerate(terms):
            quoted = quote_boolean_term(term)
            if quoted:
                candidates.append((-weight, comp_idx, term_idx, quoted))

    candidates.sort()
    seen: set[str] = set()
    ranked: list[tuple[float, str]] = []
    for neg_weight, _, _, term in candidates:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append((-neg_weight, term))
        if len(ranked) >= max_terms:
            break

    must_terms = [term for weight, term in ranked if weight >= 1.35]
    should_terms = [term for weight, term in ranked if 0.65 <= weight < 1.35]
    optional_terms = [term for weight, term in ranked if 0.35 <= weight < 0.65]

    if not must_terms and should_terms:
        must_terms = should_terms[:1]
        should_terms = should_terms[1:]

    clauses: list[str] = []
    if must_terms:
        clauses.append(" AND ".join(must_terms[:4]))
    if should_terms:
        clauses.append(f"({' OR '.join(should_terms[:10])})")
    if optional_terms:
        clauses.append(f"({' OR '.join(optional_terms[:6])})")
    return " AND ".join(clauses)


def build_hypothesis_components(
    *,
    hypothesis: str = "",
    research_config: Optional[dict] = None,
    co_occurring_conditions: Optional[list[str]] = None,
    query: str = "",
) -> list[HypothesisComponent]:
    """Build weighted components from hypothesis parts, search domains, and query terms."""
    components: list[HypothesisComponent] = []
    research = research_config or {}
    search = research.get("search") or {}
    relevance = research.get("relevance") or {}
    research_question = str(
        relevance.get("research_question") or research.get("description") or ""
    ).strip()

    base_terms = _unique_terms([str(term) for term in search.get("base_terms", [])])
    if base_terms:
        label = f"Subject: {' / '.join(base_terms[:3])}"
        components.append(HypothesisComponent(
            id=_component_id("subject", " ".join(base_terms[:3])),
            label=label,
            description="Primary subject terms from the research configuration.",
            keywords=base_terms,
            weight=1.0,
        ))

    for domain in search.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        name = str(domain.get("name") or "").strip()
        terms = _unique_terms([
            *[str(term) for term in domain.get("terms", []) or []],
            *[str(term) for term in domain.get("require_also", []) or []],
        ])
        if not name or not terms:
            continue
        components.append(HypothesisComponent(
            id=_component_id("domain", name),
            label=name,
            description=f"Search domain from config: {name}",
            keywords=terms,
            weight=1.0,
        ))

    for condition in co_occurring_conditions or []:
        cond_clean = str(condition).strip()
        if not cond_clean:
            continue
        components.append(HypothesisComponent(
            id=_component_id("condition", cond_clean),
            label=cond_clean,
            description=f"Co-occurring term from the research configuration: {cond_clean}",
            # Only the configured term: relationship words ("comorbid") would
            # match every condition and leak clinical vocabulary into searches.
            keywords=[cond_clean],
            weight=1.0,
        ))

    for comp in build_search_term_components(query or research_question or hypothesis):
        components.append(comp)

    deduped: list[HypothesisComponent] = []
    seen_ids: set[str] = set()
    for comp in components:
        if comp.id in seen_ids:
            continue
        seen_ids.add(comp.id)
        deduped.append(comp)
    return deduped


def build_weighted_component_query_terms(
    components: list[HypothesisComponent],
) -> list[str]:
    """Return query terms ordered by current component weight.

    Zero-weight components are treated as explicitly disabled.
    """
    weighted: list[tuple[float, int, int, str]] = []
    for idx, component in enumerate(components):
        weight = float(getattr(component, "weight", 1.0) or 0.0)
        if weight <= 0:
            continue
        keywords = getattr(component, "keywords", []) or []
        terms = list(keywords) if keywords else [component.label]
        seen: set[str] = set()
        for term_idx, term in enumerate(terms):
            cleaned = str(term).strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            weighted.append((-weight, idx, term_idx, cleaned))

    weighted.sort()
    result: list[str] = []
    seen_global: set[str] = set()
    for _, _, _, term in weighted:
        key = term.lower()
        if key not in seen_global:
            seen_global.add(key)
            result.append(term)
    return result


def update_component_weights_from_keyword_analysis(
    session: AnalysisSession,
    *,
    min_weight: float = 0.2,
    max_weight: float = 3.0,
) -> dict[str, dict[str, float]]:
    """Update component weights from keyword-analysis evidence.

    Weight semantics:
    - matched, relevant, supportive terms rise above 1.0;
    - terms that do not appear in useful records decay below 1.0;
    - terms associated with contradictory evidence decay further.
    """
    links_by_component: dict[str, list[EvidenceLink]] = {}
    for link in session.evidence_links:
        links_by_component.setdefault(link.component_id, []).append(link)

    stats: dict[str, dict[str, float]] = {}
    for component in session.hypothesis_components:
        links = links_by_component.get(component.id, [])
        if links:
            matched = [link for link in links if link.matched_keywords or link.relevance_score > 0]
            coverage = len(matched) / len(links)
            mean_relevance = sum(link.relevance_score for link in links) / len(links)
            mean_confidence = sum(link.confidence for link in links) / len(links)
            support = sum(1 for link in links if link.direction == EvidenceDirection.SUPPORTS)
            contradict = sum(1 for link in links if link.direction == EvidenceDirection.CONTRADICTS)
            direction_balance = (support - contradict) / len(links)
        else:
            coverage = 0.0
            mean_relevance = 0.0
            mean_confidence = 0.0
            direction_balance = 0.0

        utility = (
            0.35 * mean_relevance
            + 0.30 * coverage
            + 0.20 * mean_confidence
            + 0.15 * direction_balance
        )
        if direction_balance < 0:
            utility -= 0.75 * abs(direction_balance) * coverage
        new_weight = max(min_weight, min(max_weight, 0.5 + 1.5 * utility))
        component.weight = round(new_weight, 3)
        component.description = (
            f"{component.description.split(' | Keyword analysis:')[0]} | "
            f"Keyword analysis: utility={utility:.3f}, coverage={coverage:.2f}, "
            f"relevance={mean_relevance:.2f}, confidence={mean_confidence:.2f}, "
            f"direction={direction_balance:.2f}"
        )
        stats[component.id] = {
            "utility": round(utility, 6),
            "coverage": round(coverage, 6),
            "mean_relevance": round(mean_relevance, 6),
            "mean_confidence": round(mean_confidence, 6),
            "direction_balance": round(direction_balance, 6),
            "weight": component.weight,
        }
    return stats


def score_article_against_component(
    article: Article,
    component: HypothesisComponent,
    all_components: Optional[list[HypothesisComponent]] = None,
) -> EvidenceLink:
    """Score one article against one hypothesis component."""
    text = article.text_content or article.abstract or article.title
    if not text.strip():
        return EvidenceLink(
            article_id=article.id,
            component_id=component.id,
            direction=EvidenceDirection.NEUTRAL,
            relevance_score=0.0,
            confidence=0.0,
        )

    # Keyword frequency
    kw_score = _keyword_frequency_score(text, component.keywords)

    # Co-occurrence with other components
    cooc = 0.0
    if all_components:
        other_kws = []
        for c in all_components:
            if c.id != component.id:
                other_kws.extend(c.keywords)
        cooc = _cooccurrence_score(text, component.keywords, other_kws)

    # Direction
    dir_score = _directional_score(text)

    # Combined relevance (0-1)
    relevance = 0.5 * kw_score + 0.3 * cooc + 0.2 * abs(dir_score)

    # Determine direction
    if relevance < 0.1:
        direction = EvidenceDirection.NEUTRAL
    elif dir_score > 0.2:
        direction = EvidenceDirection.SUPPORTS
    elif dir_score < -0.2:
        direction = EvidenceDirection.CONTRADICTS
    else:
        direction = EvidenceDirection.TANGENTIAL

    excerpts = _extract_key_excerpts(text, component.keywords)
    normalized_text = _normalize_text(text)

    return EvidenceLink(
        article_id=article.id,
        component_id=component.id,
        direction=direction,
        relevance_score=relevance,
        confidence=min(kw_score + cooc, 1.0),
        key_excerpts=excerpts,
        matched_keywords=[kw for kw in component.keywords
                          if kw.lower() in normalized_text],
    )


def compute_alignment(
    evidence_links: list[EvidenceLink],
    component: HypothesisComponent,
    articles: dict[str, Article],
) -> AlignmentResult:
    """Aggregate evidence links into an alignment score for one component."""
    comp_links = [el for el in evidence_links if el.component_id == component.id]
    if not comp_links:
        return AlignmentResult(
            component_id=component.id,
            component_label=component.label,
        )

    supporting = [el for el in comp_links if el.direction == EvidenceDirection.SUPPORTS]
    contradicting = [el for el in comp_links if el.direction == EvidenceDirection.CONTRADICTS]

    n = len(comp_links)
    support_score = (len(supporting) - len(contradicting)) / n if n else 0.0
    confidence = sum(el.confidence for el in comp_links) / n if n else 0.0

    return AlignmentResult(
        component_id=component.id,
        component_label=component.label,
        support_score=support_score,
        evidence_count=n,
        supporting_count=len(supporting),
        contradicting_count=len(contradicting),
        neutral_count=n - len(supporting) - len(contradicting),
        confidence=confidence,
        top_supporting=[el.article_id for el in sorted(
            supporting, key=lambda x: x.relevance_score, reverse=True)[:5]],
        top_contradicting=[el.article_id for el in sorted(
            contradicting, key=lambda x: x.relevance_score, reverse=True)[:3]],
    )


def compute_trends(
    evidence_links: list[EvidenceLink],
    articles: dict[str, Article],
    component: HypothesisComponent,
) -> list[TrendPoint]:
    """Compute publication trend data for one component."""
    comp_links = [el for el in evidence_links if el.component_id == component.id]
    year_data: dict[int, list[EvidenceLink]] = {}
    for el in comp_links:
        art = articles.get(el.article_id)
        yr = art.year if art and art.year else 0
        if yr > 1900:
            year_data.setdefault(yr, []).append(el)

    points = []
    cum_citations = 0
    for yr in sorted(year_data):
        links = year_data[yr]
        avg_score = sum(el.relevance_score for el in links) / len(links) if links else 0
        link_citations = 0
        for el in links:
            art = articles.get(el.article_id)
            if art is None:
                art = Article(id="", title="")
            link_citations += art.citations
        cum_citations += link_citations
        points.append(TrendPoint(
            year=yr,
            article_count=len(links),
            avg_support_score=avg_score,
            cumulative_citations=cum_citations,
            component_id=component.id,
        ))
    return points


# ── Confidence calibration ───────────────────────────────────────────────

def calibrate_confidence(
    predicted: list[float],
    observed: list[int],
    method: str = "isotonic",
    n_bins: int = 10,
) -> dict:
    """Calibrate confidence scores against ground truth labels.

    Parameters
    ----------
    predicted : list of raw confidence scores (0-1)
    observed : list of binary ground truth labels (0 or 1)
    method : "isotonic" (default) or "platt" (logistic sigmoid)
    n_bins : number of bins for Expected Calibration Error

    Returns
    -------
    dict with:
        calibrated : list of calibrated probabilities
        ece : Expected Calibration Error (lower is better)
        method : method used
        bin_stats : list of {bin_center, bin_count, mean_predicted, mean_observed}
    """
    import numpy as np

    pred = np.array(predicted, dtype=np.float64)
    obs = np.array(observed, dtype=np.float64)

    if len(pred) < 3 or len(pred) != len(obs):
        return {
            "calibrated": list(predicted),
            "ece": 1.0,
            "method": method,
            "bin_stats": [],
        }

    if method == "platt":
        # Platt scaling: fit logistic regression P(y=1|f) = 1/(1+exp(Af+B))
        from scipy.optimize import minimize

        def _neg_log_likelihood(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * pred + b))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(obs * np.log(p) + (1 - obs) * np.log(1 - p))

        result = minimize(_neg_log_likelihood, [0.0, 0.0], method="Nelder-Mead")
        a_opt, b_opt = result.x
        calibrated = 1.0 / (1.0 + np.exp(a_opt * pred + b_opt))
    else:
        # Isotonic regression: non-parametric monotonic fit
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip")
        calibrated = ir.fit_transform(pred, obs)

    calibrated = np.clip(calibrated, 0.0, 1.0)

    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_stats = []
    ece = 0.0
    for i in range(n_bins):
        mask = (calibrated >= bin_edges[i]) & (calibrated < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = mask | (calibrated == bin_edges[i + 1])
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            mean_pred = float(np.mean(calibrated[mask]))
            mean_obs = float(np.mean(obs[mask]))
            ece += (bin_count / len(pred)) * abs(mean_pred - mean_obs)
            bin_stats.append({
                "bin_center": round((bin_edges[i] + bin_edges[i + 1]) / 2, 2),
                "bin_count": bin_count,
                "mean_predicted": round(mean_pred, 4),
                "mean_observed": round(mean_obs, 4),
            })

    return {
        "calibrated": [round(float(c), 6) for c in calibrated],
        "ece": round(float(ece), 6),
        "method": method,
        "bin_stats": bin_stats,
    }


def run_full_analysis(
    articles,
    components: list[HypothesisComponent],
    progress_cb=None,
) -> AnalysisSession:
    """Run keyword analysis on all articles × all components.

    Returns a populated AnalysisSession.
    """
    # Coerce articles into the canonical dict[str, Article] shape.
    # The GUI pipeline can pass either a dict (id -> Article) or a list of Article.
    if isinstance(articles, list):
        articles_by_id: dict[str, Article] = {}
        for idx, a in enumerate(articles):
            if isinstance(a, Article):
                art_id = a.id
                article = a
            elif isinstance(a, dict):
                art_id = (
                    a.get("id")
                    or a.get("doi")
                    or a.get("pmid")
                    or (a.get("title", "") or "")[:60]
                )
                article = Article(
                    id=art_id,
                    title=a.get("title", ""),
                    authors=a.get("authors", []) or [],
                    abstract=a.get("abstract", "") or "",
                    full_text=a.get("full_text", "") or "",
                    year=_coerce_year(a.get("year")),
                    journal=a.get("journal", "") or "",
                    doi=a.get("doi", "") or "",
                    citations=a.get("citationCount", 0) or a.get("citations", 0) or 0,
                    url=a.get("url", "") or "",
                    pmid=a.get("pmid", "") or "",
                    keywords=a.get("keywords", []) or [],
                    has_full_text=bool(a.get("has_full_text", False)),
                    open_access=bool(a.get("open_access", False)),
                    full_text_url=a.get("full_text_url", "") or "",
                )
            else:
                raise TypeError(
                    "run_full_analysis expected articles to be a dict[str, Article] or list[Article]; "
                    f"got list item {idx} of type {type(a).__name__}"
                )

            if not art_id:
                art_id = f"article_{idx}"
                article.id = art_id
            articles_by_id[art_id] = article
        articles = articles_by_id
    elif not isinstance(articles, dict):
        raise TypeError(
            "run_full_analysis expected articles to be dict[str, Article] or list[Article]; "
            f"got {type(articles).__name__}"
        )

    session = AnalysisSession()
    session.hypothesis_components = components
    session.articles = articles
    session.total_articles_fetched = len(articles)

    logger.info("═" * 60)
    logger.info("KEYWORD ANALYSIS: %d articles × %d components", len(articles), len(components))
    logger.info("═" * 60)

    # Score every article against every component
    evidence_links: list[EvidenceLink] = []
    total = len(articles) * len(components)
    done = 0
    for art_id, article in articles.items():
        for comp in components:
            link = score_article_against_component(article, comp, components)
            evidence_links.append(link)
            done += 1
            if done % 50 == 0:
                logger.info("[%d/%d] Scoring article×component pairs...", done, total)
                if progress_cb:
                    progress_cb(done, total)

    session.evidence_links = evidence_links
    session.total_articles_analyzed = len(articles)

    logger.info("Scored %d article×component pairs", len(evidence_links))

    # Compute alignment per component
    for comp in components:
        alignment = compute_alignment(evidence_links, comp, articles)
        session.alignment_results.append(alignment)
        logger.info("  %s: support=%.2f (%d supporting, %d contradicting, %d neutral)",
                     comp.label, alignment.support_score,
                     alignment.supporting_count, alignment.contradicting_count,
                     alignment.neutral_count)

    # Compute trends
    for comp in components:
        trends = compute_trends(evidence_links, articles, comp)
        session.trend_data.extend(trends)

    supporting_total = sum(a.supporting_count for a in session.alignment_results)
    contradicting_total = sum(a.contradicting_count for a in session.alignment_results)
    logger.info("═" * 60)
    logger.info("ANALYSIS COMPLETE: %d supporting, %d contradicting",
                supporting_total, contradicting_total)
    logger.info("═" * 60)

    return session
