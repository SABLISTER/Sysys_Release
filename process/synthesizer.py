"""Claim clustering and narrative synthesis."""
import logging
import re
import zlib
from typing import Any, cast
from collections import Counter, defaultdict

import numpy as np

from core.claims import claim_flag, claim_list, claim_value

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model cache
# ---------------------------------------------------------------------------
_model_cache: dict = {}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
NARRATIVE_PROMPT = """You are synthesizing research findings from a cluster of {paper_count} academic papers for a systematic evidence review.

Research question:
{research_question}

Key mechanisms discussed: {mechanisms}
Populations, constructs, or contextual factors mentioned: {conditions}
Key entities or systems involved: {entities}
Papers directly informing the research question: {primary_support_pct}%

Paper summaries:
{paper_summaries}

Write a cohesive 2-3 paragraph narrative synthesizing these findings. Focus on:
1. What the papers collectively show about the configured research question and described mechanisms
2. Areas of agreement and disagreement
3. Strength of the evidence

Write in academic style suitable for a review paper."""

ABSTRACT_PROMPT = """You are drafting a structured abstract for a systematic review.

Research question:
{research_question}

Total papers analyzed: {total_papers}

Cluster summaries:
{cluster_summaries}

Write a structured abstract with these sections:
- Background (1-2 sentences)
- Objective (1 sentence)
- Methods (2-3 sentences summarizing the pipeline)
- Results (3-5 sentences, the key findings across all clusters)
- Conclusions (2-3 sentences)

Keep it under 350 words."""


from process.claim_extractor import (
    _claim_think_mode as _synthesis_think_mode,
    _research_question_from_config,
)
from core.config import DEFAULT_EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load sentence-transformers model (cached).  Returns None if not installed."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        logger.info("Loaded sentence-transformers model: %s", model_name)
        return model
    except ImportError:
        logger.warning(
            "sentence_transformers not installed — using TF-IDF fallback for clustering")
        return None


def _paper_to_text(paper: dict) -> str:
    """Build text representation for embedding."""
    claims_value = paper.get("claims")
    claims = claims_value if isinstance(claims_value, dict) else {}
    parts = [paper.get("title", "")]
    parts.append(paper.get("abstract", ""))
    parts.extend(paper.get("findings", []) or claims.get("findings", []))
    parts.extend(paper.get("mechanisms", []) or claims.get("mechanisms", []))
    return " ".join(p for p in parts if p)


_claim_value = claim_value


def _as_list(value) -> list:
    """Normalize common scalar/list claim values to a clean list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _paper_id(paper: dict, fallback: str = "") -> str:
    for key in ("id", "article_id", "doi", "pmid", "pmcid", "paperId", "title"):
        value = paper.get(key)
        if value:
            return str(value)
    return fallback


def _paper_title(paper: dict) -> str:
    return str(paper.get("title") or paper.get("article_title") or "Untitled")


def _paper_score(paper: dict) -> float:
    for key in (
        "relevance_score",
        "score",
        "llm_relevance",
        "support_score",
        "combined_score",
    ):
        value = paper.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    if _claim_value(paper, "supports_primary_question", False):
        return 3.0
    return 0.0


def _paper_direction(paper: dict) -> str:
    for key in ("direction", "llm_direction", "evidence_direction"):
        value = paper.get(key)
        if value:
            text = str(value).lower()
            if "contradict" in text:
                return "contradicts"
            if "support" in text:
                return "supports"
            if "neutral" in text:
                return "neutral"
    if _claim_value(paper, "supports_primary_question", False):
        return "supports"
    return "neutral"


def _paper_quote(paper: dict) -> str:
    for key in ("key_quote", "quoted_span", "quote", "excerpt"):
        value = _claim_value(paper, key, "")
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    spans = _as_list(paper.get("evidence_spans") or _claim_value(paper, "evidence_spans", []))
    return spans[0] if spans else ""


def _method_lookup(method_profiles: list[dict] | dict | None) -> dict[str, dict]:
    if not method_profiles:
        return {}
    rows = method_profiles.values() if isinstance(method_profiles, dict) else method_profiles
    lookup: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("article_id") or row.get("id") or "")
        if pid:
            lookup[pid] = row
    return lookup


def _grade_summary(grade_results: list[dict] | dict | None) -> dict:
    if not grade_results:
        return {}
    rows = grade_results.values() if isinstance(grade_results, dict) else grade_results
    counts: Counter[str] = Counter()
    examples: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cert = (
            row.get("overall_certainty")
            or row.get("certainty")
            or row.get("overall")
            or ""
        )
        if isinstance(cert, dict):
            cert = cert.get("value") or cert.get("label") or ""
        cert_text = str(cert).strip()
        if cert_text:
            counts[cert_text] += 1
        label = row.get("component_label") or row.get("component") or row.get("label")
        if label and cert_text and len(examples) < 8:
            examples.append({"component": str(label), "certainty": cert_text})
    return {"certainty_counts": dict(sorted(counts.items())), "examples": examples}


def _top_terms(papers: list[dict], field: str, limit: int = 12) -> list[dict]:
    counter: Counter[str] = Counter()
    for paper in papers:
        for value in _as_list(_claim_value(paper, field, [])):
            key = " ".join(value.split())
            if key:
                counter[key] += 1
    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(limit)
    ]


def _paper_rank_key(p: dict) -> tuple:
    """Shared sort key: supporting papers first, then score, citations, year, title."""
    return (
        _paper_direction(p) == "supports",
        _paper_score(p),
        int(p.get("citations") or p.get("citation_count") or 0),
        str(p.get("year") or ""),
        _paper_title(p).lower(),
    )


def _rank_papers(papers: list[dict], limit: int = 8) -> list[dict]:
    ranked = sorted(papers, key=_paper_rank_key, reverse=True)
    rows = []
    for paper in ranked[:limit]:
        rows.append({
            "id": _paper_id(paper),
            "title": _paper_title(paper),
            "year": paper.get("year"),
            "score": _paper_score(paper),
            "direction": _paper_direction(paper),
            "citations": paper.get("citations") or paper.get("citation_count") or 0,
            "quote": _paper_quote(paper),
            "findings": _as_list(_claim_value(paper, "findings", []))[:3],
            "mechanisms": _as_list(_claim_value(paper, "mechanisms", []))[:5],
        })
    return rows


def build_hypothesis_explanation(
    clusters: list[dict],
    narratives: list[dict],
    abstract: str,
    *,
    research_config: dict | None = None,
    method_profiles: list[dict] | dict | None = None,
    grade_results: list[dict] | dict | None = None,
) -> dict:
    """Build a deterministic provenance explanation for the synthesis."""
    all_papers: list[dict] = []
    for cluster in clusters:
        all_papers.extend(cluster.get("papers", []) or [])

    method_by_id = _method_lookup(method_profiles)
    method_counts: Counter[str] = Counter()
    for paper in all_papers:
        profile = method_by_id.get(_paper_id(paper)) or method_by_id.get(str(paper.get("article_id") or ""))
        if profile:
            method_counts[str(profile.get("study_method_type") or "unknown")] += 1

    narrative_by_cluster = {
        n.get("cluster_id"): n
        for n in narratives
        if isinstance(n, dict)
    }

    cluster_rows = []
    for cluster in clusters:
        papers = cluster.get("papers", []) or []
        support_count = sum(1 for p in papers if _paper_direction(p) == "supports")
        contradict_count = sum(1 for p in papers if _paper_direction(p) == "contradicts")
        neutral_count = len(papers) - support_count - contradict_count
        avg_score = (
            sum(_paper_score(p) for p in papers) / len(papers)
            if papers else 0.0
        )
        cluster_rows.append({
            "cluster_id": cluster.get("cluster_id"),
            "paper_count": len(papers),
            "avg_relevance_score": round(avg_score, 3),
            "supporting_count": support_count,
            "contradicting_count": contradict_count,
            "neutral_count": neutral_count,
            "mechanisms": cluster.get("mechanisms", [])[:12],
            "conditions": cluster.get("conditions", [])[:12],
            "key_entities": claim_list(cluster, "key_entities")[:12],
            "top_papers": _rank_papers(papers, limit=6),
            "narrative_excerpt": (
                narrative_by_cluster.get(cluster.get("cluster_id"), {}).get("narrative", "")[:600]
            ),
        })

    cluster_rows.sort(
        key=lambda row: (
            row["supporting_count"],
            row["avg_relevance_score"],
            row["paper_count"],
        ),
        reverse=True,
    )

    support_total = sum(row["supporting_count"] for row in cluster_rows)
    contradict_total = sum(row["contradicting_count"] for row in cluster_rows)
    neutral_total = sum(row["neutral_count"] for row in cluster_rows)

    caveats: list[str] = []
    if not all_papers:
        caveats.append("No papers were available for deterministic provenance.")
    if contradict_total:
        caveats.append(
            f"{contradict_total} paper(s) were marked as contradicting or directionally opposed."
        )
    if support_total and neutral_total > support_total:
        caveats.append("Neutral or weakly directional evidence outnumbers direct support.")
    if not method_counts:
        caveats.append("Method validation output was not available for study-design weighting.")
    if not grade_results:
        caveats.append("GRADE output was not available when this explanation was generated.")

    explanation = {
        "schema_version": 1,
        "research_question": _research_question_from_config(research_config),
        "total_papers": len(all_papers),
        "cluster_count": len(clusters),
        "supporting_count": support_total,
        "contradicting_count": contradict_total,
        "neutral_count": neutral_total,
        "top_mechanisms": _top_terms(all_papers, "mechanisms"),
        "top_conditions": _top_terms(all_papers, "conditions"),
        "top_key_entities": _top_terms(all_papers, "key_entities"),
        "top_findings": _top_terms(all_papers, "findings"),
        "method_mix": dict(sorted(method_counts.items())),
        "grade_summary": _grade_summary(grade_results),
        "clusters": cluster_rows,
        "top_papers_overall": _rank_papers(all_papers, limit=10),
        "caveats": caveats,
        "abstract_excerpt": (abstract or "")[:1200],
    }
    return explanation


def render_hypothesis_explanation_markdown(explanation: dict) -> str:
    """Render the deterministic explanation artifact as readable Markdown."""
    lines = [
        "# Why This Hypothesis?",
        "",
        "## Research Question",
        "",
        str(explanation.get("research_question") or "Not specified."),
        "",
        "## Evidence Basis",
        "",
        f"- Papers in synthesis: {explanation.get('total_papers', 0)}",
        f"- Clusters: {explanation.get('cluster_count', 0)}",
        f"- Supporting signals: {explanation.get('supporting_count', 0)}",
        f"- Contradicting signals: {explanation.get('contradicting_count', 0)}",
        f"- Neutral or unclear signals: {explanation.get('neutral_count', 0)}",
        "",
    ]

    for title, key in (
        ("Top Mechanisms", "top_mechanisms"),
        ("Top Conditions or Contexts", "top_conditions"),
        ("Top Key Entities", "top_key_entities"),
    ):
        rows = explanation.get(key) or []
        if rows:
            lines.extend([f"## {title}", ""])
            for row in rows[:10]:
                lines.append(f"- {row['term']} (n={row['count']})")
            lines.append("")

    method_mix = explanation.get("method_mix") or {}
    if method_mix:
        lines.extend(["## Method Mix", ""])
        for method, count in method_mix.items():
            lines.append(f"- {method}: {count}")
        lines.append("")

    grade_summary = explanation.get("grade_summary") or {}
    if grade_summary.get("certainty_counts"):
        lines.extend(["## GRADE Snapshot", ""])
        for certainty, count in grade_summary["certainty_counts"].items():
            lines.append(f"- {certainty}: {count}")
        lines.append("")

    lines.extend(["## Top Contributing Clusters", ""])
    for cluster in explanation.get("clusters", [])[:6]:
        cid = cluster.get("cluster_id")
        lines.append(
            f"### Cluster {cid} - {cluster.get('paper_count', 0)} papers"
        )
        lines.append(
            f"Average score: {cluster.get('avg_relevance_score', 0)}; "
            f"supporting: {cluster.get('supporting_count', 0)}; "
            f"contradicting: {cluster.get('contradicting_count', 0)}"
        )
        terms = []
        for key in ("mechanisms", "conditions", "key_entities"):
            terms.extend(claim_list(cluster, key))
        if terms:
            lines.append(f"Key terms: {', '.join(terms[:10])}")
        for paper in cluster.get("top_papers", [])[:3]:
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "n.d."
            score = paper.get("score", 0)
            direction = paper.get("direction", "neutral")
            lines.append(f"- {title} ({year}) - {direction}, score {score:g}")
            quote = paper.get("quote")
            if quote:
                lines.append(f"  Quote: {quote[:280]}")
        lines.append("")

    caveats = explanation.get("caveats") or []
    if caveats:
        lines.extend(["## Deterministic Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
        lines.append("")

    if explanation.get("abstract_excerpt"):
        lines.extend(["## Synthesis Excerpt", "", str(explanation["abstract_excerpt"]).strip(), ""])

    return "\n".join(lines).rstrip() + "\n"


def _source_identity_key(paper: dict) -> str:
    """Return a stable key for collapsing source citations."""
    for key in ("doi", "pmid", "pmc", "pmcid", "s2_id", "oa_id", "wos_id", "scopus_id", "id", "article_id"):
        value = paper.get(key)
        if value:
            text = str(value).strip().lower()
            for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
            if text:
                return f"{key}:{text}"
    title = re.sub(r"\W+", " ", _paper_title(paper).lower()).strip()
    year = str(paper.get("year") or "nd")[:4]
    return f"title:{title}|year:{year}"


def _format_authors_short(authors) -> str:
    if isinstance(authors, list):
        clean = [str(author).strip() for author in authors if str(author).strip()]
    elif authors:
        clean = [part.strip() for part in str(authors).split(";") if part.strip()]
    else:
        clean = []
    if not clean:
        return "Unknown authors"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{clean[0]} et al."


def _format_source_reference(source: dict) -> str:
    authors = _format_authors_short(source.get("authors"))
    year = source.get("year") or "n.d."
    title = source.get("title") or "Untitled"
    venue = source.get("journal") or source.get("venue") or ""
    doi = source.get("doi") or ""
    parts = [f"{authors} ({year}). {title}."]
    if venue:
        parts.append(str(venue).rstrip(".") + ".")
    if doi:
        parts.append(f"doi:{doi}")
    return " ".join(parts)


def build_synthesis_source_citations(clusters: list[dict]) -> dict:
    """Build a source-citation map for the exact papers used in synthesis."""
    from process.export import assign_citation_keys

    sources_by_identity: dict[str, dict] = {}
    cluster_source_identities: dict[str, list[str]] = defaultdict(list)

    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id"))
        for paper in cluster.get("papers", []) or []:
            identity = _source_identity_key(paper)
            if identity not in cluster_source_identities[cluster_id]:
                cluster_source_identities[cluster_id].append(identity)

            if identity not in sources_by_identity:
                sources_by_identity[identity] = {
                    "identity": identity,
                    "id": _paper_id(paper),
                    "article_id": paper.get("article_id") or paper.get("id") or "",
                    "title": _paper_title(paper),
                    "authors": paper.get("authors") or [],
                    "year": paper.get("year"),
                    "journal": paper.get("journal") or paper.get("venue") or "",
                    "venue": paper.get("venue") or paper.get("journal") or "",
                    "doi": paper.get("doi") or "",
                    "pmid": paper.get("pmid") or "",
                    "pmc": paper.get("pmc") or paper.get("pmcid") or "",
                    "s2_id": paper.get("s2_id") or paper.get("paperId") or "",
                    "source": paper.get("source") or "",
                    "url": paper.get("url") or "",
                    "abstract": paper.get("abstract") or "",
                    "relevance_score": _paper_score(paper),
                    "direction": _paper_direction(paper),
                    "cluster_ids": [cluster_id],
                }
            elif cluster_id not in sources_by_identity[identity]["cluster_ids"]:
                sources_by_identity[identity]["cluster_ids"].append(cluster_id)

    source_records = list(sources_by_identity.values())

    keyed_sources = []
    citation_key_by_identity: dict[str, str] = {}
    for citation_key, source in assign_citation_keys(source_records):
        row = dict(source)
        row["citation_key"] = citation_key
        row["reference"] = _format_source_reference(row)
        keyed_sources.append(row)
        citation_key_by_identity[row["identity"]] = citation_key

    cluster_rows = []
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id"))
        source_keys = [
            citation_key_by_identity[identity]
            for identity in cluster_source_identities.get(cluster_id, [])
            if identity in citation_key_by_identity
        ]
        cluster_rows.append({
            "cluster_id": cluster.get("cluster_id"),
            "paper_count": cluster.get("paper_count", len(cluster.get("papers", []) or [])),
            "mechanisms": cluster.get("mechanisms", [])[:12],
            "conditions": cluster.get("conditions", [])[:12],
            "source_citation_keys": source_keys,
        })

    return {
        "schema_version": 1,
        "kind": "synthesis_source_citations",
        "source_count": len(keyed_sources),
        "cluster_count": len(cluster_rows),
        "clusters": cluster_rows,
        "sources": keyed_sources,
    }


def render_synthesis_source_citations_markdown(citations: dict) -> str:
    """Render synthesis source citations as Markdown grouped by cluster."""
    lines = [
        "# Sources Used for Synthesis",
        "",
        "These are the source papers that directly entered Stage 6 clustering, "
        "cluster narratives, and the synthesis abstract.",
        "",
        "Use the bracketed citation keys with `synthesis_sources.bib` when "
        "writing or revising the synthesis.",
        "",
        "## Summary",
        "",
        f"- Unique cited sources: {citations.get('source_count', 0)}",
        f"- Clusters represented: {citations.get('cluster_count', 0)}",
        "",
        "## Cluster Source Map",
        "",
    ]

    source_by_key = {
        source.get("citation_key"): source
        for source in citations.get("sources", [])
        if source.get("citation_key")
    }

    for cluster in citations.get("clusters", []):
        cid = cluster.get("cluster_id")
        lines.append(f"### Cluster {cid}")
        terms = []
        terms.extend(cluster.get("mechanisms") or [])
        terms.extend(cluster.get("conditions") or [])
        if terms:
            lines.append(f"Key terms: {', '.join(terms[:10])}")
        keys = cluster.get("source_citation_keys") or []
        if not keys:
            lines.append("- No source papers recorded for this cluster.")
        for key in keys:
            source = source_by_key.get(key, {})
            title = source.get("title") or "Untitled"
            year = source.get("year") or "n.d."
            doi = source.get("doi") or ""
            suffix = f" doi:{doi}" if doi else ""
            lines.append(f"- [{key}] {title} ({year}).{suffix}")
        lines.append("")

    lines.extend(["## Reference List", ""])
    for source in citations.get("sources", []):
        key = source.get("citation_key") or "unknown"
        lines.append(f"- [{key}] {source.get('reference') or _format_source_reference(source)}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_evidence_network(
    clusters: list[dict],
    *,
    top_papers_per_cluster: int = 8,
    max_terms_per_type: int = 80,
) -> dict:
    """Build a generalized evidence network from clustered papers."""
    nodes: dict[str, dict] = {}
    edge_weights: dict[tuple[str, str, str], float] = defaultdict(float)
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    term_counts: dict[str, Counter[str]] = {
        "mechanism": Counter(),
        "condition": Counter(),
        "entity": Counter(),
        "finding": Counter(),
        "method": Counter(),
    }

    def norm_id(kind: str, label: str) -> str:
        clean = " ".join(str(label).split()).lower()
        clean = "".join(ch if ch.isalnum() else "_" for ch in clean).strip("_")
        return f"{kind}:{clean or 'unknown'}"

    def add_node(node_id: str, label: str, node_type: str, **metadata) -> None:
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "count": 0,
                "metadata": {},
            }
        nodes[node_id]["count"] += int(metadata.pop("count", 1) or 1)
        nodes[node_id]["metadata"].update({k: v for k, v in metadata.items() if v not in (None, "")})

    def add_edge(source: str, target: str, edge_type: str, weight: float = 1.0) -> None:
        if source == target:
            return
        key = (source, target, edge_type)
        edge_weights[key] += float(weight or 1.0)
        edge_counts[key] += 1

    ranked_per_cluster = [
        sorted(
            cluster.get("papers", []) or [],
            key=_paper_rank_key,
            reverse=True,
        )[:top_papers_per_cluster]
        for cluster in clusters
    ]

    for cluster, ranked_originals in zip(clusters, ranked_per_cluster):
        cid = str(cluster.get("cluster_id"))
        cluster_id = f"cluster:{cid}"
        add_node(cluster_id, f"Cluster {cid}", "cluster", count=cluster.get("paper_count", 1))
        papers = cluster.get("papers", []) or []
        for paper in papers:
            for kind, field in (
                ("mechanism", "mechanisms"),
                ("condition", "conditions"),
                ("entity", "key_entities"),
                ("finding", "findings"),
            ):
                for value in _as_list(_claim_value(paper, field, [])):
                    term_counts[kind][value] += 1
            methodology = _claim_value(paper, "methodology", "")
            if methodology:
                for token in str(methodology).replace("/", " ").split(",")[:3]:
                    token = token.strip()
                    if token:
                        term_counts["method"][token] += 1

        for paper in ranked_originals:
            paper_id = norm_id("paper", _paper_id(paper))
            add_node(
                paper_id,
                _paper_title(paper),
                "paper",
                count=1,
                year=paper.get("year"),
                score=_paper_score(paper),
                direction=_paper_direction(paper),
            )
            add_edge(paper_id, cluster_id, "belongs_to", 1.0)

    allowed_terms = {
        kind: {term for term, _ in counter.most_common(max_terms_per_type)}
        for kind, counter in term_counts.items()
    }

    for cluster, ranked_originals in zip(clusters, ranked_per_cluster):
        cid = str(cluster.get("cluster_id"))
        cluster_id = f"cluster:{cid}"
        for paper in ranked_originals:
            pid = norm_id("paper", _paper_id(paper))
            direction = _paper_direction(paper)
            if direction in ("supports", "contradicts"):
                add_edge(pid, cluster_id, direction, max(_paper_score(paper), 1.0))
            for kind, field, edge_type in (
                ("mechanism", "mechanisms", "mentions"),
                ("condition", "conditions", "mentions"),
                ("entity", "key_entities", "mentions"),
                ("finding", "findings", "reports"),
            ):
                for value in _as_list(_claim_value(paper, field, [])):
                    if value not in allowed_terms[kind]:
                        continue
                    term_id = norm_id(kind, value)
                    if term_id not in nodes:
                        add_node(term_id, value, kind, count=term_counts[kind][value])
                    add_edge(pid, term_id, edge_type, 1.0)
                    add_edge(cluster_id, term_id, "cluster_theme", 1.0)
            methodology = _claim_value(paper, "methodology", "")
            for value in [v.strip() for v in str(methodology).replace("/", " ").split(",")[:3] if v.strip()]:
                if value not in allowed_terms["method"]:
                    continue
                method_id = norm_id("method", value)
                if method_id not in nodes:
                    add_node(method_id, value, "method", count=term_counts["method"][value])
                add_edge(pid, method_id, "uses_method", 1.0)

    edges = [
        {
            "source": source,
            "target": target,
            "type": edge_type,
            "weight": round(weight, 3),
            "evidence_count": edge_counts[(source, target, edge_type)],
        }
        for (source, target, edge_type), weight in edge_weights.items()
    ]
    edges.sort(key=lambda e: (e["source"], e["target"], e["type"]))
    node_list = sorted(nodes.values(), key=lambda n: (n["type"], n["label"].lower()))
    return {
        "schema_version": 1,
        "kind": "general_evidence_network",
        "nodes": node_list,
        "edges": edges,
        "summary": {
            "node_count": len(node_list),
            "edge_count": len(edges),
            "included_cluster_count": len(clusters),
            "top_papers_per_cluster": top_papers_per_cluster,
        },
    }


_HASHING_STOP_WORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his if in into is it
    its of on or she that the their then there these they this to was were what when which
    who will with would you your our we us not no than them""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hashing_embeddings(texts: list[str], n_features: int = 512) -> np.ndarray:
    """Embed texts with deterministic feature hashing (no third-party deps).

    This is the last-resort encoder used when neither ``sentence-transformers``
    nor ``sklearn`` is installed. It is a sublinear-tf, L2-normalized bag of
    hashed unigrams — far weaker than a real sentence embedding, but it is a
    genuine function of the text, so clustering still groups papers by shared
    vocabulary instead of by noise.

    Token buckets come from :func:`zlib.crc32` rather than the builtin
    :func:`hash`, because ``hash()`` on ``str`` is salted per interpreter
    process (``PYTHONHASHSEED``) and would make clusters irreproducible
    across runs.
    """
    matrix = np.zeros((len(texts), n_features), dtype=np.float64)
    for row, text in enumerate(texts):
        counts: Counter = Counter(
            tok
            for tok in _TOKEN_RE.findall(text.lower())
            if len(tok) > 2 and tok not in _HASHING_STOP_WORDS
        )
        for token, count in counts.items():
            bucket = zlib.crc32(token.encode("utf-8")) % n_features
            # Signed hashing cancels collision bias in expectation.
            sign = 1.0 if (zlib.crc32(b"s" + token.encode("utf-8")) & 1) else -1.0
            matrix[row, bucket] += sign * (1.0 + np.log(count))
        norm = np.linalg.norm(matrix[row])
        if norm > 0:
            matrix[row] /= norm
    return matrix


def _get_embeddings(
    papers: list[dict], model_name: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """Embed papers.  Uses sentence-transformers if available, else TF-IDF."""
    model = _get_embedding_model(model_name)
    texts = [_paper_to_text(p) for p in papers]

    if model is not None:
        logger.info("Encoding %d papers with sentence-transformers", len(texts))
        return np.asarray(model.encode(texts, show_progress_bar=False))

    # TF-IDF fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("Encoding %d papers with TF-IDF fallback", len(texts))
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_matrix: Any = vec.fit_transform(texts)
        return tfidf_matrix.toarray()
    except ImportError:
        logger.warning(
            "Neither sentence_transformers nor sklearn is installed — falling back to "
            "deterministic feature hashing for %d papers. Clusters will be based on "
            "shared vocabulary only, not semantics; install sentence-transformers or "
            "scikit-learn before relying on these clusters in a published review.",
            len(texts),
        )
        return _hashing_embeddings(texts)


# ---------------------------------------------------------------------------
# Simple k-means fallback (no sklearn)
# ---------------------------------------------------------------------------
def _simple_kmeans(
    X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42
) -> np.ndarray:
    """Minimal k-means via random partition.

    Returns an array of cluster labels (length = number of rows in X).

    Uses a locally seeded generator so repeated runs over the same corpus
    produce identical clusters. The global ``np.random`` state is never
    touched, so callers cannot perturb clustering by seeding elsewhere.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # Random initial assignment
    labels = np.array([i % k for i in range(n)])
    rng.shuffle(labels)

    for _iteration in range(max_iter):
        # Compute centroids
        centroids = np.zeros((k, X.shape[1]))
        for ci in range(k):
            members = X[labels == ci]
            if len(members) > 0:
                centroids[ci] = members.mean(axis=0)
            else:
                centroids[ci] = X[rng.integers(n)]

        # Reassign
        new_labels = np.argmin(
            np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return labels


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _select_k_by_silhouette(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 12,
    seed: int = 42,
) -> tuple[int, list[tuple[int, float]]]:
    """Pick the cluster count with the best mean silhouette score.

    Sweeps k in ``[k_min, min(k_max, n_samples - 1)]``, fits k-means for
    each candidate, and returns ``(best_k, [(k, score), ...])``. All fits
    share a fixed seed so the selection is reproducible.

    Falls back to ``min(k_max, n)`` with empty scores when sklearn is not
    installed (its silhouette implementation is what we rely on).
    """
    n = embeddings.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return max(1, min(k_min, n)), []
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning(
            "sklearn not available — cannot sweep silhouette scores; "
            "falling back to k=%d",
            k_max,
        )
        return k_max, []

    scores: list[tuple[int, float]] = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=cast(str, 10), random_state=seed)
        labels = km.fit_predict(embeddings)
        score = float(silhouette_score(embeddings, labels))
        scores.append((k, score))
        logger.info("  silhouette sweep: k=%d score=%.4f", k, score)

    best_k, best_score = max(scores, key=lambda t: t[1])
    logger.info("  Selected k=%d (silhouette %.4f)", best_k, best_score)
    return best_k, scores


def cluster_claims(
    papers: list[dict],
    n_clusters: int = 8,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    k_info: dict | None = None,
) -> list[dict]:
    """Cluster papers thematically using k-means on embeddings.

    Pass ``n_clusters=0`` (or less) to auto-select k by silhouette score
    (sweeping 2..min(12, n_papers-1)). When *k_info* is given as a dict it
    is filled with ``{"mode", "chosen_k", "silhouette_scores"}`` describing
    how k was determined.

    Returns list of cluster dicts::

        [{"cluster_id": 0, "papers": [...], "mechanisms": [...],
          "conditions": [...], "key_entities": [...],
          "primary_support_count": int}, ...]
    """
    if not papers:
        logger.warning("No papers to cluster — returning empty list")
        return []

    # Embed
    embeddings = _get_embeddings(papers, model_name=embedding_model)

    if n_clusters <= 0:
        # Auto mode: silhouette sweep over candidate k values
        actual_k, sweep = _select_k_by_silhouette(embeddings)
        if k_info is not None:
            k_info.update({
                "mode": "auto",
                "chosen_k": actual_k,
                "silhouette_scores": [
                    {"k": k, "score": round(s, 4)} for k, s in sweep
                ],
            })
    else:
        # Reduce n_clusters if fewer papers
        actual_k = min(n_clusters, len(papers))
        if actual_k < n_clusters:
            logger.info(
                "Reducing clusters from %d to %d (only %d papers)",
                n_clusters, actual_k, len(papers),
            )
        if actual_k < 1:
            actual_k = 1
        if k_info is not None:
            k_info.update({
                "mode": "fixed",
                "chosen_k": actual_k,
                "silhouette_scores": [],
            })

    # K-means
    try:
        from sklearn.cluster import KMeans
        logger.info("Running sklearn KMeans with k=%d on %d papers", actual_k, len(papers))
        # sklearn's own signature leaves n_init unannotated (default "auto"),
        # so pyright infers it as `str`; an int is a documented-valid value.
        km = KMeans(n_clusters=actual_k, n_init=cast(str, 10), random_state=42)
        labels = km.fit_predict(embeddings)
    except ImportError:
        logger.info(
            "sklearn not available — running simple k-means fallback with k=%d", actual_k)
        labels = _simple_kmeans(embeddings, actual_k)

    # Build cluster summaries
    clusters: list[dict] = []
    for ci in range(actual_k):
        indices = [i for i, lb in enumerate(labels) if lb == ci]
        cluster_papers = [papers[i] for i in indices]
        mechanisms: set[str] = set()
        conditions: set[str] = set()
        key_entities: set[str] = set()
        primary_support_count = 0

        for p in cluster_papers:
            mechanisms.update(_claim_value(p, "mechanisms", []) or [])
            conditions.update(_claim_value(p, "conditions", []) or [])
            key_entities.update(claim_list(p, "key_entities"))
            if claim_flag(p):
                primary_support_count += 1

        clusters.append({
            "cluster_id": ci,
            "papers": cluster_papers,
            "paper_count": len(cluster_papers),
            "mechanisms": sorted(mechanisms),
            "conditions": sorted(conditions),
            "key_entities": sorted(key_entities),
            "primary_support_count": primary_support_count,
        })

    # Log cluster sizes
    sizes = [c["paper_count"] for c in clusters]
    logger.info(
        "Clustered %d papers into %d clusters — sizes: %s",
        len(papers), actual_k, sizes,
    )
    return clusters


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------
def generate_cluster_narrative(
    model: str,
    cluster: dict,
    *,
    research_config: dict | None = None,
    use_thinking: bool = False,
    max_tokens: int = 4096,
) -> dict:
    """Generate a narrative for one cluster via LLM.

    Returns ``{"cluster_id": ..., "narrative": ..., "paper_count": ...}``.
    """
    from process.llm import chat as llm_chat

    paper_count = cluster.get("paper_count", len(cluster.get("papers", [])))
    mechanisms = ", ".join(cluster.get("mechanisms", [])) or "not specified"
    conditions = ", ".join(cluster.get("conditions", [])) or "not specified"
    entities = ", ".join(claim_list(cluster, "key_entities")) or "not specified"
    primary_count = cluster.get("primary_support_count", 0)
    primary_support_pct = round(100 * primary_count / paper_count) if paper_count else 0

    # Build per-paper summaries
    summaries: list[str] = []
    for p in cluster.get("papers", []):
        title = p.get("title", "Untitled")
        abstract_snippet = (p.get("abstract", "") or "")[:300]
        findings = "; ".join(p.get("findings", [])[:3])
        summaries.append(f"- {title}: {abstract_snippet}... Findings: {findings}")
    paper_summaries = "\n".join(summaries) if summaries else "No summaries available."

    prompt_text = NARRATIVE_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        paper_count=paper_count,
        mechanisms=mechanisms,
        conditions=conditions,
        entities=entities,
        primary_support_pct=primary_support_pct,
        paper_summaries=paper_summaries,
    )

    logger.info(
        "Generating narrative for cluster %d (%d papers, model=%s)",
        cluster["cluster_id"], paper_count, model,
    )

    messages = [{"role": "user", "content": prompt_text}]
    narrative = llm_chat(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        think=_synthesis_think_mode(model, enabled=use_thinking),
    )

    logger.info(
        "Cluster %d narrative: %d chars",
        cluster["cluster_id"], len(narrative),
    )

    return {
        "cluster_id": cluster["cluster_id"],
        "narrative": narrative,
        "paper_count": paper_count,
        "mechanisms": cluster.get("mechanisms", []),
        "conditions": cluster.get("conditions", []),
        "key_entities": claim_list(cluster, "key_entities"),
        "primary_support_count": primary_count,
    }


# ---------------------------------------------------------------------------
# Abstract draft
# ---------------------------------------------------------------------------
def generate_abstract_draft(
    model: str,
    cluster_narratives: list[dict],
    total_papers: int,
    *,
    research_config: dict | None = None,
    use_thinking: bool = False,
    max_tokens: int = 8192,
) -> str:
    """Generate a structured abstract from all cluster narratives."""
    from process.llm import chat as llm_chat

    # Build cluster summaries block
    parts: list[str] = []
    for cn in cluster_narratives:
        cid = cn.get("cluster_id", "?")
        n = cn.get("paper_count", 0)
        mechs = ", ".join(cn.get("mechanisms", []))
        snippet = (cn.get("narrative", "") or "")[:500]
        parts.append(
            f"Cluster {cid} ({n} papers, mechanisms: {mechs}):\n{snippet}"
        )
    cluster_summaries = "\n\n".join(parts) if parts else "No cluster summaries."

    prompt_text = ABSTRACT_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        total_papers=total_papers,
        cluster_summaries=cluster_summaries,
    )

    logger.info(
        "Generating abstract draft from %d cluster narratives (%d total papers, model=%s)",
        len(cluster_narratives), total_papers, model,
    )

    messages = [{"role": "user", "content": prompt_text}]
    abstract = llm_chat(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        think=_synthesis_think_mode(model, enabled=use_thinking),
    )

    logger.info("Abstract draft generated: %d chars", len(abstract))
    return abstract
