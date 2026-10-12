"""Stage 6: Cluster claims, generate narratives, draft abstract."""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from core.io import atomic_write_json, atomic_write_text
from core.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def run(
    config=None,
    claims_path: Optional[str] = None,
    n_clusters: int = 8,
    embedding_model: Optional[str] = None,
    max_tokens: int = 4096,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kw,
) -> dict:
    """Run Stage 6: cluster claims, generate per-cluster narratives, draft abstract.

    Steps
    -----
    1. Load claims (filtered or unfiltered)
    2. Cluster papers with embeddings
    3. Generate narrative per cluster via LLM
    4. Draft structured abstract via LLM
    5. Save clusters.json, narratives.json, abstract_draft.md

    Returns
    -------
    dict
        Summary with keys: total_papers, n_clusters, narratives_count,
        abstract_length, output paths.
    """
    from process.llm import set_config
    from process.synthesizer import (
        build_evidence_network,
        build_hypothesis_explanation,
        build_synthesis_source_citations,
        cluster_claims,
        generate_cluster_narrative,
        generate_abstract_draft,
        render_hypothesis_explanation_markdown,
        render_synthesis_source_citations_markdown,
    )
    from process.export import export_bibtex
    from process.paths import resolve_claims_path

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    def _cancelled() -> bool:
        if cancel_event is None:
            return False
        if hasattr(cancel_event, "is_set"):
            return cancel_event.is_set()
        return bool(cancel_event)

    # ── 0. Configure LLM ────────────────────────────────────────────────
    if config is None:
        raise ValueError("config is required but was not provided")
    set_config(config)
    model_name = config.llm_model
    _progress(f"LLM: provider={config.llm_provider}, model={model_name}")

    emb_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    # ── 1. Load claims ──────────────────────────────────────────────────
    if claims_path:
        cp = Path(claims_path)
    else:
        cp = resolve_claims_path(config)
    _progress(f"Loading claims from {cp}")

    with open(cp, encoding="utf-8") as f:
        papers = json.load(f)

    if not isinstance(papers, list):
        papers = [papers]

    _progress(f"Loaded {len(papers)} papers")

    if not papers:
        _progress("No papers found — nothing to synthesize")
        return {
            "total_papers": 0,
            "n_clusters": 0,
            "narratives_count": 0,
            "abstract_length": 0,
        }

    if _cancelled():
        _progress("Cancelled before clustering")
        return {"cancelled": True}

    # ── 2. Cluster ──────────────────────────────────────────────────────
    if n_clusters and n_clusters >= 1:
        _progress(f"Clustering {len(papers)} papers into up to {n_clusters} clusters "
                  f"(embedding model: {emb_model})")
    else:
        _progress(f"Clustering {len(papers)} papers with auto k "
                  f"(silhouette sweep, embedding model: {emb_model})")

    k_info: dict = {}
    clusters = cluster_claims(
        papers,
        n_clusters=n_clusters,
        embedding_model=emb_model,
        k_info=k_info,
    )

    if k_info.get("mode") == "auto":
        sweep = k_info.get("silhouette_scores", [])
        if sweep:
            _progress("Silhouette sweep: " + ", ".join(
                f"k={s['k']}:{s['score']:.3f}" for s in sweep))
        _progress(f"Auto-selected {k_info.get('chosen_k')} clusters by silhouette score")

    cluster_sizes = [c["paper_count"] for c in clusters]
    _progress(f"Formed {len(clusters)} clusters — sizes: {cluster_sizes}")

    if _cancelled():
        _progress("Cancelled after clustering")
        return {"cancelled": True}

    # ── 3. Generate per-cluster narratives ──────────────────────────────
    narratives: list[dict] = []
    for i, cluster in enumerate(clusters):
        if _cancelled():
            _progress(f"Cancelled during narrative generation (cluster {i})")
            return {"cancelled": True}

        _progress(
            f"Generating narrative for cluster {i + 1}/{len(clusters)} "
            f"({cluster['paper_count']} papers)"
        )
        narrative = generate_cluster_narrative(
            model=model_name,
            cluster=cluster,
            research_config=getattr(config, "research_config", None),
            max_tokens=max_tokens,
        )
        narratives.append(narrative)
        _progress(
            f"Cluster {i + 1} narrative complete: {len(narrative.get('narrative', ''))} chars"
        )

    _progress(f"All {len(narratives)} cluster narratives generated")

    if _cancelled():
        _progress("Cancelled before abstract generation")
        return {"cancelled": True}

    # ── 4. Draft abstract ───────────────────────────────────────────────
    _progress("Drafting structured abstract from cluster narratives")
    abstract = generate_abstract_draft(
        model=model_name,
        cluster_narratives=narratives,
        total_papers=len(papers),
        research_config=getattr(config, "research_config", None),
        max_tokens=max_tokens * 2,
    )
    _progress(f"Abstract draft generated: {len(abstract)} chars")

    # ── 5. Save outputs ────────────────────────────────────────────────
    config.ensure_dirs()
    out_dir = config.clusters_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # clusters.json — strip full paper dicts to keep file small
    clusters_light = []
    for c in clusters:
        cl = dict(c)
        cl["paper_titles"] = [p.get("title", "") for p in c.get("papers", [])]
        cl.pop("papers", None)
        clusters_light.append(cl)

    clusters_path = out_dir / "clusters.json"
    atomic_write_json(clusters_path, clusters_light)
    _progress(f"Saved clusters: {clusters_path}")

    narratives_path = out_dir / "narratives.json"
    atomic_write_json(narratives_path, narratives)
    _progress(f"Saved narratives: {narratives_path}")

    if k_info:
        k_selection_path = out_dir / "k_selection.json"
        atomic_write_json(k_selection_path, k_info)
        _progress(f"Saved cluster-count selection: {k_selection_path}")

    abstract_path = out_dir / "abstract_draft.md"
    atomic_write_text(abstract_path, abstract)
    _progress(f"Saved abstract draft: {abstract_path}")

    method_profiles = _load_optional_json(config.data_dir / "method_review" / "method_profiles_validated.json")
    grade_results = _load_optional_json(config.output_dir / "grade_results.json")

    explanation = build_hypothesis_explanation(
        clusters,
        narratives,
        abstract,
        research_config=getattr(config, "research_config", None),
        method_profiles=method_profiles,
        grade_results=grade_results,
    )
    explanation_path = out_dir / "hypothesis_explanation.json"
    atomic_write_json(explanation_path, explanation)
    _progress(f"Saved hypothesis explanation: {explanation_path}")

    explanation_md_path = out_dir / "hypothesis_explanation.md"
    atomic_write_text(
        explanation_md_path,
        render_hypothesis_explanation_markdown(explanation),
    )
    _progress(f"Saved hypothesis explanation Markdown: {explanation_md_path}")

    evidence_network = build_evidence_network(clusters)
    network_path = out_dir / "evidence_network.json"
    atomic_write_json(network_path, evidence_network)
    _progress(f"Saved evidence network: {network_path}")

    source_citations = build_synthesis_source_citations(clusters)
    source_citations_path = out_dir / "synthesis_sources.json"
    atomic_write_json(source_citations_path, source_citations)
    _progress(f"Saved synthesis source map: {source_citations_path}")

    source_citations_md_path = out_dir / "synthesis_sources.md"
    atomic_write_text(
        source_citations_md_path,
        render_synthesis_source_citations_markdown(source_citations),
    )
    _progress(f"Saved synthesis source citations: {source_citations_md_path}")

    source_citations_bib_path = out_dir / "synthesis_sources.bib"
    export_bibtex(source_citations.get("sources", []), source_citations_bib_path)
    _progress(f"Saved synthesis source BibTeX: {source_citations_bib_path}")

    _progress("Stage 6 complete")

    return {
        "total_papers": len(papers),
        "n_clusters": len(clusters),
        "narratives_count": len(narratives),
        "abstract_length": len(abstract),
        "k_selection": k_info,
        "clusters_path": str(clusters_path),
        "narratives_path": str(narratives_path),
        "abstract_path": str(abstract_path),
        "hypothesis_explanation_path": str(explanation_path),
        "hypothesis_explanation_md_path": str(explanation_md_path),
        "evidence_network_path": str(network_path),
        "synthesis_sources_path": str(source_citations_path),
        "synthesis_sources_md_path": str(source_citations_md_path),
        "synthesis_sources_bib_path": str(source_citations_bib_path),
    }


def check_output(config) -> dict:
    """Check whether Stage 6 outputs exist."""
    out_dir = config.clusters_dir
    clusters = out_dir / "clusters.json"
    narratives = out_dir / "narratives.json"
    abstract = out_dir / "abstract_draft.md"
    return {
        "clusters_exists": clusters.exists(),
        "narratives_exists": narratives.exists(),
        "abstract_exists": abstract.exists(),
        "hypothesis_explanation_exists": (out_dir / "hypothesis_explanation.json").exists(),
        "evidence_network_exists": (out_dir / "evidence_network.json").exists(),
        "synthesis_sources_exists": (out_dir / "synthesis_sources.json").exists(),
        "synthesis_sources_md_exists": (out_dir / "synthesis_sources.md").exists(),
        "synthesis_sources_bib_exists": (out_dir / "synthesis_sources.bib").exists(),
    }


def _load_optional_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not load optional synthesis context: %s", path)
    return None
