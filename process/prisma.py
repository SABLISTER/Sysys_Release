"""
PRISMA 2020 reporting module for the Sysys pipeline.

Collects pipeline output data into PRISMA-required counts, generates a
publication-quality PRISMA 2020 flow diagram using matplotlib, and produces
a PRISMA 2020 checklist with PRISMA-trAIce AI-transparency extensions.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.figure import Figure

from core.io import load_json as _load_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PRISMAData:
    """All counts and metadata needed for PRISMA 2020 reporting."""

    # Identification
    records_identified: int = 0
    records_per_source: dict = field(default_factory=dict)
    records_removed_duplicates: int = 0

    # Screening
    records_screened: int = 0
    records_excluded_screening: int = 0
    screening_threshold: int = 3

    # Eligibility
    reports_sought_fulltext: int = 0
    reports_not_retrieved: int = 0
    reports_assessed_eligibility: int = 0
    reports_excluded_reasons: dict = field(default_factory=dict)

    # Included
    studies_in_review: int = 0
    studies_in_synthesis: int = 0

    # AI-specific (PRISMA-trAIce)
    ai_tools: list = field(default_factory=list)
    human_oversight_points: list = field(default_factory=list)
    ai_performance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _download_log_rows(config) -> list[dict]:
    """Load full-text download records from the stage logs, if any exist."""
    candidates = [
        getattr(config, "data_dir", None) and config.data_dir / "fulltext" / "download_log.json",
        getattr(config, "data_dir", None) and config.data_dir / "fulltext_corpus" / "download_log.json",
    ]
    for path in candidates:
        if path is None:
            continue
        payload = _load_json(path)
        rows: list[dict] = []
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            for key in ("downloads", "papers", "results", "records"):
                inner = payload.get(key)
                if isinstance(inner, list):
                    rows = [row for row in inner if isinstance(row, dict)]
                    break
        if rows:
            return rows
    return []


def _report_was_retrieved(row: dict) -> bool:
    """True when a download-log row says the full text was obtained."""
    if row.get("success") is True:
        return True
    if row.get("success") is False:
        return False
    status = str(row.get("status") or row.get("download_status") or "").strip().lower()
    if status in {"ok", "success", "downloaded", "cached"}:
        return True
    if row.get("file") or row.get("saved_file") or row.get("text"):
        return status not in {"failed", "no-text", "error", "not_found", "missing"}
    return False


def collect_prisma_data(
    config,
    interrater_provider: str | None = None,
    interrater_model: str | None = None,
) -> PRISMAData:
    """Walk all pipeline output files and collect PRISMA-required counts.

    Handles missing files gracefully — returns 0 for stages that have not run.
    """
    data = PRISMAData()

    # -- Identification: raw search counts ---------------------------------
    # The AnalysisSession stores raw_search_counts, but it may also be
    # persisted as part of the session JSON.  Try the session file first,
    # then fall back to reading corpus.json length.
    session_path = config.output_dir / "analysis_session.json"
    session = _load_json(session_path)
    if isinstance(session, dict):
        rsc = session.get("raw_search_counts", {})
        if isinstance(rsc, dict):
            data.records_per_source = {str(k): int(v) for k, v in rsc.items()}
            data.records_identified = sum(data.records_per_source.values())

    # -- Screening: corpus → relevant --------------------------------------
    corpus = _load_json(config.corpus_dir / "corpus.json")
    relevant = _load_json(config.claims_dir / "relevant.json")

    if isinstance(corpus, list):
        corpus_count = len(corpus)
    elif isinstance(corpus, dict):
        corpus_count = len(corpus.get("articles", corpus))
    else:
        corpus_count = 0

    if isinstance(relevant, list):
        relevant_count = len(relevant)
    elif isinstance(relevant, dict):
        relevant_count = len(relevant.get("articles", relevant))
    else:
        relevant_count = 0

    # If we don't have raw_search_counts, approximate from corpus
    if data.records_identified == 0 and corpus_count > 0:
        data.records_identified = corpus_count

    data.records_screened = corpus_count
    data.records_removed_duplicates = max(
        data.records_identified - corpus_count, 0
    )
    data.records_excluded_screening = max(corpus_count - relevant_count, 0)
    data.screening_threshold = getattr(config, "relevance_threshold", 3)

    # -- Eligibility: claims_filtered + removal_log ------------------------
    claims_filtered = _load_json(config.claims_dir / "claims_filtered.json")
    removal_log = _load_json(config.clusters_dir / "removal_log.json")

    if isinstance(claims_filtered, list):
        claims_count = len(claims_filtered)
    elif isinstance(claims_filtered, dict):
        claims_count = len(claims_filtered)
    else:
        claims_count = 0

    download_rows = _download_log_rows(config)
    retrieved = sum(1 for row in download_rows if _report_was_retrieved(row))
    if download_rows:
        data.reports_sought_fulltext = len(download_rows)
        data.reports_not_retrieved = len(download_rows) - retrieved
        data.reports_assessed_eligibility = retrieved
    else:
        data.reports_sought_fulltext = relevant_count
        data.reports_not_retrieved = 0
        data.reports_assessed_eligibility = relevant_count

    # Parse removal reasons
    exclusion_reasons: dict[str, int] = {}
    if isinstance(removal_log, list):
        for entry in removal_log:
            if isinstance(entry, dict):
                reason = str(entry.get("reason", "unspecified"))
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
    elif isinstance(removal_log, dict):
        for _aid, reason in removal_log.items():
            r = str(reason) if not isinstance(reason, dict) else str(reason.get("reason", "unspecified"))
            exclusion_reasons[r] = exclusion_reasons.get(r, 0) + 1
    data.reports_excluded_reasons = exclusion_reasons

    # -- Included: clusters / synthesis ------------------------------------
    clusters = _load_json(config.clusters_dir / "clusters.json")

    if isinstance(clusters, list):
        # Count unique article ids across clusters
        article_ids = set()
        for cl in clusters:
            if isinstance(cl, dict):
                for member in cl.get("members", cl.get("articles", [])):
                    if isinstance(member, str):
                        article_ids.add(member)
                    elif isinstance(member, dict):
                        article_ids.add(member.get("article_id", ""))
        data.studies_in_synthesis = len(article_ids) if article_ids else len(clusters)
    elif isinstance(clusters, dict):
        data.studies_in_synthesis = len(clusters)

    data.studies_in_review = claims_count if claims_count > 0 else relevant_count
    if data.studies_in_synthesis == 0:
        data.studies_in_synthesis = data.studies_in_review

    # -- AI tools ----------------------------------------------------------
    provider = getattr(config, "llm_provider", "unknown")
    model = getattr(config, "llm_model", "unknown")
    data.ai_tools = [
        {
            "name": f"{provider} / {model}",
            "role": "relevance scoring, claim extraction, synthesis",
            "version": model,
        }
    ]

    # Interrater second rater (explicit args preferred; config attrs are a
    # fallback for non-GUI callers that set them).
    ir_provider = interrater_provider or getattr(config, "interrater_provider", None)
    if ir_provider:
        ir_model = (
            interrater_model
            or getattr(config, "interrater_model", None)
            or "unknown"
        )
        data.ai_tools.append({
            "name": f"{ir_provider} / {ir_model}",
            "role": "interrater reliability (second rater)",
            "version": ir_model,
        })

    data.human_oversight_points = [
        "Hypothesis and search strategy defined by researcher",
        "Relevance threshold set manually",
        "Span-verification provides auditable evidence trail",
        "Audit stage flags duplicate / suspicious spans for review",
        "Interrater reliability compares LLM raters",
        "Robustness analysis tests conclusion stability",
    ]

    # -- AI performance metrics --------------------------------------------
    interrater = _load_json(config.clusters_dir / "interrater_results.json")
    if isinstance(interrater, dict):
        data.ai_performance = {
            k: interrater[k]
            for k in ("kappa", "percent_agreement", "mean_agreement_score",
                       "n_sampled", "agreement_score")
            if k in interrater
        }
    elif isinstance(interrater, list) and interrater:
        # Aggregate from per-article results
        agreements = [
            r.get("agreement_score", 0)
            for r in interrater
            if isinstance(r, dict) and "agreement_score" in r
        ]
        if agreements:
            data.ai_performance["mean_agreement_score"] = (
                sum(agreements) / len(agreements)
            )
            data.ai_performance["n_sampled"] = len(agreements)

    return data


# ---------------------------------------------------------------------------
# Flow diagram
# ---------------------------------------------------------------------------

def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    header: str = "",
    fontsize: float = 8,
    header_fontsize: float = 9,
    facecolor: str = "#ffffff",
    header_color: str = "#000000",
):
    """Draw a rounded box with optional bold header line."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor,
        edgecolor="#000000",
        linewidth=1.0,
    )
    ax.add_patch(box)
    if header:
        ax.text(
            x + w / 2, y + h - 0.025, header,
            ha="center", va="top",
            fontsize=header_fontsize, fontweight="bold", color=header_color,
        )
        ax.text(
            x + w / 2, y + h / 2 - 0.015, text,
            ha="center", va="center",
            fontsize=fontsize, color="#333333",
            wrap=True,
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color="#333333",
        )


def _draw_arrow(ax, x1, y1, x2, y2):
    """Draw a simple arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#000000",
            lw=1.2,
            connectionstyle="arc3,rad=0",
        ),
    )


def _source_breakdown(data: PRISMAData) -> str:
    if not data.records_per_source:
        return ""
    parts = [f"{src}: {n}" for src, n in sorted(data.records_per_source.items())]
    return "\n".join(parts)


def _render_flow_diagram(
    data: PRISMAData,
    figsize: tuple[float, float],
    dpi: int,
    title_fontsize: int,
) -> Figure:
    """Shared PRISMA 2020 flow-diagram renderer (single layout source)."""
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.97, "PRISMA 2020 Flow Diagram",
        ha="center", va="top", fontsize=title_fontsize, fontweight="bold",
    )

    # Layout constants
    lx, lw = 0.08, 0.48   # left column
    rx, rw = 0.62, 0.32   # right column (exclusion boxes)
    bh = 0.10              # box height
    bh_small = 0.08

    # ── IDENTIFICATION ────────────────────────────────────────────────
    y_id = 0.84
    source_text = _source_breakdown(data)
    id_body = f"Records identified (n = {data.records_identified})"
    if source_text:
        id_body += "\n" + source_text
    _draw_box(ax, lx, y_id, lw, bh, id_body,
              header="IDENTIFICATION", facecolor="#f0f8ff")

    # ── Duplicates removed ────────────────────────────────────────────
    y_dedup = 0.72
    _draw_box(
        ax, lx, y_dedup, lw, bh_small,
        f"Records after duplicates removed (n = {data.records_screened})\n"
        f"[duplicates removed: {data.records_removed_duplicates}]",
    )
    _draw_arrow(ax, lx + lw / 2, y_id, lx + lw / 2, y_dedup + bh_small)

    # ── SCREENING ─────────────────────────────────────────────────────
    y_scr = 0.58
    _draw_box(
        ax, lx, y_scr, lw, bh,
        f"Records screened\n(n = {data.records_screened})",
        header="SCREENING", facecolor="#f0fff0",
    )
    _draw_arrow(ax, lx + lw / 2, y_dedup, lx + lw / 2, y_scr + bh)

    # Exclusion box (right)
    _draw_box(
        ax, rx, y_scr + 0.01, rw, bh_small,
        f"Records excluded\n(n = {data.records_excluded_screening}, "
        f"below threshold {data.screening_threshold})",
    )
    _draw_arrow(ax, lx + lw, y_scr + bh / 2, rx, y_scr + 0.01 + bh_small / 2)

    # ── ELIGIBILITY ───────────────────────────────────────────────────
    y_el = 0.42
    _draw_box(
        ax, lx, y_el, lw, bh,
        f"Full-text reports assessed\n(n = {data.reports_assessed_eligibility})",
        header="ELIGIBILITY", facecolor="#fffff0",
    )
    _draw_arrow(ax, lx + lw / 2, y_scr, lx + lw / 2, y_el + bh)

    # Exclusion reasons box (right)
    total_excl = sum(data.reports_excluded_reasons.values())
    reason_lines = [f"Reports excluded (n = {total_excl})"]
    for reason, count in sorted(
        data.reports_excluded_reasons.items(), key=lambda x: -x[1]
    )[:5]:
        reason_lines.append(f"  {reason}: {count}")
    excl_text = "\n".join(reason_lines) if reason_lines else "Reports excluded (n = 0)"
    excl_h = max(bh_small, 0.02 * len(reason_lines) + 0.04)
    _draw_box(ax, rx, y_el + 0.01, rw, excl_h, excl_text, fontsize=7)
    _draw_arrow(
        ax, lx + lw, y_el + bh / 2,
        rx, y_el + 0.01 + excl_h / 2,
    )

    # ── INCLUDED ──────────────────────────────────────────────────────
    y_inc = 0.24
    _draw_box(
        ax, lx, y_inc, lw, bh + 0.02,
        f"Studies included in review (n = {data.studies_in_review})\n"
        f"Studies included in synthesis (n = {data.studies_in_synthesis})",
        header="INCLUDED", facecolor="#fff0f5",
    )
    _draw_arrow(ax, lx + lw / 2, y_el, lx + lw / 2, y_inc + bh + 0.02)

    # ── AI transparency note ──────────────────────────────────────────
    if data.ai_tools:
        tool_names = ", ".join(t.get("name", "?") for t in data.ai_tools)
        ai_note = f"AI tools used: {tool_names}"
        ax.text(
            0.5, 0.17, ai_note,
            ha="center", va="top", fontsize=7, fontstyle="italic",
            color="#666666",
        )

    return fig


def generate_flow_diagram(
    data: PRISMAData,
    output_path: Path,
    fmt: str = "png",
) -> Path:
    """Generate a PRISMA 2020 flow diagram using matplotlib.

    Returns the *Path* to the saved file.
    """
    fig = _render_flow_diagram(data, figsize=(10, 12), dpi=150,
                               title_fontsize=14)
    # ── Save ──────────────────────────────────────────────────────────
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{fmt}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format=fmt, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("PRISMA flow diagram saved to %s", output_path)
    return output_path


def generate_flow_diagram_figure(data: PRISMAData) -> Figure:
    """Generate the PRISMA flow diagram and return the matplotlib Figure.

    Used by the GUI to embed in FigureCanvasQTAgg without saving to disk.
    """
    return _render_flow_diagram(data, figsize=(8, 10), dpi=100,
                                title_fontsize=13)


# ---------------------------------------------------------------------------
# Checklist
# ---------------------------------------------------------------------------

_PRISMA_ITEMS = [
    ("Title", 1, "Identify the report as a systematic review"),
    ("Title", 2, "Provide a structured summary including background, objectives, methods, results, and conclusions"),
    ("Introduction", 3, "Describe the rationale for the review in the context of existing knowledge"),
    ("Introduction", 4, "Provide an explicit statement of the question being addressed"),
    ("Methods", 5, "Specify the eligibility criteria and information sources"),
    ("Methods", 6, "Describe all information sources searched"),
    ("Methods", 7, "Present the full search strategy for at least one database"),
    ("Methods", 8, "Describe the process for selecting studies"),
    ("Methods", 9, "Describe the process for data extraction"),
    ("Methods", 10, "List and define all variables for which data were sought"),
    ("Methods", 11, "Describe methods used to assess risk of bias"),
    ("Methods", 12, "Describe methods for handling data and combining results"),
    ("Methods", 13, "Describe the methods of synthesis"),
    ("Methods", 14, "Describe any methods used to explore publication bias"),
    ("Methods", 15, "Describe methods of additional analyses (sensitivity, subgroup)"),
    ("Results", 16, "Give numbers of studies screened, assessed, and included, with reasons for exclusions"),
    ("Results", 17, "For each study, present characteristics and cite the study"),
    ("Results", 18, "Present data on risk of bias for each study"),
    ("Results", 19, "Present all collected data for each outcome considered"),
    ("Results", 20, "Present results of each synthesis undertaken"),
    ("Results", 21, "Present results of additional analyses"),
    ("Results", 22, "Present results of risk of bias across studies"),
    ("Discussion", 23, "Summarise main findings including strength of evidence"),
    ("Discussion", 24, "Discuss limitations at study and outcome level"),
    ("Discussion", 25, "Provide a general interpretation in the context of other evidence"),
    ("Other", 26, "Describe sources of funding and role of funders"),
    ("Other", 27, "Indicate registration and where the protocol can be accessed"),
]

_AUTO_ITEMS = {1, 5, 6, 7, 8, 13, 16, 17, 20}


def generate_checklist(data: PRISMAData) -> list[dict]:
    """Generate a PRISMA 2020 checklist (27 items + PRISMA-trAIce extension)."""
    checklist: list[dict] = []

    for section, item, description in _PRISMA_ITEMS:
        entry: dict[str, Any] = {
            "section": section,
            "item": item,
            "description": description,
            "status": "auto" if item in _AUTO_ITEMS else "manual",
            "content": "",
            "prisma_traice": False,
        }

        # Auto-fill where pipeline data is available
        if item == 1:
            entry["content"] = "Systematic review of evidence (AI-assisted)"
        elif item == 5:
            entry["content"] = (
                f"Relevance threshold >= {data.screening_threshold} on 0-5 scale. "
                f"LLM-scored abstracts with span verification."
            )
        elif item == 6:
            sources = ", ".join(data.records_per_source.keys()) if data.records_per_source else "configured databases"
            entry["content"] = f"Information sources: {sources}"
        elif item == 7:
            entry["content"] = "Search queries auto-generated from hypothesis components and co-occurring conditions"
        elif item == 8:
            entry["content"] = (
                f"LLM relevance scoring (threshold {data.screening_threshold}); "
                f"span verification; claim extraction and audit"
            )
        elif item == 13:
            entry["content"] = (
                "Narrative synthesis via LLM-generated cluster summaries, "
                "grouped by hypothesis component"
            )
        elif item == 16:
            entry["content"] = (
                f"Identified: {data.records_identified}; "
                f"Screened: {data.records_screened}; "
                f"Excluded (screening): {data.records_excluded_screening}; "
                f"Assessed (eligibility): {data.reports_assessed_eligibility}; "
                f"Included in review: {data.studies_in_review}; "
                f"Included in synthesis: {data.studies_in_synthesis}"
            )
        elif item == 17:
            entry["content"] = (
                f"{data.studies_in_review} studies with extracted claims, "
                f"study designs classified via LLM"
            )
        elif item == 20:
            entry["content"] = (
                f"{data.studies_in_synthesis} studies included in synthesis "
                f"across hypothesis components"
            )

        checklist.append(entry)

    # PRISMA-trAIce extension items
    traice_items = [
        {
            "section": "AI Transparency",
            "item": 28,
            "description": "AI tool identification — name, version, and provider of each AI tool used",
            "status": "auto",
            "content": "; ".join(
                f"{t['name']} ({t['role']})" for t in data.ai_tools
            ) if data.ai_tools else "",
            "prisma_traice": True,
        },
        {
            "section": "AI Transparency",
            "item": 29,
            "description": "AI role description — what each AI tool did at each pipeline stage",
            "status": "auto",
            "content": (
                "Relevance scoring (abstract/fulltext), claim extraction, "
                "study design classification, narrative synthesis, "
                "interrater reliability verification"
            ),
            "prisma_traice": True,
        },
        {
            "section": "AI Transparency",
            "item": 30,
            "description": "Human-AI interaction points — where human oversight occurred",
            "status": "auto",
            "content": "; ".join(data.human_oversight_points) if data.human_oversight_points else "",
            "prisma_traice": True,
        },
        {
            "section": "AI Transparency",
            "item": 31,
            "description": "AI performance metrics — agreement scores and verification rates",
            "status": "auto" if data.ai_performance else "manual",
            "content": "; ".join(
                f"{k}: {v}" for k, v in data.ai_performance.items()
            ) if data.ai_performance else "",
            "prisma_traice": True,
        },
    ]
    checklist.extend(traice_items)

    return checklist


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_checklist_markdown(checklist: list[dict], output_path: Path) -> Path:
    """Save checklist as a nicely formatted markdown file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PRISMA 2020 Checklist",
        "",
        "| # | Section | Description | Status | Content |",
        "|---|---------|-------------|--------|---------|",
    ]
    for item in checklist:
        traice = " (trAIce)" if item.get("prisma_traice") else ""
        content = item["content"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['item']} | {item['section']}{traice} | "
            f"{item['description']} | {item['status']} | {content} |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("PRISMA checklist (Markdown) saved to %s", output_path)
    return output_path


def save_checklist_csv(checklist: list[dict], output_path: Path) -> Path:
    """Save checklist as CSV for easy editing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Item", "Section", "Description", "Status", "Content", "PRISMA-trAIce"])
        for item in checklist:
            writer.writerow([
                item["item"],
                item["section"],
                item["description"],
                item["status"],
                item["content"],
                "Yes" if item.get("prisma_traice") else "No",
            ])

    logger.info("PRISMA checklist (CSV) saved to %s", output_path)
    return output_path
