"""Export utilities for Sysys pipeline outputs.

Provides BibTeX, RIS, and EndNote XML bibliography export, plus a one-click
report bundle that collects all pipeline artifacts into a single directory.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

logger = logging.getLogger(__name__)


# ── Corpus loading ────────────────────────────────────────────────────────


def load_corpus_for_export(config) -> list[dict]:
    """Load the best available corpus (claims_filtered > relevant > corpus).

    Falls back through progressively less-filtered corpus files so that
    export works at any pipeline stage.
    """
    candidates = [
        config.claims_dir / "claims_filtered.json",
        config.claims_dir / "relevant.json",
        config.corpus_dir / "corpus.json",
        config.data_dir / "corpus.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "articles" in data:
                    return data["articles"]
                return []
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load corpus from %s: %s", path, exc)
    return []


# ── Citation key generation ───────────────────────────────────────────────


def _make_citation_key(paper: dict) -> str:
    """Generate a citation key from first author surname + year."""
    existing = str(paper.get("citation_key") or "").strip()
    if existing:
        clean = re.sub(r"[^a-zA-Z0-9_:-]", "", existing)
        if clean:
            return clean

    authors = paper.get("authors", "")
    if isinstance(authors, list):
        first = authors[0] if authors else "unknown"
    else:
        first = str(authors).split(",")[0].strip() if authors else "unknown"

    # Extract surname (last word of the author name)
    surname = first.split()[-1] if first.strip() else "unknown"
    surname = re.sub(r"[^a-zA-Z]", "", surname).lower() or "unknown"

    year = str(paper.get("year", ""))[:4]
    if not year.isdigit():
        year = "nd"

    return f"{surname}{year}"


def assign_citation_keys(corpus: list[dict]) -> list[tuple[str, dict]]:
    """Assign stable, unique citation keys to paper records."""
    used_keys: dict[str, int] = {}
    keyed: list[tuple[str, dict]] = []
    for paper in corpus:
        base_key = _make_citation_key(paper)
        count = used_keys.get(base_key, 0)
        used_keys[base_key] = count + 1
        key = base_key if count == 0 else f"{base_key}{chr(97 + count)}"
        keyed.append((key, paper))
    return keyed


def _escape_bibtex(text: str) -> str:
    """Escape special LaTeX/BibTeX characters."""
    if not text:
        return ""
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _format_bibtex_authors(authors) -> str:
    """Format author list for BibTeX (Author, First and Author2, First2)."""
    if not authors:
        return ""
    if isinstance(authors, str):
        return _escape_bibtex(authors)
    return " and ".join(_escape_bibtex(str(a)) for a in authors)


# ── BibTeX export ─────────────────────────────────────────────────────────


def export_bibtex(corpus: list[dict], output_path: Path) -> Path:
    """Export corpus as BibTeX (.bib).

    Generates one @article entry per paper with title, author, year,
    journal, doi, and abstract fields.  Missing fields are omitted.
    Duplicate citation keys get a letter suffix (smith2023a, smith2023b).
    """
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".bib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for key, paper in assign_citation_keys(corpus):
        entry_lines = [f"@article{{{key},"]

        title = paper.get("title", "")
        if title:
            entry_lines.append(f"    title = {{{_escape_bibtex(title)}}},")

        authors = paper.get("authors", "")
        if authors:
            entry_lines.append(f"    author = {{{_format_bibtex_authors(authors)}}},")

        year = paper.get("year", "")
        if year:
            entry_lines.append(f"    year = {{{year}}},")

        journal = paper.get("journal", "")
        if journal:
            entry_lines.append(f"    journal = {{{_escape_bibtex(journal)}}},")

        doi = paper.get("doi", "")
        if doi:
            entry_lines.append(f"    doi = {{{doi}}},")

        abstract = paper.get("abstract", "")
        if abstract:
            entry_lines.append(f"    abstract = {{{_escape_bibtex(abstract)}}},")

        entry_lines.append("}")
        lines.append("\n".join(entry_lines))

    output_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Exported %d entries to BibTeX: %s", len(corpus), output_path)
    return output_path


# ── RIS export ────────────────────────────────────────────────────────────


def export_ris(corpus: list[dict], output_path: Path) -> Path:
    """Export corpus as RIS (.ris).

    RIS format uses tagged lines (TY, TI, AU, PY, JO, DO, AB, ER).
    Each author gets a separate AU tag.
    """
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".ris")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[str] = []

    for paper in corpus:
        lines: list[str] = ["TY  - JOUR"]

        title = paper.get("title", "")
        if title:
            lines.append(f"TI  - {title}")

        authors = paper.get("authors", "")
        if isinstance(authors, list):
            for author in authors:
                lines.append(f"AU  - {author}")
        elif authors:
            for author in str(authors).split(";"):
                author = author.strip()
                if author:
                    lines.append(f"AU  - {author}")

        year = paper.get("year", "")
        if year:
            lines.append(f"PY  - {year}")

        journal = paper.get("journal", "")
        if journal:
            lines.append(f"JO  - {journal}")

        doi = paper.get("doi", "")
        if doi:
            lines.append(f"DO  - {doi}")

        abstract = paper.get("abstract", "")
        if abstract:
            lines.append(f"AB  - {abstract}")

        lines.append("ER  - ")
        entries.append("\n".join(lines))

    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    logger.info("Exported %d entries to RIS: %s", len(corpus), output_path)
    return output_path


# ── EndNote XML export ────────────────────────────────────────────────────


def export_endnote_xml(corpus: list[dict], output_path: Path) -> Path:
    """Export corpus as EndNote XML.

    Generates an XML file following the EndNote import schema with
    record, title, authors, year, periodical, electronic-resource-num,
    and abstract elements.
    """
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".xml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_root = Element("xml")
    records = SubElement(xml_root, "records")

    for paper in corpus:
        record = SubElement(records, "record")

        # Reference type
        ref_type = SubElement(record, "ref-type", name="Journal Article")
        ref_type.text = "17"

        # Contributors / authors
        contributors = SubElement(record, "contributors")
        authors_el = SubElement(contributors, "authors")
        authors = paper.get("authors", "")
        if isinstance(authors, list):
            for author in authors:
                a_el = SubElement(authors_el, "author")
                a_el.text = str(author)
        elif authors:
            for author in str(authors).split(";"):
                author = author.strip()
                if author:
                    a_el = SubElement(authors_el, "author")
                    a_el.text = author

        # Title
        titles = SubElement(record, "titles")
        title_el = SubElement(titles, "title")
        title_el.text = paper.get("title", "")

        # Journal
        journal = paper.get("journal", "")
        if journal:
            periodical = SubElement(titles, "secondary-title")
            periodical.text = journal

        # Year
        year = paper.get("year", "")
        if year:
            dates = SubElement(record, "dates")
            year_el = SubElement(dates, "year")
            year_el.text = str(year)

        # DOI
        doi = paper.get("doi", "")
        if doi:
            ern = SubElement(record, "electronic-resource-num")
            ern.text = doi

        # Abstract
        abstract = paper.get("abstract", "")
        if abstract:
            abs_el = SubElement(record, "abstract")
            abs_el.text = abstract

    indent(xml_root, space="  ")
    tree = ElementTree(xml_root)
    tree.write(str(output_path), encoding="unicode", xml_declaration=True)
    logger.info("Exported %d entries to EndNote XML: %s", len(corpus), output_path)
    return output_path


# ── Report bundle ─────────────────────────────────────────────────────────


def _copy_synthesis_abstracts(config, output_dir: Path, copied_files: list[str]) -> None:
    """Copy synthesis abstracts without letting two sources share one filename.

    The pipeline abstract wins the ``synthesis_abstract.md`` name. When the
    cluster draft also exists, it is stored as ``abstract_draft.md``.
    """
    pipeline_abstract = config.output_dir / "synthesis_abstract.md"
    cluster_draft = config.clusters_dir / "abstract_draft.md"
    pipeline_exists = pipeline_abstract.exists()
    draft_exists = cluster_draft.exists()

    if pipeline_exists:
        try:
            shutil.copy2(pipeline_abstract, output_dir / "synthesis_abstract.md")
            copied_files.append("synthesis_abstract.md")
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", pipeline_abstract, exc)
    elif draft_exists:
        try:
            shutil.copy2(cluster_draft, output_dir / "synthesis_abstract.md")
            copied_files.append("synthesis_abstract.md")
            logger.info(
                "Copied cluster abstract draft as synthesis_abstract.md; "
                "no pipeline synthesis abstract was present"
            )
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", cluster_draft, exc)
        return

    if pipeline_exists and draft_exists:
        try:
            shutil.copy2(cluster_draft, output_dir / "abstract_draft.md")
            copied_files.append("abstract_draft.md")
            logger.info(
                "Kept cluster abstract draft as abstract_draft.md; "
                "synthesis_abstract.md already came from %s",
                pipeline_abstract,
            )
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", cluster_draft, exc)


def export_report_bundle(config, output_dir: Path) -> Path:
    """One-click export of everything the pipeline has produced.

    Creates a directory with all available pipeline artifacts:
    PRISMA reports, GRADE results, plots, coordinates, bibliography,
    robustness report, synthesis abstract, and a README explaining each file.

    Skips files that don't exist (not all stages may have run).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []

    # Mapping: (source relative to output_dir / project output, dest filename)
    file_map = [
        # PRISMA outputs
        (config.output_dir / "prisma_flow_diagram.png", "prisma_flow.png"),
        (config.output_dir / "prisma_flow_diagram.svg", "prisma_flow.svg"),
        (config.output_dir / "prisma_checklist.md", "prisma_checklist.md"),
        (config.output_dir / "prisma_checklist.csv", "prisma_checklist.csv"),
        # GRADE outputs
        (config.output_dir / "grade_sof_table.md", "grade_sof_table.md"),
        (config.output_dir / "grade_results.json", "grade_results.json"),
        # Meta-analysis plots
        (config.output_dir / "forest_plot.png", "forest_plot.png"),
        (config.output_dir / "funnel_plot.png", "funnel_plot.png"),
        # Co-occurrence network and optional neuroimaging coordinates
        (config.output_dir / "network_graph.svg", "network_graph.svg"),
        (config.output_dir / "coordinates.nimare.json", "coordinates.nimare.json"),
        (config.output_dir / "coordinates.sleuth.txt", "coordinates.sleuth.txt"),
        # Robustness & synthesis
        (config.output_dir / "robustness_report.md", "robustness_report.md"),
        (config.clusters_dir / "synthesis_sources.md", "synthesis_sources.md"),
        (config.clusters_dir / "synthesis_sources.json", "synthesis_sources.json"),
        (config.clusters_dir / "synthesis_sources.bib", "synthesis_sources.bib"),
        # NeuroQuery
        (config.output_dir / "neuroquery_convergence.json", "neuroquery_convergence.json"),
        # Full results
        (config.output_dir / "full_results.json", "full_results.json"),
    ]

    for src, dest_name in file_map:
        if src.exists():
            try:
                shutil.copy2(src, output_dir / dest_name)
                if dest_name not in copied_files:
                    copied_files.append(dest_name)
            except OSError as exc:
                logger.warning("Failed to copy %s: %s", src, exc)

    _copy_synthesis_abstracts(config, output_dir, copied_files)

    # Export bibliography from corpus
    corpus = load_corpus_for_export(config)
    if corpus:
        try:
            export_bibtex(corpus, output_dir / "corpus.bib")
            copied_files.append("corpus.bib")
        except Exception as exc:
            logger.warning("BibTeX export failed: %s", exc)
        try:
            export_ris(corpus, output_dir / "corpus.ris")
            copied_files.append("corpus.ris")
        except Exception as exc:
            logger.warning("RIS export failed: %s", exc)

    # Generate README
    _write_bundle_readme(output_dir, copied_files, config)
    copied_files.append("README.md")

    logger.info(
        "Report bundle exported to %s (%d files)", output_dir, len(copied_files)
    )
    return output_dir


def _write_bundle_readme(output_dir: Path, files: list[str], config) -> None:
    """Write a README.md explaining the report bundle contents."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    descriptions = {
        "prisma_flow.png": "PRISMA 2020 flow diagram (PNG)",
        "prisma_flow.svg": "PRISMA 2020 flow diagram (SVG, editable)",
        "prisma_checklist.md": "PRISMA 2020 checklist (Markdown)",
        "prisma_checklist.csv": "PRISMA 2020 checklist (CSV)",
        "grade_sof_table.md": "GRADE Summary of Findings table",
        "grade_results.json": "GRADE assessment results (JSON)",
        "forest_plot.png": "Forest plot — random-effects meta-analysis",
        "funnel_plot.png": "Contour-enhanced funnel plot for publication bias",
        "network_graph.svg": "Condition–entity co-occurrence network graph",
        "coordinates.nimare.json": "Neuroimaging coordinates in NiMARE format (Neuroimaging tab)",
        "coordinates.sleuth.txt": "Neuroimaging coordinates in GingerALE/Sleuth format (Neuroimaging tab)",
        "corpus.bib": "Full corpus bibliography (BibTeX)",
        "corpus.ris": "Full corpus bibliography (RIS)",
        "robustness_report.md": "Robustness analysis report",
        "synthesis_abstract.md": "Narrative synthesis abstract",
        "abstract_draft.md": "Cluster abstract draft (kept separate from the synthesis abstract)",
        "synthesis_sources.md": "Sources used in synthesis, grouped by cluster",
        "synthesis_sources.json": "Structured source map for synthesis citations",
        "synthesis_sources.bib": "Bibliography for sources used in synthesis",
        "neuroquery_convergence.json": "NeuroQuery convergence scores",
        "full_results.json": "Complete pipeline results",
    }

    lines = [
        "# Sysys Report Bundle",
        "",
        f"Generated: {timestamp}",
        f"LLM Provider: {config.llm_provider}",
        f"LLM Model: {config.llm_model}",
        "",
        "## Contents",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for fname in sorted(files):
        desc = descriptions.get(fname, "Pipeline output")
        lines.append(f"| `{fname}` | {desc} |")

    lines.extend([
        "",
        "## Pipeline Configuration",
        "",
        f"- Hypothesis: {config.hypothesis_text[:120]}...",
        f"- Relevance threshold: {config.relevance_threshold}",
        f"- Project root: {config.project_root}",
        "",
        "---",
        "*Generated by Sysys (Systematic Evidence Synthesis)*",
    ])

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
