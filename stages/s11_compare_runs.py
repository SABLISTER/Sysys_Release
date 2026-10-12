"""Stage 11: Compare pipeline runs.

Compares 2+ completed pipeline runs to measure pipeline-level stability.
Uses metrics from Stage 8 (s8_interrater.py) — kappa, Jaccard, etc.

No LLM calls — pure comparison statistics.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.claims import PRIMARY_OUTCOME_FIELD, claim_flag, claim_list, claim_value
from core.io import atomic_write_json, atomic_write_text

logger = logging.getLogger(__name__)


# ── Record helpers ──────────────────────────────────────────────────────────

def _record_key(record: dict) -> str:
    """Generate unique key for a paper. Priority: doi > pmid > s2_id > title|year."""
    doi = str(record.get("doi", "")).strip()
    if doi:
        return f"doi:{doi.lower()}"
    pmid = str(record.get("pmid", "")).strip()
    if pmid:
        return f"pmid:{pmid}"
    s2 = str(record.get("s2_id", "")).strip()
    if s2:
        return f"s2:{s2}"
    title = str(record.get("title", "")).strip().lower()[:80]
    year = str(record.get("year", ""))
    return f"title:{title}|{year}"


def _list_values(record: dict, field: str) -> set:
    """Extract a set of normalized string values from a (possibly nested) list field."""
    return {v.lower() for v in claim_list(record, field)}


def _support(record: dict, outcome_field: str = PRIMARY_OUTCOME_FIELD) -> Optional[bool]:
    """True/False for the outcome flag, or None when the record lacks it."""
    if claim_value(record, outcome_field) is None:
        return None
    return claim_flag(record, outcome_field)


def _relevance_score(record: dict) -> int:
    """Get relevance score, defaulting to 0."""
    score = record.get("relevance_score", 0)
    if score is None:
        return 0
    try:
        return int(score)
    except (ValueError, TypeError):
        return 0


def _mean(values: list) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ── Load & summarize ────────────────────────────────────────────────────────

def _load_claims(path: Path) -> list:
    """Load claims JSON from path."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def _run_summary(path: Path, records: list,
                 outcome_field: str = PRIMARY_OUTCOME_FIELD) -> dict:
    """Build summary for one run."""
    scores = [_relevance_score(r) for r in records]
    score_dist = {}
    for s in scores:
        score_dist[s] = score_dist.get(s, 0) + 1

    conditions = set()
    mechanisms = set()
    entities = set()
    support_count = 0
    support_total = 0
    for r in records:
        conditions |= _list_values(r, "conditions")
        mechanisms |= _list_values(r, "mechanisms")
        entities |= _list_values(r, "key_entities")
        supported = _support(r, outcome_field)
        if supported is not None:
            support_total += 1
            if supported:
                support_count += 1

    return {
        "path": str(path),
        "paper_count": len(records),
        "avg_relevance": _mean(scores),
        "score_distribution": dict(sorted(score_dist.items())),
        "conditions_found": sorted(conditions),
        "mechanisms_found": sorted(mechanisms),
        "entities_found": sorted(entities),
        "outcome_field": outcome_field,
        "support_rate": support_count / support_total if support_total > 0 else None,
        "support_count": support_count,
        "support_total": support_total,
    }


# ── Pairwise comparison ────────────────────────────────────────────────────

def _pairwise_compare(run_a: dict, run_b: dict,
                      outcome_field: str = PRIMARY_OUTCOME_FIELD) -> dict:
    """Compare two runs.

    run_a / run_b each have: {"summary": ..., "records": {...key: record...}}
    """
    from stages.s8_interrater import cohens_kappa, weighted_kappa, jaccard_similarity

    keys_a = set(run_a["records"].keys())
    keys_b = set(run_b["records"].keys())

    # Paper overlap
    paper_jaccard = jaccard_similarity(keys_a, keys_b)
    shared_keys = sorted(keys_a & keys_b)
    only_a = len(keys_a - keys_b)
    only_b = len(keys_b - keys_a)

    result = {
        "run_a": run_a["summary"]["path"],
        "run_b": run_b["summary"]["path"],
        "papers_a": len(keys_a),
        "papers_b": len(keys_b),
        "shared_papers": len(shared_keys),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "paper_jaccard": paper_jaccard,
    }

    if not shared_keys:
        # No overlap — all agreement metrics are undefined
        result.update({
            "score_weighted_kappa": None,
            "exact_agreement": None,
            "within_1_agreement": None,
            "support_kappa": None,
            "conditions_jaccard": None,
            "mechanisms_jaccard": None,
            "findings_jaccard": None,
            "entities_jaccard": None,
            "content_jaccard_avg": None,
        })
        return result

    # Score agreement on shared papers
    scores_a = []
    scores_b = []
    exact = 0
    within_1 = 0
    support_a = []
    support_b = []
    cond_jaccards = []
    mech_jaccards = []
    find_jaccards = []
    reg_jaccards = []

    for key in shared_keys:
        rec_a = run_a["records"][key]
        rec_b = run_b["records"][key]

        sa = _relevance_score(rec_a)
        sb = _relevance_score(rec_b)
        scores_a.append(sa)
        scores_b.append(sb)

        if sa == sb:
            exact += 1
        if abs(sa - sb) <= 1:
            within_1 += 1

        ca = _support(rec_a, outcome_field)
        cb = _support(rec_b, outcome_field)
        if ca is not None and cb is not None:
            support_a.append(str(ca))
            support_b.append(str(cb))

        # Content overlap per shared paper
        cond_jaccards.append(jaccard_similarity(
            _list_values(rec_a, "conditions"), _list_values(rec_b, "conditions")))
        mech_jaccards.append(jaccard_similarity(
            _list_values(rec_a, "mechanisms"), _list_values(rec_b, "mechanisms")))
        find_jaccards.append(jaccard_similarity(
            _list_values(rec_a, "findings"), _list_values(rec_b, "findings")))
        reg_jaccards.append(jaccard_similarity(
            _list_values(rec_a, "key_entities"), _list_values(rec_b, "key_entities")))

    n_shared = len(shared_keys)
    wk = weighted_kappa(scores_a, scores_b, max_score=5)
    exact_rate = exact / n_shared
    within_1_rate = within_1 / n_shared

    ck = cohens_kappa(support_a, support_b) if support_a else None

    cond_j = _mean(cond_jaccards)
    mech_j = _mean(mech_jaccards)
    find_j = _mean(find_jaccards)
    reg_j = _mean(reg_jaccards)
    content_vals = [v for v in [cond_j, mech_j, find_j, reg_j] if v is not None]
    content_avg = _mean(content_vals)

    result.update({
        "score_weighted_kappa": wk,
        "exact_agreement": exact_rate,
        "within_1_agreement": within_1_rate,
        "support_kappa": ck,
        "conditions_jaccard": cond_j,
        "mechanisms_jaccard": mech_j,
        "findings_jaccard": find_j,
        "entities_jaccard": reg_j,
        "content_jaccard_avg": content_avg,
    })
    return result


# ── Overall summary ─────────────────────────────────────────────────────────

def _overall_summary(pairwise: list, run_summaries: list) -> dict:
    """Average across all pairs for overall stability metrics."""
    if not pairwise:
        return {"n_pairs": 0, "interpretation": "insufficient data"}

    def _avg_field(field: str) -> Optional[float]:
        vals = [p[field] for p in pairwise if p.get(field) is not None]
        return _mean(vals)

    avg_paper_j = _avg_field("paper_jaccard")
    avg_wk = _avg_field("score_weighted_kappa")
    avg_exact = _avg_field("exact_agreement")
    avg_w1 = _avg_field("within_1_agreement")
    avg_ck = _avg_field("support_kappa")
    avg_content = _avg_field("content_jaccard_avg")

    # Composite stability score (average of available metrics, normalized 0-1)
    components = []
    if avg_paper_j is not None:
        components.append(avg_paper_j)
    if avg_wk is not None:
        components.append(max(0.0, (avg_wk + 1.0) / 2.0))  # kappa [-1,1] -> [0,1]
    if avg_exact is not None:
        components.append(avg_exact)
    if avg_content is not None:
        components.append(avg_content)

    stability = _mean(components) if components else None

    if stability is None:
        interp = "insufficient data"
    elif stability >= 0.9:
        interp = "excellent stability"
    elif stability >= 0.75:
        interp = "good stability"
    elif stability >= 0.5:
        interp = "moderate stability"
    else:
        interp = "poor stability"

    return {
        "n_runs": len(run_summaries),
        "n_pairs": len(pairwise),
        "avg_paper_jaccard": avg_paper_j,
        "avg_score_weighted_kappa": avg_wk,
        "avg_exact_agreement": avg_exact,
        "avg_within_1_agreement": avg_w1,
        "avg_support_kappa": avg_ck,
        "avg_content_jaccard": avg_content,
        "stability_score": stability,
        "interpretation": interp,
    }


# ── Report generation ───────────────────────────────────────────────────────

def _fmt(val: Any, decimals: int = 3) -> str:
    """Format a value for the report."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _report_text(results: dict) -> str:
    """Generate markdown report with tables."""
    lines = [
        "# Stage 11: Compare Runs Report",
        "",
        f"Generated: {results.get('timestamp', 'unknown')}",
        f"Number of runs compared: {results.get('n_runs', 0)}",
        "",
    ]

    # Per-run summaries
    lines.append("## Per-Run Summaries")
    lines.append("")
    lines.append("| Run | Papers | Avg Relevance | Support Rate |")
    lines.append("|-----|--------|---------------|------------------|")
    for i, s in enumerate(results.get("run_summaries", []), 1):
        lines.append(
            f"| Run {i} | {s['paper_count']} "
            f"| {_fmt(s['avg_relevance'])} "
            f"| {_fmt(s['support_rate'])} |"
        )
    lines.append("")

    # Pairwise comparisons
    pairwise = results.get("pairwise_comparisons", [])
    if pairwise:
        lines.append("## Pairwise Comparisons")
        lines.append("")
        lines.append(
            "| Pair | Shared | Jaccard | wKappa | Exact% | Within-1% "
            "| Crit-κ | Content-J |"
        )
        lines.append(
            "|------|--------|---------|--------|--------|---------- "
            "|--------|-----------|"
        )
        for i, p in enumerate(pairwise, 1):
            lines.append(
                f"| {i} "
                f"| {p['shared_papers']}/{p['papers_a']}+{p['papers_b']} "
                f"| {_fmt(p['paper_jaccard'])} "
                f"| {_fmt(p['score_weighted_kappa'])} "
                f"| {_fmt(p['exact_agreement'])} "
                f"| {_fmt(p['within_1_agreement'])} "
                f"| {_fmt(p['support_kappa'])} "
                f"| {_fmt(p['content_jaccard_avg'])} |"
            )
        lines.append("")

    # Overall stability
    overall = results.get("overall_summary", {})
    lines.append("## Overall Stability")
    lines.append("")
    lines.append(f"- **Stability score**: {_fmt(overall.get('stability_score'))}")
    lines.append(f"- **Interpretation**: {overall.get('interpretation', 'N/A')}")
    lines.append(f"- **Avg paper Jaccard**: {_fmt(overall.get('avg_paper_jaccard'))}")
    lines.append(
        f"- **Avg score weighted kappa**: "
        f"{_fmt(overall.get('avg_score_weighted_kappa'))}"
    )
    lines.append(
        f"- **Avg exact agreement**: {_fmt(overall.get('avg_exact_agreement'))}"
    )
    lines.append(
        f"- **Avg within-1 agreement**: "
        f"{_fmt(overall.get('avg_within_1_agreement'))}"
    )
    lines.append(
        f"- **Avg support kappa**: "
        f"{_fmt(overall.get('avg_support_kappa'))}"
    )
    lines.append(
        f"- **Avg content Jaccard**: {_fmt(overall.get('avg_content_jaccard'))}"
    )
    lines.append("")

    return "\n".join(lines)


# ── Main entry point ────────────────────────────────────────────────────────

def run(
    config=None,
    claims_paths=None,
    comparison_name: str = "run_compare",
    cancel_event=None,
    progress_callback=None,
    **kw,
) -> dict:
    """Compare 2+ pipeline runs.

    Parameters
    ----------
    config : Config, optional
        Pipeline configuration object.
    claims_paths : list[str|Path]
        Paths to 2+ claims JSON files to compare.
    comparison_name : str
        Name for this comparison (used in output filenames).
    cancel_event : threading.Event or similar, optional
        Set to cancel the run.
    progress_callback : callable, optional
        Called with progress message strings.
    """

    def _cancelled() -> bool:
        if cancel_event is None:
            return False
        if hasattr(cancel_event, "is_set"):
            return cancel_event.is_set()
        return bool(cancel_event)

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _progress("=" * 60)
    _progress("STAGE 11: COMPARE RUNS")
    _progress("=" * 60)

    if not claims_paths or len(claims_paths) < 2:
        raise ValueError("Need at least 2 claims files to compare")

    claims_paths = [Path(p) for p in claims_paths]
    for p in claims_paths:
        if not p.exists():
            raise FileNotFoundError(f"Claims file not found: {p}")

    _progress(f"Comparing {len(claims_paths)} runs:")
    for i, p in enumerate(claims_paths):
        _progress(f"  Run {i + 1}: {p}")

    # ── Resolve output paths ─────────────────────────────────────────
    if config and hasattr(config, "data_dir"):
        data_dir = Path(config.data_dir)
    else:
        data_dir = Path.cwd() / "data"

    if config and hasattr(config, "project_root"):
        output_dir = Path(config.project_root) / "output"
    else:
        output_dir = Path.cwd() / "output"

    slug = _slugify(comparison_name)
    results_path = data_dir / "validation" / f"{slug}_results.json"
    report_path = output_dir / f"{slug}_report.md"

    if _cancelled():
        return {"cancelled": True}

    # ── Step 1: Load each claims file ────────────────────────────────
    from stages.s9_robustness import _load_outcome
    outcome_field = _load_outcome(config).field
    _progress(f"Outcome compared: {outcome_field}")
    _progress("Loading claims files...")
    run_data = []
    for i, p in enumerate(claims_paths):
        records = _load_claims(p)
        keyed = {}
        duplicates = 0
        for rec in records:
            key = _record_key(rec)
            if key in keyed:
                duplicates += 1
            keyed[key] = rec  # last wins on duplicates within a run
        summary = _run_summary(p, records, outcome_field)
        run_data.append({
            "summary": summary,
            "records": keyed,
        })
        _progress(
            f"  Run {i + 1}: {len(records)} papers, "
            f"{len(keyed)} unique keys"
            + (f" ({duplicates} duplicates)" if duplicates else "")
        )

    if _cancelled():
        return {"cancelled": True}

    # ── Step 2: Log per-run summaries ────────────────────────────────
    _progress("-" * 40)
    _progress("Per-run summaries:")
    for i, rd in enumerate(run_data):
        s = rd["summary"]
        _progress(
            f"  Run {i + 1}: {s['paper_count']} papers, "
            f"avg relevance={_fmt(s['avg_relevance'])}, "
            f"support rate={_fmt(s['support_rate'])}, "
            f"conditions={len(s['conditions_found'])}, "
            f"mechanisms={len(s['mechanisms_found'])}, "
            f"entities={len(s['entities_found'])}"
        )

    if _cancelled():
        return {"cancelled": True}

    # ── Step 3: Pairwise comparisons ─────────────────────────────────
    _progress("-" * 40)
    _progress("Computing pairwise comparisons...")
    pairwise = []
    n_runs = len(run_data)
    pair_idx = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            if _cancelled():
                return {"cancelled": True}
            pair_idx += 1
            _progress(f"  Pair {pair_idx}: Run {i + 1} vs Run {j + 1}")
            comparison = _pairwise_compare(run_data[i], run_data[j], outcome_field)
            pairwise.append(comparison)

            _progress(
                f"    Overlap: {comparison['shared_papers']} shared, "
                f"Jaccard={_fmt(comparison['paper_jaccard'])}"
            )
            if comparison.get("score_weighted_kappa") is not None:
                _progress(
                    f"    Scores: wκ={_fmt(comparison['score_weighted_kappa'])}, "
                    f"exact={_fmt(comparison['exact_agreement'])}, "
                    f"within-1={_fmt(comparison['within_1_agreement'])}"
                )
            if comparison.get("support_kappa") is not None:
                _progress(
                    f"    Support κ={_fmt(comparison['support_kappa'])}"
                )
            if comparison.get("content_jaccard_avg") is not None:
                _progress(
                    f"    Content Jaccard avg={_fmt(comparison['content_jaccard_avg'])}"
                )

    if _cancelled():
        return {"cancelled": True}

    # ── Step 4: Overall stability ────────────────────────────────────
    _progress("-" * 40)
    _progress("Computing overall stability...")
    summaries = [rd["summary"] for rd in run_data]
    overall = _overall_summary(pairwise, summaries)

    _progress(
        f"  Stability score: {_fmt(overall.get('stability_score'))} "
        f"({overall.get('interpretation', 'N/A')})"
    )

    # ── Step 5: Build results ────────────────────────────────────────
    results = {
        "stage": "s11_compare_runs",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(run_data),
        "claims_paths": [str(p) for p in claims_paths],
        "comparison_name": comparison_name,
        "run_summaries": summaries,
        "pairwise_comparisons": pairwise,
        "overall_summary": overall,
    }

    # ── Step 6: Save outputs ─────────────────────────────────────────
    _progress("-" * 40)
    _progress("Saving results...")
    atomic_write_json(results_path, results)
    _progress(f"  Results JSON: {results_path}")

    report = _report_text(results)
    atomic_write_text(report_path, report)
    _progress(f"  Report: {report_path}")

    results["results_path"] = str(results_path)
    results["report_path"] = str(report_path)

    _progress("=" * 60)
    _progress(
        f"Stage 11 complete: {overall.get('interpretation', 'N/A')} "
        f"(score={_fmt(overall.get('stability_score'))})"
    )
    _progress("=" * 60)

    return results


# ── Check output ────────────────────────────────────────────────────────────

def check_output(config=None, comparison_name: str = "run_compare") -> dict:
    """Check whether Stage 11 output already exists."""
    if config is None:
        return {"exists": False}
    slug = _slugify(comparison_name)
    path = config.data_dir / "validation" / f"{slug}_results.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "exists": True,
                "n_runs": data.get("n_runs", 0),
                "interpretation": data.get("overall_summary", {}).get(
                    "interpretation", "unknown"
                ),
                "path": str(path),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"exists": False}
