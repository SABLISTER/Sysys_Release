"""Stage 10: Abstract number verification.

Extract every number and statistic from the synthesised abstract and
cross-check each against the actual data (claims_filtered.json) and,
optionally, robustness_results.json / interrater_summary.json.

No LLM calls — pure regex extraction + numerical comparison.
Tolerance: within 2 % relative **or** ±2 absolute ⇒ PASS.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from core.claims import claim_list
from core.io import atomic_write_json, atomic_write_text
from process.domain_patterns import contains_term
from stages.s9_robustness import (
    DEFAULT_OUTCOME,
    RobustnessOutcome,
    _extract_conditions,
    _load_condition_config,
    _load_outcome,
)

logger = logging.getLogger(__name__)

# ── Verification vocabulary ─────────────────────────────────────────────
# Domain vocabulary comes from the research config: mechanism terms from
# ``process.domain_patterns.extraction_patterns``, conditions and the support
# outcome from the same loaders Stage 9 uses. Without a config nothing
# domain-specific is assumed; only generic "support" wording is recognized.

# Words near a percentage that mark it as the share of supporting papers.
SUPPORT_CONTEXT = ["support", "inform"]


def _verify_terms(config) -> tuple[list[str], RobustnessOutcome, list[str],
                                   list[str], dict[str, str]]:
    """Resolve stage-10 vocabulary from the research config.

    Returns ``(mechanism_terms, outcome, mechanism_context, support_context,
    condition_patterns)``.
    """
    if config is None:
        return ([], DEFAULT_OUTCOME, [], list(SUPPORT_CONTEXT), {})
    from process.domain_patterns import extraction_patterns
    mechanism_terms = list(extraction_patterns(config).get("mechanisms", []))
    outcome = _load_outcome(config)
    mechanism_context = ["mechanism"] + [t.lower() for t in mechanism_terms]
    support_context = SUPPORT_CONTEXT + [t.lower() for t in outcome.fallback_terms]
    condition_patterns, _ = _load_condition_config(config)
    return (mechanism_terms, outcome, mechanism_context, support_context,
            condition_patterns)


# ── Paper helpers ───────────────────────────────────────────────────────

def _has_mechanism(paper: dict, keywords: list[str]) -> bool:
    text = " ".join([paper.get("abstract", "") or "", *claim_list(paper, "mechanisms")])
    return any(contains_term(text, kw) for kw in keywords)


# ── Number extraction from abstract ────────────────────────────────────

def _extract_abstract_numbers(abstract_text: str) -> dict:
    """Extract all numbers and statistics from abstract text.

    Returns dict with keys: paper_counts, percentages, kappa_values,
    ci_values, p_values, correlations, generic_counts.
    """
    numbers: dict[str, list] = {
        "paper_counts": [],
        "percentages": [],
        "kappa_values": [],
        "ci_values": [],
        "p_values": [],
        "correlations": [],
        "generic_counts": [],
    }

    # Paper/study counts: "245 papers", "120 studies", "80 articles"
    for m in re.finditer(
        r'(\d[\d,]*)\s+(?:papers?|studies|articles|publications?)',
        abstract_text, re.IGNORECASE,
    ):
        val = int(m.group(1).replace(",", ""))
        numbers["paper_counts"].append({
            "value": val, "raw": m.group(0), "span": m.span(),
        })

    # Percentages: "78.5%" or "78.5 %"
    for m in re.finditer(
        r'(\d+(?:\.\d+)?)\s*%',
        abstract_text,
    ):
        val = float(m.group(1))
        numbers["percentages"].append({
            "value": val, "raw": m.group(0), "span": m.span(),
        })

    # Kappa (κ): "κ = 0.72" or "kappa = 0.72" or "Cohen's κ = 0.72"
    for m in re.finditer(
        r'(?:κ|kappa)\s*=\s*(\d+(?:\.\d+)?)',
        abstract_text, re.IGNORECASE,
    ):
        val = float(m.group(1))
        numbers["kappa_values"].append({
            "value": val, "raw": m.group(0), "span": m.span(),
        })

    # Confidence intervals: "CI [0.55, 0.89]" or "CI (0.55–0.89)"
    for m in re.finditer(
        r'CI\s*[\[(\s]*(\d+(?:\.\d+)?)\s*[,\u2013\-–]\s*(\d+(?:\.\d+)?)\s*[\])]?',
        abstract_text, re.IGNORECASE,
    ):
        lo, hi = float(m.group(1)), float(m.group(2))
        numbers["ci_values"].append({
            "lower": lo, "upper": hi, "raw": m.group(0), "span": m.span(),
        })

    # P-values: "p < 0.001", "p = 0.03", "p<.05"
    for m in re.finditer(
        r'p\s*([<=<>≤≥])\s*\.?(\d+(?:\.\d+)?)',
        abstract_text, re.IGNORECASE,
    ):
        op = m.group(1)
        val = float(m.group(2))
        # Handle p<.05 → 0.05
        if val >= 1 and "." not in m.group(2):
            pass  # keep as-is (e.g. p = 1)
        numbers["p_values"].append({
            "value": val, "operator": op, "raw": m.group(0), "span": m.span(),
        })

    # Correlations: "r = 0.83"
    for m in re.finditer(
        r'\br\s*=\s*(\-?\d+(?:\.\d+)?)',
        abstract_text,
    ):
        val = float(m.group(1))
        numbers["correlations"].append({
            "value": val, "raw": m.group(0), "span": m.span(),
        })

    # Generic counts near condition/mechanism words
    for m in re.finditer(
        r'(\d+)\s+(?:of\s+)?(?:\w+\s+)?(?:conditions?|mechanisms?|clusters?|entit(?:y|ies))',
        abstract_text, re.IGNORECASE,
    ):
        val = int(m.group(1))
        numbers["generic_counts"].append({
            "value": val, "raw": m.group(0), "span": m.span(),
        })

    return numbers


# ── Tolerance check ─────────────────────────────────────────────────────

def _within_tolerance(expected: float, actual: float,
                      rel_tol: float = 0.02, abs_tol: float = 2.0) -> bool:
    """Return True if actual is within 2% relative or ±2 absolute of expected."""
    if expected == 0 and actual == 0:
        return True
    if abs(actual - expected) <= abs_tol:
        return True
    if expected != 0 and abs(actual - expected) / abs(expected) <= rel_tol:
        return True
    return False


# ── Core verification logic ─────────────────────────────────────────────

def verify(papers: list[dict], abstract_text: str,
           robustness_path: Optional[Path] = None,
           interrater_path: Optional[Path] = None,
           config=None) -> dict:
    """Core verification: extract numbers from abstract, check each.

    Pass the pipeline ``config`` so condition patterns, mechanism terms, and
    the support outcome come from the research config instead of hardcoded
    domain vocabulary. Without a config, term lists are empty and support is
    read from the extracted ``supports_primary_question`` flag.

    Returns
    -------
    dict with keys: summary, checks, extracted_numbers
    """
    checks: list[dict] = []
    extracted = _extract_abstract_numbers(abstract_text)

    (mechanism_terms, outcome, mechanism_context,
     support_context, cond_patterns) = _verify_terms(config)

    # ── Compute actual statistics from corpus ────────────────────────
    n_papers = len(papers)
    n_supporting = sum(1 for p in papers if outcome.supports(p))
    pct_supporting = (n_supporting / n_papers * 100) if n_papers else 0.0
    has_mechanism_terms = bool(mechanism_terms)
    n_mech = sum(1 for p in papers if _has_mechanism(p, mechanism_terms))
    pct_mech = (n_mech / n_papers * 100) if n_papers else 0.0
    mechanism_label = "mechanism"

    # Condition counts
    condition_counter: dict[str, int] = {}
    for p in papers:
        for cond in _extract_conditions(p, cond_patterns):
            condition_counter[cond] = condition_counter.get(cond, 0) + 1
    n_conditions = len(condition_counter)

    # ── Load optional cross-check data ───────────────────────────────
    robustness_data: Optional[dict] = None
    if robustness_path and robustness_path.exists():
        try:
            with open(robustness_path, encoding="utf-8") as f:
                robustness_data = json.load(f)
            logger.info("Loaded robustness data for cross-check: %s",
                        robustness_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load robustness data: %s", exc)

    interrater_data: Optional[dict] = None
    if interrater_path and interrater_path.exists():
        try:
            with open(interrater_path, encoding="utf-8") as f:
                interrater_data = json.load(f)
            logger.info("Loaded interrater data for cross-check: %s",
                        interrater_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load interrater data: %s", exc)

    # ── 1. Check paper/study counts ──────────────────────────────────
    for item in extracted["paper_counts"]:
        claimed = item["value"]
        ok = _within_tolerance(claimed, n_papers)
        status = "PASS" if ok else "FAIL"
        diff = n_papers - claimed
        checks.append({
            "claim": item["raw"],
            "expected": claimed,
            "actual": n_papers,
            "status": status,
            "note": f"off by {diff}" if not ok else "within tolerance",
        })
        logger.info("CHECK paper_count: claim=%d actual=%d → %s",
                     claimed, n_papers, status)

    # ── 2. Check percentages ─────────────────────────────────────────
    # Try to match each percentage to a known metric
    for item in extracted["percentages"]:
        claimed_pct = item["value"]
        raw_text = item["raw"]
        # Determine context from surrounding text
        start = max(0, item["span"][0] - 80)
        end = min(len(abstract_text), item["span"][1] + 40)
        context = abstract_text[start:end].lower()

        actual_pct: Optional[float] = None
        label = "percentage"

        if any(kw in context for kw in support_context):
            actual_pct = pct_supporting
            label = f"support ({outcome.field})"
        elif has_mechanism_terms and any(kw in context for kw in mechanism_context):
            actual_pct = pct_mech
            label = mechanism_label
        elif any(kw in context for kw in ["agree", "concordan"]):
            # Agreement percentage — try interrater data
            if interrater_data:
                actual_pct = interrater_data.get("overall_agreement_pct")
            label = "agreement"

        if actual_pct is not None:
            ok = _within_tolerance(claimed_pct, actual_pct)
            status = "PASS" if ok else "FAIL"
            checks.append({
                "claim": raw_text,
                "expected": claimed_pct,
                "actual": round(actual_pct, 2),
                "status": status,
                "note": f"{label}: diff={actual_pct - claimed_pct:.2f}pp"
                        if not ok else f"{label}: within tolerance",
            })
            logger.info("CHECK pct (%s): claim=%.1f%% actual=%.1f%% → %s",
                         label, claimed_pct, actual_pct, status)
        else:
            checks.append({
                "claim": raw_text,
                "expected": claimed_pct,
                "actual": None,
                "status": "WARN",
                "note": "could not match percentage to a known metric",
            })
            logger.info("CHECK pct: claim=%.1f%% → WARN (no matching metric)",
                         claimed_pct)

    # ── 3. Check kappa values ────────────────────────────────────────
    for item in extracted["kappa_values"]:
        claimed_kappa = item["value"]
        actual_kappa: Optional[float] = None

        if interrater_data:
            # Try weighted_kappa first, then cohens_kappa
            actual_kappa = interrater_data.get("weighted_kappa")
            if actual_kappa is None:
                actual_kappa = interrater_data.get("cohens_kappa")

        if actual_kappa is not None:
            ok = _within_tolerance(claimed_kappa, actual_kappa)
            status = "PASS" if ok else "FAIL"
            checks.append({
                "claim": item["raw"],
                "expected": claimed_kappa,
                "actual": round(actual_kappa, 4),
                "status": status,
                "note": f"diff={actual_kappa - claimed_kappa:.4f}"
                        if not ok else "within tolerance",
            })
            logger.info("CHECK kappa: claim=%.3f actual=%.3f → %s",
                         claimed_kappa, actual_kappa, status)
        else:
            checks.append({
                "claim": item["raw"],
                "expected": claimed_kappa,
                "actual": None,
                "status": "WARN",
                "note": "no interrater data available for cross-check",
            })
            logger.info("CHECK kappa: claim=%.3f → WARN (no interrater data)",
                         claimed_kappa)

    # ── 4. Check CI values ───────────────────────────────────────────
    for item in extracted["ci_values"]:
        claimed_lo = item["lower"]
        claimed_hi = item["upper"]

        actual_lo: Optional[float] = None
        actual_hi: Optional[float] = None

        if robustness_data:
            bs = robustness_data.get("bootstrap", {})
            actual_lo = bs.get("ci_lower")
            actual_hi = bs.get("ci_upper")

        if actual_lo is not None and actual_hi is not None:
            ok_lo = _within_tolerance(claimed_lo, actual_lo)
            ok_hi = _within_tolerance(claimed_hi, actual_hi)
            status = "PASS" if (ok_lo and ok_hi) else "FAIL"
            checks.append({
                "claim": item["raw"],
                "expected": [claimed_lo, claimed_hi],
                "actual": [round(actual_lo, 4), round(actual_hi, 4)],
                "status": status,
                "note": f"lo_ok={ok_lo} hi_ok={ok_hi}",
            })
            logger.info(
                "CHECK CI: claim=[%.3f,%.3f] actual=[%.3f,%.3f] → %s",
                claimed_lo, claimed_hi, actual_lo, actual_hi, status)
        else:
            checks.append({
                "claim": item["raw"],
                "expected": [claimed_lo, claimed_hi],
                "actual": None,
                "status": "WARN",
                "note": "no robustness data available for CI cross-check",
            })
            logger.info("CHECK CI: claim=[%.3f,%.3f] → WARN (no data)",
                         claimed_lo, claimed_hi)

    # ── 5. Check p-values ────────────────────────────────────────────
    for item in extracted["p_values"]:
        claimed_p = item["value"]
        actual_p: Optional[float] = None

        if robustness_data:
            pt = robustness_data.get("permutation", {})
            actual_p = pt.get("p_value")

        if actual_p is not None:
            # For p-values, check direction: if claim is p < X, actual
            # should also be < X
            op = item.get("operator", "=")
            if op in ("<", "≤", "<"):
                ok = actual_p <= claimed_p
            else:
                ok = _within_tolerance(claimed_p, actual_p)
            status = "PASS" if ok else "FAIL"
            checks.append({
                "claim": item["raw"],
                "expected": claimed_p,
                "actual": round(actual_p, 6),
                "status": status,
                "note": f"actual p={actual_p:.6f}"
                        if not ok else "within tolerance",
            })
            logger.info("CHECK p-value: claim=p%s%.4f actual=%.6f → %s",
                         op, claimed_p, actual_p, status)
        else:
            checks.append({
                "claim": item["raw"],
                "expected": claimed_p,
                "actual": None,
                "status": "WARN",
                "note": "no permutation data available for p-value cross-check",
            })
            logger.info("CHECK p-value: claim=%.4f → WARN (no data)",
                         claimed_p)

    # ── 6. Check correlations ────────────────────────────────────────
    for item in extracted["correlations"]:
        claimed_r = item["value"]
        actual_r: Optional[float] = None

        if robustness_data:
            # Try permutation observed_correlation or split_half
            pt = robustness_data.get("permutation", {})
            sh = robustness_data.get("split_half", {})
            actual_r = pt.get("observed_correlation")
            if actual_r is None:
                actual_r = sh.get("mean_correlation")

        if actual_r is not None:
            ok = _within_tolerance(claimed_r, actual_r)
            status = "PASS" if ok else "FAIL"
            checks.append({
                "claim": item["raw"],
                "expected": claimed_r,
                "actual": round(actual_r, 4),
                "status": status,
                "note": f"diff={actual_r - claimed_r:.4f}"
                        if not ok else "within tolerance",
            })
            logger.info("CHECK correlation: claim=%.3f actual=%.3f → %s",
                         claimed_r, actual_r, status)
        else:
            checks.append({
                "claim": item["raw"],
                "expected": claimed_r,
                "actual": None,
                "status": "WARN",
                "note": "no robustness data for correlation cross-check",
            })
            logger.info("CHECK correlation: claim=%.3f → WARN (no data)",
                         claimed_r)

    # ── 7. Check generic counts (conditions, clusters, etc.) ─────────
    for item in extracted["generic_counts"]:
        claimed_n = item["value"]
        raw_lower = item["raw"].lower()
        actual_n: Optional[int] = None
        label = "generic count"

        if "condition" in raw_lower:
            actual_n = n_conditions
            label = "condition count"
        elif "cluster" in raw_lower:
            # Cannot determine without cluster data; mark as WARN
            actual_n = None
            label = "cluster count"

        if actual_n is not None:
            ok = _within_tolerance(claimed_n, actual_n)
            status = "PASS" if ok else "FAIL"
            checks.append({
                "claim": item["raw"],
                "expected": claimed_n,
                "actual": actual_n,
                "status": status,
                "note": f"{label}: off by {actual_n - claimed_n}"
                        if not ok else f"{label}: within tolerance",
            })
            logger.info("CHECK %s: claim=%d actual=%d → %s",
                         label, claimed_n, actual_n, status)
        else:
            checks.append({
                "claim": item["raw"],
                "expected": claimed_n,
                "actual": None,
                "status": "WARN",
                "note": f"{label}: no reference data available",
            })
            logger.info("CHECK %s: claim=%d → WARN (no data)",
                         label, claimed_n)

    # ── Build summary ────────────────────────────────────────────────
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")

    return {
        "summary": {
            "total_checks": len(checks),
            "passed": n_pass,
            "failed": n_fail,
            "warnings": n_warn,
        },
        "checks": checks,
        "extracted_numbers": extracted,
    }


# ── Markdown report writer ──────────────────────────────────────────────

def _write_report(results: dict, path: Path) -> None:
    """Write markdown verification report with pass/fail table."""
    lines: list[str] = []
    summary = results.get("summary", {})

    lines.append("# Abstract Number Verification Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total checks:** {summary.get('total_checks', 0)}")
    lines.append(f"- **Passed:** {summary.get('passed', 0)}")
    lines.append(f"- **Failed:** {summary.get('failed', 0)}")
    lines.append(f"- **Warnings:** {summary.get('warnings', 0)}")
    lines.append("")

    checks = results.get("checks", [])
    if checks:
        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| # | Claim | Expected | Actual | Status | Note |")
        lines.append("|---|-------|----------|--------|--------|------|")
        for i, c in enumerate(checks, 1):
            claim = c.get("claim", "").replace("|", "\\|")
            expected = c.get("expected", "")
            actual = c.get("actual", "N/A")
            if actual is None:
                actual = "N/A"
            status = c.get("status", "")
            note = c.get("note", "").replace("|", "\\|")
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(status, "")
            lines.append(
                f"| {i} | {claim} | {expected} | {actual} "
                f"| {icon} {status} | {note} |"
            )
        lines.append("")

    # Numbers found
    extracted = results.get("extracted_numbers", {})
    if extracted:
        lines.append("## Extracted Numbers")
        lines.append("")
        for category, items in extracted.items():
            if items:
                lines.append(f"### {category.replace('_', ' ').title()}")
                for item in items:
                    lines.append(f"- `{item.get('raw', item)}`")
                lines.append("")

    lines.append("---")
    lines.append("*Generated by Stage 10: Abstract Number Verification*")
    lines.append("")

    atomic_write_text(path, "\n".join(lines))


# ── Main entry point ────────────────────────────────────────────────────

def run(
    claims_path: Optional[str] = None,
    abstract_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config=None,
    cancel_event=None,
    progress_callback=None,
    **kw,
) -> dict:
    """Run Stage 10: abstract number verification.

    1. Load claims data
    2. Load abstract text
    3. Extract numbers/statistics from abstract
    4. Verify each against the actual data
    5. Optionally cross-check against robustness_results.json
    6. Save verification_results.json + verification_report.md
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

    _progress("═" * 60)
    _progress("STAGE 10: ABSTRACT NUMBER VERIFICATION")
    _progress("═" * 60)

    # ── Resolve paths ────────────────────────────────────────────────
    project_root = config.project_root if config else Path.cwd()
    data_dir = project_root / "data"
    validation_dir = Path(output_dir) if output_dir else data_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    # ── Load claims corpus ───────────────────────────────────────────
    corpus_path = None
    if claims_path:
        corpus_path = Path(claims_path)
    else:
        candidates = [
            data_dir / "claims" / "claims_filtered.json",
            data_dir / "claims" / "claims.json",
            data_dir / "corpus" / "relevant.json",
        ]
        if config:
            candidates = [
                config.claims_dir / "claims_filtered.json",
                config.claims_dir / "claims.json",
                config.corpus_dir / "relevant.json",
            ] + candidates
        for cand in candidates:
            if cand.exists():
                corpus_path = cand
                break

    if corpus_path is None or not corpus_path.exists():
        _progress("ERROR: No claims corpus found. Run earlier stages first.")
        return {"error": "No claims corpus found"}

    with open(corpus_path, encoding="utf-8") as f:
        papers = json.load(f)
    if not isinstance(papers, list):
        papers = list(papers.values()) if isinstance(papers, dict) else []

    n_papers = len(papers)
    _progress(f"Loaded {n_papers} papers from {corpus_path}")

    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    # ── Load abstract text ───────────────────────────────────────────
    abs_path: Optional[Path] = None
    if abstract_path:
        abs_path = Path(abstract_path)
    else:
        candidates_abs = []
        if config:
            candidates_abs.append(config.clusters_dir / "abstract_draft.md")
            candidates_abs.append(config.output_dir / "abstract_draft.md")
        candidates_abs.extend([
            data_dir / "clusters" / "abstract_draft.md",
            project_root / "output" / "abstract_draft.md",
        ])
        for cand in candidates_abs:
            if cand.exists():
                abs_path = cand
                break

    if abs_path is None or not abs_path.exists():
        _progress("WARNING: No abstract_draft.md found — returning empty results.")
        empty_results = {
            "summary": {"total_checks": 0, "passed": 0, "failed": 0, "warnings": 0},
            "checks": [],
            "extracted_numbers": {},
            "warning": "No abstract_draft.md found",
        }
        results_path = validation_dir / "verification_results.json"
        atomic_write_json(results_path, empty_results)
        empty_results["results_path"] = str(results_path)
        return empty_results

    abstract_text = abs_path.read_text(encoding="utf-8")
    _progress(f"Loaded abstract ({len(abstract_text)} chars) from {abs_path}")

    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    # ── Resolve optional cross-check paths ───────────────────────────
    robustness_path = validation_dir / "robustness_results.json"
    if not robustness_path.exists():
        robustness_path = data_dir / "validation" / "robustness_results.json"
    if not robustness_path.exists():
        _progress("No robustness_results.json found — skipping cross-checks.")
        robustness_path = None

    interrater_path = validation_dir / "interrater_summary.json"
    if not interrater_path.exists():
        interrater_path = data_dir / "validation" / "interrater_summary.json"
    if not interrater_path.exists():
        _progress("No interrater_summary.json found — skipping kappa cross-checks.")
        interrater_path = None

    # ── Run verification ─────────────────────────────────────────────
    _progress("Extracting numbers from abstract and verifying...")
    results = verify(
        papers, abstract_text,
        robustness_path=robustness_path,
        interrater_path=interrater_path,
        config=config,
    )

    summary = results["summary"]
    _progress(
        f"Verification complete: {summary['total_checks']} checks — "
        f"{summary['passed']} passed, {summary['failed']} failed, "
        f"{summary['warnings']} warnings"
    )

    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    # ── Save outputs ─────────────────────────────────────────────────
    results_path = validation_dir / "verification_results.json"
    atomic_write_json(results_path, results)
    _progress(f"Results saved: {results_path}")
    results["results_path"] = str(results_path)

    report_path = validation_dir / "verification_report.md"
    _write_report(results, report_path)
    _progress(f"Report saved: {report_path}")
    results["report_path"] = str(report_path)

    _progress("═" * 60)
    _progress("STAGE 10 COMPLETE")
    _progress("═" * 60)

    return results


# ── Output checker ──────────────────────────────────────────────────────

def check_output(config=None) -> dict:
    """Check whether Stage 10 output already exists."""
    if config is None:
        return {"exists": False}
    path = config.data_dir / "validation" / "verification_results.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            return {
                "exists": True,
                "total_checks": summary.get("total_checks", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "path": str(path),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"exists": False}
