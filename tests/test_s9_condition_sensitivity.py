"""Stage 9 analysis 6: per-condition inclusion/exclusion sensitivity."""
from __future__ import annotations

from stages.s9_robustness import _write_report, condition_sensitivity_analysis


def _paper(title: str, supports: bool, score: int = 3) -> dict:
    return {"title": title, "abstract": "", "supports_primary_question": supports,
            "relevance_score": score}


PATTERNS = {"drought": "drought", "soil salinity": "soil salinity", "frost": "frost"}
RANKING = ["drought", "soil salinity", "frost"]
PAPERS = [
    _paper("drought yield loss", True, 5),
    _paper("drought and heat", True, 4),
    _paper("soil salinity trial", False, 2),
    _paper("irrigation baseline", False, 2),
    _paper("fertilizer baseline", True, 3),
]


def test_skipped_without_configured_conditions():
    result = condition_sensitivity_analysis(PAPERS, {}, [])
    assert result["skipped"] is True
    assert "co_occurring_conditions" in result["reason"]


def test_every_configured_condition_is_tested_with_its_own_split():
    result = condition_sensitivity_analysis(PAPERS, PATTERNS, RANKING)
    assert result["skipped"] is False
    assert list(result["conditions"]) == RANKING

    drought = result["conditions"]["drought"]
    assert (drought["n_with"], drought["n_without"]) == (2, 3)
    assert drought["support_rate_with"] == 1.0
    assert drought["support_rate_without"] == round(1 / 3, 4)
    assert drought["rate_difference"] == round(1 - 1 / 3, 4)
    assert "substantially" in drought["interpretation"]

    salinity = result["conditions"]["soil salinity"]
    assert (salinity["n_with"], salinity["n_without"]) == (1, 4)
    assert salinity["support_rate_with"] == 0.0


def test_condition_absent_from_corpus_is_reported_not_compared():
    frost = condition_sensitivity_analysis(PAPERS, PATTERNS, RANKING)["conditions"]["frost"]
    assert frost["skipped"] is True
    assert frost["n_with"] == 0
    assert "no papers" in frost["reason"]


def test_largest_shift_names_the_most_influential_condition():
    result = condition_sensitivity_analysis(PAPERS, PATTERNS, RANKING)
    # Absolute shift wins: salinity drops support by 75 points, drought raises it by ~67.
    assert result["conditions"]["soil salinity"]["rate_difference"] == -0.75
    assert result["conditions"]["drought"]["rate_difference"] == round(2 / 3, 4)
    assert result["largest_shift"] == "soil salinity"


def test_report_has_generic_section_without_hardcoded_condition(tmp_path):
    results = {"condition_sensitivity": condition_sensitivity_analysis(PAPERS, PATTERNS, RANKING)}
    path = tmp_path / "report.md"
    _write_report(results, path, len(PAPERS), RANKING)
    text = path.read_text()
    assert "## 6. Co-occurring Condition Sensitivity" in text
    assert "| drought |" in text and "| soil salinity |" in text
    assert "ADHD" not in text and "E/I" not in text
