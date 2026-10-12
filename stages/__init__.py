"""Stages package — the 11-stage SystS analysis pipeline.

Each stage module exposes a ``run(config, ...)`` coroutine/function and an
optional ``check_output(config)`` helper.

Stage map
---------
s1_search       — Literature search (multi-provider)
s2_score        — LLM relevance scoring
s2c_statistical_extract — Programmatic statistical evidence extraction
s2b_fulltext    — Full-text corpus enrichment (pre-scoring)
s3_extract      — Claim extraction
s4_validate     — Method review & span validation
s5_audit        — Span quality audit & filtering
s6_synthesize   — Clustering, narration, abstract draft
s7_fulltext     — Full-text retrieval & metric extraction
s8_interrater   — Inter-rater reliability (cross-model)
s9_robustness   — Robustness analyses
s10_verify      — Abstract number verification
s11_compare_runs — Multi-run comparison
"""

from stages import (
    s1_search,
    s2_score,
    s2c_statistical_extract,
    s2b_fulltext,
    s3_extract,
    s4_validate,
    s5_audit,
    s6_synthesize,
    s7_fulltext,
    s8_interrater,
    s9_robustness,
    s10_verify,
    s11_compare_runs,
)

__all__ = [
    "s1_search",
    "s2_score",
    "s2c_statistical_extract",
    "s2b_fulltext",
    "s3_extract",
    "s4_validate",
    "s5_audit",
    "s6_synthesize",
    "s7_fulltext",
    "s8_interrater",
    "s9_robustness",
    "s10_verify",
    "s11_compare_runs",
]
