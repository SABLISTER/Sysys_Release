"""Multi-axis structured LLM scoring and tier derivation."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import Config
from hard_mode.fulltext import score_text_bundle
from hard_mode.records import ensure_spine
from process.llm import chat, set_config

logger = logging.getLogger(__name__)


def _llm_scoring_params(config: Config, hm: dict[str, Any]) -> tuple[int, int, int, int, bool | None]:
    """Title cap, abstract cap, full-text cap, max_tokens, and think flag."""
    ls = hm.get("llm_scoring") or {}
    abs_default = int(getattr(config, "score_abstract_chars", 8000))
    abs_lim = (
        int(ls["abstract_max_chars"])
        if ls.get("abstract_max_chars") is not None
        else min(abs_default, 6000)
    )
    full_default = int(getattr(config, "score_fulltext_chars", 24000))
    full_text_max_chars = (hm.get("full_text") or {}).get("score_max_chars")
    full_lim = (
        int(full_text_max_chars) if full_text_max_chars is not None else full_default
    )
    title_lim = int(ls["title_max_chars"]) if ls.get("title_max_chars") is not None else 400
    cfg_cap = int(getattr(config, "score_max_tokens", 4096))
    default_tok = min(800, max(256, cfg_cap))
    max_tok = (
        int(ls["max_output_tokens"])
        if ls.get("max_output_tokens") is not None
        else default_tok
    )
    max_tok = max(128, min(max_tok, 4096))

    think_arg: bool | None
    if ls.get("use_reasoning") is True:
        think_arg = True
    elif ls.get("use_reasoning") is False:
        think_arg = False
    else:
        think_arg = None
    return title_lim, abs_lim, full_lim, max_tok, think_arg


def _extract_json_object(text: str) -> dict[str, Any]:
    from process.llm import parse_llm_json

    data = parse_llm_json(text)
    if data is None:
        raise ValueError("No JSON object in model output")
    return data


def _build_axis_prompt(hm: dict[str, Any]) -> str:
    axes = hm.get("axes") or []
    lines = []
    for ax in axes:
        aid = ax.get("id", "")
        name = ax.get("display_name", aid)
        desc = (ax.get("description_for_prompt") or "").strip()
        lines.append(f'- "{aid}": {name}. {desc}')
    return "\n".join(lines) if lines else ""


def _score_one_paper(config: Config, hm: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    axes = hm.get("axes") or []
    axis_ids = [str(a.get("id", "")) for a in axes if a.get("id")]
    evidence_types = hm.get("evidence_types") or [
        "empirical",
        "review",
        "theory",
        "computational",
    ]

    title_lim, abs_lim, full_lim, max_tok, think_arg = _llm_scoring_params(config, hm)
    bundle = score_text_bundle(
        config,
        hm,
        paper,
        title_limit=title_lim,
        abstract_limit=abs_lim,
        fulltext_limit=full_lim,
    )
    runtime = config.runtime_config(bundle["workload"])
    set_config(runtime)

    schema_lines = [
        "Return a single JSON object only, no markdown.",
        "{",
        '  "axis_scores": { ' + ", ".join(f'"{a}": <float 0-1>' for a in axis_ids) + " },",
        '  "overall_confidence": <float 0-1>,',
        f'  "evidence_type": one of {json.dumps(evidence_types)},',
        '  "rationale": "<= 25 words"',
        "}",
    ]

    user = "\n".join(
        [
            "Score this paper for a systematic review.",
            "",
            "Axes:",
            _build_axis_prompt(hm),
            "",
            "Title:",
            bundle["title"],
            "",
            "Abstract:",
            bundle["abstract"],
            "",
            bundle["body_label"] + ":",
            bundle["body_text"],
            "",
            "Schema:",
            "\n".join(schema_lines),
        ]
    )

    messages = [
        {
            "role": "system",
            "content": "You output only valid JSON. Scores are subjective estimates for triage, not clinical claims.",
        },
        {"role": "user", "content": user},
    ]

    model = runtime.llm_model_for(bundle["workload"])
    raw = chat(
        model,
        messages,
        temperature=0,
        max_tokens=max_tok,
        think=think_arg,
    )
    data = _extract_json_object(raw)

    axis_scores = data.get("axis_scores") or {}
    for aid in axis_ids:
        axis_scores.setdefault(aid, 0.0)
        try:
            axis_scores[aid] = max(0.0, min(1.0, float(axis_scores[aid])))
        except (TypeError, ValueError):
            axis_scores[aid] = 0.0

    try:
        overall_conf = max(0.0, min(1.0, float(data.get("overall_confidence", 0))))
    except (TypeError, ValueError):
        overall_conf = 0.0

    et = str(data.get("evidence_type", "empirical"))
    if et not in evidence_types:
        et = "empirical"

    # The "keystone" axis gates tier 1 in derive_overlap_and_tier(). Which
    # configured axis plays that role is config-driven (tier_rules keys);
    # the output keys stay "integration"/"domain_context" for schema
    # stability, but the axis IDs they read are no longer hardcoded.
    rules = hm.get("tier_rules") or {}
    keystone_axis_id = str(rules.get("tier1_keystone_axis_id", "integration"))
    context_axis_id = str(rules.get("domain_context_axis_id", "domain_context"))

    return {
        "axis_scores": axis_scores,
        "integration": float(axis_scores.get(keystone_axis_id, 0) or 0),
        "domain_context": float(axis_scores.get(context_axis_id, 0) or 0),
        "confidence": overall_conf,
        "evidence_type": et,
        "rationale": str(data.get("rationale", ""))[:500],
        "text_source": bundle["text_source"],
        "body_chars_used": bundle["body_chars"],
        "workload": bundle["workload"],
    }


def derive_overlap_and_tier(hm: dict[str, Any], machine: dict[str, Any]) -> dict[str, Any]:
    rules = hm.get("overlap_rules") or {}
    thr = float(rules.get("axis_score_threshold", 0.35))
    axis_ids = rules.get("axis_ids_for_overlap") or [
        "axis_a",
        "axis_b",
        "axis_c",
        "axis_d",
    ]
    scores = machine.get("axis_scores") or {}
    count = sum(1 for aid in axis_ids if float(scores.get(aid, 0) or 0) >= thr)
    if count >= 4:
        bucket = "4-way"
    elif count == 3:
        bucket = "3-way"
    elif count == 2:
        bucket = "2-way"
    elif count == 1:
        bucket = "1-way"
    else:
        bucket = "0-way"

    tr = hm.get("tier_rules") or {}
    strong_thr = float(tr.get("axis_strong_threshold", 0.55))
    strong_axes = sum(1 for aid in axis_ids if float(scores.get(aid, 0) or 0) >= strong_thr)
    integ = float(machine.get("integration") or 0)
    conf = float(machine.get("confidence") or 0)
    need_strong = int(tr.get("tier1_min_axes_strong", 3))
    need_int = float(tr.get("tier1_min_integration", 0.5))
    need_conf = float(tr.get("tier1_min_confidence", 0.45))
    tier2_conf = float(tr.get("tier2_min_confidence", 0.3))

    if strong_axes >= need_strong and integ >= need_int and conf >= need_conf:
        tier = 1
    elif conf >= tier2_conf:
        tier = 2
    else:
        tier = 3

    return {"overlap_bucket": bucket, "axis_overlap_count": count, "priority_tier": tier}


def score_corpus(
    config: Config,
    hm: dict[str, Any],
    papers: list[dict[str, Any]],
    *,
    skip_if_machine: bool = True,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for paper in papers:
        p = ensure_spine(dict(paper))
        if skip_if_machine and (p.get("machine") or {}).get("axis_scores"):
            derived = derive_overlap_and_tier(hm, p["machine"])
            p.setdefault("derived", {}).update(derived)
            out.append(p)
            continue
        try:
            machine = _score_one_paper(config, hm, p)
            derived = derive_overlap_and_tier(hm, machine)
            p["machine"] = machine
            p["derived"] = {**(p.get("derived") or {}), **derived}
        except Exception as e:
            logger.warning("Score failed for %s: %s", p.get("title", "")[:60], e)
            p.setdefault("machine", {})["error"] = str(e)
        out.append(p)
    return out
