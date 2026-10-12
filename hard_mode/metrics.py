"""Run metrics for hard-mode loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hard_mode.paths import hard_mode_run_dir
from core.config import Config


def _count_hits(papers: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_family: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    for p in papers:
        for h in p.get("retrieval_hits") or []:
            fid = str(h.get("query_family_id", ""))
            prov = str(h.get("provider", ""))
            by_family[fid] = by_family.get(fid, 0) + 1
            by_provider[prov] = by_provider.get(prov, 0) + 1
    return {"by_query_family": by_family, "by_provider": by_provider}


def compute_metrics(
    config: Config,
    run_id: str,
    papers: list[dict[str, Any]],
    *,
    round_label: str = "",
    dive_kind: str = "",
) -> dict[str, Any]:
    cheap_triaged = sum(1 for p in papers if (p.get("cheap") or {}).get("queue_band"))
    scored = sum(1 for p in papers if (p.get("machine") or {}).get("axis_scores"))
    reviewed = sum(
        1 for p in papers if (p.get("human") or {}).get("reviewer_label")
    )
    rel = sum(
        1
        for p in papers
        if (p.get("human") or {}).get("reviewer_label") == "relevant"
    )
    tier_counts: dict[int, int] = {}
    cheap_band_counts: dict[str, int] = {}
    for p in papers:
        t = int((p.get("derived") or {}).get("priority_tier") or 0)
        tier_counts[t] = tier_counts.get(t, 0) + 1
        band = str((p.get("cheap") or {}).get("queue_band") or "").strip()
        if band:
            cheap_band_counts[band] = cheap_band_counts.get(band, 0) + 1

    return {
        "run_id": run_id,
        "round": round_label,
        "dive_kind": dive_kind,
        "papers_total": len(papers),
        "papers_cheap_triaged": cheap_triaged,
        "papers_scored": scored,
        "human_reviewed": reviewed,
        "human_relevant": rel,
        "tier_counts": tier_counts,
        "cheap_band_counts": cheap_band_counts,
        "retrieval_hit_counts": _count_hits(papers),
    }


def write_metrics(config: Config, run_id: str, snapshot: dict[str, Any]) -> Path:
    run_dir = hard_mode_run_dir(config, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    return path


def append_metrics_log(config: Config, run_id: str, snapshot: dict[str, Any]) -> None:
    run_dir = hard_mode_run_dir(config, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")
