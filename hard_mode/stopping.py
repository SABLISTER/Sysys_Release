"""Stopping rules for iterative retrieval."""

from __future__ import annotations

from typing import Any


def should_stop_retrieval(
    hm: dict[str, Any],
    cycle_index: int,
    last_cycle_new_relevant: int,
) -> tuple[bool, str]:
    st = hm.get("stopping") or {}
    max_cycles = int(st.get("max_retrieval_rounds", 12))
    min_yield = int(st.get("min_new_reviewed_relevant_per_round", 0))

    if cycle_index >= max_cycles:
        return True, f"max_retrieval_rounds ({max_cycles}) reached"
    if cycle_index > 0 and last_cycle_new_relevant < min_yield:
        return (
            True,
            f"new relevant below threshold ({last_cycle_new_relevant} < {min_yield})",
        )
    return False, ""
