"""Acquire package — multi-provider literature retrieval clients.

Primary entry point: :func:`acquire.orchestrator.run_acquisition`.
"""

from acquire.orchestrator import run_acquisition
from acquire.deduplicator import deduplicate_papers

__all__ = [
    "run_acquisition",
    "deduplicate_papers",
]
