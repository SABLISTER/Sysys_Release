"""Compatibility imports for the renamed :mod:`process.graph_analysis`.

The old module path remains importable; select defaults.pattern_pack: neuroscience
explicitly to use the historical vocabulary.
"""
from process.graph_analysis import (
    load_patterns, extract_entities, build_cooccurrence_edges,
    compute_edge_threshold, detect_communities, temporal_network_analysis,
)

__all__ = [
    "load_patterns", "extract_entities", "build_cooccurrence_edges",
    "compute_edge_threshold", "detect_communities", "temporal_network_analysis",
]
