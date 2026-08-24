"""Offline strategy research: datasets, features, splits, metrics, backtesting.

This package is **one-directional**: it imports from ``backend`` (indicators,
strategies, the paper fill model) but ``backend`` never imports from here. The
trading server has no dependency on research code, and later phases will keep
heavy ML dependencies confined to this side of that line.
"""

RESEARCH_VERSION = "1.0.0"
