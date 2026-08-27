"""Prompt features computed from the prompt text alone.

Public entry points:

    from promptfeat import extract_features, explain_prompt
    extract_features("Who wrote The Hobbit?")

The registry in `promptfeat.registry` is the single source of truth: it drives
the flat dict, the explanations, FEATURES.md and the web UI.
"""

from .engine import compute, explain_prompt, extract_features, feature_declaration
from .registry import (
    GROUPS,
    NOT_APPLICABLE,
    OK,
    REGISTRY,
    UNAVAILABLE,
    UNDEFINED,
    UNRELIABLE,
    Feature,
    FeatureResult,
    feature_names,
    features_by_group,
    top_features,
)

FEATURE_NAMES = list(REGISTRY)
TOP30_FEATURES = [f.name for f in top_features()]

__all__ = [
    "extract_features",
    "explain_prompt",
    "compute",
    "feature_declaration",
    "REGISTRY",
    "FEATURE_NAMES",
    "TOP30_FEATURES",
    "GROUPS",
    "Feature",
    "FeatureResult",
    "feature_names",
    "features_by_group",
    "top_features",
    "OK",
    "NOT_APPLICABLE",
    "UNDEFINED",
    "UNRELIABLE",
    "UNAVAILABLE",
]
