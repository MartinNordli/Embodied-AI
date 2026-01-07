"""Helpers to build consistent SB3 policy kwargs (scaffold)."""

from __future__ import annotations


def build_policy_kwargs(*, features_extractor_class, features_extractor_kwargs: dict) -> dict:
    """Return a dict suitable for SB3 `policy_kwargs=`."""
    return {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": features_extractor_kwargs,
    }


