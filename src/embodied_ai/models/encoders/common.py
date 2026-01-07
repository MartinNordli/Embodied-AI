"""Shared encoder utilities (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderOutputSpec:
    features_dim: int


