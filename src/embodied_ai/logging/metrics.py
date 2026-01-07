"""Metric aggregation helpers (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeStats:
    episode_return: float
    length: int
    success: bool | None = None


