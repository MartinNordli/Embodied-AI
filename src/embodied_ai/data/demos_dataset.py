"""Demonstrations dataset loader (scaffold).

Target format suggestion:
- store trajectories as .npz/.h5 with arrays:
  - obs_rgb: uint8 [T, H, W, 3]
  - action: float32 [T, act_dim]
  - done: bool [T]
  - info_success: bool [T] (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoDatasetConfig:
    path: Path


class DemosDataset:
    def __init__(self, cfg: DemoDatasetConfig) -> None:
        self.cfg = cfg

    def __len__(self) -> int:
        raise NotImplementedError("Scaffold only. Implement dataset indexing.")

    def __getitem__(self, idx: int):
        raise NotImplementedError("Scaffold only. Implement sample retrieval.")


