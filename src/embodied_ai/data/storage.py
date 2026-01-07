"""Path helpers and naming conventions (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    runs_dir: Path = Path("runs")
    data_dir: Path = Path("data_local")

    @property
    def demos_dir(self) -> Path:
        return self.data_dir / "demos"


DEFAULT_PATHS = ProjectPaths()


