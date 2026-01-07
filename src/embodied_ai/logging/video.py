"""Video encoding utilities (scaffold).

Implement with imageio/opencv/moviepy once you decide dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol


class FrameLike(Protocol):
    shape: tuple[int, ...]


def write_video(*, frames: Iterable[FrameLike], out_path: Path, fps: int = 30) -> None:
    raise NotImplementedError("Scaffold only. Implement mp4/gif writer.")


