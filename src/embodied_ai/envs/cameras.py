"""Camera setup helpers (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSpec:
    name: str = "base_camera"
    width: int = 128
    height: int = 128


DEFAULT_CAMERA = CameraSpec()


