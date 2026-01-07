"""Config helpers (scaffold).

You can swap this for Hydra/OmegaConf later. For now: load plain YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("Install pyyaml to use load_yaml().") from e
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at root of YAML, got {type(data)}")
    return data


