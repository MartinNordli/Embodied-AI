"""Device selection helpers (scaffold)."""

from __future__ import annotations


def pick_device(preferred: str = "cuda") -> str:
    """Return 'cuda' if available else 'cpu' (when torch is installed)."""
    if preferred != "cuda":
        return preferred
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


