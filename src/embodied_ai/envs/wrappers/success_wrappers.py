"""Success metric wrappers (scaffold).

Goal: ensure a consistent `info["is_success"]` boolean for logging/eval.
"""

from __future__ import annotations


class SuccessWrapperNotImplementedError(NotImplementedError):
    """Raised when wrapper is referenced before being implemented."""



