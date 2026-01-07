"""Observation wrappers (scaffold).

Responsibilities:
- extract RGB / RGB-D from env observations
- resize + normalize
- optional frame stacking
"""

from __future__ import annotations


class ObsWrapperNotImplementedError(NotImplementedError):
    """Raised when wrappers are used before being implemented."""



