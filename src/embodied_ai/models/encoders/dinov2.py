"""DINOv2 encoder wiring (scaffold).

Suggested implementation:
- load backbone via torch.hub or a pinned package
- optionally freeze backbone
- add small projection head to `features_dim`
"""

from __future__ import annotations


class DinoV2EncoderNotImplementedError(NotImplementedError):
    pass


