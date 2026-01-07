"""Single source of truth for building the ManiSkill2 env + wrappers (scaffold)."""

from __future__ import annotations

from typing import Any

from embodied_ai.envs.tasks import TASKS


def make_env(task_id: str, **kwargs: Any) -> Any:
    """Create and wrap an environment.

    Replace `Any` with `gymnasium.Env` once gymnasium is installed.
    """
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id={task_id!r}. Known: {sorted(TASKS.keys())}")

    # This is intentionally a stub so the scaffold is dependency-free.
    raise NotImplementedError(
        "Scaffold only. Implement ManiSkill2 environment construction here.\n"
        "Suggested: import mani_skill2 / gymnasium and apply wrappers from envs/wrappers."
    )


