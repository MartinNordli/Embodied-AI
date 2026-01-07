"""Task registry (scaffold).

Keep all task IDs and env kwargs in one place so the rest of the code can stay task-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    max_episode_steps: int = 200


TASKS: dict[str, TaskSpec] = {
    "PickCube-v1": TaskSpec(task_id="PickCube-v1", max_episode_steps=200),
    "StackCube-v1": TaskSpec(task_id="StackCube-v1", max_episode_steps=250),
}


