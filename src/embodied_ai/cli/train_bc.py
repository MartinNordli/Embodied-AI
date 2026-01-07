"""Train Behavior Cloning (BC) on demonstration trajectories (scaffold).

Replace this with:
- loading demos via `embodied_ai.data.demos_dataset`
- training loop via `embodied_ai.algorithms.bc_trainer`
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    raise SystemExit(
        f"Scaffold only. Implement BC training. (config={args.config})"
    )


if __name__ == "__main__":
    main()


