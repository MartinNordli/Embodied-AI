"""Train Deep RL from pixels (SAC/PPO) (scaffold)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    raise SystemExit(
        f"Scaffold only. Implement RL training harness (SB3/SAC/PPO). (config={args.config})"
    )


if __name__ == "__main__":
    main()


