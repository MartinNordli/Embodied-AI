"""Collect demonstration trajectories from a scripted/demo policy (scaffold)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="data_local/demos/trajectories.npz")
    args = parser.parse_args()

    raise SystemExit(
        "Scaffold only. Implement demo rollout + trajectory saving. "
        f"(config={args.config}, episodes={args.episodes}, out={args.out})"
    )


if __name__ == "__main__":
    main()


