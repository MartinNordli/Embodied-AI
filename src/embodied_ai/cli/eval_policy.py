"""Evaluate a trained policy and export videos (scaffold)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    raise SystemExit(
        "Scaffold only. Implement eval loop + video export. "
        f"(config={args.config}, checkpoint={args.checkpoint}, episodes={args.episodes})"
    )


if __name__ == "__main__":
    main()


