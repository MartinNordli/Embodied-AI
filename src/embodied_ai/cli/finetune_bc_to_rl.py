"""Fine-tune RL starting from BC weights (scaffold)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bc_checkpoint", required=True)
    args = parser.parse_args()

    raise SystemExit(
        "Scaffold only. Implement BC→RL loading + continued training. "
        f"(config={args.config}, bc_checkpoint={args.bc_checkpoint})"
    )


if __name__ == "__main__":
    main()


