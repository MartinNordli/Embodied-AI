# Embodied AI — ManiSkill2 (Demos → BC → Deep RL)

This repository is structured for a **vision-based manipulation** project:

- Collect demonstrations in ManiSkill2
- Train **Behavior Cloning (BC)** from pixels
- Fine-tune with **Deep RL** (SAC/PPO)
- Log metrics, configs, checkpoints, and export videos

## Quick start (skeleton)

This repo currently contains a **scaffold** (project map + module stubs). The next step is to:

- pick a ManiSkill2 task id and camera setup in `src/embodied_ai/envs/`
- implement the wrappers in `src/embodied_ai/envs/wrappers/`
- implement training logic in `src/embodied_ai/algorithms/`

## Repo layout

- `configs/`: experiment configs (env/obs/algo/encoder/augmentation/randomization)
- `src/embodied_ai/`: Python package (envs, data, models, training, logging, utils)
- `scripts/`: helper scripts (install/sweeps)
- `tests/`: smoke tests for env building and dataset loading
- `docs/`: report + experiment notes
- `assets/`: small checked-in figures/videos for README/report
- `data_local/`: large local-only data (demos, pretrained weights) — ignored by git
- `runs/`: run artifacts (configs, checkpoints, videos) — ignored by git


