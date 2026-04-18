Place pretrained MaskablePPO weights here (Stable-Baselines3 format), without the `.zip` extension in CLI paths.

Example:
  checkpoints/base_model.zip

The autoresearch loop reads from this directory only; it writes working copies and best snapshots under `results/`.
