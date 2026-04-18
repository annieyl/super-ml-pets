# super-ml-pets

Reinforcement learning for Super Auto Pets (see `README_original.md` for the full upstream guide: training, deployment, assets).

## Autoresearch loop (reward tuning)

This repo includes a small **autoresearch** harness that proposes changes to `reward_config.yaml`, runs a short finetune plus evaluation, and logs results to `log.txt`. Each line in the log is one JSON object (JSONL).

### Prerequisites

Use **Python 3.8 or newer** (3.11+ recommended on Windows) so `numpy` and `scikit-learn` install from wheels without compiling.

1. Python environment with dependencies installed.

**Windows (recommended):** Stable-Baselines3 1.8 pins **`gym==0.21`**, which often fails to build under modern pip unless setuptools is pinned and `gym` is installed without build isolation. From the repo root run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_deps.ps1
```

**macOS / Linux:** `chmod +x install_deps.sh && ./install_deps.sh`

**Manual equivalent** (any OS):

```powershell
pip install -U pip
pip install "setuptools>=59.5,<65.0" wheel
pip install "gym==0.21.0" --no-build-isolation
pip install -r requirements.txt
```

**Anaconda / conda base:** install into a **dedicated virtual environment** (`python -m venv .venv` then activate). The base env already contains packages like `numba`, `gensim`, and `tables`; mixing them with `pip install -r requirements.txt` produces resolver warnings and hard conflicts (for example `numba 0.57` needs `numpy<1.25`, which this file respects—upgrading numpy in base can still break other conda tools).

If you see `ModuleNotFoundError: No module named 'sb3_contrib'`, the RL stack did not install (for example because `pip` stopped at an earlier line in `requirements.txt`). Install it explicitly, then retry the full file:

```powershell
pip install "stable-baselines3>=1.8.0,<2.0" "sb3-contrib==1.8.0"
pip install -r requirements.txt
```

2. A **pretrained MaskablePPO** checkpoint in Stable-Baselines3 format (a single `.zip` file). Put it under `checkpoints/` and use the path **without** the `.zip` extension in commands. Example file: `checkpoints/base_model.zip` → pass `checkpoints/base_model`.

Note: `*.zip` is gitignored; add a checkpoint locally or use `git add -f` if you need it in version control.

### Run the full loop

From the repository root:

```powershell
python run_loop.py --iters 5 --checkpoint checkpoints/base_model --finetune-steps 2048
```

Useful options:

| Flag | Meaning |
|------|--------|
| `--iters N` | Number of propose → patch → finetune → eval cycles (default: 3). |
| `--checkpoint PATH` | Model stem without `.zip` (default: `checkpoints/base_model`). |
| `--finetune-steps` | PPO `learn` timesteps per iteration before eval (default: 2048). |
| `--eval-rows`, `--eval-cols` | Eval grid size; default 10×10 = 100 games. |
| `--log PATH` | Append-only log (default: `log.txt`). |
| `--reward-config PATH` | YAML to patch (default: `reward_config.yaml`). |
| `--idea PATH` | Agent prompt file (default: `idea.md`). |
| `--results-root` | Where per-iteration folders go (default: `results`). |

When a run sets a new best `mean_trophies` (ties broken by `win_rate`), the loop saves `results/<run>/best_model.zip` and copies the overall best to `results/best_checkpoint/best_model.zip`.

### LLM-driven proposals (optional)

With `OPENAI_API_KEY` set, `analyze.py` calls the Chat Completions API using the prompt in `idea.md`. Override the model name with `AUTORESEARCH_MODEL` if you want.

Without an API key, the loop uses a small **offline** heuristic instead of an LLM.

### Run pieces manually

- **Evaluate only** (no training), 10×10 games, optional log append:

```powershell
python experiment.py --model checkpoints/base_model --reward-config reward_config.yaml --rows 10 --cols 10 --log log.txt
```

- **One finetune + eval + log line** (same core as one loop iteration, without proposing a patch):

```powershell
python train_run.py --checkpoint checkpoints/base_model --reward-config reward_config.yaml --log log.txt --results-dir results/manual --finetune-steps 2048 --iter 0
```

- **Propose next patch** (writes `proposal.json`; add `--apply` to merge into YAML and `--snapshot-dir DIR` to copy the YAML):

```powershell
python analyze.py --log log.txt --out proposal.json
python analyze.py --log log.txt --out proposal.json --apply --snapshot-dir results/snapshot_01
```

### Files involved

| File | Role |
|------|------|
| `idea.md` | Instructions for the hypothesis agent (when using OpenAI). |
| `reward_config.yaml` | Active reward weights; updated when patches are applied. |
| `log.txt` | Append-only JSONL experiment log. |
| `run_loop.py` | Orchestrates analyze → patch → `train_run` for N iterations. |
| `train_run.py` | Finetune checkpoint, evaluate, append log, track best. |
| `experiment.py` | Load YAML, shape rewards, run fixed rollout eval. |
| `analyze.py` | Read log, pick best, emit next `patch` JSON. |
| `checkpoints/` | Put your starting pretrained `.zip` here (e.g. `checkpoints/base_model.zip`). |
