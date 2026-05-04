"""
Orchestrate autoresearch: propose patch -> write YAML + snapshot -> finetune+eval -> log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from analyze import apply_patch_to_yaml, propose_next, snapshot_config
from train_run import train_run


@dataclass
class ExperimentSpec:
    """One playstyle / prompt experiment: separate log and results subfolder."""

    name: str
    idea_path: str
    log_path: str
    n_iters: int


def _warn_if_conda_and_venv() -> None:
    """Mixing conda (base) with a venv often breaks PyTorch DLLs on Windows (c10.dll WinError 1114)."""
    if os.environ.get("CONDA_PREFIX") and "venv" in sys.prefix.replace("\\", "/").lower():
        print(
            "WARNING: Conda is active (e.g. (base)) while sys.prefix is a venv. "
            "Run `conda deactivate` until only (.venv) remains, or recreate the venv with "
            "https://www.python.org/downloads/ Python (not conda's interpreter).",
            file=sys.stderr,
        )


def run_loop(
    n_iters: int,
    checkpoint_stem: str,
    log_path: str = "log.txt",
    reward_config_path: str = "reward_config.yaml",
    idea_path: str = "idea.md",
    results_root: str = "results",
    finetune_steps: int = 2048,
    eval_rows: int = 10,
    eval_cols: int = 10,
    seed: int = 0,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(results_root, exist_ok=True)
    last: Dict[str, Any] = {}
    run_started = time.perf_counter()

    for i in range(n_iters):
        iter_started = time.perf_counter()
        run_dir = os.path.join(results_root, f"exp_{i:04d}")
        os.makedirs(run_dir, exist_ok=True)

        proposal_path = os.path.join(run_dir, "proposal.json")
        t0 = time.perf_counter()
        proposal = propose_next(log_path, reward_config_path, idea_path, seed=seed + i)
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)
        propose_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        apply_patch_to_yaml(reward_config_path, proposal.get("patch") or {})
        snapshot_config(reward_config_path, run_dir)
        patch_seconds = time.perf_counter() - t1

        t2 = time.perf_counter()
        out = train_run(
            checkpoint_stem,
            reward_config_path,
            log_path,
            results_dir=run_dir,
            finetune_steps=finetune_steps,
            eval_rows=eval_rows,
            eval_cols=eval_cols,
            iter_index=i,
            proposal=proposal,
            seed=seed + i * 9973,
            experiment_name=experiment_name,
        )
        train_eval_seconds = time.perf_counter() - t2
        last = out

        if out.get("is_new_best"):
            best_dir = os.path.join(results_root, "best_checkpoint")
            os.makedirs(best_dir, exist_ok=True)
            src = os.path.join(run_dir, "best_model.zip")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(best_dir, "best_model.zip"))

        iter_seconds = time.perf_counter() - iter_started
        stats = out.get("summary", {})
        print(
            (
                f"[iter {i + 1}/{n_iters}] "
                f"elapsed={iter_seconds:.2f}s "
                f"(propose={propose_seconds:.2f}s, patch={patch_seconds:.2f}s, train_eval={train_eval_seconds:.2f}s) "
                f"verdict={out.get('verdict')} "
                f"win_rate={float(stats.get('win_rate', 0.0)):.3f} "
                f"mean_trophies={float(stats.get('mean_trophies', 0.0)):.3f}"
            )
        )

    return {"last": last, "total_runtime_seconds": round(time.perf_counter() - run_started, 3)}


def run_experiments(
    specs: List[ExperimentSpec],
    *,
    checkpoint_stem: str,
    reward_config_path: str,
    baseline_reward_config_path: str,
    base_results_root: str = "results",
    finetune_steps: int = 2048,
    eval_rows: int = 10,
    eval_cols: int = 10,
    base_seed: int = 0,
    truncate_logs: bool = True,
) -> Dict[str, Any]:
    """
    Run several independent autoresearch tracks (e.g. aggressive vs conservative).

    Before each track, copies ``baseline_reward_config_path`` onto ``reward_config_path`` so every
    experiment starts from the same weights. Each spec uses its own ``log_path`` and
    ``base_results_root / name`` for artifacts.
    """
    if not specs:
        return {"experiments": []}

    if not os.path.isfile(baseline_reward_config_path):
        raise FileNotFoundError(f"Baseline reward config not found: {baseline_reward_config_path}")

    summaries: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        shutil.copy2(baseline_reward_config_path, reward_config_path)
        log_dir = os.path.dirname(os.path.abspath(spec.log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        if truncate_logs:
            open(spec.log_path, "w", encoding="utf-8").close()

        exp_seed = base_seed + idx * 7919
        out = run_loop(
            n_iters=spec.n_iters,
            checkpoint_stem=checkpoint_stem,
            log_path=spec.log_path,
            reward_config_path=reward_config_path,
            idea_path=spec.idea_path,
            results_root=os.path.join(base_results_root, spec.name),
            finetune_steps=finetune_steps,
            eval_rows=eval_rows,
            eval_cols=eval_cols,
            seed=exp_seed,
            experiment_name=spec.name,
        )
        summaries.append({"name": spec.name, **out})

    return {"experiments": summaries}


def load_experiment_suite(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("experiments") or []
    specs: List[ExperimentSpec] = []
    for item in raw:
        specs.append(
            ExperimentSpec(
                name=str(item["name"]),
                idea_path=str(item["idea"]),
                log_path=str(item["log"]),
                n_iters=int(item["iters"]),
            )
        )
    return {
        "specs": specs,
        "baseline_reward_config": str(
            data.get("baseline_reward_config") or "reward_config.experiment_baseline.yaml"
        ),
        "checkpoint": str(data.get("checkpoint") or os.path.join("checkpoints", "base_model")),
        "reward_config": str(data.get("reward_config") or "reward_config.yaml"),
        "base_results_root": str(data.get("base_results_root") or "results"),
        "finetune_steps": int(data.get("finetune_steps") or 2048),
        "eval_rows": int(data.get("eval_rows") or 10),
        "eval_cols": int(data.get("eval_cols") or 10),
        "seed": int(data.get("seed") or 0),
    }


def main() -> None:
    # Keep terminal output clean during long loops.
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3, help="Number of autoresearch iterations")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "base_model"),
        help="Pretrained MaskablePPO stem (no .zip)",
    )
    parser.add_argument("--log", type=str, default="log.txt")
    parser.add_argument("--reward-config", type=str, default="reward_config.yaml")
    parser.add_argument("--idea", type=str, default="idea.md")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--finetune-steps", type=int, default=2048)
    parser.add_argument("--eval-rows", type=int, default=10)
    parser.add_argument("--eval-cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--experiment-suite",
        type=str,
        default=None,
        help="YAML suite (see experiments_suite.example.yaml) to run multiple experiments",
    )
    parser.add_argument(
        "--append-experiment-logs",
        action="store_true",
        help="With --experiment-suite, do not truncate per-experiment log files before each run",
    )
    args = parser.parse_args()
    _warn_if_conda_and_venv()

    if args.experiment_suite:
        suite = load_experiment_suite(args.experiment_suite)
        summary = run_experiments(
            suite["specs"],
            checkpoint_stem=suite["checkpoint"],
            reward_config_path=suite["reward_config"],
            baseline_reward_config_path=suite["baseline_reward_config"],
            base_results_root=suite["base_results_root"],
            finetune_steps=suite["finetune_steps"],
            eval_rows=suite["eval_rows"],
            eval_cols=suite["eval_cols"],
            base_seed=suite["seed"],
            truncate_logs=not args.append_experiment_logs,
        )
    else:
        summary = run_loop(
            n_iters=args.iters,
            checkpoint_stem=args.checkpoint,
            log_path=args.log,
            reward_config_path=args.reward_config,
            idea_path=args.idea,
            results_root=args.results_root,
            finetune_steps=args.finetune_steps,
            eval_rows=args.eval_rows,
            eval_cols=args.eval_cols,
            seed=args.seed,
        )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
