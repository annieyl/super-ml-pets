"""
Orchestrate autoresearch: propose patch -> write YAML + snapshot -> finetune+eval -> log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict

from analyze import apply_patch_to_yaml, propose_next, snapshot_config
from train_run import train_run


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
) -> Dict[str, Any]:
    os.makedirs(results_root, exist_ok=True)
    last: Dict[str, Any] = {}

    for i in range(n_iters):
        run_dir = os.path.join(results_root, f"exp_{i:04d}")
        os.makedirs(run_dir, exist_ok=True)

        proposal_path = os.path.join(run_dir, "proposal.json")
        proposal = propose_next(log_path, reward_config_path, idea_path, seed=seed + i)
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)

        apply_patch_to_yaml(reward_config_path, proposal.get("patch") or {})
        snapshot_config(reward_config_path, run_dir)

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
        )
        last = out

        if out.get("is_new_best"):
            best_dir = os.path.join(results_root, "best_checkpoint")
            os.makedirs(best_dir, exist_ok=True)
            src = os.path.join(run_dir, "best_model.zip")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(best_dir, "best_model.zip"))

    return {"last": last}


def main() -> None:
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
    args = parser.parse_args()

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
