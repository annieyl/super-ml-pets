"""
Load patched reward_config, optional short finetune on pretrained MaskablePPO, evaluate, append experiment log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, Optional, Tuple

from sb3_contrib import MaskablePPO

from analyze import best_record, read_log_records
from experiment import append_log, apply_reward_config, load_reward_config, make_env, run_evaluation


def _verdict(
    summary: Dict[str, Any],
    prior_best: Optional[Dict[str, Any]],
) -> Tuple[str, bool]:
    mt = float(summary["mean_trophies"])
    wr = float(summary["win_rate"])
    if prior_best is None:
        return "first_baseline", True
    best_mt = float(prior_best["mean_trophies"])
    best_wr = float(prior_best.get("win_rate", 0.0))
    if (mt > best_mt) or (mt == best_mt and wr > best_wr):
        return "new_best", True
    if mt >= best_mt - 1e-6:
        return "tie_best", False
    return "below_best", False


def train_run(
    checkpoint_stem: str,
    reward_config_path: str,
    log_path: str,
    results_dir: str,
    finetune_steps: int = 2048,
    eval_rows: int = 10,
    eval_cols: int = 10,
    iter_index: int = 0,
    proposal: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)

    zip_path = checkpoint_stem + (".zip" if not checkpoint_stem.endswith(".zip") else "")
    stem = checkpoint_stem[:-4] if checkpoint_stem.endswith(".zip") else checkpoint_stem
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Checkpoint not found: {zip_path}")

    cfg = load_reward_config(reward_config_path)
    env = make_env()
    apply_reward_config(env, cfg["weights"])

    model = MaskablePPO.load(stem, env=env)

    if finetune_steps > 0:
        model.learn(total_timesteps=int(finetune_steps))

    working_path = os.path.join(results_dir, f"iter_{iter_index:04d}_model.zip")
    model.save(working_path[:-4])
    del model

    records = read_log_records(log_path)
    prior = best_record(records)

    post_stem = working_path[:-4]
    summary, episodes = run_evaluation(
        post_stem,
        reward_config_path,
        n_rows=eval_rows,
        n_cols=eval_cols,
        seed=seed,
    )

    verdict, is_new_best = _verdict(summary, prior)
    full: Dict[str, Any] = {
        "kind": "train_run",
        "iter": iter_index,
        "finetune_steps": int(finetune_steps),
        "checkpoint_in": stem,
        "checkpoint_out": post_stem,
        "proposal": proposal or {},
        "verdict": verdict,
        "is_new_best": is_new_best,
        "win_rate": summary["win_rate"],
        "mean_trophies": summary["mean_trophies"],
        "std_trophies": summary["std_trophies"],
        "mean_final_stats": summary["mean_final_stats"],
        "weights": summary["weights"],
        "n_eval_games": summary["n_games"],
    }
    append_log(log_path, full)

    best_ckpt_marker = os.path.join(results_dir, "best_model_path.txt")
    if is_new_best:
        best_zip = os.path.join(results_dir, "best_model.zip")
        shutil.copy2(post_stem + ".zip", best_zip)
        with open(best_ckpt_marker, "w", encoding="utf-8") as f:
            f.write(best_zip + "\n")

    return {
        "summary": summary,
        "verdict": verdict,
        "is_new_best": is_new_best,
        "full_record": full,
        "episodes_sample": episodes[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Pretrained model path without .zip")
    parser.add_argument("--reward-config", type=str, default="reward_config.yaml")
    parser.add_argument("--log", type=str, default="log.txt")
    parser.add_argument("--results-dir", type=str, default="results/latest_run")
    parser.add_argument("--finetune-steps", type=int, default=2048)
    parser.add_argument("--eval-rows", type=int, default=10)
    parser.add_argument("--eval-cols", type=int, default=10)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = train_run(
        args.checkpoint,
        args.reward_config,
        args.log,
        args.results_dir,
        finetune_steps=args.finetune_steps,
        eval_rows=args.eval_rows,
        eval_cols=args.eval_cols,
        iter_index=args.iter,
        proposal=None,
        seed=args.seed,
    )
    print(json.dumps({"verdict": out["verdict"], "mean_trophies": out["summary"]["mean_trophies"]}, indent=2))


if __name__ == "__main__":
    main()
