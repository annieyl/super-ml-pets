"""
Run a single evaluation: load reward_config.yaml, play games with a frozen policy, return aggregate stats.
"""

from __future__ import annotations

import argparse
import json
import os
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from sapai_gym import SuperAutoPetsEnv
from sapai_gym.opponent_gen.opponent_generators import biggest_numbers_horizontal_opp_generator
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


def opponent_generator(num_turns: int):
    """Match smp.utils.opponent_generator without importing tkinter-heavy smp.utils."""
    return biggest_numbers_horizontal_opp_generator(num_turns)

DEFAULT_WEIGHTS = {
    "wins": 1.0,
    "bad_action": 1.0,
    "lives": 0.0,
    "gold": 0.0,
    "turn": 0.0,
    "team_power": 0.0,
}


def load_reward_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    weights = dict(DEFAULT_WEIGHTS)
    weights.update((data.get("weights") or {}))
    return {"weights": weights}


def _team_power_norm(player) -> float:
    total = 0
    for slot in player.team:
        if slot.empty:
            continue
        total += int(slot.pet.attack) + int(slot.pet.health)
    return float(min(total / 200.0, 1.0))


def _shaped_get_reward(self):
    w = getattr(self, "_reward_cfg", DEFAULT_WEIGHTS)
    p = self.player
    bad = float(self.bad_action_reward_sum)
    r = w.get("bad_action", 1.0) * bad
    r += w.get("wins", 1.0) * (p.wins / 10.0)
    r += w.get("lives", 0.0) * (p.lives / 10.0)
    r += w.get("gold", 0.0) * (min(p.gold, 20) / 20.0) * 0.1
    r += w.get("turn", 0.0) * (1.0 - min(p.turn, 25) / 25.0) * 0.01
    r += w.get("team_power", 0.0) * _team_power_norm(p)
    return float(r)


def apply_reward_config(env: SuperAutoPetsEnv, weights: Dict[str, float]) -> None:
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(weights)
    env._reward_cfg = merged
    env.get_reward = types.MethodType(_shaped_get_reward, env)


def make_env() -> SuperAutoPetsEnv:
    return SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)


def rollout_episode(model: MaskablePPO, env: SuperAutoPetsEnv) -> Dict[str, Any]:
    obs = env.reset()
    steps = 0
    while True:
        masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _reward, done, _info = env.step(int(action))
        steps += 1
        if done:
            break
    p = env.player
    won = p.wins >= 10
    return {
        "trophies": int(p.wins),
        "lives": int(p.lives),
        "turn": int(p.turn),
        "won": won,
        "steps": steps,
    }


def run_evaluation(
    model_path: str,
    reward_config_path: str,
    n_rows: int = 10,
    n_cols: int = 10,
    seed: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Play n_rows * n_cols full games (episodes) with the same policy checkpoint.
    Returns (summary, per_episode_records).
    """
    cfg = load_reward_config(reward_config_path)
    weights = cfg["weights"]

    env = make_env()
    apply_reward_config(env, weights)

    model = MaskablePPO.load(model_path)

    rng = np.random.RandomState(seed)
    episodes: List[Dict[str, Any]] = []
    n_games = n_rows * n_cols

    for k in range(n_games):
        env.reset()
        if seed is not None:
            env.action_space.seed(int(rng.randint(0, 2**31 - 1)))
        ep = rollout_episode(model, env)
        ep["episode_index"] = k
        episodes.append(ep)

    env.close()

    wins = sum(1 for e in episodes if e["won"])
    trophies = [e["trophies"] for e in episodes]
    summary = {
        "n_games": n_games,
        "grid": [n_rows, n_cols],
        "win_rate": wins / max(n_games, 1),
        "mean_trophies": float(np.mean(trophies)),
        "std_trophies": float(np.std(trophies)),
        "mean_final_stats": {
            "mean_lives": float(np.mean([e["lives"] for e in episodes])),
            "mean_turn": float(np.mean([e["turn"] for e in episodes])),
            "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        },
        "weights": weights,
    }
    return summary, episodes


def append_log(log_path: str, record: Dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with reward_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to MaskablePPO .zip without extension")
    parser.add_argument("--reward-config", type=str, default="reward_config.yaml")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--log", type=str, default=None, help="Append JSON summary as one line")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    zip_path = args.model + ".zip" if not args.model.endswith(".zip") else args.model
    base = args.model[:-4] if args.model.endswith(".zip") else args.model
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Model not found: {zip_path}")

    summary, _ = run_evaluation(base, args.reward_config, args.rows, args.cols, args.seed)
    print(json.dumps(summary, indent=2))

    if args.log:
        append_log(args.log, {"kind": "eval_only", **summary})


if __name__ == "__main__":
    main()
