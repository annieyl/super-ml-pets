"""
Read experiment log, identify the best run, and propose the next reward patch (JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from reward_config import DEFAULT_WEIGHTS, load_reward_config

# .env is not read by the OS or Python by default; load repo-root .env for GEMINI_API_KEY etc.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

IDEA_PATH = "idea.md"
REWARD_CONFIG_PATH = "reward_config.yaml"
DEFAULT_GEMINI_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite"
]


def read_log_records(log_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(log_path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def best_record(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    scored = [r for r in records if "mean_trophies" in r and "weights" in r]
    if not scored:
        return None
    return max(scored, key=lambda r: (r["mean_trophies"], r.get("win_rate", 0.0)))


def last_k_records(records: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    return records[-k:]


def load_idea_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))


def _gemini_generate(
    api_key: str,
    model_name: str,
    payload: Dict[str, Any],
    max_wait_seconds: float = 120.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + max_wait_seconds
    sleep_seconds = 2.0
    last_error: Optional[Exception] = None

    while True:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_error = e
            # Retry transient availability/rate-limit failures for up to max_wait_seconds.
            if e.code in (429, 500, 502, 503, 504):
                now = time.monotonic()
                if now >= deadline:
                    break
                time.sleep(min(sleep_seconds, deadline - now))
                sleep_seconds = min(sleep_seconds * 1.6, 12.0)
                continue
            raise
        except urllib.error.URLError as e:
            last_error = e
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(sleep_seconds, deadline - now))
            sleep_seconds = min(sleep_seconds * 1.6, 12.0)

    if isinstance(last_error, urllib.error.HTTPError):
        err_text = last_error.read().decode("utf-8", errors="replace")
        raise RuntimeError(err_text) from last_error
    raise RuntimeError(f"Gemini request failed after waiting up to {max_wait_seconds:.0f} seconds")


def _gemini_list_models(api_key: str) -> List[str]:
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    names: List[str] = []
    for model in body.get("models", []):
        methods = model.get("supportedGenerationMethods") or []
        if "generateContent" in methods:
            name = str(model.get("name", ""))
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if name:
                names.append(name)
    return names


def propose_with_gemini(
    idea_text: str,
    recent: List[Dict[str, Any]],
    current_weights: Dict[str, float],
    best: Optional[Dict[str, Any]] = None,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    prompt = (
        "You output only valid JSON objects as specified by the user.\n\n"
        + idea_text
        + "\n\n## Current weights\n"
        + json.dumps(current_weights, indent=2)
        + "\n\n## Best experiment so far\n"
        + json.dumps(best or {}, indent=2)
        + "\n\n## Recent experiments (oldest to newest)\n"
        + json.dumps(recent, indent=2)
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
        },
    }
    requested_model = os.environ.get("AUTORESEARCH_MODEL", model)
    try:
        body = _gemini_generate(api_key, requested_model, payload)
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        if e.code == 404:
            try:
                available = _gemini_list_models(api_key)
            except Exception:
                available = []
            preferred = [requested_model] + DEFAULT_GEMINI_CANDIDATES + available
            dedup: List[str] = []
            for m in preferred:
                if m and m not in dedup:
                    dedup.append(m)
            for candidate in dedup:
                if candidate == requested_model:
                    continue
                try:
                    body = _gemini_generate(api_key, candidate, payload)
                    break
                except urllib.error.HTTPError:
                    continue
            else:
                raise RuntimeError(err_text) from e
        else:
            raise RuntimeError(err_text) from e

    content = body["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json_object(content)


def propose_heuristic(
    recent: List[Dict[str, Any]],
    current_weights: Dict[str, float],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Offline fallback: nudge weights using very simple rules on the last run.
    """
    base = dict(DEFAULT_WEIGHTS)
    base.update(current_weights)
    last = recent[-1] if recent else None
    patch: Dict[str, float] = {}

    if last and last.get("mean_trophies", 0) < 3.0:
        patch["wins"] = min(2.0, base.get("wins", 1.0) + 0.05)
        patch["team_power"] = min(1.0, base.get("team_power", 0.0) + 0.03)
        rationale = "Low trophies suggest strengthening win and board-power signals slightly."
    elif last and last.get("mean_final_stats", {}).get("mean_lives", 5.0) < 3.0:
        patch["lives"] = min(1.0, base.get("lives", 0.0) + 0.04)
        rationale = "Bleeding lives suggests adding a small bonus for preserving lives."
    else:
        keys = ["wins", "lives", "gold", "turn", "team_power"]
        k = rng.choice(keys)
        delta = rng.choice([-0.04, -0.02, 0.02, 0.04])
        new_val = float(base.get(k, 0.0)) + delta
        new_val = max(-1.0, min(2.0, new_val))
        patch[k] = new_val
        rationale = "Exploratory small random move on one weight to probe the response surface."

    return {"patch": patch, "rationale": rationale}


def propose_next(
    log_path: str,
    reward_config_path: str,
    idea_path: str,
    seed: int = 0,
) -> Dict[str, Any]:
    records = read_log_records(log_path)
    recent = last_k_records(records, 5)
    current = load_reward_config(reward_config_path)["weights"]
    best = best_record(records)

    if os.environ.get("GEMINI_API_KEY"):
        idea = load_idea_text(idea_path)
        return propose_with_gemini(idea, recent, current, best)

    rng = random.Random(seed)
    print("can't find api key")
    return propose_heuristic(recent, current, rng)


def apply_patch_to_yaml(reward_config_path: str, patch: Dict[str, float]) -> Dict[str, float]:
    cfg = load_reward_config(reward_config_path)
    merged = dict(cfg["weights"])
    for k, v in patch.items():
        if k in DEFAULT_WEIGHTS:
            merged[k] = float(v)
    out = {"weights": merged}
    with open(reward_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=True)
    return merged


def snapshot_config(reward_config_path: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "reward_config.yaml")
    with open(reward_config_path, "r", encoding="utf-8") as f:
        txt = f.read()
    with open(dest, "w", encoding="utf-8") as f:
        f.write(txt)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="log.txt")
    parser.add_argument("--reward-config", type=str, default=REWARD_CONFIG_PATH)
    parser.add_argument("--idea", type=str, default=IDEA_PATH)
    parser.add_argument("--out", type=str, default="proposal.json", help="Write proposal JSON here")
    parser.add_argument("--apply", action="store_true", help="Patch reward_config.yaml in place")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="If set, copy YAML here after apply")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    proposal = propose_next(args.log, args.reward_config, args.idea, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(proposal, f, indent=2)

    print(json.dumps(proposal, indent=2))

    if args.apply:
        merged = apply_patch_to_yaml(args.reward_config, proposal.get("patch") or {})
        if args.snapshot_dir:
            snapshot_config(args.reward_config, args.snapshot_dir)
        print("Updated weights:", json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
