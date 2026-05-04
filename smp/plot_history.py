"""
Plot Stable-Baselines `progress.csv` or autoresearch JSONL experiment logs.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc


def plot_sb3_csv(csv_path: str, show: bool = True, outfile: Optional[str] = None) -> None:
    data = pd.read_csv(csv_path)
    # rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    x = list(range(len(data["rollout/ep_len_mean"])))
    ax[0].plot(x, data["rollout/ep_len_mean"])
    ax[0].set_title("Number of actions before game is done")
    ax[0].grid("on")
    ax[1].plot(x, data["rollout/ep_rew_mean"])
    ax[1].set_title("Mean episode reward (SB3 rollout)")
    ax[1].grid("on")
    plt.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def load_train_runs_jsonl(paths: List[str], display_labels: Optional[List[str]] = None) -> pd.DataFrame:
    rows: List[dict] = []
    for i, path in enumerate(paths):
        file_stem = os.path.splitext(os.path.basename(path))[0]
        forced = display_labels[i] if display_labels and i < len(display_labels) else None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("kind") != "train_run":
                    continue
                series = forced or r.get("experiment") or file_stem
                rows.append(
                    {
                        "series": series,
                        "iter": int(r.get("iter", 0)),
                        "mean_trophies": float(r.get("mean_trophies", 0.0)),
                        "win_rate": float(r.get("win_rate", 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def plot_experiment_jsonl(
    paths: List[str],
    labels: Optional[List[str]] = None,
    show: bool = True,
    outfile: Optional[str] = None,
) -> None:
    df = load_train_runs_jsonl(paths, display_labels=labels)
    if df.empty:
        raise ValueError("No train_run records found in JSONL file(s).")

    # rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    for name, g in df.groupby("series"):
        g = g.sort_values("iter")
        ax[0].plot(g["iter"], g["mean_trophies"], marker="o", label=name)
        ax[1].plot(g["iter"], g["win_rate"], marker="o", label=name)

    ax[0].set_title("Eval mean trophies per autoresearch iteration")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("mean trophies")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title("Eval win rate per autoresearch iteration")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("win rate")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def main() -> None:
    parser = ArgumentParser(description="Plot SB3 progress.csv or autoresearch JSONL logs.")
    parser.add_argument(
        "--csv",
        "--log",
        dest="csv",
        type=str,
        default=None,
        help="Path to Stable-Baselines progress.csv (--log is legacy alias)",
    )
    parser.add_argument(
        "--experiment-jsonl",
        type=str,
        nargs="+",
        default=None,
        metavar="LOG",
        help="One or more autoresearch JSONL logs (e.g. logs/aggressive.jsonl logs/conservative.jsonl)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Legend labels in the same order as --experiment-jsonl files",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a GUI window (use with --out)",
    )
    parser.add_argument("--out", type=str, default=None, help="Save figure to this path")
    args = parser.parse_args()

    show = not args.no_show
    default_history = "./history/history_rl_model/progress.csv"
    if args.experiment_jsonl:
        plot_experiment_jsonl(
            args.experiment_jsonl,
            labels=args.labels,
            show=show,
            outfile=args.out,
        )
    elif args.csv:
        plot_sb3_csv(args.csv, show=show, outfile=args.out)
    elif os.path.isfile(default_history):
        plot_sb3_csv(default_history, show=show, outfile=args.out)
    else:
        parser.error(
            "Provide --csv PATH, --experiment-jsonl log1 [log2 ...], "
            f"or create {default_history}"
        )


if __name__ == "__main__":
    main()
