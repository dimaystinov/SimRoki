from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path):
    if not path.exists():
        return None
    data = path.read_text(encoding="utf-8").strip()
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _extract_xy(data: Any):
    if not data:
        return [], [], [], [], []
    t = [row.get("timesteps", 0) or row.get("total_seen", 0) for row in data]
    right = [row.get("right_robot_dx", 0.0) for row in data]
    left = [row.get("left_robot_dx", 0.0) for row in data]
    mn = []
    for row in data:
        min_dx = row.get("min_robot_dx")
        if min_dx is None:
            min_dx = min(float(row.get("right_robot_dx", 0.0)), float(row.get("left_robot_dx", 0.0)))
        mn.append(min_dx)
    stage = [row.get("stage", "") for row in data]
    return t, right, left, mn, stage


def draw(train_log: Path, eval_log: Path, out_png: Path, max_points: int = 200) -> bool:
    progress = _read_json(train_log)
    eval_data = _read_json(eval_log)
    if not progress and not eval_data:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax1 = axes[0]
    if progress:
        ts = [row.get("timesteps", 0) for row in progress]
        ep_rew = [row.get("ep_reward_mean", 0.0) for row in progress]
        ep_len = [row.get("ep_len_mean", 0.0) for row in progress]
        st = [row.get("stage", "") for row in progress]

        # Keep only last window
        ts = ts[-max_points:]
        ep_rew = ep_rew[-max_points:]
        ep_len = ep_len[-max_points:]
        st = st[-max_points:]
        ax1.plot(ts, ep_rew, marker="o", color="#1f77b4", label="ep_reward_mean")
        ax1.set_title("Training reward")
        ax1.set_ylabel("reward")
        ax1.set_xlabel("timesteps")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best")

        ax1_twin = ax1.twinx()
        ax1_twin.plot(ts, ep_len, marker="x", color="#ff7f0e", alpha=0.8, label="ep_len_mean")
        ax1_twin.set_ylabel("episode len")
        ax1_twin.legend(loc="upper right")

        for i in range(1, len(ts)):
            if st[i] != st[i - 1]:
                ax1.axvline(ts[i], color="gray", linestyle="--", alpha=0.45)
                ax1.text(ts[i], max(ep_rew), st[i], rotation=45, va="bottom", ha="left", fontsize=8)

    ax2 = axes[1]
    if eval_data:
        t, right, left, mn, st = _extract_xy(eval_data)
        t = t[-max_points:]
        right = right[-max_points:]
        left = left[-max_points:]
        mn = mn[-max_points:]
        st = st[-max_points:]

        ax2.plot(t, right, marker="o", color="#2ca02c", label="right_robot_dx")
        ax2.plot(t, left, marker="s", color="#d62728", label="left_robot_dx")
        ax2.plot(t, mn, marker="^", color="#9467bd", label="min_robot_dx")
        ax2.set_title("Direction progress")
        ax2.set_xlabel("timesteps")
        ax2.set_ylabel("robot_dx")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="best")

        for i in range(1, len(t)):
            if st[i] != st[i - 1]:
                ax2.axvline(t[i], color="gray", linestyle="--", alpha=0.45)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Live training progress plotter")
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--eval-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=20.0)
    parser.add_argument("--idle-timeout", type=float, default=300.0)
    parser.add_argument("--max-points", type=int, default=240)
    args = parser.parse_args()

    print(f"plotter started: {args.train_log} / {args.eval_log} -> {args.out}")
    last_mtime = 0.0
    last_draw = 0.0
    stale = 0.0

    while True:
        latest = 0.0
        if args.train_log.exists():
            latest = max(latest, args.train_log.stat().st_mtime)
        if args.eval_log.exists():
            latest = max(latest, args.eval_log.stat().st_mtime)

        if latest > last_mtime:
            did_draw = draw(args.train_log, args.eval_log, args.out, max_points=args.max_points)
            last_draw = time.time()
            if did_draw:
                print(f"saved plot: {args.out}")
            last_mtime = latest
            stale = 0.0
        else:
            stale = time.time() - last_draw if last_draw else time.time() - last_mtime

        if stale > args.idle_timeout:
            print("stale timeout reached, stop watcher")
            break

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
