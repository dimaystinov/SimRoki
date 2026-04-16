from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from desktop_rl_env import DesktopRobotEnv
from train_walk_sac import (
    TASK_WALK,
    Actor,
    canonicalize_observation,
    decanonicalize_action,
)

ROOT = Path(r"C:\Users\root\Documents\New project\RL\KNP")
BALL_DX_INDEX = 14


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the best saved SAC walking policy.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(r"C:\Users\root\Documents\New project\RL\KNP\sac_walk_policy.pt"),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--direction", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.policy, map_location="cpu")
    actor = Actor(
        obs_size=int(checkpoint["obs_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        action_scale_deg=float(checkpoint["action_scale_deg"]),
        hidden_layers=int(checkpoint.get("hidden_layers", 2)),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    env = DesktopRobotEnv(repeat_steps=args.repeat_steps)
    residual_walk = bool(checkpoint.get("residual_walk", False))
    walk_speed_mps = float(checkpoint.get("walk_speed_mps", 0.45))
    _task = str(checkpoint.get("task", TASK_WALK))
    baseline_path = ROOT / "kmp_best_config.json"
    if residual_walk and baseline_path.exists():
        env.set_walk_config(**json.loads(baseline_path.read_text(encoding="utf-8")))
    result = env.reset_with_direction(args.direction)
    if residual_walk:
        env.set_walk_direction_speed(direction=args.direction, enabled=True, speed_mps=walk_speed_mps)
    start_robot_x = float(result.raw["observation"]["base_x"])
    start_ball_x = start_robot_x + float(result.raw["observation"]["values"][BALL_DX_INDEX])
    sign = 1.0 if args.direction >= 0.0 else -1.0
    total_reward = 0.0

    for _ in range(args.steps):
        obs = torch.tensor(canonicalize_observation(result.observation), dtype=torch.float32).flatten()
        expected_size = int(checkpoint["obs_size"])
        if obs.numel() != expected_size:
            fitted = torch.zeros(expected_size, dtype=torch.float32)
            copy_count = min(expected_size, obs.numel())
            fitted[:copy_count] = obs[:copy_count]
            obs = fitted
        obs = obs.unsqueeze(0)
        with torch.no_grad():
            canonical_action = actor.act_deterministic(obs).squeeze(0).numpy()
        action = decanonicalize_action(canonical_action, args.direction)
        result = env.step(action, direction=args.direction, residual=residual_walk)
        total_reward += result.reward
        if result.done or result.truncated:
            break

    robot_x = float(result.raw["observation"]["base_x"])
    ball_x = robot_x + float(result.raw["observation"]["values"][BALL_DX_INDEX])
    print(f"reward={total_reward:+.3f}")
    print(f"robot_dx={(robot_x - start_robot_x) * sign:+.3f} m")
    print(f"ball_dx_world={(ball_x - start_ball_x) * sign:+.3f} m")


if __name__ == "__main__":
    main()
