from __future__ import annotations

import argparse
from pathlib import Path

import torch

from desktop_rl_env import DesktopRobotEnv
from train_walk_ppo import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the best saved PPO walking policy.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\ppo_walk_policy.pt"),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--direction", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.policy, map_location="cpu")
    model = ActorCritic(
        obs_size=int(checkpoint["obs_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        action_scale_deg=float(checkpoint["action_scale_deg"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    env = DesktopRobotEnv(repeat_steps=args.repeat_steps)
    result = env.reset_with_direction(args.direction)
    start_robot_x = float(result.raw["observation"]["base_x"])
    start_ball_x = start_robot_x + float(result.raw["observation"]["values"][10])
    direction_sign = 1.0 if args.direction >= 0.0 else -1.0
    total_reward = 0.0

    for _ in range(args.steps):
        obs = torch.tensor(result.observation, dtype=torch.float32)
        with torch.no_grad():
            mean, _, _ = model(obs.unsqueeze(0))
        result = env.step(mean.squeeze(0).numpy(), direction=args.direction)
        total_reward += result.reward
        if result.done or result.truncated:
            break

    robot_x = float(result.raw["observation"]["base_x"])
    ball_x = robot_x + float(result.raw["observation"]["values"][10])
    print(f"reward={total_reward:+.3f}")
    print(f"robot_dx={(robot_x - start_robot_x) * direction_sign:+.3f} m")
    print(f"ball_dx_world={(ball_x - start_ball_x) * direction_sign:+.3f} m")


if __name__ == "__main__":
    main()
