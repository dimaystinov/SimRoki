from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gymnasium_robot_env import GymnasiumRobotEnv
from rl_signal_utils import BALL_DX_INDEX


ROOT = Path(r"C:\Users\root\Documents\New project\RL\KNP")


def make_env(
    transport: str,
    dll_path: str | None,
    config_path: str | None,
    repeat_steps: int,
    residual_walk: bool,
    walk_speed_mps: float,
    reward_mode: str,
    action_limit_deg: float,
):
    def _factory():
        return GymnasiumRobotEnv(
            transport=transport,
            dll_path=dll_path,
            config_path=config_path,
            repeat_steps=repeat_steps,
            action_limit_deg=action_limit_deg,
            residual_walk=residual_walk,
            walk_speed_mps=walk_speed_mps,
            randomize_direction=False,
            canonicalize=True,
            reward_mode=reward_mode,
        )

    return _factory


def evaluate_direction(model: PPO, vec_env, direction: float, steps: int) -> dict[str, float]:
    vec_env.env_method("set_direction", direction)
    obs = vec_env.reset()
    raw = vec_env.get_attr("last_info")[0]["raw"]["observation"]
    start_robot_x = float(raw["base_x"])
    start_ball_x = start_robot_x + float(raw["values"][BALL_DX_INDEX])
    sign = 1.0 if direction >= 0.0 else -1.0
    total_reward = 0.0
    last_info = vec_env.get_attr("last_info")[0]
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += float(rewards[0])
        last_info = infos[0]
        if bool(dones[0]):
            break
    raw = last_info["raw"]["observation"]
    robot_x = float(raw["base_x"])
    ball_x = robot_x + float(raw["values"][BALL_DX_INDEX])
    return {
        "reward": total_reward,
        "robot_dx": (robot_x - start_robot_x) * sign,
        "ball_dx": (ball_x - start_ball_x) * sign,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SB3 PPO walking policy.")
    parser.add_argument("--policy", type=Path, default=ROOT / "sb3_ppo_walk_policy.zip")
    parser.add_argument("--vecnorm", type=Path, default=ROOT / "sb3_ppo_walk_vecnormalize.pkl")
    parser.add_argument("--transport", type=str, default="ffi")
    parser.add_argument("--dll-path", type=str, default=r"C:\Users\root\Documents\New project\target\release\sim_core.dll")
    parser.add_argument("--config-path", type=str, default=r"C:\Users\root\Documents\New project\robot_config.toml")
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--walk-speed-mps", type=float, default=0.45)
    parser.add_argument("--action-limit-deg", type=float, default=45.0)
    parser.add_argument("--residual-walk", action="store_true")
    parser.add_argument("--reward-mode", type=str, choices=["shaped", "sim"], default="sim")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    vec_env = DummyVecEnv(
        [
            make_env(
                args.transport,
                args.dll_path,
                args.config_path,
                args.repeat_steps,
                args.residual_walk,
                args.walk_speed_mps,
                args.reward_mode,
                args.action_limit_deg,
            )
        ]
    )
    if args.vecnorm.exists():
        vec_env = VecNormalize.load(str(args.vecnorm), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = PPO.load(str(args.policy), env=vec_env, device="cpu")

    results = {
        "right": evaluate_direction(model, vec_env, 1.0, args.steps),
        "left": evaluate_direction(model, vec_env, -1.0, args.steps),
    }
    results["summary"] = {
        "min_robot_dx": min(results["right"]["robot_dx"], results["left"]["robot_dx"]),
        "min_ball_dx": min(results["right"]["ball_dx"], results["left"]["ball_dx"]),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
