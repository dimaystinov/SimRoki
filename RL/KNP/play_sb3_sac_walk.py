from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gymnasium_robot_env import GymnasiumRobotEnv


ROOT = Path(r"C:\Users\root\Documents\New project\RL\KNP")


def make_env(transport: str, dll_path: str | None, config_path: str | None, repeat_steps: int, residual_walk: bool, walk_speed_mps: float, reward_mode: str):
    def _factory():
        return GymnasiumRobotEnv(
            transport=transport,
            dll_path=dll_path,
            config_path=config_path,
            repeat_steps=repeat_steps,
            residual_walk=residual_walk,
            walk_speed_mps=walk_speed_mps,
            randomize_direction=False,
            canonicalize=True,
            reward_mode=reward_mode,
        )

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SB3 SAC policy on the robot simulator.")
    parser.add_argument("--policy", type=Path, default=ROOT / "sb3_sac_walk_policy.zip")
    parser.add_argument("--vecnorm", type=Path, default=ROOT / "sb3_sac_walk_vecnormalize.pkl")
    parser.add_argument("--transport", type=str, default="http")
    parser.add_argument("--dll-path", type=str, default=r"C:\Users\root\Documents\New project\target\release\sim_core.dll")
    parser.add_argument("--config-path", type=str, default=r"C:\Users\root\Documents\New project\robot_config.toml")
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--walk-speed-mps", type=float, default=0.45)
    parser.add_argument("--residual-walk", action="store_true")
    parser.add_argument("--direction", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reward-mode", type=str, choices=["shaped", "sim"], default="shaped")
    args = parser.parse_args()

    vec_env = DummyVecEnv(
        [make_env(args.transport, args.dll_path, args.config_path, args.repeat_steps, args.residual_walk, args.walk_speed_mps, args.reward_mode)]
    )
    if args.vecnorm.exists():
        vec_env = VecNormalize.load(str(args.vecnorm), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    model = SAC.load(str(args.policy), env=vec_env, device="cpu")

    obs = vec_env.reset()
    vec_env.env_method("set_direction", args.direction)
    total_reward = 0.0
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += float(rewards[0])
        if bool(dones[0]):
            break

    info = infos[0]
    raw = info["raw"]["observation"]
    print(f"reward={total_reward:+.3f}")
    print(f"base_x={float(raw['base_x']):+.3f}")
    print(f"direction={float(info['direction']):+.1f}")


if __name__ == "__main__":
    main()
