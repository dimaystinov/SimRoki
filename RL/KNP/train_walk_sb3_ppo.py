from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from gymnasium_robot_env import GymnasiumRobotEnv


ROOT = Path(__file__).resolve().parent
DEFAULT_DLL_PATH = str((ROOT.parent / "target" / "release" / "sim_core.dll").resolve())
DEFAULT_CONFIG_PATH = str((ROOT.parent / "robot_config.toml").resolve())


def make_env(
    transport: str,
    dll_path: str | None,
    config_path: str | None,
    repeat_steps: int,
    residual_walk: bool,
    walk_speed_mps: float,
    reward_mode: str,
):
    def _factory():
        env = GymnasiumRobotEnv(
            transport=transport,
            dll_path=dll_path,
            config_path=config_path,
            repeat_steps=repeat_steps,
            residual_walk=residual_walk,
            walk_speed_mps=walk_speed_mps,
            randomize_direction=True,
            canonicalize=True,
            reward_mode=reward_mode,
        )
        return Monitor(env)

    return _factory


class ProgressCallback(BaseCallback):
    def __init__(self, log_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_path = log_path
        self.rows: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            row = {"timesteps": int(self.num_timesteps)}
            if len(self.model.ep_info_buffer) > 0:
                rewards = [float(ep["r"]) for ep in self.model.ep_info_buffer]
                lengths = [float(ep["l"]) for ep in self.model.ep_info_buffer]
                row["ep_reward_mean"] = sum(rewards) / len(rewards)
                row["ep_len_mean"] = sum(lengths) / len(lengths)
            self.rows.append(row)
            self.log_path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 PPO on the robot simulator via Gymnasium + FFI.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--transport", type=str, default="ffi")
    parser.add_argument("--dll-path", type=str, default=DEFAULT_DLL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--walk-speed-mps", type=float, default=0.45)
    residual_group = parser.add_mutually_exclusive_group()
    residual_group.add_argument("--residual-walk", action="store_true", dest="residual_walk")
    residual_group.add_argument("--no-residual-walk", action="store_false", dest="residual_walk")
    parser.set_defaults(residual_walk=True)
    parser.add_argument("--reward-mode", type=str, choices=["shaped", "sim"], default="shaped")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=ROOT / "sb3_ppo_walk_policy.zip",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=Path,
        default=ROOT / "sb3_ppo_walk_vecnormalize.pkl",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=ROOT / "sb3_ppo_walk_training.json",
    )
    args = parser.parse_args()

    env_fns = [
        make_env(
            transport=args.transport,
            dll_path=args.dll_path,
            config_path=args.config_path,
            repeat_steps=args.repeat_steps,
            residual_walk=args.residual_walk,
            walk_speed_mps=args.walk_speed_mps,
            reward_mode=args.reward_mode,
        )
        for _ in range(max(1, args.num_envs))
    ]
    if args.num_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv(
        [
            make_env(
                transport=args.transport,
                dll_path=args.dll_path,
                config_path=args.config_path,
                repeat_steps=args.repeat_steps,
                residual_walk=args.residual_walk,
                walk_speed_mps=args.walk_speed_mps,
                reward_mode=args.reward_mode,
            )
        ]
    )
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    progress = ProgressCallback(args.log_path)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.policy_path.parent),
        log_path=str(args.policy_path.parent / "sb3_ppo_eval"),
        eval_freq=10_000,
        n_eval_episodes=4,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=args.timesteps, callback=[progress, eval_callback])
    model.save(str(args.policy_path))
    vec_env.save(str(args.vecnorm_path))
    print(f"saved model to {args.policy_path}")
    print(f"saved vecnormalize to {args.vecnorm_path}")


if __name__ == "__main__":
    main()
