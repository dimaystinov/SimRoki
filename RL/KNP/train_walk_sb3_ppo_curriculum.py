from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from gymnasium_robot_env import GymnasiumRobotEnv
from rl_signal_utils import STAGE_BALANCE, STAGE_ENDURANCE, STAGE_SPEED_TRACKING, STAGE_STAND, STAGE_WALK


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_DLL_PATH = str((PROJECT_ROOT / "target" / "release" / "sim_core.dll").resolve())
DEFAULT_CONFIG_PATH = str((PROJECT_ROOT / "robot_config.toml").resolve())


@dataclass
class StageSpec:
    name: str
    timesteps: int
    walk_speed_mps: float
    reward_mode: str
    speed_min_mps: float
    speed_max_mps: float


def make_env(
    transport: str,
    dll_path: str | None,
    config_path: str | None,
    repeat_steps: int,
    residual_walk: bool,
    walk_speed_mps: float,
    reward_mode: str,
    settle_steps: int,
    action_limit_deg: float,
    randomize_direction: bool,
    walk_config: dict[str, float] | None,
):
    def _factory():
        env = GymnasiumRobotEnv(
            transport=transport,
            dll_path=dll_path,
            config_path=config_path,
            repeat_steps=repeat_steps,
            action_limit_deg=action_limit_deg,
            residual_walk=residual_walk,
            walk_speed_mps=walk_speed_mps,
            randomize_direction=randomize_direction,
            canonicalize=True,
            reward_mode=reward_mode,
            stage=STAGE_STAND,
            include_previous_action=True,
            settle_steps=settle_steps,
        )
        if walk_config is not None:
            env.env.set_walk_config(**walk_config)
        return Monitor(env)

    return _factory


class CurriculumProgressCallback(BaseCallback):
    def __init__(self, stage_name: str, log_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stage_name = stage_name
        self.log_path = log_path
        self.rows: list[dict[str, float | int | str]] = []

    def _on_step(self) -> bool:
        if self.n_calls % 5000 == 0:
            row: dict[str, float | int | str] = {
                "stage": self.stage_name,
                "timesteps": int(self.num_timesteps),
            }
            if len(self.model.ep_info_buffer) > 0:
                rewards = [float(ep["r"]) for ep in self.model.ep_info_buffer]
                lengths = [float(ep["l"]) for ep in self.model.ep_info_buffer]
                row["ep_reward_mean"] = sum(rewards) / len(rewards)
                row["ep_len_mean"] = sum(lengths) / len(lengths)
            self.rows.append(row)
            self.log_path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        return True


class LongEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        stage_name: str,
        eval_env: VecNormalize,
        log_path: Path,
        best_policy_path: Path,
        best_vecnorm_path: Path,
        eval_steps: int,
        eval_freq: int = 10000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.stage_name = stage_name
        self.eval_env = eval_env
        self.log_path = log_path
        self.best_policy_path = best_policy_path
        self.best_vecnorm_path = best_vecnorm_path
        self.eval_steps = eval_steps
        self.eval_freq = max(1, eval_freq)
        self.best_min_robot_dx = -float("inf")
        self.rows: list[dict[str, float | int | str]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        metrics = self._evaluate()
        row: dict[str, float | int | str] = {"stage": self.stage_name, "timesteps": int(self.num_timesteps), **metrics}
        self.rows.append(row)
        self.log_path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        min_robot_dx = float(metrics["min_robot_dx"])
        if self.stage_name in {STAGE_WALK, STAGE_SPEED_TRACKING, STAGE_ENDURANCE} and min_robot_dx > self.best_min_robot_dx:
            self.best_min_robot_dx = min_robot_dx
            self.model.save(str(self.best_policy_path))
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                vec_norm.save(str(self.best_vecnorm_path))
            if self.verbose:
                print(f"[long-eval] new best min_robot_dx={min_robot_dx:.3f} saved to {self.best_policy_path}")
        return True

    def _evaluate(self) -> dict[str, float]:
        current_vecnorm = self.model.get_vec_normalize_env()
        if current_vecnorm is not None:
            self.eval_env.obs_rms = current_vecnorm.obs_rms.copy()
        right_dx = self._evaluate_direction(1.0)
        left_dx = self._evaluate_direction(-1.0)
        return {
            "right_robot_dx": float(right_dx),
            "left_robot_dx": float(left_dx),
            "min_robot_dx": float(min(right_dx, left_dx)),
        }

    def _evaluate_direction(self, direction: float) -> float:
        self.eval_env.env_method("set_direction", direction)
        obs = self.eval_env.reset()
        raw = self.eval_env.get_attr("last_info")[0]["raw"]["observation"]
        start_robot_x = float(raw["base_x"])
        sign = 1.0 if direction >= 0.0 else -1.0
        last_info = self.eval_env.get_attr("last_info")[0]
        for _ in range(self.eval_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.eval_env.step(action)
            last_info = infos[0]
            if bool(dones[0]):
                break
        raw = last_info["raw"]["observation"]
        robot_x = float(raw["base_x"])
        return (robot_x - start_robot_x) * sign


def set_stage(vec_env, stage: StageSpec) -> None:
    vec_env.env_method("set_stage", stage.name)
    vec_env.env_method("set_walk_speed", stage.walk_speed_mps)
    vec_env.env_method("set_speed_range", stage.speed_min_mps, stage.speed_max_mps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train staged PPO curriculum for stable 2D biped walking.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--action-limit-deg", type=float, default=45.0)
    parser.add_argument("--transport", type=str, default="ffi")
    parser.add_argument("--dll-path", type=str, default=DEFAULT_DLL_PATH)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    residual_group = parser.add_mutually_exclusive_group()
    residual_group.add_argument("--residual-walk", action="store_true", dest="residual_walk")
    residual_group.add_argument("--no-residual-walk", action="store_false", dest="residual_walk")
    parser.set_defaults(residual_walk=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stand-steps", type=int, default=30000)
    parser.add_argument("--balance-steps", type=int, default=40000)
    parser.add_argument("--walk-steps", type=int, default=60000)
    parser.add_argument("--speed-steps", type=int, default=80000)
    parser.add_argument("--endurance-steps", type=int, default=100000)
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument("--randomize-direction", action="store_true", dest="randomize_direction")
    direction_group.add_argument("--fixed-direction", action="store_false", dest="randomize_direction")
    parser.set_defaults(randomize_direction=False)
    parser.add_argument("--settle-steps", type=int, default=12)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-steps", type=int, default=600)
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=ROOT / "sb3_ppo_curriculum_policy.zip",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=Path,
        default=ROOT / "sb3_ppo_curriculum_vecnormalize.pkl",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=ROOT / "sb3_ppo_curriculum_training.json",
    )
    parser.add_argument(
        "--best-policy-path",
        type=Path,
        default=ROOT / "sb3_ppo_curriculum_best_long.zip",
    )
    parser.add_argument(
        "--best-vecnorm-path",
        type=Path,
        default=ROOT / "sb3_ppo_curriculum_best_long_vecnormalize.pkl",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--resume-vecnorm-from",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--walk-config-path",
        type=Path,
        default=ROOT / "kmp_best_config.json",
        help="Optional JSON file with walk config values to seed the built-in gait.",
    )
    args = parser.parse_args()
    walk_config: dict[str, float] | None = None
    if args.walk_config_path.exists():
        walk_config = json.loads(args.walk_config_path.read_text(encoding="utf-8"))

    env_fns = [
        make_env(
            transport=args.transport,
            dll_path=args.dll_path,
            config_path=args.config_path,
            repeat_steps=args.repeat_steps,
            action_limit_deg=args.action_limit_deg,
            residual_walk=args.residual_walk,
            walk_speed_mps=0.45,
            reward_mode="shaped",
            randomize_direction=args.randomize_direction,
            settle_steps=args.settle_steps,
            walk_config=walk_config,
        )
        for _ in range(max(1, args.num_envs))
    ]
    if args.num_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    if args.resume_vecnorm_from and args.resume_vecnorm_from.exists():
        vec_env = VecNormalize.load(str(args.resume_vecnorm_from), vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv(
        [
            make_env(
                transport=args.transport,
                dll_path=args.dll_path,
                config_path=args.config_path,
                repeat_steps=args.repeat_steps,
                action_limit_deg=args.action_limit_deg,
                residual_walk=args.residual_walk,
                walk_speed_mps=0.45,
                reward_mode="shaped",
                randomize_direction=False,
                settle_steps=args.settle_steps,
                walk_config=walk_config,
            )
        ]
    )
    if args.resume_vecnorm_from and args.resume_vecnorm_from.exists():
        eval_env = VecNormalize.load(str(args.resume_vecnorm_from), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    if args.resume_from and args.resume_from.exists():
        model = PPO.load(str(args.resume_from), env=vec_env, device="cpu")
        model.lr_schedule = lambda _: args.learning_rate
        model.ent_coef = args.ent_coef
        policy_kwargs = None
    else:
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
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
            verbose=1,
            seed=args.seed,
            device="cpu",
        )
        policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))

    history: list[dict[str, int | str]] = []
    stages = [
        StageSpec(STAGE_STAND, args.stand_steps, 0.0, "shaped", 0.0, 0.0),
        StageSpec(STAGE_BALANCE, args.balance_steps, 0.05, "shaped", 0.03, 0.08),
        StageSpec(STAGE_WALK, args.walk_steps, 0.16, "shaped", 0.12, 0.18),
        StageSpec(STAGE_SPEED_TRACKING, args.speed_steps, 0.22, "shaped", 0.14, 0.26),
        StageSpec(STAGE_ENDURANCE, args.endurance_steps, 0.25, "shaped", 0.18, 0.28),
    ]
    long_eval_callback = LongEvalCallback(
        stage_name=STAGE_STAND,
        eval_env=eval_env,
        log_path=args.log_path.with_name(f"{args.log_path.stem}_eval{args.log_path.suffix}"),
        best_policy_path=args.best_policy_path,
        best_vecnorm_path=args.best_vecnorm_path,
        eval_steps=args.eval_steps,
        eval_freq=args.eval_freq,
        verbose=1,
    )

    total_seen = 0
    for stage in stages:
        set_stage(vec_env, stage)
        set_stage(eval_env, stage)
        progress_callback = CurriculumProgressCallback(stage.name, args.log_path)
        long_eval_callback.stage_name = stage.name
        model.learn(total_timesteps=stage.timesteps, callback=[progress_callback, long_eval_callback], reset_num_timesteps=False)
        total_seen += stage.timesteps
        history.append({"stage": stage.name, "timesteps": stage.timesteps, "total_seen": total_seen})
        args.log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    model.save(str(args.policy_path))
    vec_env.save(str(args.vecnorm_path))
    print(f"saved model to {args.policy_path}")
    print(f"saved vecnormalize to {args.vecnorm_path}")


if __name__ == "__main__":
    main()
