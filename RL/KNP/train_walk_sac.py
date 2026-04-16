from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from desktop_rl_env import DesktopRobotEnv
from rl_signal_utils import (
    BALL_DX_INDEX,
    TASK_DRIBBLE,
    TASK_WALK,
    canonicalize_observation,
    decanonicalize_action,
    shaped_reward,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ROOT = Path(r"C:\Users\root\Documents\New project\RL\KNP")
CURRICULUM_SPEEDS = (0.20, 0.35, 0.50, 0.65)


def build_mlp(input_size: int, hidden_size: int, hidden_layers: int, output_size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = input_size
    for _ in range(max(1, hidden_layers)):
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.SiLU())
        in_features = hidden_size
    layers.append(nn.Linear(in_features, output_size))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, action_scale_deg: float, hidden_layers: int = 2) -> None:
        super().__init__()
        self.action_scale_deg = action_scale_deg
        self.hidden_layers = hidden_layers
        self.backbone = build_mlp(obs_size, hidden_size, hidden_layers, hidden_size)
        self.mean = nn.Linear(hidden_size, 4)
        self.log_std = nn.Linear(hidden_size, 4)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale_deg
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.action_scale_deg


class Critic(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, hidden_layers: int = 2) -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.q = build_mlp(obs_size + 4, hidden_size, hidden_layers, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, batch_size)

    def __len__(self) -> int:
        return len(self.data)

def sample_target_speed(max_speed_mps: float) -> float:
    speed = random.choice(CURRICULUM_SPEEDS)
    return float(min(speed, max_speed_mps))


def evaluate_policy(
    env: DesktopRobotEnv,
    actor: Actor,
    direction: float,
    horizon: int,
    device: torch.device,
    residual_walk: bool,
    walk_speed_mps: float,
    task: str,
) -> dict[str, float]:
    result = env.reset_with_direction(direction)
    if residual_walk:
        env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=walk_speed_mps)
    start_robot_x = float(result.raw["observation"]["base_x"])
    start_ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + start_robot_x
    total_reward = 0.0
    sign = 1.0 if direction >= 0.0 else -1.0
    prev_result = result
    for _ in range(horizon):
        obs = torch.tensor(canonicalize_observation(result.observation), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            canonical_action = actor.act_deterministic(obs).squeeze(0).cpu().numpy()
        action = decanonicalize_action(canonical_action, direction)
        result = env.step(action, direction=direction, residual=residual_walk)
        total_reward += shaped_reward(
            prev_result.observation,
            float(prev_result.raw["observation"]["base_x"]),
            result.observation,
            float(result.raw["observation"]["base_x"]),
            action,
            direction,
            walk_speed_mps,
            task,
            result.done,
        )
        prev_result = result
        if result.done or result.truncated:
            break
    robot_x = float(result.raw["observation"]["base_x"])
    ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + robot_x
    return {
        "reward": total_reward,
        "robot_dx": (robot_x - start_robot_x) * sign,
        "ball_dx": (ball_x - start_ball_x) * sign,
    }


def train_sac(
    steps: int,
    repeat_steps: int,
    hidden_size: int,
    hidden_layers: int,
    action_scale_deg: float,
    learning_rate: float,
    log_path: Path,
    policy_path: Path,
    transport: str,
    dll_path: str | None,
    config_path: str | None,
    residual_walk: bool,
    walk_speed_mps: float,
    num_envs: int,
    resume_from: Path | None,
    task: str,
) -> None:
    device = torch.device("cpu")
    baseline_walk_config: dict[str, float] = {}
    if residual_walk:
        baseline_path = ROOT / "kmp_best_config.json"
        if baseline_path.exists():
            baseline_walk_config = json.loads(baseline_path.read_text(encoding="utf-8"))
    envs = [
        DesktopRobotEnv(
            repeat_steps=repeat_steps,
            transport=transport,
            dll_path=dll_path,
            config_path=config_path,
        )
        for _ in range(max(1, num_envs))
    ]
    for env in envs:
        if baseline_walk_config:
            env.set_walk_config(**baseline_walk_config)
    first_speed = sample_target_speed(baseline_walk_config.get("max_speed_mps", walk_speed_mps if walk_speed_mps > 0 else 0.65))
    first = envs[0].reset_with_direction(1.0)
    if residual_walk:
        envs[0].set_walk_direction_speed(direction=1.0, enabled=True, speed_mps=first_speed)
    obs_size = first.observation.shape[0]

    actor = Actor(obs_size, hidden_size, action_scale_deg, hidden_layers=hidden_layers).to(device)
    critic1 = Critic(obs_size, hidden_size, hidden_layers=hidden_layers).to(device)
    critic2 = Critic(obs_size, hidden_size, hidden_layers=hidden_layers).to(device)
    target1 = Critic(obs_size, hidden_size, hidden_layers=hidden_layers).to(device)
    target2 = Critic(obs_size, hidden_size, hidden_layers=hidden_layers).to(device)
    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=learning_rate)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=learning_rate)

    alpha = 0.15
    gamma = 0.99
    tau = 0.01
    batch_size = 64
    warmup_steps = 300
    update_after = 300
    update_every = 2

    replay = ReplayBuffer(50_000)
    history: list[dict] = []
    best_score = -float("inf")

    if resume_from is not None and resume_from.exists():
        checkpoint = torch.load(resume_from, map_location=device)
        if "actor_state_dict" in checkpoint:
            actor.load_state_dict(checkpoint["actor_state_dict"])
        if "critic1_state_dict" in checkpoint:
            critic1.load_state_dict(checkpoint["critic1_state_dict"])
        if "critic2_state_dict" in checkpoint:
            critic2.load_state_dict(checkpoint["critic2_state_dict"])
        if "target1_state_dict" in checkpoint:
            target1.load_state_dict(checkpoint["target1_state_dict"])
        else:
            target1.load_state_dict(critic1.state_dict())
        if "target2_state_dict" in checkpoint:
            target2.load_state_dict(checkpoint["target2_state_dict"])
        else:
            target2.load_state_dict(critic2.state_dict())
        if "actor_opt_state_dict" in checkpoint:
            actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
        if "critic1_opt_state_dict" in checkpoint:
            critic1_opt.load_state_dict(checkpoint["critic1_opt_state_dict"])
        if "critic2_opt_state_dict" in checkpoint:
            critic2_opt.load_state_dict(checkpoint["critic2_opt_state_dict"])
        best_score = float(checkpoint.get("best_score", best_score))

    directions: list[float] = []
    target_speeds: list[float] = []
    results = []
    max_speed_limit = baseline_walk_config.get("max_speed_mps", walk_speed_mps if walk_speed_mps > 0 else 0.65)
    for idx, env in enumerate(envs):
        direction = 1.0 if idx % 2 == 0 else -1.0
        result = env.reset_with_direction(direction)
        target_speed = sample_target_speed(max_speed_limit)
        if residual_walk:
            env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=target_speed)
        directions.append(direction)
        target_speeds.append(target_speed)
        results.append(result)

    for step in range(1, steps + 1):
        for env_idx, env in enumerate(envs):
            direction = directions[env_idx]
            target_speed = target_speeds[env_idx]
            result = results[env_idx]
            canonical_obs = canonicalize_observation(result.observation)

            if step <= warmup_steps:
                canonical_action = np.random.uniform(-action_scale_deg, action_scale_deg, size=(4,)).astype(np.float32)
            else:
                obs_t = torch.tensor(canonical_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    canonical_action, _ = actor.sample(obs_t)
                canonical_action = canonical_action.squeeze(0).cpu().numpy()

            action = decanonicalize_action(canonical_action, direction)
            next_result = env.step(action, direction=direction, residual=residual_walk)
            done = float(next_result.done or next_result.truncated)
            reward = shaped_reward(
                result.observation,
                float(result.raw["observation"]["base_x"]),
                next_result.observation,
                float(next_result.raw["observation"]["base_x"]),
                action,
                direction,
                target_speed,
                task,
                bool(done),
            )
            replay.add(
                Transition(
                    obs=canonical_obs.copy(),
                    action=np.asarray(canonical_action, dtype=np.float32).copy(),
                    reward=reward,
                    next_obs=canonicalize_observation(next_result.observation),
                    done=done,
                )
            )
            results[env_idx] = next_result

            if done:
                direction = -direction
                next_result = env.reset_with_direction(direction)
                target_speed = sample_target_speed(max_speed_limit)
                if residual_walk:
                    env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=target_speed)
                directions[env_idx] = direction
                target_speeds[env_idx] = target_speed
                results[env_idx] = next_result

        if step >= update_after and len(replay) >= batch_size and step % update_every == 0:
            for _ in range(update_every):
                batch = replay.sample(batch_size)
                obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor(np.stack([t.action for t in batch]), dtype=torch.float32, device=device)
                rewards = torch.tensor([[t.reward] for t in batch], dtype=torch.float32, device=device)
                next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device)
                dones = torch.tensor([[t.done] for t in batch], dtype=torch.float32, device=device)

                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(next_obs)
                    target_q = torch.min(target1(next_obs, next_action), target2(next_obs, next_action)) - alpha * next_log_prob
                    target_value = rewards + gamma * (1.0 - dones) * target_q

                q1 = critic1(obs, actions)
                q2 = critic2(obs, actions)
                critic1_loss = F.mse_loss(q1, target_value)
                critic2_loss = F.mse_loss(q2, target_value)
                critic1_opt.zero_grad()
                critic1_loss.backward()
                critic1_opt.step()
                critic2_opt.zero_grad()
                critic2_loss.backward()
                critic2_opt.step()

                new_action, log_prob = actor.sample(obs)
                q_new = torch.min(critic1(obs, new_action), critic2(obs, new_action))
                actor_loss = (alpha * log_prob - q_new).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                for target_param, param in zip(target1.parameters(), critic1.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)
                for target_param, param in zip(target2.parameters(), critic2.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)

        if step % 250 == 0 or step == 1:
            evals = []
            for eval_speed in (0.20, 0.35, 0.50, min(0.65, max_speed_limit)):
                evals.append(
                    {
                        "speed": eval_speed,
                        "right": evaluate_policy(envs[0], actor, 1.0, 600, device, residual_walk, eval_speed, task),
                        "left": evaluate_policy(envs[0], actor, -1.0, 600, device, residual_walk, eval_speed, task),
                    }
                )
            min_robot_dx = min(min(item["right"]["robot_dx"], item["left"]["robot_dx"]) for item in evals)
            mean_robot_dx = sum(item["right"]["robot_dx"] + item["left"]["robot_dx"] for item in evals) / (2.0 * len(evals))
            min_ball_dx = min(min(item["right"]["ball_dx"], item["left"]["ball_dx"]) for item in evals)
            if task == TASK_DRIBBLE:
                score = 3.0 * min_robot_dx + 2.0 * mean_robot_dx + 1.0 * min_ball_dx
            else:
                score = 4.0 * min_robot_dx + 1.5 * mean_robot_dx
            history.append(
                {
                    "step": step,
                    "evals": evals,
                    "score": score,
                    "buffer_size": len(replay),
                    "num_envs": len(envs),
                    "task": task,
                }
            )
            print(
                f"step={step:05d} "
                f"min_robot_dx={min_robot_dx:+.3f} "
                f"mean_robot_dx={mean_robot_dx:+.3f} "
                f"min_ball_dx={min_ball_dx:+.3f} "
                f"score={score:+.3f}"
            )
            if score > best_score:
                best_score = score
                torch.save(
                    {
                        "actor_state_dict": actor.state_dict(),
                        "critic1_state_dict": critic1.state_dict(),
                        "critic2_state_dict": critic2.state_dict(),
                        "target1_state_dict": target1.state_dict(),
                        "target2_state_dict": target2.state_dict(),
                        "actor_opt_state_dict": actor_opt.state_dict(),
                        "critic1_opt_state_dict": critic1_opt.state_dict(),
                        "critic2_opt_state_dict": critic2_opt.state_dict(),
                        "obs_size": obs_size,
                        "hidden_size": hidden_size,
                        "action_scale_deg": action_scale_deg,
                        "hidden_layers": hidden_layers,
                        "best_score": score,
                        "residual_walk": residual_walk,
                        "walk_speed_mps": walk_speed_mps,
                        "task": task,
                        "evals": evals,
                    },
                    str(policy_path),
                )
            log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved policy to {policy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible SAC training against the desktop robot simulator.")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--action-scale-deg", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--transport", type=str, default="auto")
    parser.add_argument("--dll-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default="robot_config.toml")
    parser.add_argument("--task", type=str, choices=[TASK_WALK, TASK_DRIBBLE], default=TASK_WALK)
    parser.add_argument("--residual-walk", action="store_true")
    parser.add_argument("--walk-speed-mps", type=float, default=0.45)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=Path(r"C:\Users\root\Documents\New project\RL\KNP\sac_walk_policy.pt"),
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path(r"C:\Users\root\Documents\New project\RL\KNP\sac_walk_policy.pt"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path(r"C:\Users\root\Documents\New project\RL\KNP\sac_walk_training.json"),
    )
    args = parser.parse_args()
    train_sac(
        steps=args.steps,
        repeat_steps=args.repeat_steps,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        action_scale_deg=args.action_scale_deg,
        learning_rate=args.lr,
        log_path=args.log_path,
        policy_path=args.policy_path,
        transport=args.transport,
        dll_path=args.dll_path,
        config_path=args.config_path,
        residual_walk=args.residual_walk,
        walk_speed_mps=args.walk_speed_mps,
        num_envs=args.num_envs,
        resume_from=args.resume_from,
        task=args.task,
    )


if __name__ == "__main__":
    main()
