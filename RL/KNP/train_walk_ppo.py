from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from desktop_rl_env import DesktopRobotEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, action_scale_deg: float) -> None:
        super().__init__()
        self.action_scale_deg = action_scale_deg
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_size, 4)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((4,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = torch.tanh(self.actor_mean(hidden)) * self.action_scale_deg
        value = self.critic(hidden).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor


def collect_rollout(env: DesktopRobotEnv, model: ActorCritic, horizon: int, device: torch.device) -> RolloutBatch:
    obs_list = []
    actions_list = []
    log_prob_list = []
    rewards_list = []
    dones_list = []
    values_list = []

    current = env.reset_with_direction(1.0)
    obs = torch.tensor(current.observation, dtype=torch.float32, device=device)
    direction = 1.0

    for _ in range(horizon):
        with torch.no_grad():
            mean, std, value = model(obs.unsqueeze(0))
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        result = env.step(action.squeeze(0).cpu().numpy(), direction=direction)

        obs_list.append(obs)
        actions_list.append(action.squeeze(0))
        log_prob_list.append(log_prob.squeeze(0))
        rewards_list.append(torch.tensor(result.reward, dtype=torch.float32, device=device))
        dones_list.append(torch.tensor(float(result.done or result.truncated), dtype=torch.float32, device=device))
        values_list.append(value.squeeze(0))

        if result.done or result.truncated:
            direction = -direction
            reset = env.reset_with_direction(direction)
            obs = torch.tensor(reset.observation, dtype=torch.float32, device=device)
        else:
            obs = torch.tensor(result.observation, dtype=torch.float32, device=device)

    with torch.no_grad():
        _, _, next_value = model(obs.unsqueeze(0))

    return RolloutBatch(
        obs=torch.stack(obs_list),
        actions=torch.stack(actions_list),
        log_probs=torch.stack(log_prob_list),
        rewards=torch.stack(rewards_list),
        dones=torch.stack(dones_list),
        values=torch.stack(values_list),
        next_value=next_value.squeeze(0),
    )


def compute_gae(batch: RolloutBatch, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(batch.rewards)
    last_advantage = torch.zeros((), device=batch.rewards.device)
    next_value = batch.next_value
    for t in reversed(range(batch.rewards.shape[0])):
        mask = 1.0 - batch.dones[t]
        delta = batch.rewards[t] + gamma * next_value * mask - batch.values[t]
        last_advantage = delta + gamma * lam * mask * last_advantage
        advantages[t] = last_advantage
        next_value = batch.values[t]
    returns = advantages + batch.values
    return advantages, returns


def evaluate_policy(
    env: DesktopRobotEnv,
    model: ActorCritic,
    direction: float,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    result = env.reset_with_direction(direction)
    start_robot_x = float(result.raw["observation"]["base_x"])
    start_ball_x = start_robot_x + float(result.raw["observation"]["values"][10])
    total_reward = 0.0
    direction_sign = 1.0 if direction >= 0.0 else -1.0

    for _ in range(steps):
        obs = torch.tensor(result.observation, dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, _, _ = model(obs.unsqueeze(0))
        result = env.step(mean.squeeze(0).cpu().numpy(), direction=direction)
        total_reward += result.reward
        if result.done or result.truncated:
            break

    robot_x = float(result.raw["observation"]["base_x"])
    ball_x = robot_x + float(result.raw["observation"]["values"][10])
    return {
        "reward": total_reward,
        "robot_dx": (robot_x - start_robot_x) * direction_sign,
        "ball_dx": (ball_x - start_ball_x) * direction_sign,
    }


def train_visible(
    updates: int,
    horizon: int,
    repeat_steps: int,
    hidden_size: int,
    action_scale_deg: float,
    learning_rate: float,
    log_path: Path,
    transport: str,
    dll_path: str | None,
    config_path: str | None,
) -> None:
    device = torch.device("cpu")
    env = DesktopRobotEnv(
        repeat_steps=repeat_steps,
        transport=transport,
        dll_path=dll_path,
        config_path=config_path,
    )
    initial = env.reset_with_direction(1.0)
    model = ActorCritic(
        obs_size=initial.observation.shape[0],
        hidden_size=hidden_size,
        action_scale_deg=action_scale_deg,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.99
    lam = 0.95
    clip = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    minibatch_size = min(64, horizon)
    epochs = 6
    best_score = -float("inf")
    history: list[dict] = []

    print(f"obs_size={initial.observation.shape[0]} action_size=4")
    for update in range(1, updates + 1):
        batch = collect_rollout(env, model, horizon, device)
        advantages, returns = compute_gae(batch, gamma=gamma, lam=lam)
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        indices = np.arange(horizon)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, horizon, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                mb_obs = batch.obs[mb_idx]
                mb_actions = batch.actions[mb_idx]
                mb_old_log_probs = batch.log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                mean, std, values = model(mb_obs)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratios = (log_probs - mb_old_log_probs).exp()
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())

        avg_reward = float(batch.rewards.mean().item())
        avg_value = float(batch.values.mean().item())
        eval_right = evaluate_policy(env, model, 1.0, horizon, device)
        eval_left = evaluate_policy(env, model, -1.0, horizon, device)
        eval_score = min(eval_right["robot_dx"], eval_left["robot_dx"]) + 1.5 * min(eval_right["ball_dx"], eval_left["ball_dx"])
        history.append(
            {
                "update": update,
                "avg_reward": avg_reward,
                "avg_value": avg_value,
                "eval_right": eval_right,
                "eval_left": eval_left,
                "eval_score": eval_score,
            }
        )
        if eval_score > best_score:
            best_score = eval_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_size": initial.observation.shape[0],
                    "hidden_size": hidden_size,
                    "action_scale_deg": action_scale_deg,
                    "observation_names": env.observation_names,
                    "action_names": env.action_names,
                    "best_score": eval_score,
                    "eval_right": eval_right,
                    "eval_left": eval_left,
                },
                "C:\\Users\\root\\Documents\\New project\\RL\\KNP\\ppo_walk_policy.pt",
            )
        print(
            f"update={update:04d} avg_reward={avg_reward:+.4f} avg_value={avg_value:+.4f} "
            f"policy_loss={total_policy_loss:.4f} value_loss={total_value_loss:.4f} entropy={total_entropy:.4f} "
            f"eval_right={eval_right['robot_dx']:+.3f}/{eval_right['ball_dx']:+.3f} "
            f"eval_left={eval_left['robot_dx']:+.3f}/{eval_left['ball_dx']:+.3f}"
        )
        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("saved policy to RL/KNP/ppo_walk_policy.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible PPO training against the desktop robot simulator.")
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--action-scale-deg", type=float, default=35.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--transport", type=str, default="auto")
    parser.add_argument("--dll-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default="robot_config.toml")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\ppo_walk_training.json"),
    )
    args = parser.parse_args()
    train_visible(
        updates=args.updates,
        horizon=args.horizon,
        repeat_steps=args.repeat_steps,
        hidden_size=args.hidden_size,
        action_scale_deg=args.action_scale_deg,
        learning_rate=args.lr,
        log_path=args.log_path,
        transport=args.transport,
        dll_path=args.dll_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
