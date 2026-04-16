from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from desktop_rl_env import DesktopRobotEnv
from play_best_knp_walk_kick import ReplayKnpStylePolicy, decanonicalize_action
from train_walk_ppo import ActorCritic
from train_walk_sac import (
    TASK_WALK,
    Actor as SacActor,
    canonicalize_observation as sac_canonicalize_observation,
    decanonicalize_action as sac_decanonicalize_action,
)


ROOT = Path(r"C:\Users\root\Documents\New project\RL\KNP")
BALL_DX_INDEX = 14


def fit_observation(obs, expected_size: int) -> torch.Tensor:
    tensor = torch.tensor(obs, dtype=torch.float32).flatten()
    if tensor.numel() == expected_size:
        return tensor.unsqueeze(0)
    fitted = torch.zeros(expected_size, dtype=torch.float32)
    copy_count = min(expected_size, tensor.numel())
    fitted[:copy_count] = tensor[:copy_count]
    return fitted.unsqueeze(0)


def evaluate_kmp(
    direction: float,
    duration_s: float = 4.0,
    transport: str = "auto",
    dll_path: str | None = None,
    config_path: str | None = "robot_config.toml",
) -> dict[str, float]:
    env = DesktopRobotEnv(transport=transport, dll_path=dll_path, config_path=config_path)
    kmp_config_path = ROOT / "kmp_best_config.json"
    if kmp_config_path.exists():
        env.set_walk_config(**json.loads(kmp_config_path.read_text(encoding="utf-8")))
    env.reset_with_direction(direction)
    start = env.state()
    start_robot_x = float(start["base"]["x"])
    start_ball_x = float(start["ball"]["x"])
    sign = 1.0 if direction >= 0.0 else -1.0
    env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=0.35)
    env.advance(duration_s)
    finish = env.state()
    env.set_walk_direction_speed(direction=direction, enabled=False, speed_mps=0.35)
    return {
        "robot_dx": (float(finish["base"]["x"]) - start_robot_x) * sign,
        "ball_dx": (float(finish["ball"]["x"]) - start_ball_x) * sign,
        "torso_angle": abs(float(finish["base"]["angle"])),
        "torso_height": float(finish["base"]["y"]),
    }


def evaluate_knp(
    direction: float,
    steps: int = 240,
    repeat_steps: int = 2,
    transport: str = "auto",
    dll_path: str | None = None,
    config_path: str | None = "robot_config.toml",
) -> dict[str, float]:
    env = DesktopRobotEnv(repeat_steps=repeat_steps, transport=transport, dll_path=dll_path, config_path=config_path)
    agent = ReplayKnpStylePolicy(ROOT / "knp_walk_kick_best.npz")
    kmp_config_path = ROOT / "kmp_best_config.json"
    if agent.residual_walk and kmp_config_path.exists():
        env.set_walk_config(**json.loads(kmp_config_path.read_text(encoding="utf-8")))
    result = env.reset_with_direction(direction)
    if agent.residual_walk:
        env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=agent.walk_speed_mps)
    start_robot_x = float(result.raw["observation"]["base_x"])
    start_ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + start_robot_x
    sign = 1.0 if direction >= 0.0 else -1.0
    total_reward = 0.0
    agent.reset()
    for _ in range(steps):
        action = decanonicalize_action(agent.act(result.observation), direction)
        result = env.step(
            action,
            direction=direction,
            residual=agent.residual_walk,
        )
        total_reward += result.reward
        if result.done or result.truncated:
            break
    robot_x = float(result.raw["observation"]["base_x"])
    ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + robot_x
    return {"reward": total_reward, "robot_dx": (robot_x - start_robot_x) * sign, "ball_dx": (ball_x - start_ball_x) * sign}


def evaluate_ppo(
    direction: float,
    steps: int = 240,
    repeat_steps: int = 2,
    transport: str = "auto",
    dll_path: str | None = None,
    config_path: str | None = "robot_config.toml",
) -> dict[str, float]:
    checkpoint = torch.load(ROOT / "ppo_walk_policy.pt", map_location="cpu")
    model = ActorCritic(
        obs_size=int(checkpoint["obs_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        action_scale_deg=float(checkpoint["action_scale_deg"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    env = DesktopRobotEnv(repeat_steps=repeat_steps, transport=transport, dll_path=dll_path, config_path=config_path)
    result = env.reset_with_direction(direction)
    start_robot_x = float(result.raw["observation"]["base_x"])
    start_ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + start_robot_x
    sign = 1.0 if direction >= 0.0 else -1.0
    total_reward = 0.0
    for _ in range(steps):
        obs = fit_observation(result.observation, int(checkpoint["obs_size"]))
        with torch.no_grad():
            action, _, _ = model(obs)
        result = env.step(action.squeeze(0).numpy(), direction=direction)
        total_reward += result.reward
        if result.done or result.truncated:
            break
    robot_x = float(result.raw["observation"]["base_x"])
    ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + robot_x
    return {"reward": total_reward, "robot_dx": (robot_x - start_robot_x) * sign, "ball_dx": (ball_x - start_ball_x) * sign}


def evaluate_sac(
    direction: float,
    steps: int = 240,
    repeat_steps: int = 2,
    transport: str = "auto",
    dll_path: str | None = None,
    config_path: str | None = "robot_config.toml",
    policy_path: Path | None = None,
) -> dict[str, float]:
    checkpoint = torch.load(policy_path or (ROOT / "sac_walk_policy.pt"), map_location="cpu")
    actor = SacActor(
        obs_size=int(checkpoint["obs_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        action_scale_deg=float(checkpoint["action_scale_deg"]),
        hidden_layers=int(checkpoint.get("hidden_layers", 2)),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    env = DesktopRobotEnv(repeat_steps=repeat_steps, transport=transport, dll_path=dll_path, config_path=config_path)
    residual_walk = bool(checkpoint.get("residual_walk", False))
    walk_speed_mps = float(checkpoint.get("walk_speed_mps", 0.45))
    task = str(checkpoint.get("task", TASK_WALK))
    baseline_path = ROOT / "kmp_best_config.json"
    if residual_walk and baseline_path.exists():
        env.set_walk_config(**json.loads(baseline_path.read_text(encoding="utf-8")))
    result = env.reset_with_direction(direction)
    if residual_walk:
        env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=walk_speed_mps)
    start_robot_x = float(result.raw["observation"]["base_x"])
    start_ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + start_robot_x
    sign = 1.0 if direction >= 0.0 else -1.0
    total_reward = 0.0
    for _ in range(steps):
        canonical_obs = sac_canonicalize_observation(result.observation)
        obs = fit_observation(canonical_obs, int(checkpoint["obs_size"]))
        with torch.no_grad():
            canonical_action = actor.act_deterministic(obs).squeeze(0).numpy()
        action = sac_decanonicalize_action(canonical_action, direction)
        result = env.step(action, direction=direction, residual=residual_walk)
        total_reward += result.reward
        if result.done or result.truncated:
            break
    robot_x = float(result.raw["observation"]["base_x"])
    ball_x = float(result.raw["observation"]["values"][BALL_DX_INDEX]) + robot_x
    return {"reward": total_reward, "robot_dx": (robot_x - start_robot_x) * sign, "ball_dx": (ball_x - start_ball_x) * sign}


def safe_eval(name: str, fn) -> dict[str, object]:
    try:
        return {"right": fn(1.0), "left": fn(-1.0)}
    except FileNotFoundError:
        return {"error": f"{name} policy file not found"}
    except Exception as exc:
        return {"error": f"{name} evaluation failed: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KMP, KNP, PPO and SAC on the hybrid simulator env.")
    parser.add_argument("--transport", type=str, default="auto")
    parser.add_argument("--dll-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default="robot_config.toml")
    parser.add_argument("--sac-policy", type=Path, default=ROOT / "sac_walk_policy.pt")
    args = parser.parse_args()

    results = {
        "kmp": safe_eval("kmp", lambda direction: evaluate_kmp(direction, transport=args.transport, dll_path=args.dll_path, config_path=args.config_path)),
        "knp": safe_eval("knp", lambda direction: evaluate_knp(direction, transport=args.transport, dll_path=args.dll_path, config_path=args.config_path)),
        "ppo": safe_eval("ppo", lambda direction: evaluate_ppo(direction, transport=args.transport, dll_path=args.dll_path, config_path=args.config_path)),
        "sac": safe_eval("sac", lambda direction: evaluate_sac(direction, transport=args.transport, dll_path=args.dll_path, config_path=args.config_path, policy_path=args.sac_policy)),
    }

    summary: dict[str, float] = {}
    for name, data in results.items():
        if "error" in data:
            continue
        summary[f"{name}_min_robot_dx"] = min(data["right"]["robot_dx"], data["left"]["robot_dx"])
        summary[f"{name}_min_ball_dx"] = min(data["right"]["ball_dx"], data["left"]["ball_dx"])
    results["summary"] = summary

    out_path = ROOT / "walk_policy_comparison.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
