from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gymnasium is required for GymnasiumRobotEnv") from exc

from desktop_rl_env import DesktopRobotEnv
from rl_signal_utils import (
    STAGE_BALANCE,
    STAGE_ENDURANCE,
    STAGE_SPEED_TRACKING,
    STAGE_STAND,
    STAGE_WALK,
    TASK_WALK,
    canonicalize_observation,
    curriculum_reward,
    decanonicalize_action,
    shaped_reward,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DLL_PATH = str((ROOT / "target" / "release" / "sim_core.dll").resolve())
DEFAULT_CONFIG_PATH = str((ROOT / "robot_config.toml").resolve())


class GymnasiumRobotEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        transport: str = "auto",
        base_url: str = "http://127.0.0.1:8080",
        dll_path: str | None = DEFAULT_DLL_PATH,
        config_path: str | None = DEFAULT_CONFIG_PATH,
        repeat_steps: int = 4,
        action_limit_deg: float = 120.0,
        direction: float = 1.0,
        residual_walk: bool = True,
        walk_speed_mps: float = 0.45,
        randomize_direction: bool = True,
        canonicalize: bool = True,
        task: str = TASK_WALK,
        reward_mode: str = "shaped",
        stage: str = STAGE_WALK,
        speed_min_mps: float = 0.15,
        speed_max_mps: float = 0.65,
        include_previous_action: bool = True,
        settle_steps: int = 12,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.direction = 1.0 if direction >= 0.0 else -1.0
        self.residual_walk = bool(residual_walk)
        self.walk_speed_mps = float(walk_speed_mps)
        self.randomize_direction = bool(randomize_direction)
        self.canonicalize = bool(canonicalize)
        self.task = task
        self.reward_mode = reward_mode
        self.stage = stage
        self.speed_min_mps = float(speed_min_mps)
        self.speed_max_mps = float(speed_max_mps)
        self.include_previous_action = bool(include_previous_action)
        self.settle_steps = int(settle_steps)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.current_walk_enabled = self.residual_walk
        self.current_walk_speed_mps = self.walk_speed_mps
        self.env = DesktopRobotEnv(
            transport=transport,
            base_url=base_url,
            dll_path=dll_path,
            config_path=config_path,
            repeat_steps=repeat_steps,
        )
        initial = self.env.reset_with_direction(self.direction)
        if self.residual_walk:
            self.env.set_walk_direction_speed(direction=self.direction, enabled=True, speed_mps=self.walk_speed_mps)
        self._previous = initial
        initial_obs = self._encode_observation(initial.observation)
        obs_size = int(initial_obs.shape[0])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.full((4,), -action_limit_deg, dtype=np.float32),
            high=np.full((4,), action_limit_deg, dtype=np.float32),
            dtype=np.float32,
        )
        self.last_info: dict[str, Any] = {
            "transport": self.env.backend_name,
            "observation_names": list(self.env.observation_names),
            "action_names": list(self.env.action_names),
            "raw": initial.raw,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        direction = self.direction
        if options and "direction" in options:
            direction = 1.0 if float(options["direction"]) >= 0.0 else -1.0
            self.direction = direction
        elif self.randomize_direction:
            direction = 1.0 if self.np_random.random() >= 0.5 else -1.0
            self.direction = direction
        result = self.env.reset_with_direction(direction)
        self.last_action = np.zeros(4, dtype=np.float32)
        if self.settle_steps > 0:
            settle_action = np.zeros(4, dtype=np.float32)
            for _ in range(self.settle_steps):
                result = self.env.step(
                    action_deg=settle_action,
                    direction=self.direction,
                    residual=False,
                    walk_enabled=False,
                    walk_speed_mps=0.0,
                )
                if result.done or result.truncated:
                    result = self.env.reset_with_direction(direction)
                    continue
        self._apply_stage_command(direction)
        result = self.env.step(
            action_deg=np.zeros(4, dtype=np.float32),
            direction=self.direction,
            residual=self.residual_walk,
            walk_enabled=self.current_walk_enabled,
            walk_speed_mps=self.current_walk_speed_mps if self.residual_walk else None,
        )
        self._previous = result
        observation = self._encode_observation(result.observation)
        info = {
            "transport": self.env.backend_name,
            "observation_names": list(self.env.observation_names),
            "action_names": list(self.env.action_names),
            "episode_time": result.episode_time,
            "raw": result.raw,
            "direction": direction,
        }
        self.last_info = info
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        policy_action = np.asarray(action, dtype=np.float32)
        if policy_action.shape != (4,):
            policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)[:4]
        if policy_action.shape[0] != 4:
            raise ValueError(f"action must have 4 elements, got shape {policy_action.shape}")
        policy_action = np.clip(policy_action, self.action_space.low, self.action_space.high)
        raw_action = policy_action
        if self.canonicalize:
            raw_action = decanonicalize_action(policy_action, self.direction)
        result = self.env.step(
            action_deg=raw_action,
            direction=self.direction,
            residual=self.residual_walk,
            walk_enabled=self.current_walk_enabled,
            walk_speed_mps=self.current_walk_speed_mps if self.residual_walk else None,
        )
        if self.reward_mode == "sim":
            reward = float(result.reward)
        else:
            reward = curriculum_reward(
                self._previous.observation,
                float(self._previous.raw["observation"]["base_x"]),
                result.observation,
                float(result.raw["observation"]["base_x"]),
                policy_action,
                self.last_action,
                self.direction,
                self.current_walk_speed_mps,
                self.task,
                self.stage,
                result.done,
            )
        self.last_action = policy_action.reshape(4)
        self._previous = result
        info = {
            "transport": self.env.backend_name,
            "observation_names": list(self.env.observation_names),
            "action_names": list(self.env.action_names),
            "episode_time": result.episode_time,
            "raw": result.raw,
            "direction": self.direction,
        }
        self.last_info = info
        return self._encode_observation(result.observation), reward, result.done, result.truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.env.close()

    def set_direction(self, direction: float) -> None:
        self.direction = 1.0 if direction >= 0.0 else -1.0

    def set_stage(self, stage: str) -> None:
        self.stage = stage

    def set_speed_range(self, min_speed_mps: float, max_speed_mps: float) -> None:
        self.speed_min_mps = float(min_speed_mps)
        self.speed_max_mps = float(max_speed_mps)

    def set_walk_speed(self, walk_speed_mps: float) -> None:
        self.walk_speed_mps = float(walk_speed_mps)

    def _encode_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if self.canonicalize:
            obs = canonicalize_observation(obs)
        if self.include_previous_action:
            return np.concatenate([obs, self.last_action.astype(np.float32)], axis=0)
        return obs

    def _apply_stage_command(self, direction: float) -> None:
        if self.stage == STAGE_STAND:
            self.current_walk_enabled = False
            self.current_walk_speed_mps = 0.0
        elif self.stage == STAGE_BALANCE:
            self.current_walk_enabled = True
            self.current_walk_speed_mps = float(self.np_random.uniform(self.speed_min_mps, self.speed_max_mps))
        elif self.stage == STAGE_SPEED_TRACKING:
            self.current_walk_enabled = True
            self.current_walk_speed_mps = float(self.np_random.uniform(self.speed_min_mps, self.speed_max_mps))
        elif self.stage == STAGE_ENDURANCE:
            self.current_walk_enabled = True
            self.current_walk_speed_mps = float(self.np_random.uniform(self.speed_min_mps, self.speed_max_mps))
        else:
            self.current_walk_enabled = True
            self.current_walk_speed_mps = self.walk_speed_mps
        if self.residual_walk:
            self.env.set_walk_direction_speed(
                direction=direction,
                enabled=self.current_walk_enabled,
                speed_mps=self.current_walk_speed_mps,
            )
