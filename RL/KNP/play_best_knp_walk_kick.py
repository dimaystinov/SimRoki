from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from knp.neuron_traits import BLIFATNeuronParameters

from desktop_rl_env import DesktopRobotEnv

ROOT = Path(r"C:\Users\root\Documents\New project\RL\KNP")
OBS_IDX = {
    "target_direction": 0,
    "torso_height": 1,
    "torso_angle": 2,
    "torso_vx": 3,
    "torso_vy": 4,
    "torso_omega": 5,
    "walk_speed": 6,
    "walk_target_speed": 7,
    "walk_phase_sin": 8,
    "walk_phase_cos": 9,
    "com_dx": 10,
    "com_dy": 11,
    "left_contact": 12,
    "right_contact": 13,
    "ball_dx": 14,
    "ball_dy": 15,
}
JOINT_BLOCK_START = 16
JOINT_BLOCK_SIZE = 4


def canonicalize_observation(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).copy()
    if obs.shape[0] < (JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 4) or obs[OBS_IDX["target_direction"]] >= 0.0:
        obs[0] = abs(obs[0]) if obs.shape[0] > 0 else 1.0
        return obs

    canon = obs.copy()
    canon[OBS_IDX["target_direction"]] = 1.0
    canon[OBS_IDX["torso_angle"]] = -obs[OBS_IDX["torso_angle"]]
    canon[OBS_IDX["torso_vx"]] = -obs[OBS_IDX["torso_vx"]]
    canon[OBS_IDX["torso_omega"]] = -obs[OBS_IDX["torso_omega"]]
    canon[OBS_IDX["com_dx"]] = -obs[OBS_IDX["com_dx"]]
    canon[OBS_IDX["ball_dx"]] = -obs[OBS_IDX["ball_dx"]]
    canon[OBS_IDX["left_contact"]] = obs[OBS_IDX["right_contact"]]
    canon[OBS_IDX["right_contact"]] = obs[OBS_IDX["left_contact"]]

    right_hip = obs[JOINT_BLOCK_START:JOINT_BLOCK_START + JOINT_BLOCK_SIZE].copy()
    right_knee = obs[JOINT_BLOCK_START + JOINT_BLOCK_SIZE:JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 2].copy()
    left_hip = obs[JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 2:JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 3].copy()
    left_knee = obs[JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 3:JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 4].copy()

    canon[JOINT_BLOCK_START:JOINT_BLOCK_START + JOINT_BLOCK_SIZE] = -left_hip
    canon[JOINT_BLOCK_START + JOINT_BLOCK_SIZE:JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 2] = -left_knee
    canon[JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 2:JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 3] = -right_hip
    canon[JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 3:JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 4] = -right_knee
    return canon


def decanonicalize_action(action: np.ndarray, direction: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(4)
    if direction >= 0.0:
        return action
    return np.array(
        [
            -action[2],
            -action[3],
            -action[0],
            -action[1],
        ],
        dtype=np.float32,
    )


class ReplayKnpStylePolicy:
    def __init__(self, policy_path: Path, action_scale_deg: float = 20.0) -> None:
        data = np.load(policy_path)
        self.w_in = data["w_in"].astype(np.float32)
        if "w_rec" in data:
            self.w_rec = data["w_rec"].astype(np.float32)
        else:
            self.w_rec = np.zeros((self.w_in.shape[1], self.w_in.shape[1]), dtype=np.float32)
        self.w_out = data["w_out"].astype(np.float32)
        self.bias_hidden = data["bias_hidden"].astype(np.float32)
        self.bias_out = data["bias_out"].astype(np.float32)
        if "action_scale_deg" in data:
            self.action_scale_deg = float(np.asarray(data["action_scale_deg"]).reshape(-1)[0])
        else:
            self.action_scale_deg = action_scale_deg

        self.hidden_params = BLIFATNeuronParameters()
        self.output_params = BLIFATNeuronParameters()
        self.hidden_threshold = float(self.hidden_params.activation_threshold) * 0.85
        self.output_threshold = float(self.output_params.activation_threshold) * 0.80
        self.hidden_decay = 0.92
        self.output_decay = 0.94
        self.reset_value = float(self.hidden_params.potential_reset_value)

        self.hidden_potential = np.zeros(self.w_in.shape[1], dtype=np.float32)
        self.output_potential = np.zeros(self.w_out.shape[1], dtype=np.float32)
        self.last_hidden_spikes = np.zeros(self.w_in.shape[1], dtype=np.float32)
        self.phase = 0.0
        self.phase_speed = float(data["phase_speed"][0]) if "phase_speed" in data else 0.22
        self.gait_amplitude = data["gait_amplitude"].astype(np.float32) if "gait_amplitude" in data else np.array([22.0, 18.0, 22.0, 18.0], dtype=np.float32)
        self.gait_offset = data["gait_offset"].astype(np.float32) if "gait_offset" in data else np.array([6.0, 18.0, -6.0, 18.0], dtype=np.float32)
        self.residual_walk = bool(float(np.asarray(data["residual_walk"]).reshape(-1)[0])) if "residual_walk" in data else False
        self.walk_speed_mps = float(np.asarray(data["walk_speed_mps"]).reshape(-1)[0]) if "walk_speed_mps" in data else 0.35

    def reset(self) -> None:
        self.hidden_potential.fill(0.0)
        self.output_potential.fill(0.0)
        self.last_hidden_spikes.fill(0.0)
        self.phase = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = canonicalize_observation(obs)
        if obs.shape[0] != self.w_in.shape[0]:
            resized = np.zeros(self.w_in.shape[0], dtype=np.float32)
            copy_count = min(resized.shape[0], obs.shape[0])
            resized[:copy_count] = obs[:copy_count]
            obs = resized
        obs_norm = obs / (np.linalg.norm(obs) + 1e-6)
        direction = 1.0
        walk_speed = float(obs[OBS_IDX["walk_speed"]]) if obs.shape[0] > OBS_IDX["walk_speed"] else 0.0
        phase_sin_obs = float(obs[OBS_IDX["walk_phase_sin"]]) if obs.shape[0] > OBS_IDX["walk_phase_sin"] else 0.0
        phase_cos_obs = float(obs[OBS_IDX["walk_phase_cos"]]) if obs.shape[0] > OBS_IDX["walk_phase_cos"] else 1.0

        hidden_current = obs_norm @ self.w_in + self.last_hidden_spikes @ self.w_rec + self.bias_hidden
        self.hidden_potential = self.hidden_potential * self.hidden_decay + hidden_current
        hidden_spikes = (self.hidden_potential >= self.hidden_threshold).astype(np.float32)
        self.hidden_potential[hidden_spikes > 0] = self.reset_value
        self.last_hidden_spikes = hidden_spikes

        output_current = hidden_spikes @ self.w_out + self.bias_out
        self.output_potential = self.output_potential * self.output_decay + output_current
        output_spikes = (self.output_potential >= self.output_threshold).astype(np.float32)
        self.output_potential[output_spikes > 0] = self.reset_value

        motor_drive = np.tanh(self.output_potential * 0.35 + output_spikes * 0.65)
        self.phase += self.phase_speed * (0.75 + 0.5 * max(0.0, walk_speed))
        phase_sin = 0.65 * phase_sin_obs + 0.35 * np.sin(self.phase)
        phase_cos = 0.65 * phase_cos_obs + 0.35 * np.cos(self.phase)
        gait = np.array(
            [
                direction * self.gait_offset[0] + direction * self.gait_amplitude[0] * phase_sin,
                self.gait_offset[1] + self.gait_amplitude[1] * max(0.0, -phase_sin) + 1.5 * max(0.0, phase_cos),
                direction * self.gait_offset[2] - direction * self.gait_amplitude[2] * phase_sin,
                self.gait_offset[3] + self.gait_amplitude[3] * max(0.0, phase_sin) + 1.5 * max(0.0, -phase_cos),
            ],
            dtype=np.float32,
        )
        correction = motor_drive * self.action_scale_deg
        return gait + correction


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the best KNP-style walk-and-kick policy.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\knp_walk_kick_best.npz"),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--action-scale-deg", type=float, default=20.0)
    parser.add_argument("--direction", type=float, default=1.0)
    parser.add_argument("--transport", type=str, default="auto")
    parser.add_argument("--dll-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default="robot_config.toml")
    args = parser.parse_args()

    env = DesktopRobotEnv(
        repeat_steps=args.repeat_steps,
        transport=args.transport,
        dll_path=args.dll_path,
        config_path=args.config_path,
    )
    agent = ReplayKnpStylePolicy(args.policy, action_scale_deg=args.action_scale_deg)
    if agent.residual_walk:
        baseline_path = ROOT / "kmp_best_config.json"
        if baseline_path.exists():
            env.set_walk_config(**json.loads(baseline_path.read_text(encoding="utf-8")))
    step_result = env.reset_with_direction(args.direction)
    if agent.residual_walk:
        env.set_walk_direction_speed(direction=args.direction, enabled=True, speed_mps=agent.walk_speed_mps)
    start_robot_x = float(step_result.raw["observation"]["base_x"])
    start_ball_x = start_robot_x + float(step_result.raw["observation"]["values"][OBS_IDX["ball_dx"]])
    agent.reset()

    for _ in range(args.steps):
        action = decanonicalize_action(agent.act(step_result.observation), args.direction)
        step_result = env.step(
            action,
            direction=args.direction,
            residual=agent.residual_walk,
        )
        if step_result.done or step_result.truncated:
            break

    robot_x = float(step_result.raw["observation"]["base_x"])
    ball_x = robot_x + float(step_result.raw["observation"]["values"][OBS_IDX["ball_dx"]])
    direction_sign = 1.0 if args.direction >= 0.0 else -1.0
    print(f"robot_dx={(robot_x - start_robot_x) * direction_sign:+.3f} m")
    print(f"ball_dx_world={(ball_x - start_ball_x) * direction_sign:+.3f} m")


if __name__ == "__main__":
    main()
