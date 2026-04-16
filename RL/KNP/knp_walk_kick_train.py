from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass
class TrainingStats:
    episode: int
    reward: float
    robot_dx: float
    ball_dx_world: float
    robot_to_ball_dx: float
    steps: int


class KnpStyleSNNPolicy:
    """
    Practical fallback controller:
    uses KNP neuron parameter classes, but integrates membrane dynamics in Python,
    because the installed wheel backend runtime cannot currently load a backend.
    """

    def __init__(self, obs_size: int, hidden_size: int = 192, action_scale_deg: float = 8.0) -> None:
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.action_size = 4
        self.action_scale_deg = action_scale_deg

        self.hidden_params = BLIFATNeuronParameters()
        self.output_params = BLIFATNeuronParameters()
        self.hidden_threshold = float(self.hidden_params.activation_threshold) * 0.85
        self.output_threshold = float(self.output_params.activation_threshold) * 0.80
        self.hidden_decay = 0.92
        self.output_decay = 0.94
        self.reset_value = float(self.hidden_params.potential_reset_value)

        rng = np.random.default_rng(42)
        self.w_in = rng.normal(0.0, 0.18, size=(obs_size, hidden_size)).astype(np.float32)
        self.w_rec = rng.normal(0.0, 0.08, size=(hidden_size, hidden_size)).astype(np.float32)
        self.w_out = rng.normal(0.0, 0.16, size=(hidden_size, self.action_size)).astype(np.float32)
        self.bias_hidden = np.zeros(hidden_size, dtype=np.float32)
        self.bias_out = np.zeros(self.action_size, dtype=np.float32)

        self.hidden_potential = np.zeros(hidden_size, dtype=np.float32)
        self.output_potential = np.zeros(self.action_size, dtype=np.float32)
        self.last_hidden_spikes = np.zeros(hidden_size, dtype=np.float32)
        self.last_actions = np.zeros(self.action_size, dtype=np.float32)
        self.phase = 0.0
        self.phase_speed = 0.19
        self.gait_amplitude = np.array([10.0, 12.0, 10.0, 12.0], dtype=np.float32)
        self.gait_offset = np.array([2.0, 12.0, -2.0, 12.0], dtype=np.float32)

    def reset_state(self) -> None:
        self.hidden_potential.fill(0.0)
        self.output_potential.fill(0.0)
        self.last_hidden_spikes.fill(0.0)
        self.last_actions.fill(0.0)
        self.phase = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = canonicalize_observation(obs)
        obs_norm = obs / (np.linalg.norm(obs) + 1e-6)
        direction = 1.0
        walk_speed = float(obs[OBS_IDX["walk_speed"]]) if obs.shape[0] > OBS_IDX["walk_speed"] else 0.0
        phase_sin_obs = float(obs[OBS_IDX["walk_phase_sin"]]) if obs.shape[0] > OBS_IDX["walk_phase_sin"] else 0.0
        phase_cos_obs = float(obs[OBS_IDX["walk_phase_cos"]]) if obs.shape[0] > OBS_IDX["walk_phase_cos"] else 1.0

        hidden_current = obs_norm @ self.w_in + self.last_hidden_spikes @ self.w_rec + self.bias_hidden
        self.hidden_potential = self.hidden_potential * self.hidden_decay + hidden_current
        hidden_spikes = (self.hidden_potential >= self.hidden_threshold).astype(np.float32)
        self.hidden_potential[hidden_spikes > 0] = self.reset_value

        output_current = hidden_spikes @ self.w_out + self.bias_out
        self.output_potential = self.output_potential * self.output_decay + output_current
        output_spikes = (self.output_potential >= self.output_threshold).astype(np.float32)
        self.output_potential[output_spikes > 0] = self.reset_value

        # Mix membrane and spikes to get smoother motor control than pure binary spikes.
        motor_drive = np.tanh(self.output_potential * 0.35 + output_spikes * 0.65)
        self.phase += self.phase_speed * (0.75 + 0.5 * max(0.0, walk_speed))
        phase_sin_internal = np.sin(self.phase)
        phase_sin = 0.65 * phase_sin_obs + 0.35 * phase_sin_internal
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
        actions = gait + correction

        self.last_hidden_spikes = hidden_spikes
        self.last_actions = actions
        return actions.astype(np.float32)

    def update(self, obs: np.ndarray, reward: float, info: dict) -> None:
        obs = canonicalize_observation(obs)
        obs_norm = obs / (np.linalg.norm(obs) + 1e-6)
        direction = 1.0

        ball_dx = float(info["observation"]["values"][OBS_IDX["ball_dx"]]) * direction
        forward_term = reward
        kick_bonus = max(0.0, info["breakdown"].get("ball_progress", 0.0))
        forward_bonus = max(0.0, info["breakdown"].get("forward_progress", 0.0))
        upright_bonus = max(0.0, info["breakdown"].get("upright_bonus", 0.0))
        height_bonus = max(0.0, info["breakdown"].get("height_bonus", 0.0))
        contact_bonus = max(0.0, info["breakdown"].get("contact_bonus", 0.0))
        torque_penalty = max(0.0, info["breakdown"].get("torque_penalty", 0.0))
        stability_bonus = upright_bonus + height_bonus + 0.5 * contact_bonus

        reinforce = 0.0010 * forward_term + 0.0030 * forward_bonus + 0.0040 * stability_bonus + 0.0010 * kick_bonus
        anti = 0.0012 * max(0.0, -reward) + 0.0015 * torque_penalty

        active_hidden = np.where(self.last_hidden_spikes > 0, 1.0, 0.2)
        self.w_in += reinforce * np.outer(obs_norm, active_hidden)
        self.w_rec += 0.25 * reinforce * np.outer(active_hidden, active_hidden)
        self.w_out += reinforce * np.outer(np.where(self.last_hidden_spikes > 0, 1.0, 0.1), self.last_actions / self.action_scale_deg)

        # Encourage rightward motion and ball contact when the ball is in front.
        direction_hint = np.array([0.4, -0.2, -0.4, 0.2], dtype=np.float32) * direction
        if ball_dx > 0.0:
            self.bias_out += (0.0004 + 0.0010 * kick_bonus) * direction_hint
        if forward_bonus > 0.0:
            self.gait_offset += (0.0008 + 0.0015 * forward_bonus) * np.array([direction, 0.0, -direction, 0.0], dtype=np.float32)
            self.gait_amplitude += 0.004 * forward_bonus * np.array([0.8, 0.6, 0.8, 0.6], dtype=np.float32)
        if stability_bonus > 0.0:
            self.phase_speed = float(np.clip(self.phase_speed + 0.0002 * stability_bonus, 0.14, 0.28))

        if anti > 0.0:
            self.w_in -= anti * np.outer(obs_norm, np.ones(self.hidden_size, dtype=np.float32))
            self.w_rec -= 0.25 * anti * np.outer(active_hidden, active_hidden)
            self.w_out -= anti * np.outer(np.ones(self.hidden_size, dtype=np.float32), self.last_actions / self.action_scale_deg)
            self.gait_amplitude -= 0.10 * anti * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            self.phase_speed = float(np.clip(self.phase_speed - 0.001 * anti, 0.14, 0.28))

        self.w_in = np.clip(self.w_in, -2.5, 2.5)
        self.w_rec = np.clip(self.w_rec, -1.5, 1.5)
        self.w_out = np.clip(self.w_out, -2.5, 2.5)
        self.bias_out = np.clip(self.bias_out, -1.0, 1.0)
        self.gait_amplitude = np.clip(self.gait_amplitude, np.array([6.0, 8.0, 6.0, 8.0]), np.array([18.0, 24.0, 18.0, 24.0]))
        self.gait_offset = np.clip(self.gait_offset, np.array([-8.0, 6.0, -8.0, 6.0]), np.array([8.0, 24.0, 8.0, 24.0]))

    def load_npz(self, path: Path) -> None:
        checkpoint = np.load(path)
        old_w_in = checkpoint["w_in"].astype(np.float32)
        old_hidden_size = int(old_w_in.shape[1])
        new_w_in = np.zeros((self.obs_size, self.hidden_size), dtype=np.float32)
        row_count = min(self.obs_size, old_w_in.shape[0])
        col_count = min(self.hidden_size, old_hidden_size)
        new_w_in[:row_count, :col_count] = old_w_in[:row_count, :col_count]
        self.w_in = new_w_in

        old_w_rec = checkpoint["w_rec"].astype(np.float32) if "w_rec" in checkpoint else np.zeros((old_hidden_size, old_hidden_size), dtype=np.float32)
        self.w_rec = np.zeros((self.hidden_size, self.hidden_size), dtype=np.float32)
        self.w_rec[:col_count, :col_count] = old_w_rec[:col_count, :col_count]

        old_w_out = checkpoint["w_out"].astype(np.float32)
        self.w_out = np.zeros((self.hidden_size, self.action_size), dtype=np.float32)
        self.w_out[:col_count, :] = old_w_out[:col_count, :]

        old_bias_hidden = checkpoint["bias_hidden"].astype(np.float32)
        self.bias_hidden = np.zeros(self.hidden_size, dtype=np.float32)
        self.bias_hidden[:col_count] = old_bias_hidden[:col_count]
        self.bias_out = checkpoint["bias_out"].astype(np.float32)
        self.gait_amplitude = checkpoint["gait_amplitude"].astype(np.float32)
        self.gait_offset = checkpoint["gait_offset"].astype(np.float32)
        if "phase_speed" in checkpoint:
            self.phase_speed = float(np.asarray(checkpoint["phase_speed"]).reshape(-1)[0])
        self.reset_state()


def train_visible_knp(
    episodes: int,
    max_steps: int,
    repeat_steps: int,
    action_scale_deg: float,
    hidden_size: int,
    log_path: Path,
    transport: str,
    dll_path: str | None,
    config_path: str | None,
    resume_from: Path | None,
    residual_walk: bool,
    walk_speed_mps: float,
    eval_steps: int,
) -> None:
    baseline_walk_config: dict[str, float] = {}
    if residual_walk:
        baseline_path = ROOT / "kmp_best_config.json"
        if baseline_path.exists():
            baseline_walk_config = json.loads(baseline_path.read_text(encoding="utf-8"))

    env = DesktopRobotEnv(
        repeat_steps=repeat_steps,
        transport=transport,
        dll_path=dll_path,
        config_path=config_path,
    )
    if baseline_walk_config:
        env.set_walk_config(**baseline_walk_config)
    first = env.reset_with_direction(1.0)
    policy = KnpStyleSNNPolicy(
        obs_size=first.observation.shape[0],
        hidden_size=hidden_size,
        action_scale_deg=action_scale_deg,
    )
    if resume_from is not None and resume_from.exists():
        policy.load_npz(resume_from)

    history: list[dict] = []
    best_score = -float("inf")

    print("KNP walking+kicking training started")
    print(f"obs_size={first.observation.shape[0]} action_size=4 hidden_size={hidden_size}")

    def evaluate_current_policy(policy: KnpStyleSNNPolicy, direction: float, steps: int) -> tuple[float, float]:
        eval_env = DesktopRobotEnv(
            repeat_steps=repeat_steps,
            transport=transport,
            dll_path=dll_path,
            config_path=config_path,
        )
        if baseline_walk_config:
            eval_env.set_walk_config(**baseline_walk_config)
        eval_result = eval_env.reset_with_direction(direction)
        if residual_walk:
            eval_env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=walk_speed_mps)
        start_robot_x = float(eval_result.raw["observation"]["base_x"])
        start_ball_x = start_robot_x + float(eval_result.raw["observation"]["values"][OBS_IDX["ball_dx"]])
        policy.reset_state()
        last_result = eval_result
        for _ in range(steps):
            action = decanonicalize_action(policy.act(last_result.observation), direction)
            last_result = eval_env.step(
                action,
                direction=direction,
                residual=residual_walk,
            )
            if last_result.done or last_result.truncated:
                break
        robot_x = float(last_result.raw["observation"]["base_x"])
        ball_x = robot_x + float(last_result.raw["observation"]["values"][OBS_IDX["ball_dx"]])
        direction_sign = 1.0 if direction >= 0.0 else -1.0
        return (robot_x - start_robot_x) * direction_sign, (ball_x - start_ball_x) * direction_sign

    for episode in range(1, episodes + 1):
        policy.reset_state()
        direction = 1.0 if episode % 2 == 1 else -1.0
        step_result = env.reset_with_direction(direction)
        if residual_walk:
            env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=walk_speed_mps)
        start_robot_x = float(step_result.raw["observation"]["base_x"])
        start_ball_dx = float(step_result.raw["observation"]["values"][OBS_IDX["ball_dx"]])
        start_ball_x = start_robot_x + start_ball_dx
        episode_reward = 0.0
        final_raw = step_result.raw
        steps_taken = 0

        for step_idx in range(max_steps):
            action_deg = decanonicalize_action(policy.act(step_result.observation), direction)
            step_result = env.step(
                action_deg,
                direction=direction,
                residual=residual_walk,
            )
            policy.update(step_result.observation, step_result.reward, step_result.raw)
            episode_reward += step_result.reward
            final_raw = step_result.raw
            steps_taken = step_idx + 1

            if step_result.done or step_result.truncated:
                break

        robot_x = float(final_raw["observation"]["base_x"])
        ball_dx = float(final_raw["observation"]["values"][OBS_IDX["ball_dx"]])
        ball_x = robot_x + ball_dx
        direction_sign = 1.0 if direction >= 0.0 else -1.0
        stats = TrainingStats(
            episode=episode,
            reward=episode_reward,
            robot_dx=(robot_x - start_robot_x) * direction_sign,
            ball_dx_world=(ball_x - start_ball_x) * direction_sign,
            robot_to_ball_dx=ball_dx * direction_sign,
            steps=steps_taken,
        )
        history.append(stats.__dict__)

        replay_right = evaluate_current_policy(policy, 1.0, steps=eval_steps)
        replay_left = evaluate_current_policy(policy, -1.0, steps=eval_steps)
        replay_robot_dx = min(replay_right[0], replay_left[0])
        replay_ball_dx = min(replay_right[1], replay_left[1])
        score = 3.0 * replay_robot_dx + 0.2 * replay_ball_dx + 0.01 * episode_reward
        if score > best_score:
            best_score = score
            np.savez(
                "C:\\Users\\root\\Documents\\New project\\RL\\KNP\\knp_walk_kick_best.npz",
                w_in=policy.w_in,
                w_rec=policy.w_rec,
                w_out=policy.w_out,
                bias_hidden=policy.bias_hidden,
                bias_out=policy.bias_out,
                gait_amplitude=policy.gait_amplitude,
                gait_offset=policy.gait_offset,
                phase_speed=np.array([policy.phase_speed], dtype=np.float32),
                action_scale_deg=np.array([policy.action_scale_deg], dtype=np.float32),
                residual_walk=np.array([1.0 if residual_walk else 0.0], dtype=np.float32),
                walk_speed_mps=np.array([walk_speed_mps], dtype=np.float32),
                score=np.array([score], dtype=np.float32),
                reward=np.array([episode_reward], dtype=np.float32),
                robot_dx=np.array([stats.robot_dx], dtype=np.float32),
                ball_dx_world=np.array([stats.ball_dx_world], dtype=np.float32),
                replay_robot_dx=np.array([replay_robot_dx], dtype=np.float32),
                replay_ball_dx=np.array([replay_ball_dx], dtype=np.float32),
                replay_right_robot_dx=np.array([replay_right[0]], dtype=np.float32),
                replay_right_ball_dx=np.array([replay_right[1]], dtype=np.float32),
                replay_left_robot_dx=np.array([replay_left[0]], dtype=np.float32),
                replay_left_ball_dx=np.array([replay_left[1]], dtype=np.float32),
            )

        if episode % 5 == 0 or episode == 1:
            print(
                f"episode={episode:04d} dir={'right' if direction > 0 else 'left'} reward={episode_reward:+.3f} "
                f"robot_dx={stats.robot_dx:+.3f} "
                f"ball_dx_world={stats.ball_dx_world:+.3f} "
                f"replay_right={replay_right[0]:+.3f}/{replay_right[1]:+.3f} "
                f"replay_left={replay_left[0]:+.3f}/{replay_left[1]:+.3f} "
                f"ball_gap={stats.robot_to_ball_dx:.3f} steps={steps_taken}"
            )
            log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved history to {log_path}")
    print("saved best policy to RL/KNP/knp_walk_kick_best.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible KNP-style spiking training for walking forward and kicking the ball.")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--action-scale-deg", type=float, default=6.0)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--transport", type=str, default="auto")
    parser.add_argument("--dll-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default="robot_config.toml")
    parser.add_argument("--residual-walk", action="store_true")
    parser.add_argument("--walk-speed-mps", type=float, default=0.35)
    parser.add_argument("--eval-steps", type=int, default=600)
    parser.add_argument("--resume-from", type=Path, default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\knp_walk_kick_best.npz"))
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\knp_walk_kick_training.json"),
    )
    args = parser.parse_args()
    train_visible_knp(
        episodes=args.episodes,
        max_steps=args.max_steps,
        repeat_steps=args.repeat_steps,
        action_scale_deg=args.action_scale_deg,
        hidden_size=args.hidden_size,
        log_path=args.log_path,
        transport=args.transport,
        dll_path=args.dll_path,
        config_path=args.config_path,
        resume_from=args.resume_from,
        residual_walk=args.residual_walk,
        walk_speed_mps=args.walk_speed_mps,
        eval_steps=args.eval_steps,
    )


if __name__ == "__main__":
    main()
