from __future__ import annotations

import numpy as np

WALK_PHASE_SIN_INDEX = 8
WALK_PHASE_COS_INDEX = 9
BALL_DX_INDEX = 14
COM_DX_INDEX = 10
TORSO_HEIGHT_INDEX = 1
TORSO_ANGLE_INDEX = 2
TORSO_VX_INDEX = 3
TORSO_VY_INDEX = 4
TORSO_OMEGA_INDEX = 5
LEFT_CONTACT_INDEX = 12
RIGHT_CONTACT_INDEX = 13
LEFT_FOOT_DX_INDEX = 16
LEFT_FOOT_Y_INDEX = 17
RIGHT_FOOT_DX_INDEX = 18
RIGHT_FOOT_Y_INDEX = 19
LEGACY_JOINT_BLOCK_START = 16
JOINT_BLOCK_START = 20
JOINT_BLOCK_SIZE = 4
TASK_WALK = "walk"
TASK_DRIBBLE = "dribble"
STAGE_STAND = "stand"
STAGE_BALANCE = "balance"
STAGE_WALK = "walk"
STAGE_SPEED_TRACKING = "speed_tracking"
STAGE_ENDURANCE = "endurance"


def _forward_torso_lean_bonus(torso_angle: float, walk_speed_mps: float) -> float:
    angle = float(torso_angle)
    target = -0.03 - min(abs(float(walk_speed_mps)), 1.0) * 0.035
    target = max(target, -0.08)
    lean_error = abs(angle - target)
    return float(0.8 * (1.0 - min(lean_error / 0.14, 1.5)))


def _com_forward_shift_bonus(observation: np.ndarray, walk_speed_mps: float) -> float:
    obs = np.asarray(observation, dtype=np.float32)
    com_dx = float(obs[COM_DX_INDEX])
    target = -0.06 - min(abs(float(walk_speed_mps)), 1.0) * 0.028
    target = max(target, -0.10)
    com_error = abs(com_dx - target)
    return float(0.5 * (1.0 - min(com_error / 0.28, 1.5)))


def _contact_state(observation: np.ndarray) -> int:
    obs = np.asarray(observation, dtype=np.float32)
    left_contact = float(obs[LEFT_CONTACT_INDEX]) >= 0.5
    right_contact = float(obs[RIGHT_CONTACT_INDEX]) >= 0.5
    if left_contact and right_contact:
        return 2
    if left_contact:
        return 1
    if right_contact:
        return -1
    return 0


def _canonicalize_for_reward(observation: np.ndarray, direction: float) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if float(direction) >= 0.0:
        return obs
    return canonicalize_observation(obs)


def _joint_block_start(observation: np.ndarray) -> int | None:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[0] >= (JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 4):
        return JOINT_BLOCK_START
    if obs.shape[0] >= (LEGACY_JOINT_BLOCK_START + JOINT_BLOCK_SIZE * 4):
        return LEGACY_JOINT_BLOCK_START
    return None


def _has_foot_features(observation: np.ndarray) -> bool:
    obs = np.asarray(observation, dtype=np.float32)
    return obs.shape[0] >= JOINT_BLOCK_START


def _phase_fraction(observation: np.ndarray) -> float:
    obs = np.asarray(observation, dtype=np.float32)
    phase = float(np.arctan2(float(obs[WALK_PHASE_SIN_INDEX]), float(obs[WALK_PHASE_COS_INDEX])))
    if phase < 0.0:
        phase += float(2.0 * np.pi)
    return phase / float(2.0 * np.pi)


def _phase_bucket(observation: np.ndarray) -> tuple[str, float]:
    phase = _phase_fraction(observation)
    double_support = 0.12
    if phase < double_support:
        return "double_left", phase / max(double_support, 1e-6)
    if phase < 0.5:
        return "single_left", (phase - double_support) / max(0.5 - double_support, 1e-6)
    if phase < 0.5 + double_support:
        return "double_right", (phase - 0.5) / max(double_support, 1e-6)
    return "single_right", (phase - (0.5 + double_support)) / max(0.5 - double_support, 1e-6)


def _phase_clock_contact_bonus(observation: np.ndarray) -> float:
    obs = np.asarray(observation, dtype=np.float32)
    bucket, _ = _phase_bucket(obs)
    left_contact = 1.0 if float(obs[LEFT_CONTACT_INDEX]) >= 0.5 else 0.0
    right_contact = 1.0 if float(obs[RIGHT_CONTACT_INDEX]) >= 0.5 else 0.0
    if bucket == "single_left":
        expected_left, expected_right = 1.0, 0.0
    elif bucket == "single_right":
        expected_left, expected_right = 0.0, 1.0
    else:
        expected_left, expected_right = 1.0, 1.0
    bonus = 0.18 * (1.0 - abs(left_contact - expected_left))
    bonus += 0.18 * (1.0 - abs(right_contact - expected_right))
    if "single" in bucket and left_contact + right_contact > 1.5:
        bonus -= 0.08
    if "double" in bucket and left_contact + right_contact < 1.5:
        bonus -= 0.10
    return float(bonus)


def _swing_geometry_bonus(observation: np.ndarray, walk_speed_mps: float) -> float:
    obs = np.asarray(observation, dtype=np.float32)
    if not _has_foot_features(obs):
        return 0.0
    bucket, progress = _phase_bucket(obs)
    left_contact = float(obs[LEFT_CONTACT_INDEX]) >= 0.5
    right_contact = float(obs[RIGHT_CONTACT_INDEX]) >= 0.5
    speed_ratio = min(abs(float(walk_speed_mps)), 1.0)
    target_support_dx = -0.03 - 0.045 * speed_ratio
    target_step_span = 0.10 + 0.05 * speed_ratio
    target_clearance = 0.030 + 0.025 * speed_ratio

    if bucket == "single_left":
        stance_dx = float(obs[LEFT_FOOT_DX_INDEX])
        swing_dx = float(obs[RIGHT_FOOT_DX_INDEX])
        swing_y = float(obs[RIGHT_FOOT_Y_INDEX])
        stance_contact = left_contact
        swing_contact = right_contact
    elif bucket == "single_right":
        stance_dx = float(obs[RIGHT_FOOT_DX_INDEX])
        swing_dx = float(obs[LEFT_FOOT_DX_INDEX])
        swing_y = float(obs[LEFT_FOOT_Y_INDEX])
        stance_contact = right_contact
        swing_contact = left_contact
    else:
        left_dx = float(obs[LEFT_FOOT_DX_INDEX])
        right_dx = float(obs[RIGHT_FOOT_DX_INDEX])
        span = abs(right_dx - left_dx)
        span_score = 1.0 - min(abs(span - target_step_span) / 0.14, 1.5)
        return float(0.10 * span_score)

    bonus = 0.0
    if stance_contact:
        support_score = 1.0 - min(abs(stance_dx - target_support_dx) / 0.12, 1.5)
        bonus += 0.18 * support_score
    else:
        bonus -= 0.18

    if not swing_contact:
        if progress < 0.70:
            clearance_score = 1.0 - min(abs(swing_y - target_clearance) / 0.045, 1.5)
            bonus += 0.20 * clearance_score
            if swing_y < 0.012:
                bonus -= 0.14
        else:
            step_span = swing_dx - stance_dx
            span_score = 1.0 - min(abs(step_span - target_step_span) / 0.12, 1.5)
            bonus += 0.18 * span_score
            if swing_y < 0.010 and step_span < 0.05:
                bonus -= 0.12
    else:
        bonus -= 0.10
    return float(bonus)


def _human_like_gait_phase_bonus(
    previous_observation: np.ndarray,
    next_observation: np.ndarray,
    walk_speed_mps: float,
    direction: float,
) -> float:
    prev_obs = _canonicalize_for_reward(previous_observation, direction)
    next_obs = _canonicalize_for_reward(next_observation, direction)
    prev_state = _contact_state(prev_obs)
    next_state = _contact_state(next_obs)
    bonus = 0.0

    if next_state == 2:  # double support
        if prev_state == 2:
            bonus -= 0.06
        elif prev_state in (1, -1):
            bonus += 0.14 + 0.06 * _com_forward_shift_bonus(next_obs, walk_speed_mps)
        else:
            bonus += 0.03
    elif next_state in (1, -1):  # single support
        if prev_state in (1, -1):
            if next_state == prev_state:
                bonus -= 0.52
            else:
                bonus += 0.55
        elif prev_state == 2:
            bonus += 0.26 + 0.18 * _com_forward_shift_bonus(next_obs, walk_speed_mps)
        else:
            bonus -= 0.22
    else:
        bonus -= 0.70

    if next_state != 0:
        target = -0.055 - min(abs(float(walk_speed_mps)), 1.0) * 0.025
        com_error = abs(float(next_obs[COM_DX_INDEX]) - target)
        bonus += 0.14 * (1.0 - min(com_error / 0.30, 1.5))

    bonus += _phase_clock_contact_bonus(next_obs)
    bonus += _swing_geometry_bonus(next_obs, walk_speed_mps)
    return float(bonus)


def _stability_bonus(
    observation: np.ndarray,
    previous_observation: np.ndarray | None = None,
    walk_speed_mps: float = 0.25,
    direction: float = 1.0,
) -> float:
    obs = _canonicalize_for_reward(observation, direction)
    torso_height = float(obs[TORSO_HEIGHT_INDEX])
    torso_angle = float(obs[TORSO_ANGLE_INDEX])
    abs_torso_angle = abs(torso_angle)
    torso_vx = abs(float(obs[TORSO_VX_INDEX]))
    torso_vy = abs(float(obs[TORSO_VY_INDEX]))
    torso_omega = abs(float(obs[TORSO_OMEGA_INDEX]))
    left_contact = float(obs[LEFT_CONTACT_INDEX])
    right_contact = float(obs[RIGHT_CONTACT_INDEX])
    contact_sum = left_contact + right_contact

    stability = 0.0
    stability -= 4.0 * abs_torso_angle
    stability -= 0.35 * torso_vx
    stability -= 0.45 * torso_vy
    stability -= 0.90 * torso_omega
    stability -= 2.8 * abs(torso_height - 0.98)

    if contact_sum >= 2.0:
        stability += 0.35
    elif contact_sum >= 1.0:
        stability += 0.10
    else:
        stability -= 0.80

    if previous_observation is not None:
        prev = _canonicalize_for_reward(previous_observation, direction)
        prev_contact_sum = float(prev[LEFT_CONTACT_INDEX]) + float(prev[RIGHT_CONTACT_INDEX])
        if prev_contact_sum >= 1.0 and contact_sum < 0.5:
            stability -= 0.70

        left_foot_y = float(obs[LEFT_FOOT_Y_INDEX])
        right_foot_y = float(obs[RIGHT_FOOT_Y_INDEX])
        if left_contact >= 0.5 and left_foot_y > 0.018:
            stability -= 0.08
        if right_contact >= 0.5 and right_foot_y > 0.018:
            stability -= 0.08
        prev_left = float(prev[LEFT_CONTACT_INDEX]) >= 0.5
        prev_right = float(prev[RIGHT_CONTACT_INDEX]) >= 0.5
        left = float(obs[LEFT_CONTACT_INDEX]) >= 0.5
        right = float(obs[RIGHT_CONTACT_INDEX]) >= 0.5
        if prev_left == left and prev_right == right and prev_contact_sum == 1.0 and contact_sum == 1.0:
            stability -= 0.75

        if (left and right):
            if prev_left and prev_right and prev_contact_sum == 2.0 and contact_sum == 2.0:
                stability -= 0.08

    stability += _forward_torso_lean_bonus(torso_angle, walk_speed_mps)
    stability += _com_forward_shift_bonus(obs, walk_speed_mps)
    joint_block_start = _joint_block_start(obs)
    if joint_block_start is None:
        return float(stability)
    hip_symmetry = 1.0 - min(
        abs(
            float(obs[joint_block_start])
            + float(obs[joint_block_start + JOINT_BLOCK_SIZE * 2])
        )
        / 0.65,
        1.5,
    )
    knee_symmetry = 1.0 - min(
        abs(
            float(obs[joint_block_start + JOINT_BLOCK_SIZE])
            + float(obs[joint_block_start + JOINT_BLOCK_SIZE * 3])
        )
        / 1.0,
        1.5,
    )
    stability += 0.25 * hip_symmetry
    stability += 0.15 * knee_symmetry

    return float(stability)


def canonicalize_observation(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).copy()
    joint_block_start = _joint_block_start(obs)
    if joint_block_start is None or obs[0] >= 0.0:
        obs[0] = abs(obs[0]) if obs.shape[0] > 0 else 1.0
        return obs

    canon = obs.copy()
    canon[0] = 1.0
    canon[TORSO_ANGLE_INDEX] = -obs[TORSO_ANGLE_INDEX]
    canon[TORSO_VX_INDEX] = -obs[TORSO_VX_INDEX]
    canon[TORSO_OMEGA_INDEX] = -obs[TORSO_OMEGA_INDEX]
    canon[10] = -obs[10]
    canon[BALL_DX_INDEX] = -obs[BALL_DX_INDEX]
    canon[LEFT_CONTACT_INDEX] = obs[RIGHT_CONTACT_INDEX]
    canon[RIGHT_CONTACT_INDEX] = obs[LEFT_CONTACT_INDEX]
    if _has_foot_features(obs):
        canon[LEFT_FOOT_DX_INDEX] = -obs[RIGHT_FOOT_DX_INDEX]
        canon[LEFT_FOOT_Y_INDEX] = obs[RIGHT_FOOT_Y_INDEX]
        canon[RIGHT_FOOT_DX_INDEX] = -obs[LEFT_FOOT_DX_INDEX]
        canon[RIGHT_FOOT_Y_INDEX] = obs[LEFT_FOOT_Y_INDEX]

    right_hip = obs[joint_block_start : joint_block_start + JOINT_BLOCK_SIZE].copy()
    right_knee = obs[joint_block_start + JOINT_BLOCK_SIZE : joint_block_start + JOINT_BLOCK_SIZE * 2].copy()
    left_hip = obs[joint_block_start + JOINT_BLOCK_SIZE * 2 : joint_block_start + JOINT_BLOCK_SIZE * 3].copy()
    left_knee = obs[joint_block_start + JOINT_BLOCK_SIZE * 3 : joint_block_start + JOINT_BLOCK_SIZE * 4].copy()

    canon[joint_block_start : joint_block_start + JOINT_BLOCK_SIZE] = -left_hip
    canon[joint_block_start + JOINT_BLOCK_SIZE : joint_block_start + JOINT_BLOCK_SIZE * 2] = -left_knee
    canon[joint_block_start + JOINT_BLOCK_SIZE * 2 : joint_block_start + JOINT_BLOCK_SIZE * 3] = -right_hip
    canon[joint_block_start + JOINT_BLOCK_SIZE * 3 : joint_block_start + JOINT_BLOCK_SIZE * 4] = -right_knee
    return canon


def decanonicalize_action(action: np.ndarray, direction: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(4)
    if direction >= 0.0:
        return action
    return np.array(
        [-action[2], -action[3], -action[0], -action[1]],
        dtype=np.float32,
    )


def walking_shaping(
    observation: np.ndarray,
    base_x: float,
    direction: float,
    walk_speed_mps: float,
) -> float:
    obs = _canonicalize_for_reward(observation, direction)
    joint_block_start = _joint_block_start(obs)
    if joint_block_start is None:
        return 0.0
    signed_base_x = float(base_x) * (1.0 if direction >= 0.0 else -1.0)
    torso_height = float(obs[TORSO_HEIGHT_INDEX])
    torso_angle = float(obs[TORSO_ANGLE_INDEX])
    torso_vx = float(obs[TORSO_VX_INDEX])
    torso_vy = float(obs[TORSO_VY_INDEX])
    torso_omega = float(obs[TORSO_OMEGA_INDEX])
    left_contact = float(obs[LEFT_CONTACT_INDEX])
    right_contact = float(obs[RIGHT_CONTACT_INDEX])

    right_hip = float(obs[joint_block_start + 0])
    right_knee = float(obs[joint_block_start + JOINT_BLOCK_SIZE + 0])
    left_hip = float(obs[joint_block_start + JOINT_BLOCK_SIZE * 2 + 0])
    left_knee = float(obs[joint_block_start + JOINT_BLOCK_SIZE * 3 + 0])

    speed_tracking = 1.0 - min(abs(torso_vx - walk_speed_mps) / max(walk_speed_mps, 0.15), 1.5)
    contact_sum = left_contact + right_contact
    support_quality = 0.30 if contact_sum >= 2.0 else (0.45 if contact_sum >= 1.0 else -0.60)
    symmetry = 1.0 - min(abs(right_hip + left_hip) / 0.7, 1.5)
    knee_symmetry = 1.0 - min(abs(right_knee + left_knee) / 1.0, 1.5)

    return float(
        6.0 * signed_base_x
        + 1.8 * speed_tracking
        - 2.4 * abs(torso_angle)
        - 0.30 * abs(torso_vy)
        - 0.20 * abs(torso_omega)
        - 0.60 * abs(torso_height - 1.03)
        + support_quality
        + 0.25 * symmetry
        + 0.15 * knee_symmetry
        + _stability_bonus(obs, walk_speed_mps=walk_speed_mps)
    )


def dribble_shaping(
    observation: np.ndarray,
    base_x: float,
    direction: float,
    walk_speed_mps: float,
) -> float:
    walk = walking_shaping(observation, base_x, direction, walk_speed_mps)
    obs = _canonicalize_for_reward(observation, direction)
    ball_dx = float(obs[BALL_DX_INDEX])
    ball_dy = float(obs[BALL_DX_INDEX + 1])
    torso_vx = float(obs[TORSO_VX_INDEX])
    ball_target_dx = 0.45
    ball_distance_penalty = abs(ball_dx - ball_target_dx)
    ball_height_penalty = abs(ball_dy + 0.93)
    ball_control = 1.4 - 2.2 * ball_distance_penalty - 0.35 * ball_height_penalty
    carry_bonus = 0.35 if 0.15 <= ball_dx <= 0.85 else -0.35
    speed_match = max(0.0, 1.0 - abs(torso_vx - walk_speed_mps) / max(walk_speed_mps, 0.15))
    return float(walk + ball_control + carry_bonus + 0.25 * speed_match)


def shaped_reward(
    previous_observation: np.ndarray,
    previous_base_x: float,
    next_observation: np.ndarray,
    next_base_x: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    direction: float,
    walk_speed_mps: float,
    task: str,
    done: bool,
) -> float:
    shaping_fn = dribble_shaping if task == TASK_DRIBBLE else walking_shaping
    reward = shaping_fn(
        next_observation,
        next_base_x,
        direction,
        walk_speed_mps,
    ) - shaping_fn(
        previous_observation,
        previous_base_x,
        direction,
        walk_speed_mps,
    )
    reward += _human_like_gait_phase_bonus(previous_observation, next_observation, walk_speed_mps, direction)
    reward -= 0.0035 * float(np.abs(action).sum())
    reward -= 0.0015 * float(np.square(action).sum())
    reward -= 0.0020 * float(np.abs(np.asarray(action, dtype=np.float32) - np.asarray(previous_action, dtype=np.float32)).sum())
    reward += _stability_bonus(
        next_observation,
        previous_observation,
        walk_speed_mps=walk_speed_mps,
        direction=direction,
    )
    if done:
        reward -= 100.0
    return float(reward)


def stand_reward(
    next_observation: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    done: bool,
) -> float:
    obs = np.asarray(next_observation, dtype=np.float32)
    torso_height = float(obs[TORSO_HEIGHT_INDEX])
    torso_angle = float(obs[TORSO_ANGLE_INDEX])
    torso_vx = float(obs[TORSO_VX_INDEX])
    torso_vy = float(obs[TORSO_VY_INDEX])
    torso_omega = float(obs[TORSO_OMEGA_INDEX])
    left_contact = float(obs[LEFT_CONTACT_INDEX])
    right_contact = float(obs[RIGHT_CONTACT_INDEX])
    support_bonus = 0.5 if (left_contact + right_contact) >= 2.0 else (0.2 if (left_contact + right_contact) >= 1.0 else -1.0)
    reward = (
        1.5
        - 3.8 * abs(torso_angle)
        - 0.9 * abs(torso_vx)
        - 0.6 * abs(torso_vy)
        - 0.5 * abs(torso_omega)
        - 1.8 * abs(torso_height - 1.03)
        + support_bonus
        - 0.0030 * float(np.abs(action).sum())
        - 0.0010 * float(np.square(action).sum())
        - 0.0025 * float(np.abs(np.asarray(action, dtype=np.float32) - np.asarray(previous_action, dtype=np.float32)).sum())
        + _stability_bonus(obs)
    )
    if done:
        reward -= 100.0
    return float(reward)


def balance_reward(
    next_observation: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    direction: float,
    walk_speed_mps: float,
    done: bool,
) -> float:
    obs = np.asarray(next_observation, dtype=np.float32)
    torso_vx = float(obs[TORSO_VX_INDEX]) * (1.0 if direction >= 0.0 else -1.0)
    speed_tracking = 1.0 - min(abs(torso_vx - walk_speed_mps) / max(walk_speed_mps, 0.08), 2.0)
    reward = stand_reward(next_observation, action, previous_action, False) + 0.8 * speed_tracking
    reward += _stability_bonus(
        next_observation,
        walk_speed_mps=walk_speed_mps,
        direction=direction,
    )
    if done:
        reward -= 100.0
    return float(reward)


def endurance_reward(
    previous_observation: np.ndarray,
    previous_base_x: float,
    next_observation: np.ndarray,
    next_base_x: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    direction: float,
    walk_speed_mps: float,
    done: bool,
) -> float:
    obs = _canonicalize_for_reward(next_observation, direction)
    torso_height = float(obs[TORSO_HEIGHT_INDEX])
    torso_angle = float(obs[TORSO_ANGLE_INDEX])
    torso_vx = float(obs[TORSO_VX_INDEX])
    torso_vy = float(obs[TORSO_VY_INDEX])
    torso_omega = float(obs[TORSO_OMEGA_INDEX])
    left_contact = float(obs[LEFT_CONTACT_INDEX])
    right_contact = float(obs[RIGHT_CONTACT_INDEX])
    progress = (float(next_base_x) - float(previous_base_x)) * (1.0 if direction >= 0.0 else -1.0)
    contact_sum = left_contact + right_contact
    support_quality = 0.55 if contact_sum >= 1.0 else -0.85
    double_support_bonus = 0.20 if contact_sum >= 2.0 else 0.0
    speed_tracking = 1.0 - min(abs(torso_vx - walk_speed_mps) / max(walk_speed_mps, 0.12), 1.5)
    reward = (
        5.5 * progress
        + 0.45
        + 1.3 * speed_tracking
        - 4.2 * abs(torso_angle)
        - 1.8 * max(0.0, abs(torso_angle) - 0.22)
        - 0.45 * abs(torso_vy)
        - 0.30 * abs(torso_omega)
        - 1.75 * abs(torso_height - 1.00)
        + support_quality
        + double_support_bonus
        - 0.0025 * float(np.abs(action).sum())
        - 0.0015 * float(np.square(action).sum())
        - 0.0035 * float(np.abs(np.asarray(action, dtype=np.float32) - np.asarray(previous_action, dtype=np.float32)).sum())
        + _stability_bonus(
            obs,
            previous_observation,
            walk_speed_mps=walk_speed_mps,
            direction=direction,
        )
        + _human_like_gait_phase_bonus(previous_observation, next_observation, walk_speed_mps, direction)
    )
    if done:
        reward -= 100.0
    return float(reward)


def curriculum_reward(
    previous_observation: np.ndarray,
    previous_base_x: float,
    next_observation: np.ndarray,
    next_base_x: float,
    action: np.ndarray,
    previous_action: np.ndarray,
    direction: float,
    walk_speed_mps: float,
    task: str,
    stage: str,
    done: bool,
) -> float:
    if stage == STAGE_STAND:
        return stand_reward(next_observation, action, previous_action, done)
    if stage == STAGE_BALANCE:
        return balance_reward(next_observation, action, previous_action, direction, walk_speed_mps, done)
    if stage == STAGE_ENDURANCE:
        return endurance_reward(
            previous_observation,
            previous_base_x,
            next_observation,
            next_base_x,
            action,
            previous_action,
            direction,
            walk_speed_mps,
            done,
        )
    return shaped_reward(
        previous_observation,
        previous_base_x,
        next_observation,
        next_base_x,
        action,
        previous_action,
        direction,
        walk_speed_mps,
        task,
        done,
    )
