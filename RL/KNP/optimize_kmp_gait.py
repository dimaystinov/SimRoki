from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from desktop_rl_env import DesktopRobotEnv


ROOT = Path(r"C:\Users\root\Documents\New project\RL\KNP")
OUT_PATH = ROOT / "kmp_best_config.json"
LOG_PATH = ROOT / "kmp_optimization_history.json"

BASE_CONFIG = {
    "nominal_speed_mps": 0.28,
    "max_speed_mps": 0.55,
    "max_accel_mps2": 0.75,
    "cycle_frequency_hz": 1.05,
    "max_cycle_frequency_hz": 1.60,
    "nominal_step_length_m": 0.085,
    "step_length_gain": 0.05,
    "nominal_step_height_m": 0.028,
    "run_step_height_m": 0.05,
    "stance_duty_factor": 0.68,
    "torso_pitch_kp": 0.75,
    "torso_pitch_kd": 0.32,
    "hip_upright_gain": 1.6,
    "hip_upright_damping": 0.24,
    "torso_upright_limit_rad": 0.10,
    "torso_forward_lean_per_speed": 0.03,
    "torso_forward_lean_max_rad": 0.03,
    "velocity_kp": 0.34,
    "pelvis_height_target_m": 0.84,
    "stance_foot_spread_m": 0.13,
    "foot_separation_min_m": 0.15,
    "kick_trigger_distance_m": 1.20,
}

BOUNDS = {
    "nominal_speed_mps": (0.15, 0.7),
    "max_speed_mps": (0.2, 1.0),
    "max_accel_mps2": (0.2, 2.0),
    "cycle_frequency_hz": (0.5, 2.5),
    "max_cycle_frequency_hz": (0.8, 4.0),
    "nominal_step_length_m": (0.04, 0.24),
    "step_length_gain": (0.02, 0.30),
    "nominal_step_height_m": (0.01, 0.10),
    "run_step_height_m": (0.03, 0.18),
    "stance_duty_factor": (0.55, 0.80),
    "torso_pitch_kp": (0.10, 1.20),
    "torso_pitch_kd": (0.02, 0.50),
    "hip_upright_gain": (0.2, 2.5),
    "hip_upright_damping": (0.02, 0.6),
    "torso_upright_limit_rad": (0.04, 0.25),
    "torso_forward_lean_per_speed": (0.0, 0.3),
    "torso_forward_lean_max_rad": (0.0, 0.16),
    "velocity_kp": (0.02, 0.60),
    "pelvis_height_target_m": (0.70, 0.92),
    "stance_foot_spread_m": (0.04, 0.20),
    "foot_separation_min_m": (0.08, 0.24),
    "kick_trigger_distance_m": (0.20, 1.20),
}


def clip_cfg(cfg: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in cfg.items():
        lo, hi = BOUNDS[key]
        out[key] = max(lo, min(hi, float(value)))
    out["max_speed_mps"] = max(out["max_speed_mps"], out["nominal_speed_mps"])
    out["max_cycle_frequency_hz"] = max(out["max_cycle_frequency_hz"], out["cycle_frequency_hz"])
    out["run_step_height_m"] = max(out["run_step_height_m"], out["nominal_step_height_m"])
    out["torso_forward_lean_max_rad"] = min(out["torso_forward_lean_max_rad"], out["torso_upright_limit_rad"])
    return out


def mutate_cfg(base: dict[str, float], sigma: float, rng: random.Random) -> dict[str, float]:
    candidate = {}
    for key, value in base.items():
        lo, hi = BOUNDS[key]
        scale = (hi - lo) * sigma
        candidate[key] = value + rng.gauss(0.0, scale)
    return clip_cfg(candidate)


def evaluate_direction(
    env: DesktopRobotEnv,
    config: dict[str, float],
    direction: float,
    duration_s: float,
    sample_dt: float,
) -> dict[str, float]:
    env.set_walk_config(**config)
    env.reset_with_direction(direction)
    start = env.state()
    start_x = float(start["base"]["x"])
    start_ball_x = float(start["ball"]["x"])
    sign = 1.0 if direction >= 0.0 else -1.0

    env.set_walk_direction_speed(direction=direction, enabled=True, speed_mps=config["nominal_speed_mps"])
    samples: list[dict[str, float]] = []
    elapsed = 0.0
    while elapsed < duration_s:
        env.advance(sample_dt)
        elapsed += sample_dt
        state = env.state()
        dx = (float(state["base"]["x"]) - start_x) * sign
        ball_dx = (float(state["ball"]["x"]) - start_ball_x) * sign
        angle = abs(float(state["base"]["angle"]))
        height = float(state["base"]["y"])
        contacts = int(bool(state["contacts"]["left_foot"])) + int(bool(state["contacts"]["right_foot"]))
        samples.append(
            {
                "dx": dx,
                "ball_dx": ball_dx,
                "angle": angle,
                "height": height,
                "contacts": contacts,
            }
        )
        if height < 0.35 or angle > 1.35:
            break
    env.set_walk_direction_speed(direction=direction, enabled=False, speed_mps=config["nominal_speed_mps"])

    final = samples[-1] if samples else {"dx": 0.0, "ball_dx": 0.0, "angle": math.pi, "height": 0.0, "contacts": 0}
    alive_fraction = len(samples) * sample_dt / duration_s
    avg_contacts = sum(s["contacts"] for s in samples) / max(len(samples), 1)
    score = (
        final["dx"] * 12.0
        + alive_fraction * 20.0
        + avg_contacts * 1.0
        - final["angle"] * 5.0
        - max(0.0, 0.72 - final["height"]) * 12.0
    )
    return {
        "score": score,
        "dx": final["dx"],
        "ball_dx": final["ball_dx"],
        "angle": final["angle"],
        "height": final["height"],
        "alive_fraction": alive_fraction,
        "avg_contacts": avg_contacts,
    }


def evaluate_candidate(env: DesktopRobotEnv, config: dict[str, float], duration_s: float, sample_dt: float) -> dict[str, object]:
    right = evaluate_direction(env, config, 1.0, duration_s, sample_dt)
    left = evaluate_direction(env, config, -1.0, duration_s, sample_dt)
    min_dx = min(float(right["dx"]), float(left["dx"]))
    min_ball_dx = min(float(right["ball_dx"]), float(left["ball_dx"]))
    min_alive = min(float(right["alive_fraction"]), float(left["alive_fraction"]))
    max_angle = max(float(right["angle"]), float(left["angle"]))
    min_height = min(float(right["height"]), float(left["height"]))
    aggregate = (
        min_dx * 22.0
        + min_alive * 24.0
        - max_angle * 8.0
        - max(0.0, 0.78 - min_height) * 16.0
    )
    return {"score": aggregate, "right": right, "left": left, "config": config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the built-in KMP gait controller for long stable walking and ball kicking.")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--sample-dt", type=float, default=0.15)
    parser.add_argument("--transport", type=str, default="ffi")
    parser.add_argument("--dll-path", type=str, default=r"C:\Users\root\Documents\New project\target\release\sim_core.dll")
    parser.add_argument("--config-path", type=str, default=r"C:\Users\root\Documents\New project\robot_config.toml")
    args = parser.parse_args()

    rng = random.Random(42)
    env = DesktopRobotEnv(transport=args.transport, dll_path=args.dll_path, config_path=args.config_path)
    best = evaluate_candidate(env, BASE_CONFIG, args.duration, args.sample_dt)
    history = [best]
    sigma = 0.18

    print("KMP optimization started")
    print(json.dumps(best, indent=2))

    for iteration in range(1, args.iterations + 1):
        candidate_cfg = mutate_cfg(best["config"], sigma=sigma, rng=rng)
        candidate = evaluate_candidate(env, candidate_cfg, args.duration, args.sample_dt)
        history.append(candidate)
        if float(candidate["score"]) > float(best["score"]):
            best = candidate
            sigma = max(0.05, sigma * 0.92)
        else:
            sigma = min(0.30, sigma * 1.03)

        print(
            f"iter={iteration:03d} score={candidate['score']:+.3f} "
            f"right={candidate['right']['dx']:+.3f}/{candidate['right']['ball_dx']:+.3f} "
            f"left={candidate['left']['dx']:+.3f}/{candidate['left']['ball_dx']:+.3f} "
            f"best={best['score']:+.3f}"
        )
        OUT_PATH.write_text(json.dumps(best["config"], indent=2), encoding="utf-8")
        LOG_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("best config:")
    print(json.dumps(best["config"], indent=2))


if __name__ == "__main__":
    main()
