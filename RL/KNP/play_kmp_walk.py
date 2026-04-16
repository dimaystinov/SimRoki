from __future__ import annotations

import argparse
import json
from pathlib import Path

from desktop_rl_env import DesktopRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the built-in KMP gait controller.")
    parser.add_argument("--direction", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"C:\Users\root\Documents\New project\RL\KNP\kmp_best_config.json"),
    )
    args = parser.parse_args()

    env = DesktopRobotEnv()
    if args.config.exists():
        env.set_walk_config(**json.loads(args.config.read_text(encoding="utf-8")))
    env.reset_with_direction(args.direction)
    start = env.state()
    start_x = float(start["base"]["x"])
    start_ball_x = float(start["ball"]["x"])
    env.set_walk_direction_speed(direction=args.direction, enabled=True, speed_mps=args.speed)
    env.advance(args.duration)
    finish = env.state()
    env.set_walk_direction_speed(direction=args.direction, enabled=False, speed_mps=args.speed)

    sign = 1.0 if args.direction >= 0.0 else -1.0
    robot_dx = (float(finish["base"]["x"]) - start_x) * sign
    ball_dx = (float(finish["ball"]["x"]) - start_ball_x) * sign
    print(f"robot_dx={robot_dx:+.3f} m")
    print(f"ball_dx_world={ball_dx:+.3f} m")
    print(f"torso_angle={float(finish['base']['angle']):+.3f} rad")
    print(f"contacts L/R={finish['contacts']['left_foot']}/{finish['contacts']['right_foot']}")


if __name__ == "__main__":
    main()
