from __future__ import annotations

import numpy as np

from gymnasium_robot_env import GymnasiumRobotEnv


def main() -> None:
    env = GymnasiumRobotEnv(transport="auto", config_path="robot_config.toml", repeat_steps=4, direction=1.0)
    try:
        obs, info = env.reset()
        print("transport:", info["transport"])
        print("obs size:", obs.shape[0])
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        print("reward:", round(float(reward), 6))
        print("terminated:", terminated, "truncated:", truncated)
    finally:
        env.close()


if __name__ == "__main__":
    main()
