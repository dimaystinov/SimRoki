from __future__ import annotations

import numpy as np

from desktop_rl_env import DesktopRobotEnv


def main() -> None:
    env = DesktopRobotEnv(transport="auto", config_path="robot_config.toml", repeat_steps=4)
    try:
        reset = env.reset_with_direction(1.0)
        print("transport:", env.backend_name)
        print("obs size:", reset.observation.shape[0])
        step = env.step(np.zeros(4, dtype=np.float32), direction=1.0)
        print("reward:", round(step.reward, 6))
        print("done:", step.done, "truncated:", step.truncated)
    finally:
        env.close()


if __name__ == "__main__":
    main()
