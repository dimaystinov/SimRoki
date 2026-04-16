from __future__ import annotations

from robot_sim import SimulatorFFIClient


def main() -> None:
    with SimulatorFFIClient(config_path="robot_config.toml") as sim:
        state = sim.get_state()
        print("scene:", state["scene"])
        print("time:", round(state["time"], 4))
        sim.set_joint("right_hip", -0.2)
        sim.step_for_seconds(0.1)
        obs = sim.rl_observation()
        print("obs size:", len(obs["values"]))
        print("target direction:", obs["target_direction"])


if __name__ == "__main__":
    main()
