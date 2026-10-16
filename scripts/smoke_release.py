"""Exercise the actual DLL and SDK extracted from the Windows release archive."""
import math
from pathlib import Path
import sys
import tempfile
import zipfile


def main():
    # Windows keeps a loaded DLL locked until this process exits.
    with tempfile.TemporaryDirectory(prefix="simroki-release-", ignore_cleanup_errors=True) as directory:
        with zipfile.ZipFile(sys.argv[1]) as archive:
            archive.extractall(directory)
        mac = sys.platform == "darwin"
        root = Path(directory) / ("SimRoki-macos-arm64" if mac else "SimRoki-windows-x64")
        config_root = root / "SimRoki.app/Contents/Resources" if mac else root
        for name in (("SimRoki.app/Contents/MacOS/native_app", "SimRoki.app/Contents/MacOS/SimRoki", "SimRoki.app/Contents/Info.plist") if mac else ("native_app.exe", "sim_core.dll", "Start-SimRoki.cmd", "robot_config.toml", "motion_sequence.json")):
            assert (root / name).stat().st_size > 0, name
        sys.path.insert(0, str(root / "python-sdk"))
        from robot_sim import SimulatorFFIClient

        # Use the packaged SDK's automatic DLL discovery, from outside its directory.
        with SimulatorFFIClient(config_path=str(config_root / "robot_config.toml")) as sim:
            before = sim.get_state()
            assert before["scene"] == "robot", before["scene"]
            sim.resume()
            sim.set_joint("right_hip", -0.2)
            sim.step_for_seconds(0.1)
            after = sim.get_state()
            assert after["time"] > before["time"], "Simulation did not advance"
            observation = sim.rl_observation()["values"]
            assert observation and all(math.isfinite(x) for x in observation)
            sim.pause()
            assert sim.get_state()["paused"] is True
            sim.set_scene("ball")
            sim.reset_ball()
            assert sim.get_state()["scene"] == "ball"
            sim.reset()
            assert sim.get_state()["scene"] == "robot"
        print("PASS: packaged DLL loads, steps physics, returns finite observations, pauses and resets")


if __name__ == "__main__":
    main()
