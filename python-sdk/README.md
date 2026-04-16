# Robot Simulator Python SDK

This folder contains the Python-side tools for controlling the desktop Rust simulator over the local HTTP server or directly through the Rust FFI library.

The simulator itself is desktop-first now. Python control is optional and talks to the running desktop app at `http://127.0.0.1:8080`.

## What is here

- `robot_sim/client.py`: small Python client for the local simulator API
- `robot_sim/ffi_client.py`: direct `ctypes` wrapper for `sim_core.dll`
- `robot_sim/cli.py`: command-line tool `simctl`
- `models.py`: pose and gait payload models
- `servo_sliders.py`: simple external slider GUI kept as an optional tool

## Default server

- `http://127.0.0.1:8080`

## Supported CLI commands

- `simctl state`
- `simctl reset`
- `simctl pause`
- `simctl resume`
- `simctl joint set --name right_hip --angle 0.1`
- `simctl pose set --file pose.json`
- `simctl gait send --file gait.json`

The available joint names are:

- `right_hip`
- `right_knee`
- `left_hip`
- `left_knee`

All joint angles sent from Python are in radians.

## Usage

If you want to run without installation:

```powershell
$env:PYTHONPATH="C:\Users\root\Documents\New project\python-sdk"
python -m robot_sim.cli state
```

Examples:

```powershell
$env:PYTHONPATH="C:\Users\root\Documents\New project\python-sdk"
python -m robot_sim.cli joint set --name right_hip --angle -0.2
python -m robot_sim.cli pose set --file pose.json
python -m robot_sim.cli gait send --file gait.json
```

## Python API

```python
from robot_sim import SimulatorClient, Pose, Gait, GaitPhase

client = SimulatorClient()

state = client.get_state()
client.set_joint("right_knee", 1.1)
client.set_pose(
    Pose(
        base_x=0.0,
        base_y=1.0,
        base_yaw=0.0,
        joints={
            "right_hip": -0.15,
            "right_knee": 1.15,
            "left_hip": -0.15,
            "left_knee": 1.15,
        },
    )
)
```

## Direct FFI API

Build the shared library first:

```powershell
cargo build -p sim_core --release
```

This produces:

- [target/release/sim_core.dll](C:/Users/root/Documents/New%20project/target/release/sim_core.dll)

Example:

```python
from robot_sim import SimulatorFFIClient

with SimulatorFFIClient(config_path="robot_config.toml") as sim:
    state = sim.get_state()
    sim.set_joint("right_hip", -0.2)
    sim.step_for_seconds(0.1)
    obs = sim.rl_observation()
```

The FFI wrapper exposes direct equivalents of the main simulator operations:

- `get_state`
- `pause`
- `resume`
- `reset`
- `reset_ball`
- `set_scene`
- `set_joint`
- `set_targets`
- `set_pose`
- `send_gait`
- `send_motion_sequence_deg`
- `set_servo_gains`
- `set_zero_to_current_pose`
- `set_robot_suspended`
- `set_suspend_clearance`
- `set_walk_direction`
- `set_walk_config`
- `rl_reset`
- `rl_observation`
- `rl_step`
- `save_config`

If the DLL is not passed explicitly, the wrapper searches in this order:

- environment variable `ROBOT_SIM_DLL`
- current working directory
- `target/release/sim_core.dll`
- `target/debug/sim_core.dll`

## Build wheel

Build the Python wheel from [python-sdk](C:/Users/root/Documents/New%20project/python-sdk):

```powershell
python -m pip install build
python -m build
```

The resulting wheel appears in:

- [python-sdk/dist](C:/Users/root/Documents/New%20project/python-sdk/dist)

Install it:

```powershell
pip install .\dist\robot_sim_client-0.3.0-py3-none-win_amd64.whl
```

## Build native wheel for the current environment

To build a Windows wheel that already contains `sim_core.dll` inside the package:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_native_wheel.ps1
```

This creates a platform-specific wheel for the current environment in:

- [python-sdk/dist](C:/Users/root/Documents/New%20project/python-sdk/dist)

After installation, `SimulatorFFIClient` can load the packaged DLL automatically, without setting `ROBOT_SIM_DLL`.

## Payload examples

### Pose

```json
{
  "base": { "x": 0.0, "y": 1.0, "yaw": 0.0 },
  "joints": {
    "right_hip": -0.15,
    "right_knee": 1.15,
    "left_hip": -0.15,
    "left_knee": 1.15
  }
}
```

### Gait

```json
{
  "name": "walk",
  "cycle_s": 0.8,
  "phases": [
    {
      "duration": 0.2,
      "joints": {
        "right_hip": 0.2,
        "right_knee": 1.0,
        "left_hip": -0.1,
        "left_knee": 1.2
      }
    }
  ]
}
```

## Notes

- The desktop app already contains built-in sliders and PID controls, so Python is no longer required for normal manual use.
- External Python commands temporarily take control priority when they are sent to the simulator.
- The simulator owns the actual servo dynamics, joint limits, torque clamping, masses, and link geometry.
- `/state` can be used for logging, analysis, or external control loops.
