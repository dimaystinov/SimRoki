# Five Link Robot Simulator

Desktop-native 2D simulator of a five-link biped robot built on `rapier2d`, with native Rust rendering and a local control server embedded in the desktop app.

## Current status

Implemented right now:

- `sim_core/`: physics world, robot model, config loading, servo control, state export
- `native_app/`: native desktop window on `macroquad`
- local control server on `http://127.0.0.1:8080`
- optional `python-sdk/` for external control, while desktop control works standalone

Robot model:

- 5 physical links:
  - `torso`
  - `left_thigh`
  - `left_shin`
  - `right_thigh`
  - `right_shin`
- 4 actuated joints:
  - `right_hip`
  - `right_knee`
  - `left_hip`
  - `left_knee`

The servo angle is the relative angle between neighboring links:

- hip: `thigh` relative to `torso`
- knee: `shin` relative to `thigh`

## Config file

The simulator now reads its startup parameters from:

- [robot_config.toml](C:/Users/root/Documents/New%20project/robot_config.toml)

Configurable values include:

- gravity and fixed physics step
- ground size and friction
- torso, thigh, and shin geometry
- link masses
- link frictions
- body damping
- initial body positions
- suspend anchor clearance
- servo `kp`, `ki`, `kd`
- `max_torque`
- integral limit
- servo zero positions
- startup joint targets

Notes:

- the config is the startup source of truth
- the desktop app has a `Save config` button to write the current runtime config back to `robot_config.toml`
- `Use current pose as zero` updates the runtime servo zeros, and those zeros are used by the sliders

## Desktop app features

- native desktop window, no browser UI
- infinite square grid
- infinite ground support line
- mouse pan
- wheel zoom with smooth interpolation
- built-in control panel inside the simulator window
- real-time joint sliders
- current and target joint angles shown in degrees
- joint angle labels rendered near the robot
- `Use current pose as zero`
- slider control relative to the chosen zero pose
- `Save config`
- `Suspend top point` debug mode
- higher suspend anchor for servo debugging
- automatic fallback to built-in control when no external API control is active

## Servo and control

The robot is controlled by a simple shared PID-style servo controller for all 4 joints.

Current runtime-adjustable gains:

- `kp`: `-20 .. 20`
- `ki`: `-5 .. 5`
- `kd`: `-1 .. 1`
- `max_torque`: positive limit only

Angle limits:

- joint targets are now clamped only to `±180°`
- there are no extra hip/knee-specific software limits inside the servo target path

Notes:

- default `kd = 0`
- `max_torque` is in `N·m`
- `ki` is active in the controller, not just visualized
- the final applied torque is clamped by `max_torque`

## Geometry

Current default link lengths:

- `torso`: `0.68 m`
- `left_thigh`: `0.46 m`
- `left_shin`: `0.50 m`
- `right_thigh`: `0.46 m`
- `right_shin`: `0.50 m`

Current default link masses:

- `torso`: `0.31824002 kg`
- `left_thigh`: `0.14076002 kg`
- `left_shin`: `0.12600 kg`
- `right_thigh`: `0.14076002 kg`
- `right_shin`: `0.12600 kg`

The desktop UI shows both current masses and current lengths.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_desktop.ps1
```

You can also run directly:

```powershell
cargo run -p native_app
```

The app opens a native window and starts the local server bound only to `127.0.0.1`.

## Built-in controls

- `Space`: pause/resume
- `R`: reset robot
- `B`: reset ball smoke-test
- `1`: robot scene
- `2`: ball scene
- `F`: follow/recenter behavior
- mouse wheel: zoom
- middle mouse or drag-pan mode: move around the canvas

## Local API

Current local desktop control API:

- `GET /state`
- `POST /reset`
- `POST /reset/ball`
- `POST /pause`
- `POST /resume`
- `POST /scene`
- `POST /joint/angle`
- `POST /pose`
- `POST /gait`
- `POST /servo/targets`

`/state` includes:

- joint angles
- joint targets
- joint torques
- contacts
- link masses
- link lengths
- servo zero offsets

## Python control

The simulator can also be controlled externally from Python through the local server.

Examples:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\control_stand.ps1
python -m robot_sim.cli state
python -m robot_sim.cli joint set --name right_hip --angle -0.2
python -m robot_sim.cli pose set --file pose.json
python -m robot_sim.cli gait send --file gait.json
```

If you want imports without installation:

```powershell
$env:PYTHONPATH="C:\Users\root\Documents\New project\python-sdk"
python -m robot_sim.cli state
```

## Local git restore point

The current startup baseline is saved in local git as:

- commit: `e53e550`
- tag: `start-state`

Useful commands:

```powershell
git checkout start-state
```

Return `master` to the saved startup state:

```powershell
git checkout master
git reset --hard start-state
```

## Verification

Verified during development with:

- `cargo check`
- `python -m py_compile python-sdk\robot_sim\client.py python-sdk\robot_sim\models.py python-sdk\robot_sim\cli.py`
