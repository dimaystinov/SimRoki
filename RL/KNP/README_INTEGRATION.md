# RL Integration Notes

The up-to-date high-level RL overview and current benchmark results now live in:

- [RL/README.md](C:/Users/root/Documents/New%20project/RL/README.md)

This folder now contains the local Python environment plus a hybrid bridge layer from Python to the simulator.

## What is ready

- local Miniconda environment in `.conda`
- `knp` wheel installed
- `torch`, `torchvision`, `torchaudio` installed
- `desktop_rl_env.py`: hybrid Python environment wrapper over the simulator
- `gymnasium_robot_env.py`: Gymnasium-compatible wrapper over the same environment
- `knp_walk_scaffold.py`: KNP-side scaffold check
- `knp_walk_kick_train.py`: visible KNP-style walking-and-kicking trainer
- `torch_walk_debug.py`: PyTorch-side debug rollout

## Hybrid transport

The RL bridge now supports two transports:

- `ffi`: direct in-process calls through [target/release/sim_core.dll](C:/Users/root/Documents/New%20project/target/release/sim_core.dll)
- `http`: fallback to the desktop server on `127.0.0.1:8080`

The default mode is:

- `transport="auto"`

In this mode Python tries FFI first, then falls back to HTTP if the DLL is unavailable.

## RL API exposed by the simulator

The desktop simulator now exposes:

- `POST /rl/reset`
- `GET /rl/observation`
- `POST /rl/step`

`/rl/step` payload:

```json
{
  "action_deg": [0.0, 0.0, 0.0, 0.0],
  "repeat_steps": 4
}
```

Actions are relative joint targets in degrees around the configured servo zero positions.

Servo order is fixed:

- `[1] right_hip`
- `[2] right_knee`
- `[3] left_hip`
- `[4] left_knee`

## Observation vector

The observation returned by `/rl/reset`, `/rl/observation`, and `/rl/step` contains:

- torso height
- torso angle
- torso linear velocity
- torso angular velocity
- center of mass offset from torso
- left/right foot contact flags
- ball offset from torso
- for each of the 4 joints:
  - angle relative to zero
  - angular velocity
  - target relative to zero
  - normalized torque

The response also includes the observation names list, so Python can map features reliably.

## Reward and episode termination

The reward currently combines:

- forward progress
- alive bonus
- upright bonus
- height bonus
- ground contact bonus
- torque penalty
- action delta penalty

Episode ends when:

- torso drops too low
- torso tilt becomes too large
- episode timeout is reached

These thresholds and weights now live in `robot_config.toml` under `[rl]`.

## Recommended protocol

Current recommendation for this project:

1. use `ffi` or `auto` during local RL training for minimum latency
2. keep `http` available for visual debugging against a running desktop window
3. move to headless mode later without changing the high-level Python environment API

Binary TCP is still not the preferred next step on one machine.

## Important note about the current KNP wheel

The installed `knp` wheel imports correctly, and its Python classes are available.

However, in the current environment the backend runtime does not load through `BackendLoader.load(...)`.
So right now:

- true backend-driven KNP execution is blocked by the wheel/runtime
- the working path is a KNP-style spiking controller in Python that still uses installed KNP neuron classes and runs against the live desktop simulator

This is why the current walking trainer is:

- visible
- trainable
- connected to the real simulator
- but not yet executed through a successfully loaded KNP backend plugin

## Quick start

Start the desktop simulator first:

```powershell
cargo run -p native_app
```

Then run a PyTorch debug rollout:

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\torch_walk_debug.py"
```

Or run the KNP scaffold check:

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\knp_walk_scaffold.py"
```

Run visible KNP-style walk-and-kick training:

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\knp_walk_kick_train.py"
```

Hybrid smoke test:

```powershell
$env:PYTHONPATH="C:\Users\root\Documents\New project\python-sdk;C:\Users\root\Documents\New project\RL\KNP"
python "C:\Users\root\Documents\New project\RL\KNP\hybrid_env_smoke_test.py"
```
