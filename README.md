# Five Link Robot Simulator

## Windows download

Download `SimRoki-windows-x64.zip` from [Releases](https://github.com/dimaystinov/SimRoki/releases/latest),
extract the entire archive, and run `Start-SimRoki.cmd`. The portable desktop app does not need Rust or Python.
The archive also includes `sim_core.dll`, configuration, motions, and the optional Python SDK.

Windows releases are built and smoke-tested by `.github/workflows/windows-release.yml` when a `v*` tag is pushed.
Manual workflow runs build and verify an artifact without publishing a release.
The build uses `Cargo.lock`, MSVC x64, and a statically linked C runtime.
Each release includes a SHA-256 checksum. Interactive graphics are not manually tested by the workflow.

Desktop-native 2D simulator of a five-link biped robot built on `rapier2d`, with native Rust rendering and a local control server embedded in the desktop app.

## Current status

Implemented right now:

- `sim_core/`: physics world, robot model, config loading, servo control, state export
- `native_app/`: native desktop window on `macroquad`
- local control server on `http://127.0.0.1:8080`
- optional `python-sdk/` for external control, while desktop control works standalone
- direct FFI layer from `sim_core` for Python without HTTP

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

## RABBIT model and walking research

The reference morphology is the **planar five-link, four-actuator RABBIT biped**.
The detailed [RABBIT model and implementation notes](docs/rabbit-model.md) explain
kinematics, dynamics, impact/relabeling, trajectory optimization, forward torso
pitch, 20 cm steps, speed/cadence coupling, and differences from Rapier contacts.

Primary references and source code:

- [RABBIT: A Testbed for Advanced Control Theory (2003)](https://doi.org/10.1109/MCS.2003.1234651)
  — mechanism, experimental parameters and control architecture.
- [Hybrid Zero Dynamics of Planar Biped Walkers (2003)](https://doi.org/10.1109/TAC.2002.806653)
  — hybrid model, virtual constraints and stability theory.
- [Feedback Control of Dynamic Bipedal Robot Locomotion: book, MATLAB models and errata](https://grizzle.robotics.umich.edu/publications/biped-book).
- [Virtual Constraints and Hybrid Zero Dynamics](https://arxiv.org/html/1706.01127).
- [OptimTraj five-link biped source](https://github.com/MatthewPeterKelly/OptimTraj/tree/master/demo/fiveLinkBiped)
  and [Kelly's direct-collocation tutorial](https://doi.org/10.1137/16M1062569).

`planner/model.py` derives a separate point-foot model using **SimRoki's own**
masses and geometry. `planner/optimize.py` computes periodic motor trajectories
with dynamics, contact, impact and actuator constraints. The native **RABBIT planned walk**
button runs state-phased virtual constraints with startup, contact-driven leg
switching, bounded torques and a double-support stop. The existing **Walk right**
IK controller remains available. See [controls, API and measured results](docs/walking-controller.md)
and [reproducible planner checks](planner/README.md).

At the top of the right panel, **Start automatic walk ~1 m/s** starts continuous walking. The same button becomes **Stop automatic walk** for gradual braking (about 12–15 seconds from full speed). A status line shows walking, braking, or stopped.

The **Start automatic walk ~1 m/s** button selects a higher-step profile: native trials
measure approximately 0.97–1.00 m/s, 27 cm steps, 5.5 cm foot lift and 14° forward
trunk lean. The slow RABBIT orbit retains 20 cm nominal steps (17–18 cm measured),
and the IK reference measures 22–23 cm. The arrow buttons at the upper left and
right independently hide/show the information and servo panels; Robot mode has
no solid background. Requested/nominal speed is not yet
an accurate measured-speed command. The point-foot return map has a numerical
spectral radius of about 0.818; this is not a formal stability proof for Rapier.
See [implementation status and remaining limits](docs/implementation-plan.md).

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
- RL rollout timing, termination thresholds, and reward weights

Notes:

- the config is the startup source of truth
- the desktop app has a `Save config` button to write the current runtime config back to `robot_config.toml`
- the calibrated `robot_config.toml` is also embedded at build time as the standard configuration, including the initial stance and servo gains. Launching without an external config uses these same tuned values.
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
- `POST /rl/reset`
- `GET /rl/observation`
- `POST /rl/step`

`/state` includes:

- joint angles
- joint targets
- joint torques
- contacts
- link masses
- link lengths
- servo zero offsets

`/rl/step` uses:

- `action_deg`: four relative servo targets in degrees around the configured zero pose
- `repeat_steps`: how many fixed simulation steps to execute for one RL control action

RL reward shaping and termination thresholds live in:

- `[rl]` section of [robot_config.toml](C:/Users/root/Documents/New%20project/robot_config.toml)

## RL Preparation

The desktop simulator is now prepared for visual RL integration before moving to headless mode.

What is ready:

- RL reset/step/observation API
- observation vector with stable feature names
- reward breakdown from the simulator side
- done/truncated flags from the simulator side
- action semantics fixed to 4 joints in degrees around servo zero
- Python bridge prepared in [RL/KNP/desktop_rl_env.py](C:/Users/root/Documents/New%20project/RL/KNP/desktop_rl_env.py)
- KNP scaffold in [RL/KNP/knp_walk_scaffold.py](C:/Users/root/Documents/New%20project/RL/KNP/knp_walk_scaffold.py)
- PyTorch debug rollout in [RL/KNP/torch_walk_debug.py](C:/Users/root/Documents/New%20project/RL/KNP/torch_walk_debug.py)
- integration notes in [RL/KNP/README_INTEGRATION.md](C:/Users/root/Documents/New%20project/RL/KNP/README_INTEGRATION.md)

Current transport recommendation:

- visual debug phase: local HTTP + JSON
- high-throughput training phase: in-process binding over `sim_core` FFI or shared memory

For this project, binary TCP is not the preferred next step on the same machine.

## FFI bridge

`sim_core` now builds both as a normal Rust library and as a shared library for Python FFI.

Build:

```powershell
cargo build -p sim_core --release
```

Result:

- [target/release/sim_core.dll](C:/Users/root/Documents/New%20project/target/release/sim_core.dll)

Python can then call the simulator directly through:

- [python-sdk/robot_sim/ffi_client.py](C:/Users/root/Documents/New%20project/python-sdk/robot_sim/ffi_client.py)

This wrapper mirrors the main desktop API operations without going through `127.0.0.1:8080`.

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

### Music

The right panel has a separate **Music: play / resume** / **Music: pause** button for
GHOSTFACE PLAYA — Why Not. Pause preserves the current playback position; resume
continues from it. The track loops automatically and is embedded in both desktop
release executables. Music starts paused and is independent of walking controls.

Automatic walking is regression-tested for ten consecutive 60-second walks, each
followed by a complete stop and five seconds at rest, without resetting the robot.

The ball now starts exactly **10 m ahead**. In automatic walking, the robot times
a bounded extra swing-leg torque pulse to strike on the move, without commanding
a slowdown. Assistance is limited to one swing per approach and rearms after
separation, preventing repeated boosts while chasing the ball. Ground support
excludes contact with the ball.

The robot-scene ball has a physical mass of **0.1 kg (100 g)**, configurable with
`robot.ball_mass_kg` in `robot_config.toml`. Its black radius follows the physics
body angle, making rotation and sliding visible. Contact friction remains enabled.

**Start ROBOFEST 2026 (100 m)** at the top of the right panel resets the course
and starts automatic walking without requiring motion JSON. The timer freezes
when the robot crosses 100 m; ball crossing is recorded, and the robot brakes
after finishing. Press the same button for a fresh attempt. Ball friction is
0.06; ball/surface friction is 4.05 (`physics.ball_ground_friction`).
Ball contact coefficients multiply, giving an effective coefficient of 0.243.
Foot/surface friction remains calibrated at 1.35, independently of the ball.

### Gait quality in ROBOFEST

The collapsible **Mean gait** HUD shows the whole-attempt score (0–100),
walking/running classification, and eligibility, even with both panels collapsed.
The evaluator runs at the physics timestep; pausing does not add samples. The
race stores the score at finish. Its competition heuristics are not a scientific
measure of human resemblance; this five-link model has no arms.

The score combines duration-weighted posture (40 points), alternating landings
(30), and duration-weighted foot-only support (30). Torso lean up to 35 degrees
and hip height above 65% of leg length receive full posture credit. Flight is
allowed: alternating-foot running is eligible and receives no flight penalty.
At least eight landings are required to qualify. The bottom 10% of each shin is
the allowed contact region; contact with the ball is excluded from ground support.

Persistent non-foot support (0.15 s continuously or 0.5 s total), crawling
(hip below 45% leg length or torso beyond 70 degrees for 0.5 s), fewer than 60%
alternating transitions after eight landings, external suspension, or moving
above 0.2 m/s for 3 s without a landing invalidates the attempt. Invalidity stays
latched and the score is capped at 49, so later upright motion cannot erase a
violation. ROBOFEST stops the attempt and displays the reason. The thresholds
are simulator rules, not official ROBOFEST regulations. Changes to robot anatomy
require revalidation of the contact-region rules.

Music playback switches the scene background instantly at the **20 user-marked
boundaries** in `assets/audio/why-not-boundaries.json`: original background →
cyan neon → orange neon → random neon accents without consecutive repeats.
The random sequence is selected once per application launch and retained during
pauses and track loops. Only the background, grid, and floor glow change; the
robot and ball retain their original rendering and physics. The first cut is at 37.007 s. The irregular 4/16-second gaps
are preserved exactly, without inferred extra cuts or snapping to a beat grid.
Timing follows audio playback, not simulation time. Pausing freezes the color;
resuming continues at the same position; looping restarts the cue sheet. There
is no fade. Loop length is measured from decoded samples to avoid accumulating
MP3 metadata timing error.

### MP4 recording

The red recording button in the lower-right corner starts recording the simulator window.
Click it again to finish and save an H.264/AAC MP4 in `~/Movies/SimRoki`
(`%USERPROFILE%/Videos/SimRoki` on Windows). The application's music is included
with its current playback position, volume, pauses and looping; no microphone or
other applications are captured. Recording uses FFmpeg, available on PATH or in
`/opt/homebrew/bin` or `/usr/local/bin` on macOS. Video is limited to 1920 pixels
wide at 30 fps. Stop recording and wait for `Saved:` before closing the app.

### Repeatable simulation

Physics and control advance at a fixed 120 Hz; render frames never change the
integration step. The regression suite compares the complete state every simulated
second at 15, 30, 60 and 120 render FPS and with frame stalls up to half a second.
A slow computer may display fewer frames or lag behind real time, but physics
steps are not dropped. Identical results require identical configuration and
commands at the same simulation steps; wall-clock mouse clicks are not a replay.
