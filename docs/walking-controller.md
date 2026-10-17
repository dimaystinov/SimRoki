# Walking controllers

Two controllers are available in the same native simulator. Both use actual
Rapier torso angle/angular velocity, measured contacts, four internal motor
torques capped at 2.55 N m, and no external stabilizing trunk torque. Neither
teleports the robot after a fall. Reset is an explicit action.

## IK reference: Walk right / right arrow

The tested forward gait alternates single and double support. Quintic foot
trajectories are converted to hip/knee targets using inverse kinematics for the
actual link lengths. Landing position responds to measured speed; torso angle
and angular-rate feedback correct the hips. Forward torso lean has the correct
negative world-angle sign. Both knees bend in the same direction.

Measured forward speed is filtered with a 0.20 s time constant and blended into
cadence after the startup transient. Faster forward motion increases cadence up
to the configured maximum. Contact must occur before support transfers.

Four 120 s trials, after settling for 0/1/5/10 s, measured respectively:

| Settle | Distance | Mean actual step | Maximum torso tilt |
|---|---:|---:|---:|
| 0 s | 61.09 m | 22.3 cm | 18.6° |
| 1 s | 59.62 m | 22.1 cm | 22.4° |
| 5 s | 61.77 m | 22.7 cm | 21.1° |
| 10 s | 60.38 m | 22.3 cm | 22.9° |

Requested speed was 0.25 m/s. Actual speed is approximately 0.5 m/s, so this is
not a calibrated speed servo. The initial sagittal foot separation is about
25 cm (23.8 cm after settling). The ball is placed outside the walking test area
so the automatic kick behavior cannot invalidate the walking trial.

## Planned RABBIT mode: RABBIT planned walk

`planner/model.py` derives constrained five-link dynamics and an inelastic impact
map from the current robot. `planner/optimize.py` calculates periodic trajectories
with nominal 20 cm steps and motor/contact constraints. The JSON files in
`motions/` contain angles, velocities and torques. The Rust implementation checks
model compatibility before loading a plan.

The runtime controls four virtual constraints using partial feedback
linearization. Phase comes from horizontal hip progress. Thus faster actual
movement advances the reference motor trajectory faster without depending on a
fixed animation clock. A measured new-foot contact switches the stance leg.
The startup controller builds momentum from rest, then uses the 0.5 m/s nominal
orbit and gradually blends toward the selected orbit at touchdowns. The nominal
speed is a parameter of the ideal point-foot plan, not a promise of Rapier speed.

Stop first blends a fast gait back toward the startup orbit, then waits for
double support and low forward speed before holding the measured joints with
damping. Braking is gradual, not an instantaneous freeze. A stalled step or severe fall disables the planned controller. The
explicit `reset_to_orbit` experiment initializes position **and velocity** once;
it is never used automatically for startup or recovery.

Four start/walk/stop trials (60 s walking + 60 s stopped, after 0/1/5/10 s settling)
passed. Distances were 8.65 / 8.43 / 8.51 / 8.25 m, maximum torso pitch
9.40 / 7.51 / 9.19 / 7.91°, and minimum torso height at least 1.289 m. Final
horizontal speed was below 0.02 m/s. Separate 120 s walking trials remained
upright with approximately 17–18 cm mean steps. Actual steady speed is about
0.13 m/s. The 20 cm ideal orbit therefore still has a measurable contact-model
error in Rapier; forcing a later support transfer did not solve this and was
not retained. Higher nominal-speed settings can shorten actual steps further.

### API

- `POST /walk/rabbit` with `{"enabled":true}` starts the planned controller.
- Optional `"nominal_speed_mps":0.3` through `1.0` changes its reference at
  touchdowns. Repeating this request does not restart the controller.
- `{"enabled":false}` requests a stop at double support.
- `{"enabled":true,"reset_to_orbit":true}` is an explicit experiment reset.
- `GET /state` exposes `walk_controller`, `planned_status`, `planned_phase`,
  `planned_steps`, actual base motion, joint torques and foot contacts.

### Reproduce

```sh
cargo test --release --locked --workspace
cargo run --release -p sim_core --example rabbit_probe
cargo run --release -p sim_core --example rabbit_probe -- --stop
cargo run --release -p sim_core --example rabbit_probe -- --accelerate
```

The numerical planner and model checks are described in
[planner/README.md](../planner/README.md). Primary papers, equations and parameter
comparisons are in [rabbit-model.md](rabbit-model.md).

Validation is for forward motion on level ground with the checked-in model and
120 Hz physics. Reverse walking, obstacles, arbitrary mass changes and a physical
robot are outside these results. Local numerical stability of the point-foot
model does not establish stability of Rapier's finite rectangular contacts.

## Faster, higher-step profile

`motions/rabbit-20cm-060-high.json` adds an optimized profile with nominal speed
0.6 m/s, minimum mid-swing clearance 3.5 cm and a forward-trunk reference
objective of 0.10 rad (previous objective 0.06 rad). Settings above 0.5 blend
toward this profile at touchdown. Three 120 s native trials after 1/5/10 s
settling measured steady actual speed 0.232–0.234 m/s, mean steps about 19.9 cm
and peak swing-foot height 3.0–3.1 cm. The previous 0.5 profile measured about
0.180 m/s and 1.1 cm foot lift. Starting immediately without settling can
converge to a shorter-step gait, so the numerical tests also retain that limitation.

The six-tenths nominal value must not be presented as actual forward speed.
Use `/state` or displacement over elapsed simulation time for measurement.


## Approximately 1 m/s with higher steps and more forward lean

`motions/rabbit-25cm-100-high.json` uses a 25 cm nominal step, 1.0 m/s nominal
speed, 6 cm minimum mid-swing clearance and a 0.20 rad forward-lean objective.
The optimizer reserves 3% of motor speed/torque and friction limits, preventing
interpolation between mesh points from exceeding the physical constraints.
Independent dense integration and contact checks are in the adjacent validation
report. Runtime tracking gains increase toward Kp=400, Kd=15 for this profile;
low-speed gains remain Kp=100, Kd=20. Torques remain limited to 2.55 N m.

Four 120 s native trials starting after 0/1/5/10 s standing measured steady speed
0.97–0.99 m/s, peak foot lift about 5.5 cm, mean forward lean 13.8° and torso
height above 1.28 m. Actual steps are about 27 cm, longer than the original
20 cm request. These are measured flat-ground trials, not a guarantee for
arbitrary initial conditions. The fast command ramps over successive contacts. Stopping from full speed
takes approximately 15–17 seconds in the tested starts; it follows the startup
orbit in reverse while retaining strong tracking, then holds at double support.

Use **RABBIT fast ~1 m/s** or:

```sh
curl -X POST http://127.0.0.1:8080/walk/rabbit \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"nominal_speed_mps":1.0}'
```

The left information overlay has no solid background. Arrow buttons at the
upper left and right independently collapse and expand the two panels.
