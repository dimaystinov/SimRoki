# implementation_notes

## Target system
Current robot:
- planar 5-link biped
- torso + two thighs + two shins
- 4 actuated joints: L/R hip, L/R knee
- no ankles
- RL from Python
- goal: stable and human-like walking first, later speed changes, recovery, running, jumping, ball interaction

## Core recommendation
For this exact setup, the best practical order is:
1. **Baseline PPO locomotion** with simple reward and command-conditioned velocity tracking.
2. Add **phase variable / gait clock** to stabilize gait timing.
3. Add **biomechanics-inspired rewards** to reduce crouch, scuffing, and erratic timing.
4. Add **style prior** later (AMP or imitation-like reward) only after baseline walking is robust.
5. Only then expand to recovery, running, and finally ball interaction.

## Controller architecture
### Recommended first controller
- Policy: MLP PPO policy.
- Action space: target joint position offsets or desired joint angles for 4 joints.
- Low-level actuation: PD tracking in simulator.

### Why not direct torques first
Direct torque control can work, but for a first 2D custom biped it usually increases instability and reward sensitivity. Position targets + PD is the fastest path to a working gait.

## Observation design
Recommended observation vector:
- torso pitch
- torso angular velocity
- hip angles (2)
- knee angles (2)
- hip angular velocities (2)
- knee angular velocities (2)
- COM height (if available)
- root horizontal velocity
- commanded forward velocity
- gait phase features: sin(phi), cos(phi)
- previous action
- contact indicators for left/right foot

Optional later:
- short history stack of 2–4 observations
- filtered contact durations
- estimated step timer

## Action design
### First version
- 4 continuous actions in [-1, 1]
- map to desired joint angle offsets around nominal standing pose

### Better version after baseline
- residual action around a nominal gait generator:
  - a hand-designed or optimized gait gives q_ref(phi)
  - policy outputs delta_q
  - final command = q_ref(phi) + delta_q

This is a very strong path for your robot because the morphology is small and structured.

## Reward design
### Stage 1: make it walk
Use a compact reward:
- + forward velocity tracking
- + torso upright reward
- + alive reward
- - action smoothness penalty
- - joint velocity penalty
- - torque or effort penalty
- large termination penalty on fall

### Stage 2: make it walk like a human
Add:
- step timing / cadence consistency reward
- alternating contact reward (left-right stepping)
- minimum swing foot clearance reward
- no-leg-crossing geometric penalty
- stance-knee-extension reward (careful; too strong causes stiffness)
- symmetry penalty between left and right gait statistics
- torso-height stability reward

### Do not do this early
- Do not start with 15+ reward terms.
- Do not hard-code every gait feature at once.
- Do not make “human-like” reward stronger than basic stability before walking exists.

## Curriculum design
### Recommended curriculum
1. Standing balance.
2. Small forward velocity commands.
3. Wider velocity range.
4. Random push perturbations.
5. Variable episode starts and phase starts.
6. Uneven timing and command changes.

### Command curriculum
- Start with fixed low speed.
- Then sample v_cmd in a narrow interval.
- Gradually widen to include stop/go and speed ramps.

## Human-like gait priorities for your model
Because there are no ankles yet, do **not** over-optimize for human heel-to-toe roll. Your model cannot realize that naturally.

Prioritize instead:
- upright torso
- regular alternating steps
- no leg crossing
- clean swing foot clearance
- consistent step timing
- non-crouched stance when possible
- smooth acceleration/deceleration

## Fall recovery
For the current 5-link planar model, full getting-up may be physically limited depending on contact model and available actuation.

Practical recommendation:
- First implement **push recovery while still upright**.
- Then add **partial recovery** from large pitch excursions.
- Only treat full ground get-up as a later stage or after adding more joints/contact options.

## Running and jumping
Do not train these now.

Introduce only after:
- walking works over a range of speeds
- acceleration/deceleration is smooth
- gait timing is stable
- reward no longer needs major retuning every run

Then use either:
- gait-conditioned policy with walk/run mode, or
- residual policy on top of a reference gait generator

## Ball interaction
Ball interaction should come last.

Recommended order:
1. approach a target position while walking
2. approach a moving target while walking
3. touch a ball with one foot
4. controlled kick
5. only then dribble-like repeated interaction

The two-stage structure from humanoid dribbling papers is still valid in 2D:
- stage A: locomotion competence
- stage B: task-specific ball reward

## Suggested experiment matrix
### Baselines
- PPO, no phase clock
- PPO + phase clock
- PPO + phase clock + contact-timing rewards
- PPO residual around nominal gait

### Metrics
Track at least:
- episode length
- fall rate
- velocity tracking RMSE
- mean step period
- step period variance
- torso pitch RMS
- energy / action magnitude
- symmetry score
- foot scuff count
- number of leg-crossing events

## Best immediate next step
Implement:
- PPO
- position-target actions + PD
- observation with sin/cos phase
- reward = velocity tracking + torso upright + smoothness + fall termination + alternating contacts
- command curriculum over forward speed

That is the fastest route to a useful first walking controller for your custom 2D biped.
