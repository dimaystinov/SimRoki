# 2D Biped Technical Plan

## Goal
Build a stable, human-like walking controller for the current planar 5-link biped:
- torso
- left/right thigh
- left/right shin
- 4 actuated joints: left/right hip, left/right knee

The immediate objective is stable locomotion. Running, recovery, and ball interaction come later.

## Current honest status
- Best current controller: `KNP residual`
- Best current worst-case walking distance: about `+0.50 m`
- `SB3 SAC` works through Gymnasium + FFI, but is still less stable than `KNP`
- Ball interaction is intentionally disabled during locomotion-first training by spawning the ball far away

## Core strategy
The project should not rely on pure end-to-end RL from scratch.

Recommended stack:
1. Structured gait prior in the simulator
2. Phase clock in the observation
3. Command-conditioned walking objective
4. Residual policy over nominal gait
5. Human-likeness only after stability is reliable

## Training stages

### Stage 0: Environment integrity
Required before heavy training:
- FFI and HTTP parity checks
- fixed start pose
- stable PD position control
- deterministic reset behavior
- ball removed from the locomotion task

Done or mostly done:
- FFI is the primary training backend
- ball is spawned far for walk-first training
- Gymnasium wrapper exists

### Stage 1: Stable walk baseline
Primary baseline:
- `PPO`
- `Gymnasium + FFI`
- residual control over the existing gait prior
- fixed walking command first

Observation:
- torso pitch
- torso angular velocity
- root horizontal velocity
- gait phase `sin(phi), cos(phi)`
- joint relative angles
- joint angular velocities
- contact indicators
- previous action

Action:
- 4 continuous joint target offsets
- PD tracking remains in the simulator

Reward:
- forward velocity tracking
- alive bonus
- upright torso
- action smoothness
- effort / torque penalty
- fall penalty

Success criteria:
- positive walking distance both left and right
- lower fall rate than current KNP baseline
- reduced torso oscillation

### Stage 2: Human-like gait shaping
Only after Stage 1 is stable:
- alternating contact reward
- swing foot clearance reward
- no-leg-crossing penalty
- step timing consistency
- torso height stability
- symmetry term between left/right gait statistics

Do not over-weight these before walking is already stable.

### Stage 3: Speed control
After stable baseline:
- train on a command curriculum over forward speed
- include stop, slow walk, medium walk, faster walk
- measure tracking error and transition smoothness

### Stage 4: Recovery
After speed control:
- push perturbations while upright
- partial recovery from large torso pitch excursions
- full get-up only later if morphology/contact supports it

### Stage 5: Style prior
Only after stable walk exists:
- DeepMimic-style tracking reward
or
- AMP-style discriminator reward

This stage should improve human-likeness, not create locomotion from nothing.

### Stage 6: Ball interaction
Only after walking and speed control are solved:
1. approach target while walking
2. approach moving target
3. single contact / touch
4. controlled kick
5. dribble-like repeated interaction

## Metrics to track
- episode length
- fall rate
- right-direction walking distance
- left-direction walking distance
- min walking distance across directions
- torso pitch RMS
- torso height variance
- action magnitude
- contact alternation quality
- symmetry score
- leg-crossing events

## Immediate execution plan
1. Train `SB3 PPO` on Gymnasium + FFI with residual walk
2. Compare against current `KNP` and `SB3 SAC`
3. If PPO beats SAC but not KNP:
   move PPO to staged curriculum
4. If PPO becomes the best stable learner:
   keep PPO as baseline and add human-like shaping

## Decision rule
- If a method does not improve worst-case left/right walking distance, it is not the mainline.
- Mainline training stays with the best stable baseline.
- Human-likeness improvements only continue on top of the best stable baseline.
