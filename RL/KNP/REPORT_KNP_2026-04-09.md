# KNP Training Report

Date:
`2026-04-09`

## Run setup

Environment:

- simulator: desktop `native_app`
- training method: `KNP-style spiking controller`
- Python env:
  `C:\Users\root\Documents\New project\RL\KNP\.conda`

Training command used:

```powershell
& "C:\Users\root\Documents\New project\RL\KNP\.conda\python.exe" `
  "C:\Users\root\Documents\New project\RL\KNP\knp_walk_kick_train.py" `
  --episodes 80 `
  --max-steps 220 `
  --repeat-steps 2 `
  --action-scale-deg 24 `
  --hidden-size 64 `
  --log-path "C:\Users\root\Documents\New project\RL\KNP\knp_walk_kick_training_apr09.json"
```

Artifacts:

- training log:
  [knp_train_run_apr09.log](C:/Users/root/Documents/New%20project/RL/KNP/knp_train_run_apr09.log)
- training history:
  [knp_walk_kick_training_apr09.json](C:/Users/root/Documents/New%20project/RL/KNP/knp_walk_kick_training_apr09.json)
- saved best KNP policy:
  [knp_walk_kick_best.npz](C:/Users/root/Documents/New%20project/RL/KNP/knp_walk_kick_best.npz)
- benchmark comparison:
  [walk_policy_comparison.json](C:/Users/root/Documents/New%20project/RL/KNP/walk_policy_comparison.json)

## KNP training result

Episodes:

- `80`

Aggregate training stats:

- `best_reward = 53.3453`
- `best_robot_dx = 0.3668 m`
- `best_ball_dx_world = 0.0000 m`
- `mean_reward_all = 10.3134`
- `mean_robot_dx_all = 0.0329 m`
- `mean_ball_dx_all = -0.8595 m`
- `last10_mean_reward = 9.8969`
- `last10_mean_robot_dx = -0.0460 m`
- `last10_mean_ball_dx = -0.8847 m`

Directional means during training:

- right episodes:
  - `mean_reward = 5.4108`
  - `mean_robot_dx = 0.3103 m`
  - `mean_ball_dx = 0.0000 m`
- left episodes:
  - `mean_reward = 15.2160`
  - `mean_robot_dx = -0.2445 m`
  - `mean_ball_dx = -1.7190 m`

Interpretation:

- KNP learned a short and fairly repeatable forward drift to the right.
- On left-direction episodes it remained unstable and mostly moved the wrong way.
- Ball interaction under KNP remained weak compared with the model-based baseline.
- The end of training did not show convergence toward a stable two-sided gait.

## Fresh benchmark after training

`KMP`

- right:
  - `robot_dx = 0.2355 m`
  - `ball_dx = 4683.2026 m`
- left:
  - `robot_dx = 0.6508 m`
  - `ball_dx = 4683.2026 m`
- worst-direction summary:
  - `min_robot_dx = 0.2355 m`
  - `min_ball_dx = 4683.2026 m`

`KNP`

- right:
  - `reward = 23.9690`
  - `robot_dx = 1.9480 m`
  - `ball_dx = 1.5750 m`
- left:
  - `reward = 6.5931`
  - `robot_dx = -0.3969 m`
  - `ball_dx = 1.5732 m`
- worst-direction summary:
  - `min_robot_dx = -0.3969 m`
  - `min_ball_dx = 1.5732 m`

`PPO`

- worst-direction summary:
  - `min_robot_dx = -0.6010 m`
  - `min_ball_dx = 1.5732 m`

`SAC`

- worst-direction summary:
  - `min_robot_dx = 0.0123 m`
  - `min_ball_dx = 1.5732 m`

## Comparison conclusion

Current ranking by robust two-sided walking:

1. `KMP`
2. `SAC`
3. `KNP`
4. `PPO`

Current ranking by ball strike distance:

1. `KMP`
2. `KNP / PPO / SAC` are far behind and effectively do not compete on kick distance

Main conclusion:

- `KNP` is implemented and trainable, but on the current environment it is not the best controller for stable two-sided walking.
- `KNP` beats `PPO` on worst-direction walking in this run, but still fails the “universal stable gait” requirement because left-direction performance remains negative.
- `SAC` is already more balanced than KNP in the worst direction, though still much weaker than KMP.
- `KMP` remains the best practical controller right now because it combines:
  - positive motion in both directions
  - strong recovery behavior
  - very strong ball kicking

## Recommended next step

If the goal is to improve `KNP` further, the highest-value next change is:

- train KNP as a residual controller on top of `KMP`, not from scratch

That would let KNP learn:

- speed adjustment
- balance correction
- left/right symmetry correction
- kick timing refinement

without forcing it to rediscover the entire locomotion pattern alone.
