# open_papers_fulltext

> Note: this file is an **open-access corpus pack** for Codex. It contains structured metadata, abstracts/available text summaries, and direct PDF links for open-access papers. It is not a universal raw dump of every copyrighted PDF.

## 1) Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
- Authors: Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter
- Year: 2021
- URL: https://arxiv.org/abs/2109.11978
- PDF: https://proceedings.mlr.press/v164/rudin22a/rudin22a.pdf
- Open access: yes
- Core ideas:
  - GPU-parallel RL with thousands of simultaneous environments.
  - Automatic curriculum over terrain difficulty and command difficulty.
  - Demonstrates that locomotion policies can be trained in minutes instead of days.
- Codex notes:
  - For a 2D biped, the key takeaway is not the robot morphology but the training loop.
  - Reuse: vectorized envs, curriculum schedule, command randomization, early termination.

## 2) DeepMimic
- Authors: Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne
- Year: 2018
- URL: https://xbpeng.github.io/projects/DeepMimic/index.html
- PDF: https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf
- Open access: yes
- Core ideas:
  - Motion imitation reward broken into pose, velocity, end-effector, and center-of-mass terms.
  - PPO-based training for physically simulated characters.
  - Strong reference for using a phase variable and motion-tracking reward.
- Codex notes:
  - For 2D bipeds, simplify to pose tracking + torso upright + step timing.
  - Exact full-body imitation is unnecessary at first; use only locomotion-relevant parts.

## 3) AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
- Authors: Xue Bin Peng et al.
- Year: 2021
- URL: https://arxiv.org/abs/2104.02180
- PDF: https://arxiv.org/pdf/2104.02180.pdf
- Open access: yes
- Core ideas:
  - A discriminator learns a style reward from motion clips.
  - The policy receives both task reward and style reward.
  - Avoids hand-crafting many naturalness terms.
- Codex notes:
  - Strong candidate for adding “human-like” gait after basic stability works.
  - In 2D, the discriminator input can be compact: torso angle, hip/knee angles, joint velocities, contact pattern, phase.

## 4) Reduced-Order Model-Guided Reinforcement Learning for Demonstration-Free Humanoid Locomotion
- Year: 2025
- URL: https://arxiv.org/abs/2509.19023
- PDF: http://arxiv.org/pdf/2509.19023.pdf
- Open access: yes
- Core ideas:
  - Train a reduced-order model first.
  - Use it to guide a higher-dimensional locomotion policy.
  - Aims for natural locomotion without mocap.
- Codex notes:
  - Extremely relevant to your case: your current robot is already a reduced locomotion model.
  - You can invert the paper’s philosophy: learn good gait structure in 2D first, then transfer the design principles when scaling up.

## 5) HumanMimic
- URL: https://arxiv.org/abs/2309.14225
- PDF: http://arxiv.org/pdf/2309.14225.pdf
- Open access: yes
- Core ideas:
  - Wasserstein adversarial imitation for more stable training.
  - Focus on natural locomotion and transitions.
- Codex notes:
  - Worth studying if vanilla AMP becomes unstable.

## 6) Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion
- URL: https://arxiv.org/abs/2505.20619
- Open access: yes
- Core ideas:
  - Multi-phase curriculum.
  - Gait-conditioned reward routing.
  - Unified policy over different gait regimes.
- Codex notes:
  - Very relevant for later speed changes and walk/run transition logic.
  - Also useful as inspiration for encoding stance/swing phase in 2D.

## 7) Learning Speed-Adaptive Walking Agent Using Imitation Learning with Physics-Informed Simulation
- URL: https://arxiv.org/abs/2412.03949
- Open access: yes
- Core ideas:
  - Speed-adaptive locomotion using physics-informed data.
  - Connects biomechanics and locomotion learning.
- Codex notes:
  - Useful once your policy can already walk and you want controlled acceleration/deceleration.

## 8) Humanoid Locomotion as Next Token Prediction
- URL: https://arxiv.org/abs/2402.19469
- PDF: https://arxiv.org/pdf/2402.19469.pdf
- Open access: yes
- Core ideas:
  - Treat control as sequence modeling.
  - Train a causal model over motion/control trajectories.
- Codex notes:
  - Not first-line for your simulator, but important for long-term scaling.

## 9) Learning Getting-Up Policies for Real-World Humanoid Robots
- URL: https://arxiv.org/abs/2502.12152
- Open access: yes
- Core ideas:
  - Multi-stage curriculum for getting-up behavior.
- Codex notes:
  - Read only after stable walking exists.

## 10) Learning Agile Humanoid Dribbling Through Legged Locomotion
- URL: https://arxiv.org/abs/2505.12679
- Open access: yes
- Core ideas:
  - Two-stage curriculum: locomotion first, ball interaction second.
- Codex notes:
  - Excellent template for adding a ball later.
