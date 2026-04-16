# biped_2d_reading_map

## Scope
This map is curated for a **2D planar biped** with:
- torso
- left/right thigh
- left/right shin
- 4 actuated joints: left/right hip and left/right knee
- no ankles yet
- physics simulation + RL from Python

It intentionally filters out most full-humanoid papers unless they teach something directly useful for a planar underactuated biped.

## Read first (directly useful now)

### 1) Hybrid Zero Dynamics of Planar Bipedal Walking
- Authors: Grizzle, Westervelt, et al.
- Year: foundational
- Type: classical theory
- Why read: canonical mathematical framework for **planar biped** gait design; your current model is very close to the standard HZD teaching setup.
- Use now: yes
- Priority: critical
- Link: https://grizzle.robotics.umich.edu/files/Grizzle_Westervelt_HZD_IsidoriFest.pdf

### 2) FROST*: Fast Robot Optimization and Simulation Toolkit
- Authors: Hereid et al.
- Type: toolkit / optimization
- Why read: practical route to generating periodic walking trajectories and virtual-constraint gaits.
- Use now: yes
- Priority: critical
- Link: https://cyberboticslab.com/publication/hereid-2017-frost/hereid-2017-frost.pdf

### 3) Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
- Authors: Rudin et al.
- Year: 2021
- Type: peer-reviewed
- Why read: best reference for modern RL training infrastructure, curriculum, and massive parallelism.
- Use now: yes, especially training stack and curriculum ideas
- Priority: critical
- Link: https://arxiv.org/abs/2109.11978
- PDF: https://proceedings.mlr.press/v164/rudin22a/rudin22a.pdf

### 4) Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking
- Authors: van Marum et al.
- Year: 2024
- Type: arXiv / technical research
- Why read: one of the cleanest reward-design papers; especially useful for contact timing, airtime, and velocity tracking.
- Use now: yes
- Priority: critical
- Link: https://arxiv.org/abs/2404.19173

### 5) AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
- Authors: Peng et al.
- Year: 2021
- Type: peer-reviewed
- Why read: strongest route to human-like motion without manually encoding every gait detail.
- Use now: yes, but after baseline PPO works
- Priority: high
- Link: https://arxiv.org/abs/2104.02180

### 6) DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills
- Authors: Peng et al.
- Year: 2018
- Type: peer-reviewed
- Why read: core imitation-learning paper; teaches reference tracking rewards and motion imitation structure.
- Use now: yes, for style-learning ideas
- Priority: high
- Link: https://xbpeng.github.io/projects/DeepMimic/index.html
- PDF: https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf

### 7) Reduced-Order Model-Guided Reinforcement Learning for Demonstration-Free Humanoid Locomotion
- Authors: 2025 paper
- Type: arXiv
- Why read: especially relevant because it uses a reduced-order model to guide RL toward natural locomotion without mocap.
- Use now: yes conceptually
- Priority: high
- Link: https://arxiv.org/abs/2509.19023

### 8) HumanMimic: Learning Natural Locomotion and Transitions for Humanoid Robot via Wasserstein Adversarial Imitation
- Type: arXiv
- Why read: better understanding of how to get natural transitions and more stable adversarial imitation.
- Use now: later
- Priority: medium-high
- Link: https://arxiv.org/abs/2309.14225

### 9) Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion
- Type: arXiv
- Why read: useful for explicit gait phase structure, command-conditioned walking, and later walk-run transitions.
- Use now: yes for curriculum and phase reward routing
- Priority: high
- Link: https://arxiv.org/abs/2505.20619

### 10) Learning Speed-Adaptive Walking Agent Using Imitation Learning with Physics-Informed Simulation
- Type: arXiv
- Why read: direct relevance to acceleration/deceleration and command-conditioned gait changes.
- Use now: later
- Priority: medium
- Link: https://arxiv.org/abs/2412.03949

## Read second (useful when scaling beyond the planar model)

### 11) Real-world humanoid locomotion with reinforcement learning
- Authors: Berkeley / Agility Robotics
- Why read: clean sim-to-real locomotion pipeline; less directly about planar control, more about scaling methodology.
- Link: https://www.science.org/doi/10.1126/scirobotics.adi9579

### 12) Humanoid Locomotion as Next Token Prediction
- Why read: future-facing data-driven policy learning; not the best first method for your current simulator.
- Link: https://arxiv.org/abs/2402.19469

### 13) Learning Getting-Up Policies for Real-World Humanoid Robots
- Why read: later for fall recovery, after walking is solved.
- Link: https://arxiv.org/abs/2502.12152

### 14) RuN: Residual Policy for Natural Humanoid Locomotion
- Why read: later for walk-run transition ideas.
- Link: https://arxiv.org/abs/2509.20696

### 15) Learning Agile Humanoid Dribbling Through Legged Locomotion
- Why read: relevant only after stable velocity-controlled locomotion exists.
- Link: https://arxiv.org/abs/2505.12679

## Suggested order for your exact project
1. HZD planar biped paper
2. FROST toolkit paper
3. Rudin 2021 (parallel RL + curriculum)
4. Reward design paper (Digit)
5. DeepMimic
6. AMP
7. ROM-guided RL
8. Gait-conditioned RL
9. Speed-adaptive walking
10. Fall recovery papers
