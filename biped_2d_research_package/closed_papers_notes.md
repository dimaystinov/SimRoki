# closed_papers_notes

## Purpose
This file captures the most important **closed or partially closed** sources with detailed notes so Codex can still use the main ideas without needing raw full text.

## 1) Real-world humanoid locomotion with reinforcement learning
- Venue: Science Robotics, 2024
- URL: https://www.science.org/doi/10.1126/scirobotics.adi9579
- Access status: publisher page; not safe to assume full text is openly extractable in this workflow.
- Why it matters:
  - One of the cleanest real-world demonstrations of RL humanoid walking.
  - Shows that robust locomotion can emerge from relatively clean proprioceptive policy design plus sim randomization.
- Main takeaways:
  - Policy runs at a lower rate than the inner motor loop.
  - Domain randomization and latency handling matter a lot.
  - For your 2D simulator, the most reusable parts are observation design, command-conditioned training, and reward simplicity.
- Use for your project:
  - Keep policy outputs simple.
  - Decouple high-level policy rate from low-level actuator integration.
  - Add latency/randomization before scaling to a larger robot.

## 2) Natural Humanoid Walk Using Reinforcement Learning (Figure AI)
- URL: https://www.figure.ai/news/reinforcement-learning-walking
- Access status: company technical post, not peer-reviewed
- Scientific confidence: lower than peer-reviewed papers, but technically informative
- Why it matters:
  - Clear industry description of how “naturalness” is engineered, not just stability.
- Main takeaways:
  - Human walking references are used as style priors.
  - Reward balances command tracking, power/efficiency, and style.
  - Zero-shot sim-to-real is a central target.
- Use for your project:
  - Treat naturalness as a separate objective from stability.
  - Keep a modular path to later add imitation or style rewards.

## 3) DeepMimic / AMP downstream papers behind paywalls or partial access
- Note:
  - Even when the central arXiv versions are open, some venue pages are closed.
  - Prefer arXiv/open copies whenever possible.
- Use for your project:
  - Do not build your corpus around publisher pages if an arXiv version exists.

## 4) Real-hardware humanoid papers with limited open text
- Typical limitation:
  - Hardware papers often provide abstract + media + partial descriptions but not machine-friendly full text.
- Practical recommendation:
  - Store metadata + extracted notes + direct DOI/arXiv links.
  - Let Codex use the notes plus your implementation docs instead of trying to ingest every publisher PDF.
