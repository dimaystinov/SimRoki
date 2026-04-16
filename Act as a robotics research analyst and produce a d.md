<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Act as a robotics research analyst and produce a deep technical literature review focused on humanoid robot locomotion learning.

My goal:
I want to understand the best scientifically grounded methods for training a humanoid robot to walk in a stable, human-like way. I care not only about “not falling”, but about realistic human-like gait: upright torso, natural step timing, no leg crossing, acceleration, deceleration, speed changes, recovery after falls, and later running, jumping, and interacting with a ball (approach / kick / dribble).

My current practical context:
Right now I have a simplified planar 5-link biped in simulation:

- torso
- left thigh
- left shin
- right thigh
- right shin
- 4 actuated joints: left/right hip and left/right knee
- no ankles yet
- physics simulation
- RL from Python
Later I may extend this toward a more complete humanoid.

I want you to find the best scientific and technical sources for this problem, including:

- peer-reviewed papers
- strong arXiv papers
- technical reports
- company research if serious and technical

Please prioritize sources from or related to:

- Boston Dynamics / Hyundai
- Tesla Optimus
- Agility Robotics
- Figure
- Unitree
- NVIDIA
- DeepMind / Google
- ETH Zurich
- CMU
- Berkeley
- Stanford
- MIT
- Tsinghua
- other strong humanoid locomotion groups

Important:
I do NOT want marketing summaries.
I want real scientific/technical material.
If a company has no formal paper, include the most technical source available, but clearly mark it as lower scientific confidence.

Please answer in the following structure:

1. Executive summary

- What are currently the best methods for humanoid gait learning?
- Which methods are best for:
    - stable walking
    - human-like walking
    - adaptive walking with speed changes
    - recovery after falls
    - running / jumping
    - ball interaction while walking

2. Must-read source list
Give me 20–35 of the most important sources.
For each source include:

- full title
- authors
- year
- venue
- direct link
- direct PDF/arXiv link if possible
- source type: peer-reviewed / arXiv / tech report / company
- short summary (3–6 sentences)
- what exact problem it solves
- whether results are in simulation only or on a real robot
- why it matters for humanoid gait learning

3. Method taxonomy
Compare these approaches specifically for humanoid locomotion:

- end-to-end reinforcement learning
- imitation learning from motion capture / demonstrations
- model-based control
- MPC
- whole-body control
- trajectory optimization
- hybrid zero dynamics
- phase-based gait controllers
- inverse kinematics + inverse dynamics
- residual RL on top of classical controllers
- privileged learning / teacher-student
- hierarchical RL
- locomotion with reference motion tracking
- sim-to-real pipelines
For each method explain:
- strengths
- weaknesses
- data efficiency
- stability
- human-likeness
- sim-to-real potential
- whether it is suitable for a simplified 5-link planar biped
- whether it is suitable for a full humanoid

4. Human-like gait specifically
I want a section focused on how to achieve human-like gait, not just stable gait.
Please find sources on:

- gait phase structure
- double support / single support
- torso posture
- foot placement
- cadence / step length / step timing
- acceleration and deceleration
- recovery behavior
- biomechanics-inspired reward design
- motion imitation from human walking data
Explain:
- which features of human gait matter most
- which of those can be transferred to robots
- which papers best connect biomechanics and control / RL

5. Best papers for my exact case
I have a 2D planar 5-link biped with only hips and knees for now.
Please identify which methods are most realistic and effective for THIS setup.
I want:

- the best methods for a simplified biped first
- how to scale them later to a full humanoid
- which papers are directly usable now
- which papers are more long-term

6. Recommended training recipe
Based on the literature, propose the best practical training stack for my project.
I want a concrete recommendation, not just options.
Please include:

- controller architecture
- whether to use classical gait controller + residual RL
- whether to use imitation learning
- whether to use SAC / PPO / other methods
- whether to use Gymnasium-compatible setup
- observation design
- action design
- reward design
- curriculum design
- fall recovery design
- speed control design
- when to introduce ball interaction
- when to introduce running / jumping
- how to evaluate success

7. Ranking
Give me a ranked list of the top 5 overall methods for my case.
For each method explain:

- why it ranks there
- expected difficulty
- expected training cost
- expected stability
- expected human-likeness
- expected scalability to a full humanoid

8. Final recommendation
Give me one final best-method recommendation for my project:

- what to build first
- what to train second
- what to avoid
- what is most likely to actually work in practice

Requirements:

- Prefer sources from 2018–2026, but include older foundational papers if essential.
- Be explicit when something is established science versus speculative.
- Separate “works in simulation” from “works on real hardware”.
- Include direct links whenever possible.
- Do not hallucinate papers.
- If multiple papers are variants of the same idea, tell me which one is the most important to read first.
- If possible, include a small comparison table.

At the end, give me:

- a “minimum viable reading list” of 8 papers
- an “advanced reading list” of 15 more papers
- a “practical implementation roadmap” for my simulator

Here's the deep technical literature review on humanoid robot locomotion learning. It covers all 8 sections you requested with full citation tracing. Let me give you a direct summary of the most actionable findings:

***

## Key Findings at a Glance

**The current field consensus** is: **PPO in massively parallel GPU simulation + motion imitation style rewards (AMP/DeepMimic) + privileged teacher–student for sim-to-real**. This triad beats any single approach.[^1][^2][^3]

**For your 5-link planar biped right now**, the single best practical recommendation is: **PPO + phase clock in observation + biomechanically-grounded reward shaping**, specifically implementing the minimal reward function validated on real hardware in  — velocity tracking + single-support reward (with grace period) + air-time reward + torque penalty. This is the lowest-risk path to stable and reasonably natural gait without any mocap or classical gait generator.[^4]

**The most important single paper to read first** is *Revisiting Reward Design for Robust Humanoid Standing and Walking*  — it tells you exactly what to put in the reward function, validated on a real robot. Second is *AMP*  for why adversarial style rewards beat hand-crafted naturalness terms at scale.[^5][^4]

***

The full report covers:

- **30+ annotated sources** with full citation details, direct arXiv/PDF links, sim vs. real classification
- **Complete method taxonomy table** (14 approaches × 7 dimensions)
- **Biomechanics → reward design** mapping (which human gait features matter, which can be transferred)
- **5-link biped vs. full humanoid** suitability for each method
- **6-phase training recipe** with concrete reward formulas, architecture choices, and curriculum logic
- **Ball interaction, running, and fall recovery** integration timelines
- **Minimum viable (8 papers)** and **advanced (15 papers)** reading lists
- **Month-by-month implementation roadmap** with validation milestones
<span style="display:none">[^10][^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^11][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^12][^120][^121][^122][^123][^124][^125][^126][^127][^128][^129][^13][^130][^131][^132][^133][^134][^135][^136][^137][^138][^139][^14][^140][^141][^142][^143][^144][^145][^146][^147][^148][^149][^15][^150][^151][^152][^153][^154][^155][^156][^157][^158][^159][^16][^160][^161][^162][^163][^164][^165][^166][^167][^168][^169][^17][^170][^171][^172][^173][^174][^175][^176][^177][^178][^179][^18][^180][^181][^182][^183][^184][^185][^186][^187][^188][^189][^19][^190][^191][^192][^193][^194][^195][^196][^197][^198][^199][^20][^200][^201][^202][^203][^204][^205][^206][^207][^208][^209][^21][^210][^211][^212][^213][^214][^215][^216][^217][^218][^219][^22][^220][^221][^222][^223][^224][^225][^226][^227][^228][^229][^23][^230][^231][^232][^233][^234][^235][^236][^237][^238][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^6][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^7][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^8][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^9][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://www.figure.ai/news/reinforcement-learning-walking

[^2]: https://proceedings.mlr.press/v164/rudin22a/rudin22a.pdf

[^3]: https://www.science.org/doi/10.1126/scirobotics.adi9579

[^4]: https://www.reddit.com/r/reinforcementlearning/comments/qwrz9c/is_human_like_walking_obtainable_with_ppo_and_a/

[^5]: https://arxiv.org/html/2404.17070v1

[^6]: http://proceedings.mlr.press/v100/xie20a/xie20a.pdf

[^7]: https://dl.acm.org/doi/10.1145/3197517.3201311

[^8]: https://arxiv.org/abs/2104.02180

[^9]: https://arxiv.org/abs/2509.19023

[^10]: https://arxiv.org/html/2505.20619v1

[^11]: https://arxiv.org/html/2502.12152v1

[^12]: https://arxiv.org/html/2502.20061v2

[^13]: https://arxiv.org/html/2511.07407v1

[^14]: https://hugwbc.github.io/resources/HugWBC.pdf

[^15]: https://arxiv.org/html/2509.20696v1

[^16]: https://arxiv.org/html/2505.12679v1

[^17]: https://xbpeng.github.io/projects/DeepMimic/index.html

[^18]: https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf

[^19]: https://dl.acm.org/doi/10.1145/3476576.3476723

[^20]: https://dl.acm.org/doi/10.1145/3450626.3459670

[^21]: https://arxiv.org/abs/2205.01906

[^22]: https://arxiv.org/abs/2109.11978

[^23]: https://arxiv.org/pdf/2402.19469.pdf

[^24]: https://arxiv.org/abs/2402.19469

[^25]: https://microsoft.github.io/MoCapAct/

[^26]: https://ieeexplore.ieee.org/document/9981973/

[^27]: https://arxiv.org/abs/2203.15103

[^28]: https://gwern.net/doc/reinforcement-learning/meta-learning/2022-miki.pdf

[^29]: https://www.research-collection.ethz.ch/bitstreams/39226ce8-b331-49db-930a-e43a811ff433/download

[^30]: https://leggedrobotics.github.io/rl-blindloco/

[^31]: https://ar5iv.labs.arxiv.org/html/2208.07363

[^32]: https://www.microsoft.com/en-us/research/blog/mocapact-training-humanoid-robots-to-move-like-jagger/

[^33]: https://www.themoonlight.io/en/review/gait-conditioned-reinforcement-learning-with-multi-phase-curriculum-for-humanoid-locomotion

[^34]: http://arxiv.org/pdf/2401.16889.pdf

[^35]: https://arxiv.org/html/2509.19023

[^36]: http://arxiv.org/pdf/2309.14225.pdf

[^37]: https://arxiv.org/html/2404.19173v1

[^38]: https://arxiv.org/html/2502.12152v2

[^39]: https://techxplore.com/news/2025-02-humanoid-robots-swiftly-fall-framework.html

[^40]: https://arxiv.org/html/2412.03949v1

[^41]: https://arxiv.org/abs/2512.07248

[^42]: https://arxiv.org/html/2512.07248v1

[^43]: https://arxiv.org/html/2404.17070v5

[^44]: https://www.frontiersin.org/articles/10.3389/frobt.2025.1678567/full

[^45]: https://ieeexplore.ieee.org/document/11404507/

[^46]: http://arxiv.org/pdf/2103.14295.pdf

[^47]: https://arxiv.org/html/2602.21666v1

[^48]: https://arxiv.org/abs/2512.01996

[^49]: https://grizzle.robotics.umich.edu/files/Grizzle_Westervelt_HZD_IsidoriFest.pdf

[^50]: https://arxiv.org/abs/1910.01748

[^51]: https://arxiv.org/pdf/2311.02496.pdf

[^52]: http://www.roboticsproceedings.org/rss14/p54.pdf

[^53]: https://www.roboticsproceedings.org/rss14/p54.pdf

[^54]: https://arxiv.org/html/2405.17227v1

[^55]: https://jmesopen.com/index.php/jmesopen/article/view/25

[^56]: https://cyberboticslab.com/publication/hereid-2017-frost/hereid-2017-frost.pdf

[^57]: https://arxiv.org/abs/2402.06783

[^58]: https://arxiv.org/html/2504.13619v1

[^59]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8334844/

[^60]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5222530/

[^61]: https://ieeexplore.ieee.org/document/11349020/

[^62]: https://farama.org/Gymnasium-MuJoCo-v5_Environments

[^63]: https://www.sfu.ca/~kabhishe/posts/posts/summary_tog_deepmimic_2018/

[^64]: https://arxiv.org/html/2502.08378v1

[^65]: https://arxiv.org/abs/2506.12851

[^66]: https://dl.acm.org/doi/10.1145/3747865

[^67]: https://ieeexplore.ieee.org/document/11419765/

[^68]: https://dl.acm.org/doi/10.1145/3756884.3768416

[^69]: https://www.mdpi.com/2227-7390/13/21/3445

[^70]: https://arxiv.org/abs/2310.06226

[^71]: http://arxiv.org/pdf/2405.08726.pdf

[^72]: https://arxiv.org/abs/2407.11658

[^73]: https://arxiv.org/pdf/2310.06226.pdf

[^74]: https://arxiv.org/pdf/2309.09167.pdf

[^75]: http://arxiv.org/pdf/2402.16796v1.pdf

[^76]: https://arxiv.org/html/2312.09757v1

[^77]: https://www.sciencedirect.com/science/article/abs/pii/S0952197624020785

[^78]: https://github.com/YanjieZe/awesome-humanoid-robot-learning

[^79]: https://arxiv.org/html/2504.14305v3

[^80]: https://www.ijcai.org/Proceedings/07/Papers/336.pdf

[^81]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12975443/

[^82]: https://github.com/kuds/mesozoic-labs

[^83]: https://techxplore.com/news/2026-02-humanoid-robots-falling-algorithm.html

[^84]: https://discovery.researcher.life/topic/motion-imitation/15884942?page=1\&topic_name=Motion+Imitation

[^85]: https://research.tue.nl/files/199497309/Zhou_F.pdf

[^86]: https://ieeexplore.ieee.org/document/10752370/

[^87]: https://arxiv.org/abs/2505.12619

[^88]: https://www.semanticscholar.org/paper/89850513826304a0a24cd8ff3bfa8be227d897b6

[^89]: https://dl.acm.org/doi/10.1145/3099564.3099567

[^90]: https://dl.acm.org/doi/10.1145/3610548.3618187

[^91]: https://www.semanticscholar.org/paper/029128fc05ef1a1b6a4094703ab87071edbe7b0b

[^92]: https://arxiv.org/pdf/2502.10980.pdf

[^93]: http://arxiv.org/pdf/2405.01284.pdf

[^94]: https://arxiv.org/html/2503.14637v1

[^95]: https://arxiv.org/pdf/2109.13338.pdf

[^96]: https://arxiv.org/pdf/1803.03719.pdf

[^97]: https://arxiv.org/pdf/2402.05421.pdf

[^98]: https://github.com/xbpeng/DeepMimic

[^99]: https://www.semanticscholar.org/paper/DeepMimic-Peng-Abbeel/1b9ce6abc0f3024b88fcd4dbd0c10cf5bcf7d38d

[^100]: https://arxiv.org/html/2502.10894v1

[^101]: https://arxiv.org/html/2505.04961v1

[^102]: https://arxiv.org/html/2504.06585v1

[^103]: https://dl.acm.org/doi/10.1145/3274247.3274506

[^104]: https://www.youtube.com/watch?v=8sO7VS3q8d0

[^105]: https://hulks.de/_files/PA_Luis-Scheuch.pdf

[^106]: https://ar5iv.labs.arxiv.org/html/2109.11978

[^107]: https://ieeexplore.ieee.org/document/9362343/

[^108]: https://kilthub.cmu.edu/articles/Geometric_Control_and_Learning_for_Dynamic_Legged_Robots/12181962/1

[^109]: https://ieeexplore.ieee.org/document/11049016/

[^110]: https://arxiv.org/abs/2507.13662

[^111]: https://ieeexplore.ieee.org/document/10000225/

[^112]: https://arxiv.org/abs/2207.07835

[^113]: https://www.semanticscholar.org/paper/c86fdb8d0c7ff36f3f3251a3bb209d84a91f1d8c

[^114]: https://www.semanticscholar.org/paper/1b44f0ab0e8793e6a81a622276b7168d62e6c5b1

[^115]: http://hdl.handle.net/20.500.11850/272432

[^116]: https://arxiv.org/html/2403.15993v2

[^117]: https://arxiv.org/pdf/2403.02486.pdf

[^118]: http://arxiv.org/pdf/2309.13172.pdf

[^119]: http://arxiv.org/pdf/1807.09905.pdf

[^120]: https://journals.sagepub.com/doi/pdf/10.1177/02783649221102473

[^121]: http://arxiv.org/pdf/1802.03498.pdf

[^122]: https://hybrid-robotics.berkeley.edu/publications/ICRA2022_Bayesian-Optimization_Cassie-Walking.pdf

[^123]: https://par.nsf.gov/servlets/purl/10565066

[^124]: https://arxiv.org/pdf/2302.09450.pdf

[^125]: http://ames.caltech.edu/reher2019dynamic.pdf

[^126]: https://www.youtube.com/watch?v=tUMcLbCjr0w

[^127]: https://arxiv.org/html/2402.19469v1

[^128]: https://www.notateslaapp.com/news/2732/tesla-engineers-reveal-how-optimus-learns-and-show-off-its-dance-moves-video

[^129]: https://grizzle.robotics.umich.edu/files/Rapid_Bipedal_Gait_Design_Using_C_FROST__IROS.pdf

[^130]: https://www.linkedin.com/pulse/teslas-optimus-robot-learning-from-youtube-videos-video-based-baek-ws4xc

[^131]: https://proceedings.neurips.cc/paper_files/paper/2024/file/90afd20dc776bc8849c31d61a0763a0b-Paper-Conference.pdf

[^132]: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/5BE9A427CDC882D0B006E6F1F56A0A1A/S0263574721001363a.pdf/variable-time-interval-trajectory-optimization-based-dynamic-walking-control-of-bipedal-robot.pdf

[^133]: https://www.youtube.com/watch?v=cPG5XPDAlF0

[^134]: https://ieeexplore.ieee.org/document/11203191/

[^135]: https://arxiv.org/abs/2407.02282

[^136]: https://www.mdpi.com/2076-3417/16/7/3448

[^137]: https://ieeexplore.ieee.org/document/10167753/

[^138]: https://arxiv.org/abs/2304.10888

[^139]: https://arxiv.org/abs/2510.09543

[^140]: https://ieeexplore.ieee.org/document/11298411/

[^141]: https://ieeexplore.ieee.org/document/11346990/

[^142]: https://arxiv.org/abs/2203.14912

[^143]: https://arxiv.org/pdf/2305.18743.pdf

[^144]: https://arxiv.org/pdf/2206.13142.pdf

[^145]: http://arxiv.org/pdf/2407.02282.pdf

[^146]: https://arxiv.org/pdf/2401.05018.pdf

[^147]: http://arxiv.org/pdf/2403.04954.pdf

[^148]: https://arxiv.org/html/2410.03246v2

[^149]: https://xbpeng.github.io/projects/AMP_Locomotion/index.html

[^150]: https://www.youtube.com/watch?v=wySUxZN_KbM

[^151]: https://www.emergentmind.com/topics/adversarial-motion-priors-amp

[^152]: https://www.research-collection.ethz.ch/entities/publication/95c84760-2274-43ee-9cda-efa913804d0b

[^153]: https://crl.ethz.ch/papers/Learning_to_Walk_in_Costume__Adversarial_Motion_Priors_for_Aesthetically_Constrained_Humanoids___Humanoids_2025.pdf

[^154]: https://escholarship.org/content/qt9rh2t6tt/qt9rh2t6tt.pdf

[^155]: https://www.ias.informatik.tu-darmstadt.de/uploads/Teaching/HumanoidRoboticsSeminar/hr_lu_liu.pdf

[^156]: https://arxiv.org/html/2405.01792v1

[^157]: https://arxiv.org/html/2506.20487v5

[^158]: https://www.semanticscholar.org/paper/fda19cfce6ff2e146b1aeef88c422c5772ed7b0d

[^159]: https://arxiv.org/abs/2602.15827

[^160]: https://arxiv.org/abs/2602.02960

[^161]: https://arxiv.org/pdf/2305.17109.pdf

[^162]: http://arxiv.org/pdf/1705.08439.pdf

[^163]: http://arxiv.org/pdf/1703.02710.pdf

[^164]: https://arxiv.org/html/2410.09163v2

[^165]: https://arxiv.org/pdf/2502.13569.pdf

[^166]: https://arxiv.org/pdf/2309.02976.pdf

[^167]: https://arxiv.org/pdf/2310.06606.pdf

[^168]: https://arxiv.org/pdf/1902.06007.pdf

[^169]: https://arxiv.org/html/2601.12799v1

[^170]: https://spj.science.org/doi/10.34133/research.1123

[^171]: https://github.com/robotlearning123/awesome-isaac-gym

[^172]: https://arxiv.org/html/2309.14594v2

[^173]: https://arxiv.org/html/2507.08303v3

[^174]: https://developer.nvidia.com/isaac-gym

[^175]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9680482/

[^176]: https://arxiv.org/html/2506.22827v1

[^177]: https://www.edge-ai-vision.com/2025/04/r²d²-advancing-robot-mobility-and-whole-body-control-with-novel-workflows-and-ai-foundation-models-from-nvidia-research/

[^178]: https://www.sciencedirect.com/science/article/pii/S258900422500464X

[^179]: https://github.com/InternRobotics/HoST

[^180]: https://simulately.wiki/docs/simulators/IsaacGym/

[^181]: https://arxiv.org/pdf/1909.08124.pdf

[^182]: https://arxiv.org/pdf/2308.10962.pdf

[^183]: https://arxiv.org/html/2406.13115v1

[^184]: https://www.frontiersin.org/articles/10.3389/frobt.2018.00058/pdf

[^185]: https://arxiv.org/pdf/2209.02995.pdf

[^186]: http://www.ijimai.org/journal/node/1878

[^187]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11499933/

[^188]: https://scholar.google.fi/citations?user=Ul2F7OwAAAAJ\&hl=vi

[^189]: http://ames.caltech.edu/HZD_bookchapter.pdf

[^190]: https://rsl.ethz.ch/the-lab/news/2020/10/learning-locomotion-over-challenging-terrain.html

[^191]: https://arxiv.org/html/2504.01165v1

[^192]: https://www.youtube.com/watch?v=QDU_FicBPDo

[^193]: https://fibo.kmutt.ac.th/wp-content/uploads/2025/06/2012_Amnart_Study-of-Teen-Sized-Humanoid-Robot-Kicking-Behavior.pdf

[^194]: https://search.proquest.com/openview/491b0f122fb7c36f989589731847958d/1?pq-origsite=gscholar\&cbl=18750\&diss=y

[^195]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12292580/

[^196]: http://arxiv.org/pdf/2405.15677.pdf

[^197]: https://arxiv.org/html/2503.09015v1

[^198]: https://arxiv.org/html/2407.02633

[^199]: http://arxiv.org/pdf/2111.12602.pdf

[^200]: https://arxiv.org/pdf/2502.15956.pdf

[^201]: https://arxiv.org/pdf/2103.14274.pdf

[^202]: https://neurips.cc/virtual/2024/poster/95871

[^203]: https://humanoid-next-token-prediction.github.io

[^204]: https://arxiv.org/html/2505.18429v1

[^205]: https://techtransfer.universityofcalifornia.edu/NCD/33980.html

[^206]: https://www.nature.com/articles/s41598-024-79292-4

[^207]: https://github.com/patrick-llgc/Learning-Deep-Learning/blob/master/paper_notes/locomotion_next_token_pred.md

[^208]: https://huggingface.co/datasets/microsoft/mocapact-data

[^209]: https://arxiv.org/pdf/2207.12644.pdf

[^210]: http://arxiv.org/pdf/2109.12665.pdf

[^211]: https://arxiv.org/html/2502.08844v1

[^212]: http://arxiv.org/pdf/1908.05224.pdf

[^213]: https://arxiv.org/pdf/2112.06061.pdf

[^214]: https://arxiv.org/pdf/2501.16590.pdf

[^215]: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v5.py

[^216]: https://www.nature.com/articles/s44182-025-00043-2.pdf

[^217]: https://playground.mujoco.org/assets/playground_technical_report.pdf

[^218]: https://github.com/ProfessorNova/PPO-Humanoid

[^219]: https://www.roboticsproceedings.org/rss21/p020.pdf

[^220]: https://stable-baselines3.readthedocs.io/en/master/misc/projects.html

[^221]: https://groups.inf.ed.ac.uk/advr/papers/reinforcement_learning_for_robot_locomotion.pdf

[^222]: https://www.reddit.com/r/reinforcementlearning/comments/12acvnh/stable_baselines_3_ppo_proper_rewarding/

[^223]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-87.pdf

[^224]: https://www.raphaelcousin.com/courses/data-science-practice/project/bipedal-walker

[^225]: https://arxiv.org/html/2410.08655v1

[^226]: https://arxiv.org/html/2411.04408v1

[^227]: https://arxiv.org/pdf/2302.13137.pdf

[^228]: https://arxiv.org/html/2503.04969v1

[^229]: http://arxiv.org/pdf/2502.17322.pdf

[^230]: https://arxiv.org/html/2502.03122v1

[^231]: https://humanoid-getup.github.io

[^232]: https://arxiv.org/abs/2512.12230

[^233]: https://www.themoonlight.io/en/review/learning-getting-up-policies-for-real-world-humanoid-robots

[^234]: https://arxiv.org/html/2310.01408v3

[^235]: https://huggingface.co/papers/2502.12152

[^236]: https://motusnova.com/wp-content/uploads/2021/04/2005-Bharadwaj-K.-Sugar-T.-G.-Koeneman-J.-B.-Koeneman-E.-J.-Design-of-a-Robotic-Gait-Trainer-using-Spring-Over-Muscle-Actuators-for-Ankle-Stroke-Rehabilitation.-Journal-of-Biomechanical-Engineering.pdf

[^237]: https://ui.adsabs.harvard.edu/abs/2025arXiv250212152H/abstract

[^238]: https://www.youtube.com/watch?v=hmV4v_EnB0E

