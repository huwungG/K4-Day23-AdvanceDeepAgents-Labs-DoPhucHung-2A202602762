# Survey Report: World Models

## TL;DR
- World models are best understood as internal predictive models of an environment that support planning, imagination, and decision-making; the modern framing is rooted in early recurrent model-building and curiosity work and was revived in deep RL by *World Models* (2018). [1][2][3]
- The field has diversified from latent dynamics and recurrent state-space models toward transformer-, JEPA-, and video-generation-based systems, often blending action conditioning, compact latents, and multimodal inputs to improve long-horizon simulation and control. [3][4][5][6][7][8][9]
- Evaluation has shifted away from open-loop visual prediction alone toward closed-loop usefulness: task success, controllability, rollout stability, memory consistency, and decision-making gains are now central. [10][11][12][13][14][15][16]
- In robotics and embodied AI, world models are increasingly treated as world-action models that jointly predict future state and generate actions, with benchmarks and systems stressing sim-to-real transfer and interactive use. [9][16][17][18][19][20]

## Background
The term “world model” is used in at least two closely related ways: as a predictive model of how the environment evolves under actions, and as an internal representation that helps an agent understand and reason about the world. The modern deep-RL usage is usually traced to Ha and Schmidhuber’s *World Models* paper, which showed that an agent can learn a compressed spatio-temporal latent representation and even be trained inside a hallucinated dream before transfer back to the real environment. Earlier IDSIA work from 1990 already described recurrent world models for planning and intrinsic motivation, suggesting that “curiosity” and “mental simulation” were present from the start. [1][2][21][22][3]

Surveys on model-based reinforcement learning place world models inside a broader family of learned dynamics models used to reduce sample complexity through planning or imagined rollouts. Across sources, the core distinction is not just whether the model predicts well, but whether it is useful for downstream decision-making. That split becomes important later in the survey, because many recent systems are optimized less for pixel fidelity and more for controllable interaction, policy improvement, or long-horizon simulation. [21][22][13]

## From latent dynamics to generative simulation
A first major lineage uses latent dynamics models: the environment is compressed into a compact state and the future is predicted in latent space rather than pixel space. Surveys describe this as a major family within model-based RL, and the *World Models* line exemplifies the pattern of learning a compact representation that can be rolled forward for planning. JEPA-style methods push the idea further by predicting representations directly and explicitly avoiding reconstruction of task-irrelevant pixels; EB-JEPA frames this as useful for image learning, video prediction, and action-conditioned world modeling. [2][22][3][6]

A second lineage treats world models as generative simulators, often using video-generation backbones. The survey *Video Generation Models as World Models* explicitly frames video generation as world modeling and identifies efficiency axes spanning modeling paradigms, architectures, and inference algorithms. It also highlights transformer-based video/world models such as VideoGPT, VideoPoet, Genie, and iVideoGPT. In the same spirit, *WorldDreamer* maps visual inputs to discrete tokens and predicts masked tokens with a Transformer, while allowing language and action signals to enter through cross-attention. [4][5][23]

These two lines are converging rather than competing. The sources repeatedly emphasize compact latent spaces, action conditioning, and the use of generative decoders to recover visually coherent outputs. In practice, many recent world models are no longer purely “predict the next frame” systems; they are hybrid predictive-control systems that trade off visual realism, action fidelity, and rollout stability. [3][4][5][6][8][9]

## World-action models for embodied control
Recent embodied AI work is moving from world models as passive predictors to world-action models that also generate actions. The *World Action Models* overview defines this family as embodied foundation models that unify forward predictive modeling with coupled action generation, and it organizes methods into cascaded and joint architectures. Related systems such as OpenWAM and InternW0-Δ emphasize modularity, explicit world-to-action information flow, and large heterogeneous robot-and-human datasets, reflecting a shift toward pretrained dynamics priors that can be adapted for control. [17][18]

Robot-facing systems increasingly aim for transfer across embodiments and tasks. For example, OpenWAM-alpha is reported as pretrained on roughly 6,400 hours of egocentric human and robot data and evaluated on multiple simulation benchmarks plus real-robot experiments, while InternW0-Δ reports more than 20K hours of processed heterogeneous data and strong performance on robot benchmarks. Even systems framed primarily as simulators, such as SyncWorld and A2World, focus on zero-shot simulation, action-conditioned simulation, and policy improvement rather than only video synthesis. [17][18][19]

A second embodied trend is persistent memory and interaction-aware state. ActWorld argues that the navigation-to-interaction gap comes from missing dense interaction data and recency-biased memory, and it responds with action-aware memory and a persistent memory bank. This aligns with newer systems that try to preserve object identity, causal events, and hidden state across longer horizons, which is essential when the agent must act, observe, and revise its plan in the same rollout. [12][19][24]

## Evaluation: from open-loop prediction to closed-loop utility
Evaluation has become one of the sharpest fault lines in the field. Older or simpler protocols often rewarded open-loop visual plausibility, but recent benchmarks argue that visual quality alone is a weak proxy for whether a world model helps an agent act. *How Should World Models Be Evaluated?* makes the decision-making case explicitly, recommending metrics such as interventional action fidelity, rollout validity, policy-ranking agreement, optimization lift, exploitability, and uncertainty calibration. [13]

Several new benchmarks operationalize that shift. *World-in-World* uses a closed-loop planning protocol with proposal, simulation, and revision and reports that task success matters more than visual quality. *WorldRoamBench* introduces four axes—action following, visual drift/stability, interaction physics, and memory—and argues that trajectory-level metrics can hide failures. *MIND*, *WBench*, and *WorldArena* similarly separate perceptual fidelity from functional utility, introducing memory consistency tests, multi-turn interaction metrics, and closed-loop action-planning scores. [10][11][12][14][15][16]

The practical takeaway is that world model quality is now multi-objective. A system can look visually strong but still fail at controllability, memory, or policy usefulness; conversely, a model can support planning while producing imperfect visuals. The benchmark literature is converging on the idea that a world model should be judged by what it enables an agent to do, not only by how realistic its rollouts appear. [10][11][13][14][15][16]

## Trends and open problems
The clearest recent trend is the rise of interactive, action-conditioned, and multimodal world models. Hugging Face and arXiv sources alike emphasize that recent systems learn from unlabeled video, combine state prediction with action generation, and increasingly rely on transformers or diffusion backbones. This is paired with a stronger emphasis on long-horizon interaction, memory, and sim-to-real transfer in robotics. [7][8][9][16][17][18][20]

Open problems remain substantial. First, there is no universal evaluation protocol: benchmarks disagree on whether to prioritize visual fidelity, controllability, memory, or policy utility, and this makes cross-paper comparisons difficult. Second, action granularity and closed-loop deployment are still fragile, especially when models trained on passive video must be used as simulators or planners. Third, uncertainty calibration and generalization under distribution shift remain central issues in model-based RL and in embodied settings. [22][11][13][14][15][16]

A final dispute is conceptual: some sources define world models narrowly as predictive environment simulators, while others extend the term to cognitive brains, latent representations, or full world-action systems. The field is therefore moving in two directions at once—toward more precise benchmarks and engineering, but also toward a broader concept of agents that can predict, imagine, and act in the same internal model. [3][6][17][20]

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018)
[2] Recurrent World Models Facilitate Policy Evolution. arxiv. https://arxiv.org/abs/1809.01999 (2018-09-04)
[3] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. arxiv. https://arxiv.org/abs/2606.00133 (2026-05-28)
[4] Video Generation Models as World Models: Efficient Paradigms, Architectures and Algorithms. arxiv. https://arxiv.org/abs/2603.28489 (2026-07-04)
[5] WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens. arxiv. https://arxiv.org/abs/2401.09985 (2024)
[6] EB-JEPA: open-source library for Joint-Embedding Predictive Architectures for representations and world models. arxiv. https://arxiv.org/abs/2602.03604 (2026)
[7] VideoWorld: Exploring Knowledge Learning from Unlabeled Videos. hf-search. https://huggingface.co/papers/2501.09781 (2025-01-16)
[8] VideoWorld 2: Learning Transferable Knowledge from Real-world Videos. hf-search. https://huggingface.co/papers/2602.10102 (2026-02-10)
[9] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation. hf-search. https://huggingface.co/papers/2511.09057 (2025-11-12)
[10] WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of Interactive World Models. hf-daily. https://huggingface.co/papers/2606.31672 (2026-07-06)
[11] World-in-World: World Models in a Closed-Loop World. hf-daily. https://huggingface.co/papers/2510.18135 (2025-10-20)
[12] World-in-World: World Models in a Closed-Loop World. web. https://world-in-world.github.io/ (2025)
[13] How Should World Models Be Evaluated? A Decision-Making-Centric Position. web. https://arxiv.org/abs/2606.15032 (2026)
[14] MIND: Benchmarking Memory Consistency and Action Control in World Models. web. https://arxiv.org/abs/2602.08025v2 (2026)
[15] WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation. web. https://arxiv.org/abs/2605.25874v1 (2026)
[16] WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models. web. https://arxiv.org/abs/2602.08971v1 (2026)
[17] OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining. arxiv. https://arxiv.org/abs/2609.07398 (2026)
[18] InternW0-Δ: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data. arxiv. https://arxiv.org/abs/2609.31394 (2026)
[19] SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators. arxiv. https://arxiv.org/abs/2609.09155 (2026)
[20] Embodied AI Agents: Modeling the World. hf-search. https://huggingface.co/papers/2506.22355 (2025-07-07)
[21] Model-based Reinforcement Learning: A Survey. arxiv. https://arxiv.org/abs/2006.16712 (2020)
[22] High-Accuracy Model-Based Reinforcement Learning, a Survey. arxiv. https://arxiv.org/abs/2107.08241 (2021)
[23] WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens. web. https://arxiv.org/html/2401.09985 (2024)
[24] Programmable World Model. arxiv. https://arxiv.org/abs/2609.10540 (2026)
