# Reinforcement Learning for LLM Reasoning: A Survey

## TL;DR
- Reinforcement learning became a central post-training tool for LLM reasoning because it can optimize against verifiable outcomes while preserving policy stability with methods like PPO and KL regularization; the classic RLHF pipeline is SFT → reward model → PPO [1][2][3].
- The newest reasoning-focused methods increasingly move from sparse final-answer rewards toward step-level process supervision, verifier-guided rewards, and self-correction loops, reflecting concerns about credit assignment and reward sparsity [4][5][6][7][8].
- DeepSeek-R1 is the clearest modern case study: it combines pure RL and cold-start SFT/RL stages, reports large gains on math/code/instruction benchmarks, and shows that reasoning traces can be distilled into smaller models [9][10][11][12].
- Evaluation remains centered on math, code, and instruction-following benchmarks, with task-specific protocols and sampling settings; strong gains are most consistent on verifiable tasks such as AIME, MATH-500, LiveCodeBench, and Codeforces [9][10].
- In 2025, the field broadened from frontier reasoning quality toward efficiency, cross-domain generalization, and agentic/tool-use training, as shown by HF Daily/Search papers on memory constraints, selective rollouts, entropy effects, and agentic RL [13][14][15][16][17].

## Background
Reinforcement learning entered LLM post-training first as a way to align models with human preferences, not specifically to improve reasoning. The now-standard RLHF recipe is to collect demonstrations and preference rankings, train a reward model, and then optimize the policy with PPO while constraining drift using a KL penalty or clipping [1][2][3]. Surveys of RL-enhanced LLMs describe this as a general post-training framework, while later reasoning surveys argue that the same machinery can be repurposed to search for better reasoning trajectories through trial and error [18][4][5].

For reasoning, the appeal of RL is that it can optimize over outputs that are hard to supervise token-by-token. Instead of only imitating traces, RL can reward success on math, code, or verifiable answer tasks, and process reward models can provide denser feedback on intermediate steps [4][5]. This makes RL especially attractive for multi-step reasoning, where the quality of an entire trajectory matters more than any individual token [4][5].

## From RLHF to reinforced reasoning
Classical RLHF established the mechanics that later reasoning systems inherited. InstructGPT’s pipeline uses supervised fine-tuning to initialize the model, then trains a reward model from human comparisons, then applies PPO to maximize the reward model score [1]. The same paper emphasizes practical safeguards: the policy is nudged toward the SFT model with a per-token KL penalty, and the result is improved helpfulness, truthfulness, and reduced toxicity [1]. PPO itself was attractive because it allows multiple minibatch updates while avoiding overly large policy jumps [2][3].

What changed in reasoning-oriented RL is the reward source and the target behavior. Instead of using broad human preference judgments alone, reasoning work increasingly relies on verifiable outcomes, process rewards, or structured checks [4][5]. The surveys retrieved here explicitly frame RL as a way to discover high-quality reasoning trajectories, while process reward models are highlighted as a response to the sparsity of final-answer-only supervision [4][5].

A second shift is that the strongest reasoning systems often combine RL with staged training rather than relying on one monolithic loop. DeepSeek-R1, for example, uses a pure-RL R1-Zero stage, then adds cold-start reasoning examples for the full R1 model, then mixes rejection sampling and supervised data from a stronger base model before a final RL stage [9][10][11]. This staged recipe suggests that RL is most effective when paired with carefully chosen initialization and synthetic or distilled data [10][11][12].

## Reward design: outcome, process, and self-verification
A major theme across recent work is that outcome-only rewards are too sparse for robust reasoning. HF Search papers on self-verification and process supervision explicitly target this problem: S^2R trains models to self-verify and self-correct through an RL framework, while P2S proposes probabilistic process supervision to provide fine-grained rewards in general-domain reasoning [6][8]. The logic is consistent across methods: if a model only learns whether the final answer was right, it may not learn which intermediate step mattered [6][8].

Verifier-guided approaches push this further by turning intermediate reasoning steps into explicit reward signals. The retrieved HF paper on verifiable process reward models uses deterministic rule-based verifiers to score intermediate steps, and it positions this as an advance over outcome verification alone [7]. The broader survey literature similarly describes process reward models as a key development because they make reward denser and better aligned with multi-step reasoning [4][5].

Self-improvement is the other side of the same coin. Instead of requiring a human or judge for every trajectory, recent methods try to make the model its own verifier or use task transformations that induce self-verifiable rewards [6][8]. This is especially important for open-ended tasks, where explicit correctness checks are harder than in math or code [6]. The research direction here is not just better optimization but also better task formulation: if a task can be transformed into something self-verifiable, RL becomes more scalable [6][7][8].

## Benchmarks and empirical patterns
The benchmark story is remarkably consistent: RL-for-reasoning is mostly evaluated on math, code, and instruction-following tasks that admit some form of automatic or semi-automatic verification [9][10][19]. DeepSeek-R1 is the strongest concrete example in the retrieved sources. Its evaluation suite spans AIME 2024, MATH-500, CNMO 2024, LiveCodeBench, Codeforces, SWE-Bench Verified, Aider-Polyglot, IF-Eval, AlpacaEval 2.0, and ArenaHard, with task-specific protocol choices such as multiple sampled responses for pass@1, CoT formatting for LiveCodeBench, agentless evaluation for SWE-Bench Verified, and final-summary-only judging for open-ended benchmarks [9][10].

The headline scores reported for the final model are substantial: 79.8 on AIME 2024, 97.3 on MATH-500, 65.9 on LiveCodeBench, 96.6 percentile and 2029 rating on Codeforces, 50.8 resolved on SWE-Bench Verified, 61.7 on Aider-Polyglot, 83.3 on IF-Eval, 87.6 on AlpacaEval 2.0, and 92.3 on ArenaHard [9]. The release also reports strong distilled models, such as DeepSeek-R1-Distill-Qwen-32B, which retains much of the reasoning behavior at smaller scale [10][11].

The pattern behind these numbers matters as much as the scores themselves. The retrieved sources repeatedly note that reasoning-oriented RL produces its clearest gains on math and code, where rewards are more verifiable, while instruction-following improves but remains more fragile and may require later SFT or mixed training [9][10]. The DeepSeek materials also show a trade-off: R1-Zero can discover reasoning behaviors through pure RL, but the full R1 pipeline is needed to improve readability and language consistency [10][11].

## Recent trends: efficiency, breadth, and deployment
By 2025, the field had moved beyond “just make the model better at math.” HF Daily/Search items show a strong push toward making RL for reasoning cheaper and broader. One line of work reduces memory and compute costs through parameter-efficient training and selective rollout schemes, suggesting that not every prompt or trajectory deserves the same amount of exploration [13][15]. Another line studies token entropy and argues that minority, high-entropy tokens disproportionately affect reasoning gains, which implies that training could be made more targeted [16].

Another trend is cross-domain generalization. The HF Search summary on cross-domain RL reasoning argues for diverse corpora rather than one-size-fits-all math-only training, indicating that reasoning corpora are being redesigned to cover broader task mixtures [14]. At the same time, agentic reasoning is emerging as a new RL target: the latest HF Search paper emphasizes end-to-end tool-use trajectories, exploration strategies, and deliberate control over how often tools are called [17].

Open-source reproduction and distillation are also now central. Open-R1 aims to fully reproduce DeepSeek-R1 and explicitly breaks the work into R1-Distill replication, a pure-RL R1-Zero pipeline, and end-to-end base-to-RL training [10][11]. Related community efforts report curated reasoning traces, distillation recipes, and smaller checkpoints that can run on cheaper hardware, showing that the field is not just about frontier capability but also about packaging reasoning into deployable models [11][12][20].

## Trends and open problems
The last two years changed the research agenda in three ways. First, RL for reasoning has become more empirical and benchmark-driven, with DeepSeek-R1-style results showing that pure RL and staged RL can deliver large gains on verifiable tasks [9][10]. Second, reward design has become more granular: process rewards, self-verification, and verifier-guided signals are increasingly favored over final-answer-only rewards [4][5][6][7][8]. Third, the community is now optimizing for efficiency and generality, not only peak scores, via selective rollouts, memory-aware optimization, and cross-domain corpora [13][14][15][16][17].

Several open problems remain unresolved. One is credit assignment: even with process rewards, it is still unclear which intermediate signals are sufficient for long-horizon reasoning and which just correlate with good outputs [4][5][7][8]. Another is coverage: many of the strongest successes are still on math and code, while open-ended tasks and real-world agentic settings remain harder to verify and reward reliably [5][6][14][17]. A third issue is transfer and compression: distillation makes reasoning more deployable, but it is not yet clear how much of RL-induced reasoning survives model scaling down or domain shift [10][11][12].

The broader dispute is whether future progress should come from stronger RL optimization, better reward modeling, or better data and task design. The current literature suggests that all three matter: RL supplies exploration and selection, process supervision improves credit assignment, and curated reasoning corpora and distillation make the resulting capabilities usable [4][6][14][10][11].

## References
[1] Training language models to follow instructions with human feedback. arxiv. https://arxiv.org/abs/2203.02155 (2022-03-04)
[2] Proximal Policy Optimization Algorithms. arxiv. https://arxiv.org/abs/1707.06347 (2017-07-20)
[3] Proximal Policy Optimization — Spinning Up documentation. web. https://spinningup.openai.com/en/latest/algorithms/ppo.html (n.d.)
[4] Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. web. https://arxiv.org/html/2501.09686 (n.d.)
[5] Toward large reasoning models: A survey of reinforced reasoning with large language models. web. https://www.cell.com/patterns/pdf/S2666-3899(25)00218-1.pdf (n.d.)
[6] S^2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning. hf-search. https://huggingface.co/papers/2502.12853 (2025-02-18)
[7] Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning. hf-search. https://huggingface.co/papers/2601.17223 (2026-01-23)
[8] P2S: Probabilistic Process Supervision for General-Domain Reasoning Question Answering. hf-search. https://huggingface.co/papers/2601.20649 (2026-01-28)
[9] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/abs/2501.12948 (2025-01-22)
[10] Open-R1: a fully open reproduction of DeepSeek-R1. web. https://huggingface.co/blog/open-r1 (2025-01-28)
[11] huggingface/open-r1. web. https://github.com/huggingface/open-r1?tab=readme-ov-file (2025-01-24)
[12] Re-Distilling Smaller DeepSeek R1 Models for Better Performance. web. https://dropbox.github.io/r1_redistill_blogpost/ (n.d.)
[13] Reinforcement Learning for LLM Reasoning Under Memory Constraints. hf-search. https://huggingface.co/papers/2504.20834 (2025-04-29)
[14] Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective. hf-search. https://huggingface.co/papers/2506.14965 (2025-06-17)
[15] Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts. hf-search. https://huggingface.co/papers/2506.02177 (2025-06-02)
[16] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning. hf-search. https://huggingface.co/papers/2506.01939 (2025-06-02)
[17] Demystifying Reinforcement Learning in Agentic Reasoning. hf-search. https://huggingface.co/papers/2510.11701 (2025-10-13)
[18] Reinforcement Learning Enhanced LLMs: A Survey. web. https://arxiv.org/html/2412.10400 (n.d.)
[19] A Survey of Reinforcement Learning for Large Reasoning Models. web. https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs/blob/main/README.md (2025-09-12)
[20] Simple Reinforcement Learning for Reasoning. web. https://github.com/hkust-nlp/simpleRL-reason?tab=readme-ov-file (2025-01-25)
