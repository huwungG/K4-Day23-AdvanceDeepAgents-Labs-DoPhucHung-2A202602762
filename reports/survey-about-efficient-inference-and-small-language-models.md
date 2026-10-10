# Survey on Efficient Inference and Small Language Models

## TL;DR
- Compression surveys from 2023–2025 consistently identify quantization, pruning, knowledge distillation, and low-rank/architecture changes as the main levers for making LLMs smaller and faster, with recent surveys emphasizing that many methods are most useful when they can be applied without expensive finetuning [1][2][3].
- Hugging Face’s recent paper feed shows the same trend from a deployment angle: recent SLM surveys, MiniCPM, and speculative-decoding papers increasingly pair compact or quantized draft models with serving-aware verification to reduce latency and memory use [4][5][6][7][8][9].
- Benchmarking has become multi-objective: SLM studies now report latency, throughput, memory, power, energy, and thermal behavior alongside task quality, and results show the frontier depends heavily on architecture, context length, and hardware [10][11][12][13][14][15].
- The newest systems papers move beyond “smaller model = faster inference” toward joint design of model, cache, batching, and decoding strategy; long-context and memory-bound cases are now central rather than edge cases [16][17][18][19][20][21].

## Background
Efficient inference for language models is no longer a single technique story. The literature now separates at least two coupled goals: reducing model footprint during storage or training, and reducing serving cost during autoregressive decoding. Survey papers on compression frame the main levers as quantization, pruning, distillation, low-rank approximation, and compact architecture design [1][2][22][16]. Web surveys from 2025 add that structured pruning, parameter sharing, neural architecture search, and hybrid compression-plus-tuning workflows are increasingly part of the practical toolbox [3][23][24].

At the same time, small language models (SLMs) are being positioned not just as “mini versions” of large models but as a separate deployment class for mobile, edge, and latency-sensitive settings. Hugging Face survey entries explicitly connect SLMs to low-compute environments and to evaluation across reasoning, coding, and runtime cost [4][5][25]. Concrete model papers such as MiniCPM show that careful training strategy can keep smaller models competitive with larger ones [6].

## Compression families: what shrinks the model itself
Across surveys, quantization is the most direct memory-saving method because it lowers precision from 16/32-bit arithmetic to 8-bit, 4-bit, or even lower representations [1][3][23]. The recurring distinction is post-training quantization versus quantization-aware training, with the former favored when retraining budget is limited [1][2]. Surveys also emphasize that quantization can be applied to weights alone, to weights plus activations, or to the KV cache; the last category matters because cache memory becomes a dominant bottleneck for long prompts and long generation [1][16].

Pruning attacks redundancy structurally, but the literature distinguishes unstructured pruning from structured pruning. The practical advantage of structured pruning is that it can actually translate into speedups on common hardware by removing whole heads, neurons, or layers instead of scattered weights [1][2][16][23]. Recent surveys also call out pruning as especially relevant to edge deployments, where memory and energy budgets are strict [22][3][24].

Distillation remains the canonical “behavior-preserving shrinkage” method: a larger teacher transfers its outputs or internal signals to a smaller student [1][3][24]. The common argument is that distillation is attractive when task behavior must stay close to the original model, even if capacity is reduced [2]. Low-rank factorization and parameter sharing form a separate family that changes how weights are represented rather than merely zeroing or rounding them [26][3][24]. Finally, architectural redesign and neural architecture search appear as a complementary route for making the model inherently compact and hardware-aware [26][2][23].

## From compression to inference speedups
A major lesson from the serving surveys is that compression does not automatically equal speed. The 2023–2024 serving literature stresses that low-bit methods can fail to speed up if kernel support, memory movement, or scheduling are poor [16]. That is why efficient inference papers increasingly pair model-side techniques with serving-side optimizations such as parallel computation, memory management, request scheduling, and kernel optimization [16]. In other words, the model must fit the system.

This is especially visible in speculative decoding. The basic idea is to use a draft model to propose multiple tokens and then verify them with a larger target model, reducing sequential bottlenecks [8]. The 2024–2025 Hugging Face stream shows many variations on this theme: dynamic speculation length [21], token recycling [20], reward-guided speculation [9], and hierarchical draft/verify frameworks that explicitly combine quantization and speculative decoding [7]. The recent QuantSpec and LongSpec-style papers highlight why the field has shifted toward memory-aware drafting: long-context inference is often limited by KV-cache growth, mismatch between short-context training and long-context serving, and inefficient tree attention [17][19].

The practical takeaway is that the fastest methods are increasingly hybrid. S3D uses self-speculation and mid-layer skipping for low-memory GPUs, while other methods use quantized draft models or quantized KV caches to keep the speculative path cheap [18][17]. The 2025 hierarchical framework reports a 2.78x speedup for a 4-bit Llama-3-70B setup on an A100 and 1.31x over EAGLE-2, illustrating that “small” in this context often means smaller auxiliary models or caches rather than a single tiny end model [7].

## Small language models as a deployment class
SLM surveys from 2024–2025 argue that small models deserve their own category because the key question is not only final accuracy but also whether a model can serve on-device, on mobile hardware, or under tight latency and energy constraints [4][5][27]. MiniCPM is a representative example of the competitive story: it is presented as a family of small models built with scalable training strategies, and its paper card claims performance comparable to larger models [6].

What changed in the last two years is that SLM papers are increasingly tied to concrete deployment constraints rather than abstract parameter counts. Evaluation discussions now include vocabulary size, KV cache, GQA versus MHA, and context length, because these factors affect memory and runtime independently of raw parameter count [10][12]. Surveys also emphasize that architecture can matter as much as size for latency, particularly in smaller-model regimes where GPU utilization and fixed overheads dominate [10][11][12].

The Hugging Face paper feed mirrors this shift. It prominently surfaces SLM surveys and compact-model reports, but also recent efficiency papers that make the deployment connection explicit: quantized speculative decoding, self-speculative decoding for low-memory GPUs, and reward-guided reasoning decoders [4][5][7][18][9]. This suggests that the field’s center of gravity has moved from merely shrinking model weights to co-designing model, cache, and decoding strategy for a target device [17][18][19].

## Evaluation, benchmarks, and what “efficient” now means
The evaluation literature now treats “efficient” as multi-dimensional. The SLM survey/measurement papers benchmark not only task capability but also prefill and decode latency, memory footprint, and energy consumption [10][28][12]. One study of 70 open-source SLMs in the 100M–5B range reports that latency scales with model size, but architecture can cause large differences even at similar sizes, and quantization helps most on decode latency when prompt length and method align [10]. Another edge-focused benchmark across 68 SLMs and two edge boards reports that architecture can affect latency more than size in some regimes and that vocabulary/context length strongly influence memory usage [12].

The benchmark ecosystem is also standardizing around serving metrics. MLPerf Inference v5.1 introduces a small-LLM benchmark around Llama 3.1 8B and CNN-DailyMail summarization, with metrics including TTFT, TPOT, and tokens/s, plus different envelopes for server, interactive, and edge settings [29][30]. Meanwhile, smaller community benchmarks such as smolperfbenchmark and TinyBench emphasize sustained throughput, energy per token, power, and thermal throttling on real devices [13][14].

These benchmark choices matter because they change which methods look best. Serving studies on Pareto throughput show that batch size has a knee beyond which latency worsens without much throughput gain, and that replication can help very small models when GPUs are underutilized [11]. On CPUs, inference-acceleration work similarly finds that batching and parallel processing can lower power while improving throughput [31]. The upshot is that “efficient inference” is now a hardware-and-workload-specific notion rather than a universal model ranking.

## Trends and open problems
The strongest trend in the last two years is hybridization. Compression no longer lives in isolation: quantization is paired with speculative decoding, pruning is paired with hardware-aware architecture choices, and SLM training is paired with deployment-specific evaluation [2][16][23][7][17][18]. The second trend is that memory has become first-class. KV cache quantization, constant-sized caches, and cache-aware draft models appear repeatedly because long-context serving often fails on memory before it fails on arithmetic throughput [1][17][19].

Open problems remain substantial. First, speedups are still brittle: some low-bit or speculative methods help only on certain GPUs, batch sizes, or context lengths [16][7][18][11]. Second, benchmarks are not yet unified across cloud, edge, CPU, and mobile settings, so comparisons can be misleading unless hardware and workload are fixed [10][29][12][13][14][31]. Third, there is still a gap between “smaller” and “better”: compression can preserve average accuracy while degrading robustness, reasoning depth, or long-context behavior, which is why surveys continue to emphasize task-specific tradeoffs [4][5][10].

A concise reading of the field is this: the community has moved from compressing weights to compressing the whole inference pathway. The next gains are likely to come from joint model/system design, better long-context support, and evaluation standards that reward sustained, device-aware efficiency rather than just peak throughput [16][17][19][29][13].

## References
[1] A Survey on Model Compression for Large Language Models. arxiv. https://arxiv.org/abs/2308.07633 (n.d.)
[2] Model Compression and Efficient Inference for Large Language Models: A Survey. arxiv. https://arxiv.org/abs/2402.09748 (n.d.)
[3] A survey of model compression techniques: past, present, and future. web. https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1518965/full (2025-03-20)
[4] A Survey of Small Language Models. hf-search. https://huggingface.co/papers/2410.20011 (2024-10-25)
[5] Small Language Models: Architectures, Techniques, Evaluation, Problems and Future Adaptation. hf-search. https://huggingface.co/papers/2505.19529 (2025-05-26)
[6] MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. hf-search. https://huggingface.co/papers/2404.06395 (2024-04-09)
[7] Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design. hf-search. https://huggingface.co/papers/2505.22179 (2025-05-28)
[8] Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding. hf-search. https://huggingface.co/papers/2401.07851 (2024-01-15)
[9] Reward-Guided Speculative Decoding for Efficient LLM Reasoning. hf-daily. https://huggingface.co/papers/2501.19324 (2025-01-31)
[10] Small Language Models: Survey, Measurements, and Insights. web. https://arxiv.org/abs/2409.15790 (2025-02-26)
[11] Towards Pareto Optimal Throughput in Small Language Model Serving. web. https://arxiv.org/html/2404.03353 (n.d.)
[12] Demystifying Small Language Models for Edge Deployment. web. https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2025.acl-long.718.pdf (n.d.)
[13] smolperfbenchmark · Edge Leaderboard. web. https://smolperfbenchmark.vercel.app/ (n.d.)
[14] TinyBench. web. https://github.com/conscious-engines/TinyBench (n.d.)
[15] LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators. hf-search. https://huggingface.co/papers/2411.00136 (2024-10-31)
[16] Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems. arxiv. https://arxiv.org/abs/2312.15234v1 (2023-12-23)
[17] QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache. web. https://huggingface.co/papers/2502.10424 (2025-02-05)
[18] S3D: A Simple and Cost-Effective Self-Speculative Decoding Scheme for Low-Memory GPUs. web. https://huggingface.co/papers/2405.20314 (2024-05-30)
[19] LongSpec: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification. web. https://huggingface.co/papers/2502.17421 (2025-02-24)
[20] Token Recycling. hf-search. https://huggingface.co/papers/2408.08696 (2024-08-16)
[21] Accelerating Speculative Decoding using Dynamic Speculation Length. hf-search. https://huggingface.co/papers/2405.04304 (2024-05-07)
[22] Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques. arxiv. https://arxiv.org/abs/2505.02309v2 (2025-05-05)
[23] A review of state-of-the-art techniques for large language model compression. web. https://link.springer.com/article/10.1007/s40747-025-02019-z (2025-08-01)
[24] Efficient Compressing and Tuning Methods for Large Language Models: A Systematic Literature Review. web. https://dl.acm.org/doi/10.1145/3728636 (2025-05-06)
[25] Small Language Models: Survey, Measurements, and Insights. hf-search. https://huggingface.co/papers/2409.15790 (2024-09-24)
[26] A Comprehensive Survey of Compression Algorithms for Language Models. arxiv. https://arxiv.org/abs/2401.15347v1 (n.d.)
[27] What is the Role of Small Models in the LLM Era: A Survey. web. https://huggingface.co/papers/2409.06857 (2024-09-10)
[28] Small Language Models: Survey, Measurements, and Insights. web. https://github.com/ubiquitouslearning/slm_survey (2024-09-23)
[29] MLPerf Inference 5.1: Benchmarking Small LLMs with Llama3.1-8B. web. https://mlcommons.org/2025/09/small-llm-inference-5-1/ (2025-09-09)
[30] MLCommons Releases New MLPerf Inference v5.1 Benchmark Results. web. https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/ (2025-09-09)
[31] Inference Acceleration for Large Language Models on CPUs. hf-search. https://huggingface.co/papers/2406.07553 (2024-03-04)
