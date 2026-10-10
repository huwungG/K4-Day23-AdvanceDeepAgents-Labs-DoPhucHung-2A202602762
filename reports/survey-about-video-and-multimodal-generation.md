# Survey of Video and Multimodal Generation

## TL;DR
- Video generation has moved from GAN-era models toward diffusion-dominant systems, with autoregressive and multimodal unified models now re-emerging as serious alternatives; recent surveys frame the field around GANs, diffusion, and AR model families rather than a single winning paradigm [1][2][3].
- The strongest recent method trend is latent diffusion transformers for text-to-video and image-to-video, with Hugging Face papers such as LTX-Video, Stable Video Diffusion, VideoLCM, and AnimateLCM highlighting the push toward real-time or few-step synthesis [4][5][6][7].
- Multimodal generation is broadening from text-conditioned video to audio-video, talking video, and unified image/video understanding-generation systems, suggesting that “video generation” is increasingly a cross-modal generation problem [8][9][10].
- Evaluation remains unsettled: FVD was an important early step, but newer benchmarks and metrics repeatedly show that static-image-style scores miss temporal coherence, prompt adherence, and world-consistency failures [11][12][13][14][15][16].
- Open problems now cluster around controllability, synchronization, long-horizon consistency, 3D/world grounding, and safety/provenance, including watermarking and multimodal risk detection [17][18][19][20][21][22][23].

## Background
Video generation has evolved through a fairly clear arc. Survey work traces the field from GAN-based generation to diffusion models and then to autoregressive and multimodal systems [1][24][3]. Across the surveyed sources, the key conceptual split is that diffusion models learn a denoising trajectory from noise to data, while autoregressive models factor generation into sequential next-element prediction over pixels, tokens, latents, or other representations [2][3][25]. That distinction matters because it shapes both quality and efficiency tradeoffs: diffusion has dominated high-fidelity generation, while AR and hybrid systems have re-entered the discussion for flexibility and unified multimodal decoding [1][3][26].

A second background shift is that “video generation” is no longer only text-to-video. Recent surveys and model papers increasingly treat it as a multimodal synthesis stack that spans image-to-video, audio-video, talking-head generation, editing, and world-model-style planning [27][8][9][17][19]. This broader framing is one reason the literature now emphasizes unified latent spaces and multi-condition control rather than only prompt-to-video capability [27][8][17][21].

## Method families and architectural trends
The dominant family in current high-performing systems is latent diffusion, often with transformer backbones. Stable Video Diffusion popularized a training recipe that combines text-to-image pretraining, video pretraining, and high-quality video fine-tuning, while LTX-Video pushes the same latent-diffusion idea toward real-time generation and supports both text-to-video and image-to-video use cases [4][7]. OpenVid-1M and its associated MVDiT model show a complementary direction: better datasets and stronger use of text tokens and visual tokens inside a multi-modal video diffusion transformer [27]. In other words, recent progress is not only about larger models; it is also about better latent representations, better data, and better fusion of structure and semantics [4][7][27].

Efficiency is the second major theme. VideoLCM and AnimateLCM adapt consistency-model ideas to video, aiming for very few denoising steps without collapsing fidelity [5][6]. The appeal here is straightforward: video generation is computationally expensive, so methods that preserve quality while reducing sampling steps are especially valuable for interactive applications and wider deployment [5][6]. The same efficiency pressure appears in Divot, which turns diffusion into both a video tokenizer and a de-tokenizer, linking representation learning and generation more tightly than conventional pipelines [28].

A third trend is that the representation of video itself is becoming more abstract. The long-video survey excerpt organizes methods into token-stream autoregressive families and diffusion-based transformer families, suggesting that tokenization and latent compression are becoming foundational design choices rather than implementation details [26]. This is also visible in Divot, where video tokens are tied to comprehension as well as generation [28]. The architectural implication is that future systems may be judged less by whether they are “diffusion” or “AR” and more by how well they manage video tokens, temporal context, and cross-modal structure [3][26][28].

## Multimodal generation and conditioning
A major change in the last two years is the expansion from text-only control to richer multimodal conditioning. UniForm is a unified diffusion transformer that generates audio and video simultaneously in a single latent space, and LetsTalk fuses image, audio, and video modalities for talking-video synthesis [8][9]. These models show that multimodal generation is not just auxiliary conditioning; it can be the core synthesis objective [8][9].

Broader vision-language diffusion work reinforces that diffusion is moving beyond video alone. Dual Diffusion for unified image generation and understanding positions diffusion as a plausible alternative to autoregressive models in unified generation-understanding systems [10]. In the video domain, this supports a larger trend toward shared multimodal backbones that can generate, edit, and interpret content across modalities rather than treat them as separate pipelines [8][9][10][19].

Control is also becoming more fine-grained. Surveys of controllable video generation report that text prompts alone are often insufficient, and that practitioners now use camera motion, depth maps, human pose, sketches, bounding boxes, motion trajectories, identity cues, audio, and other conditions [21]. The same survey organizes the field into single-condition, multi-condition, and universal controllable generation [21]. Taken together, these sources suggest that controllable video generation is converging on a “many conditions, one model” paradigm, where the challenge is not just accepting more inputs but preserving temporal consistency and visual quality while doing so [17][21].

## Evaluation, benchmarks, and failure modes
Evaluation remains one of the weakest parts of the field. FVD was an important milestone because it extended FID-style distance into a video setting and showed strong correlation with human judgment in an early large-scale study [11]. But later work repeatedly shows that FVD, CLIPScore, and other static or framewise metrics are incomplete for video [12][14][16]. The core criticism is that these metrics either ignore temporal motion or reduce video to image-like comparisons, so they miss failure modes such as flicker, weak temporal coherence, and poor prompt adherence [12][14][15][16].

In response, benchmark design has become more granular. FETV adds fine-grained prompt categories and reports that existing automatic metrics correlate poorly with human evaluation, motivating new metrics such as UMTScore and FVD-UMT [12]. VBench decomposes quality into 16 hierarchical dimensions and includes human preference annotations to better align benchmark scores with perception [13]. EvalCrafter similarly evaluates visual quality, content quality, motion quality, and text-caption alignment with a large set of objective metrics and multiple subjective studies [29]. The direction of travel is clear: the field is moving away from single-number scores and toward multidimensional benchmark suites [12][13][29].

Recent metric work also reflects a shift from generic realism toward explicit world consistency. World Consistency Score breaks quality into object permanence, relation stability, causal compliance, and flicker penalty, directly targeting common temporal and physical failure modes [15]. That design is important because it suggests a more precise theory of what can go wrong in generated video: not only whether frames look plausible, but whether the generated world remains internally coherent over time [15][16].

## Applications, safety, and world-model directions
The application side of video generation is expanding quickly. Surveys now emphasize audio-conditioned generation, cross-modal editing, controllable storytelling, and immersive applications such as digital humans and VR [17][19]. HF papers also show a fast-moving ecosystem for audio-visual editing: Object-AVEdit targets object-level addition, replacement, and removal across audio and visual modalities, while Safe-Sora and ConceptGuard frame safety as a generation-time problem rather than only a post-hoc moderation issue [20][23][30].

A related frontier is the connection to world models and embodied or procedural generation. The Sora-as-world-model survey argues that text-to-video systems are increasingly assessed by whether they support spatial, action, and strategic intelligence, and it notes that richer inputs such as audio and localization marks may be required [17]. On the more applied side, streaming panoramic world-model work points to navigation, VR, and embodied agent training, while procedural video world models frame generation as a closed-loop execution problem [31][32]. This is an important shift: video generation is no longer only about producing clips, but about modeling environments and action consequences [17][31][32].

Safety and provenance have become central because the same models that produce richer multimodal content can also amplify misinformation, copyright, and misuse risks. Watermarking surveys describe video watermarking as especially difficult because framewise methods can miss inter-frame correlations and degrade quality [18]. Newer Hugging Face papers on graphical watermarking and multimodal risk detection indicate that the field is beginning to treat safety as a first-class design axis, especially for multimodal-to-video systems [20][22][23].

## Trends and open problems
The main trend in the last two years is convergence toward unified latent spaces and diffusion transformers that can ingest multiple conditioning signals and generate multiple modalities jointly [4][27][8][9][10]. At the same time, efficiency work has pushed the field toward few-step or real-time generation, especially via consistency models and distilled diffusion [5][6]. The evaluation literature, however, still lags behind model progress, because the strongest recurring failure modes are temporal rather than per-frame [12][13][14][15][16].

Several open problems remain unresolved. First, controllability is still brittle: adding more conditions often makes optimization and temporal coherence harder, not easier [17][21]. Second, long-horizon consistency and world grounding are not solved; the field still struggles with object permanence, causal consistency, and scene stability over many frames [15][31][32]. Third, multimodal safety is becoming more complicated as models combine text, image, audio, and video inputs, so provenance and watermarking need to handle cross-modal and temporal structure rather than single-frame artifacts [18][20][22][23]. Finally, the community still lacks a universally accepted evaluation protocol that matches human judgment across quality, alignment, motion, and trustworthiness [12][13][29][14][16].

Overall, the literature suggests that the next phase of video and multimodal generation will not be defined by a single architectural winner. Instead, progress is likely to come from tighter integration of representation learning, multimodal conditioning, efficient sampling, benchmark design, and safety mechanisms [28][27][15][21][23].

## References
[1] Evolution of Video Generative Foundations. web. https://arxiv.org/html/2604.06339v1 (n.d.)
[2] Controllable Video Generation: A Survey. web. https://arxiv.org/html/2507.16869v1 (2025-07-22)
[3] Autoregressive Models in Vision: A Survey. web. https://arxiv.org/html/2411.05902v2 (n.d.)
[4] LTX-Video: Realtime Video Latent Diffusion. hf-daily. https://huggingface.co/papers/2501.00103 (2024-12-30)
[5] AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning. hf-daily. https://huggingface.co/papers/2402.00769 (2024-02-01)
[6] VideoLCM: Video Latent Consistency Model. hf-daily. https://huggingface.co/papers/2312.09109 (2023-12-14)
[7] Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. hf-daily. https://huggingface.co/papers/2311.15127 (2023-11-25)
[8] UniForm: A Unified Diffusion Transformer for Audio-Video Generation. hf-search. https://huggingface.co/papers/2502.03897 (2025-02-06)
[9] LetsTalk: Latent Diffusion Transformer for Talking Video Synthesis. hf-search. https://huggingface.co/papers/2411.16748 (2024-11-24)
[10] Dual Diffusion for Unified Image Generation and Understanding. hf-daily. https://huggingface.co/papers/2501.00289 (2024-12-31)
[11] Towards Accurate Generative Models of Video: A New Metric & Challenges. arxiv. https://arxiv.org/abs/1812.01717 (2019-03-27)
[12] FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation. web. https://d6108366.hf-mirror.com/papers/2311.01813 (2023-11-03)
[13] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://cvpr.thecvf.com/virtual/2024/poster/30230 (n.d.)
[14] Towards A Better Metric for Text-to-Video Generation. web. https://arxiv.org/html/2401.07781v1 (2024-01-15)
[15] World Consistency Score: A Unified Metric for Video Generation Quality. arxiv. https://arxiv.org/abs/2508.00144 (2025-07-31)
[16] Evaluating Text-to-Image and Text-to-Video Synthesis with a Conditional Fréchet Distance. web. https://arxiv.org/html/2503.21721v2 (n.d.)
[17] Sora as a World Model? A Complete Survey on Text-to-Video Generation. web. https://arxiv.org/html/2403.05131v3 (n.d.)
[18] SoK: On the Role and Future of AIGC Watermarking in the Era of Gen-AI. web. https://arxiv.org/html/2411.11478 (n.d.)
[19] LLMs Meet Multimodal Generation and Editing: A Survey. web. https://arxiv.org/html/2405.19334 (n.d.)
[20] Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking. hf-search. https://huggingface.co/papers/2505.12667 (2025-05-19)
[21] Controllable Video Generation: A Survey. web. https://huggingface.co/papers/2507.16869 (2025-07-22)
[22] Multi2AV-Safety: Benchmarking Safety in Multimodal-to-Audio-Video Generation. hf-daily. https://huggingface.co/papers/2608.26535 (2026-08-27)
[23] ConceptGuard: Proactive Safety in Text-and-Image-to-Video Generation through Multimodal Risk Detection. hf-daily. https://huggingface.co/papers/2511.18780 (2025-11-24)
[24] Video diffusion generation: comprehensive review and open problems. web. https://link.springer.com/article/10.1007/s10462-025-11331-6 (2025-08-20)
[25] A Comparative Review of Autoregressive and Diffusion Models for Video Generation - 立委NLP频道. web. https://liweinlp.com/13198 (n.d.)
[26] Long Video Generation survey excerpt. web. https://arxiv.org/pdf/2507.07202 (n.d.)
[27] OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation. hf-search. https://huggingface.co/papers/2407.02371 (2024-07-02)
[28] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation. hf-search. https://huggingface.co/papers/2412.04432 (2024-12-05)
[29] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. web. https://evalcrafter.github.io/ (n.d.)
[30] Object-AVEdit: An Object-level Audio-Visual Editing Model. hf-daily. https://huggingface.co/papers/2510.00050 (2025-09-27)
[31] SPW-Nav: A Streaming Panoramic World Model for Language-Guided Navigation. hf-daily. https://huggingface.co/papers/2610.08941 (2026-10-06)
[32] WorldGuide: Goal-Directed Video World Model for Procedural Task Execution. hf-daily. https://huggingface.co/papers/2610.12459 (2026-10-08)
