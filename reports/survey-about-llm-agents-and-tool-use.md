# Survey of LLM Agents and Tool Use

## TL;DR
- Early agent/tool-use work split into two complementary paradigms: prompt-level reasoning+acting with ReAct, and self-supervised tool learning with Toolformer; both were quickly operationalized through API-centric systems such as API-Bank, Gorilla, and OpenAI function calling. [1][2][3][4][5]
- By 2024–2025, the center of gravity shifted from “can the model call a tool?” to “can the model choose, parameterize, and compose tools reliably over long trajectories?”, which is why newer benchmarks focus on trajectory satisfaction, tool-output faithfulness, and multi-turn failure modes. [6][7][8][9]
- Training strategies are also moving from simple supervised fine-tuning toward synthetic-data pipelines and reinforcement learning, with recent work arguing that reward-guided or verifier-guided RL can outperform SFT on complex tool-use generalization. [10][11][12][13][14][15]
- The ecosystem is converging around structured interfaces and deployment stacks: MCP standardizes connections to external systems, while OpenAI’s Responses/Agents stack and multimodal desktop/web agents emphasize orchestration, observability, and efficient server-side tool execution. [16][17][18][19][20][21][22][23]
- Safety has become a first-class concern because multi-turn tool use introduces new attack surfaces, including prompt injection, tool skipping, result ignoring, parameter fabrication, and compounding risk across turns. [24][25][26][27]

## Background
LLM agents are systems that combine language generation with external actions such as search, code execution, database access, browser automation, or calls to structured APIs. The key design problem is no longer just whether a model can answer a question, but whether it can plan, select tools, pass valid arguments, integrate tool outputs, and recover from errors in a multi-step interaction. Recent surveys frame evaluation along both behavioral dimensions and process dimensions, highlighting that interactive, trace-level assessment is needed for planning, memory, tool use, collaboration, reliability, and safety. [28][29]

The field’s early foundations established two different routes to tool use. ReAct interleaves reasoning traces with environment actions so a model can update its plan while querying external information, while Toolformer teaches models to decide when to call tools and how to incorporate results into next-token prediction. These lines of work turned tool use from a brittle engineering add-on into a research problem about reasoning, control, and learning. [1][2]

## From prompting to structured invocation
ReAct is best read as a prompting paradigm: it inserts thought-action-observation loops into the context so the model can reason and act in one trajectory. In the cited results, this reduced hallucination and error propagation on knowledge tasks and delivered strong gains on embodied and web tasks such as ALFWorld and WebShop. Toolformer, by contrast, makes tool use part of the model’s learned behavior by self-supervising API-call decisions; it emphasizes that the model should learn when to call a calculator, search engine, translator, or calendar, not just how to answer after the fact. Together, these works define the prompt-based and learned tool-use branches that most later systems build on. [1][2]

A second step in maturation was standardizing structured tool invocation. OpenAI’s function-calling update framed JSON-schema-based calls as a more reliable interface between models and external APIs, with models fine-tuned to recognize when a function should be called and to emit arguments in the right structure. Gorilla pushed the idea into API-benchmark form, combining retrieval with a finetuned LLaMA-based model for API calls and emphasizing that documentation understanding and constraints matter for accurate invocation. API-Bank and LangChain’s benchmark blog then made the evaluation problem explicit: planning, retrieval, composition, and trajectory length all matter, and single-step correctness hides many failure modes. [3][4][5][6]

## Training tool use: SFT, synthetic data, and RL
The main training question is how to move from brittle demonstration-following to general tool competence. A common recipe is supervised fine-tuning on curated tool-call traces, synthetic trajectories, or execution-grounded demonstrations. Several recent sources argue, however, that SFT alone often underperforms on unfamiliar tools, multi-step dependency chains, or long-horizon workflows. ToolRL reports that reward-driven training improved tool selection and application over both the base model and an SFT model, while Tool-R1 argues for RL with executable Python code and outcome-based rewards rather than rigid JSON-only calling. [10][11]

The more experimental line of work is self-improvement. Tool-R0 describes self-play-style co-evolution from zero data, where a generator proposes tasks and a solver learns from real tool calls, reporting a 92.5% relative improvement over the base model. Other systems synthesize tasks and environments instead of hand-labeling them: RandomWorld procedurally generates interactive tools and compositional data; AReaL-SEA uses synthetic tool-grounded dialogues plus verifiable rewards; SynthAgent builds mock worlds with simulated environments and rubric-based rewards; ToolACE focuses on self-evolved data for function calling; and environment-free synthetic generation simulates API interactions without a live environment. The emerging pattern is that better tool-use training increasingly depends on data generation pipelines, not just manual annotation. [12][30][31][13][32][14][33][15]

## Evaluation is shifting from answers to trajectories
A major change in the last two years is that evaluation has become process-aware. Instead of judging only the final answer, new benchmarks test whether the agent chose the right tool, passed the right arguments, respected ordering constraints, and actually used the tool output. TRAJECT-Bench captures this with trajectory-aware metrics such as tool usage, trajectory satisfaction, and retrieval rate, and reports that performance drops as trajectories get longer. ToolScan similarly classifies recurring error types rather than compressing everything into one score. [7][9]

This matters because aggregate success rates can hide qualitatively different failures. ToolFailBench shows that tool-skip, result-ignore, output-fabrication, and unnecessary-tool-use are distinct behaviors; its reported 86.33% clean tool-use rate for the best model still leaves substantial room for error. LangChain’s earlier benchmark blog already hinted at this by showing that composition and long trajectories were harder than single-call correctness, but the newer work turns that intuition into explicit diagnostic categories and step-level measures. [6][8]

## Safety, robustness, and multi-turn risk
As tool use becomes more agentic, safety evaluation has expanded from static prompt safety to dynamic action safety. AGENT-SAFETYBENCH reports 349 interaction environments, 2,000 test cases, and safety scores below 60% for 16 evaluated agents, underscoring that tool competence and risk awareness are separate abilities. SafeToolBench takes a prospective stance by asking whether a risky action should be blocked before execution, while MT-AgentRisk turns benign single-turn tasks into multi-turn attacks and finds that multi-turn settings raise attack success by an average of 16% across models. [24][25][26]

The common failure modes are instructive. Systems may skip tools, ignore tool results, fabricate parameters, over-call tools, or become vulnerable to prompt injection and instruction drift. TraceSafe extends the safety problem to multi-step tool trajectories, showing that guardrails need step-wise labels and can degrade in long-context settings. The overall implication is that safety for agents is not just content moderation; it is policy enforcement over sequences of actions. [28][29][27]

## 2024–2025 ecosystem trends and productization
The most visible ecosystem shift is the move toward standardized orchestration layers. Anthropic’s Model Context Protocol positions itself as an open standard for connecting assistants to data sources, and later ecosystem updates emphasize adoption at scale, including thousands of servers and integration with major developer tools. OpenAI’s Responses API and Agents SDK similarly package web search, file search, computer use, handoffs, guardrails, tracing, and multi-agent workflows into a managed stack. In both cases, the runtime surface is no longer just “call a function”; it is an end-to-end agent substrate. [16][17][18][19][20]

A second trend is efficient tool access. Anthropic’s code-execution guidance argues that as tool ecosystems grow, tool definitions and results can dominate the context window; loading tools on demand and doing more work in the execution environment can drastically reduce token use. That concern is mirrored in OpenAI’s emphasis on server-side tools and stateful execution. The practical lesson is that tool use is becoming an infrastructure problem as much as a modeling problem. [19][21]

Finally, the product layer is expanding beyond text-only chat into multimodal desktop and web agents. UI-TARS and Agent TARS package browser control, desktop control, screenshots, visual grounding, and MCP integration, while AgentSea presents orchestration and device access as deployable infrastructure across local, container, and cloud environments. These systems show how agent research is being translated into operational stacks that can actually run on terminals, browsers, and desktops. [22][23][34][35]

## Trends and open problems
Three trends stand out. First, the field is moving from isolated tool calls toward long-horizon workflows with state, memory, and orchestration. Second, learning is shifting toward synthetic data and RL, especially when human-labeled tool traces are scarce or too narrow. Third, evaluation and safety are becoming trajectory-level problems, because many failures only appear when you inspect the sequence of actions rather than the final answer. [10][12][28][7][24]

The main open problem is generalization: systems that look strong on one tool or one benchmark can break on novel APIs, longer compositions, or changing environments. A related unresolved issue is faithfulness to tool outputs, since models can still fabricate arguments or ignore returned evidence even when they invoke tools correctly. Safety remains disputed in practice because there is still no consensus on how to balance autonomy, utility, and conservative blocking across multi-turn interactions. [8][9][25][26][27]

A second open problem is standardization without over-constraining innovation. MCP, Responses, and similar stacks make tool calling easier to deploy, but they also raise questions about portability, observability, and vendor-specific runtime assumptions. The field now needs benchmarks and training recipes that reflect real deployments: heterogeneous tools, partial failures, dynamic environments, and explicit risk policies. [16][18][20][21][34][35]

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. web. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[3] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. web. https://aclanthology.org/2023.emnlp-main.187/ (2023-12-06)
[4] Gorilla: Large Language Model Connected with Massive APIs. web. https://shishirpatil.github.io/publications/gorilla-2023.pdf (n.d.)
[5] Function calling and other API updates. web. https://openai.com/index/function-calling-and-other-api-updates/ (2023-06-13)
[6] Benchmarking Agent Tool Use. web. https://www.langchain.com/blog/benchmarking-agent-tool-use (2023-12-19)
[7] TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use. web. https://arxiv.org/html/2510.04550 (n.d.)
[8] ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents. web. https://arxiv.org/html/2607.04686 (n.d.)
[9] ToolScan: A Benchmark for Characterizing Errors in Tool-Use LLMs. web. https://arxiv.org/html/2411.13547 (n.d.)
[10] ToolRL: Reward is All Tool Learning Needs. arxiv. https://arxiv.org/abs/2504.13958 (2025-04-16)
[11] Tool-R1: Sample-Efficient Reinforcement Learning for Agentic Tool Use. web. https://arxiv.org/html/2509.12867 (n.d.)
[12] Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data. hf-search. https://huggingface.co/papers/2602.21320 (2026-02-24)
[13] From Self-Evolving Synthetic Data to Verifiable-Reward RL. web. https://arxiv.org/html/2601.22607v3 (n.d.)
[14] Environment-free Synthetic Data Generation for API-Calling Agents. hf-search. https://huggingface.co/papers/2607.16900 (2026-07-18)
[15] ToolACE: Winning the Points of LLM Function Calling. hf-search. https://huggingface.co/papers/2409.00920 (2024-09-02)
[16] Introducing the Model Context Protocol \ Anthropic. web. https://www.anthropic.com/news/model-context-protocol (2024-11-25)
[17] Donating the Model Context Protocol and establishing the Agentic AI Foundation. web. https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation?content=Dec2024EOYShips&medium=email&messageTypeId=140367 (n.d.)
[18] New tools for building agents | OpenAI. web. https://openai.com/index/new-tools-for-building-agents/ (2025-03-11)
[19] Why we built the Responses API. web. https://developers.openai.com/blog/responses-api (n.d.)
[20] Agents SDK | OpenAI API. web. https://developers.openai.com/api/docs/guides/agents (n.d.)
[21] Code execution with MCP: building more efficient AI agents. web. https://www.anthropic.com/engineering/code-execution-with-mcp (2025-11-04)
[22] README.md at main · bytedance/UI-TARS-desktop. web. https://github.com/bytedance/UI-TARS-desktop/blob/main/README.md (n.d.)
[23] Agent TARS core README. web. https://github.com/bytedance/UI-TARS-desktop/blob/main/multimodal/agent-tars/core/README.md (n.d.)
[24] Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents. web. https://arxiv.org/html/2602.13379v2 (n.d.)
[25] AGENT-SAFETYBENCH: Evaluating the Safety of LLM Agents. web. https://arxiv.org/pdf/2412.14470 (n.d.)
[26] SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool Utilization Safety in LLMs. web. https://aclanthology.org/2025.findings-emnlp.958.pdf (n.d.)
[27] TraceSafe: A Systematic Assessment of LLM Guardrails on Multi-Step Tool-Calling Trajectories. web. https://arxiv.org/pdf/2604.07223 (2026-08-09)
[28] Evaluation and Benchmarking of LLM Agents: A Survey - arXiv. web. https://arxiv.org/html/2507.21504v1 (n.d.)
[29] A Survey on Evaluation of LLM-based Agents - arXiv. web. https://arxiv.org/html/2503.16416v2 (n.d.)
[30] Execution-First Synthetic Tool-Use Trace Generation for LLM Agents. web. https://arxiv.org/html/2607.29175 (2026-07-31)
[31] Procedural Environment Generation for Tool-Use Agents. web. https://aclanthology.org/2025.emnlp-main.936.pdf (n.d.)
[32] Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards. web. https://arxiv.org/html/2601.22511 (n.d.)
[33] Adapting Web Agents with Synthetic Supervision. hf-search. https://huggingface.co/papers/2511.06101 (2025-11-08)
[34] AgentSea Platform. web. https://www.agentsea.ai/ (n.d.)
[35] Introduction - AgentSea. web. https://docs.hub.agentsea.ai/introduction (n.d.)
