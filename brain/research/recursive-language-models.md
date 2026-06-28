---
title: Recursive Language Models (research note)
description: "Provisional note on RLMs: recursive self-querying for 10M+ token context. Low RL relevance; indexed for completeness."
type: research-note
status: provisional
sources:
  - external-sources/recursive-language-models.mdx
created: 2026-04-08
tags:
  - research
  - provisional
  - llm
  - long-context
  - inference-time-scaling
---
## Question

How do Recursive Language Models scale LLM context to 10M+ tokens via recursive self-querying, and is there any relevance to RL/MARL? (Indexed for completeness; low RL relevance.)

## Sources cited

- [Recursive Language Models](../external-sources/recursive-language-models.mdx) (Zhang, Kraska, Khattab; MIT CSAIL; arXiv:2512.24601, Dec 2025)

## Findings

### Overview

RLMs let LLMs process arbitrarily long prompts (orders of magnitude beyond context windows) by treating the prompt as part of an **external environment** the LLM interacts with via code execution and recursive self-calls. Instead of feeding a massive prompt into the network, it loads as a variable in a Python REPL; the LLM writes code to examine/decompose it and recursively calls sub-LMs on snippets.

### Key Contributions

1. **Prompt-as-environment paradigm** — long prompts live in REPL memory; the LLM writes code to peek, filter, chunk, and process symbolically; recursive sub-LM calls on selected snippets.
2. **Scales to 10M+ tokens** — ~2 orders of magnitude beyond model context (GPT-5 272K → RLM handles 1M+ effectively), where base models catastrophically fail.
3. **Task-agnostic inference** — no task-specific training; fixed system prompt; works with any base LLM (GPT-5, Qwen3-Coder-480B).
4. **Comparable/cheaper cost** — median cost often cheaper than the base model on long contexts (selective viewing vs full ingestion).

### Experimental Results

| Task | Complexity | GPT-5 | RLM(GPT-5) |
|------|-----------|-------|-----------|
| S-NIAH | constant (needle) | degrades slowly | 100% |
| OOLONG | linear | 80%→40% | 56% |
| OOLONG-Pairs | quadratic | <1% | 58% |
| BrowseComp+ (1K) | multi-hop, 6–11M tokens | cannot fit | 91% |
| CodeQA | fixed (repo) | 24% | 62% |

Base-LLM performance degrades as f(input length, task complexity); RLMs degrade much slower.

### Conceptual parallels to RL (loose)

- **Hierarchical decomposition** — recursive sub-calls resemble options / temporal abstraction.
- **Inference-time scaling** — like test-time compute (MCTS, iterative refinement).
- **Environment interaction** — REPL as environment, code as actions → agent-like.
- **Emergent strategies** — zero-shot filtering/chunking resembles emergent coordination.

## Open questions

1. Inefficient context decisions (Qwen3-Coder makes 1000s of sub-calls; GPT-5 conservative).
2. No RLM-specific training (zero-shot frontier models only).
3. Sequential/blocking sub-calls — async would cut runtime.
4. Recursion depth = 1 (sub-calls are base LMs, not RLMs).
5. Prompt brittleness around `FINAL()` tags.

### Relevance to comm-vs-ctde project

**Very low** — RLMs are LLM context scaling, not multi-agent RL (no inter-agent communication, rewards, or policies). Tangential utility only: processing large MARL paper corpora, long experiment logs, or big codebases (EPyMARL, SMAC). Contrast with [The Era of Experience](./the-era-of-experience.md): that paper emphasizes experiential learning; RLMs emphasize symbolic reasoning over long contexts. Indexed for completeness.