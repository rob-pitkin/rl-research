---
title: The Era of Experience (research note)
description: Provisional analysis of Silver & Sutton's Era of Experience thesis and its framing for experiential multi-agent learning.
type: research-note
status: provisional
sources:
  - external-sources/the-era-of-experience.mdx
created: 2026-04-08
tags:
  - research
  - provisional
  - rl
  - position-paper
  - experiential-learning
  - grounded-rewards
  - world-models
---
## Question

What is Silver & Sutton's "Era of Experience" thesis, and how does it frame experiential learning for multi-agent coordination (the comm-vs-ctde lens)?

## Sources cited

- [Welcome to the Era of Experience](../external-sources/the-era-of-experience.mdx) (David Silver, Richard S. Sutton; 2025 position paper)

## Findings

### Core thesis

AI is transitioning from the "Era of Human Data" (LLMs trained on human-generated corpora) to the "Era of Experience" (agents learning predominantly from **self-generated experiential data** through environmental interaction). To achieve superhuman intelligence, we must shift from static human data (approaching exhaustion) to continual self-improvement via grounded rewards, long-term streams of experience, rich action/observation spaces, and experiential planning — a return to core RL principles adapted for real-world deployment.

### Three Eras of AI

1. **Era of Simulation (2014–2020)** — RL in simulators with well-defined rewards (AlphaGo, Atari, StarCraft II, Dota 2). Superhuman in narrow domains; couldn't bridge sim→reality.
2. **Era of Human Data (2020–2025)** — LLMs + RLHF; unprecedented generality but a performance ceiling at human knowledge.
3. **Era of Experience (2025+)** — agents learn from self-generated data; data generation improves as the agent strengthens. Early evidence: AlphaProof (IMO medal), DeepSeek-R1.

### Four Pillars

1. **Streams of experience** — lifelong, cross-episode learning with long-term goals (vs short episodic Q&A).
2. **Rich actions and observations** — autonomous interaction (API calls, computer use, code execution, sensors, robotics) beyond human-privileged text I/O.
3. **Grounded rewards** — signals from actual consequences of actions (heart rate, exam results, CO2 levels) vs human prejudgement (RLHF). Bi-level optimization: user feedback (alignment) on top, grounded environmental signals (autonomous learning) below.
4. **Planning and reasoning about experience** — ground reasoning in **world models** that predict action consequences, rather than imitating human language chains (which inherit human biases / become an "echo chamber").

### Revisiting core RL concepts

The Era of Human Data bypassed value functions, exploration, world models, and temporal abstraction. The paper argues these should be revisited and adapted for long, grounded, autonomous interaction streams — "don't throw out the baby with the bathwater."

### Connections

- **Baselines through this lens**: [QMIX](./qmix.md) and [MAPPO](./mappo.md) are Era-of-Simulation exemplars — grounded rewards (combat outcomes), experiential (self-play in simulation).
- **Communication as experiential interaction**: [CommNet](./commnet.md) agents exchange experience-based messages; is communication a richer form of experiential interaction than shared state?
- **Counterfactuals from experience**: [COMA](./coma.md) grounds counterfactuals in actual experience.

## Open questions (critiques)

1. **Optimistic timeline** — "imminent" may overreach; sim→reality gap assumed solvable.
2. **Safety underaddressed** — bi-level optimization and adaptability are speculative; no concrete alignment mechanisms.
3. **Grounded rewards in reality** — Goodhart's Law risk (optimize measurable proxy → miss true goal).
4. **Compute** — AlphaProof needed 100M self-generated proofs; real-world experience may be costlier.
5. **Human-in-the-loop still required** — tension between autonomy and alignment.

### Relevance to comm-vs-ctde project

Provides a **philosophical/methodological frame**: the project explores fundamental experiential-learning principles (how agents learn coordination from interaction) in a controlled setting (SMAC). The paper validates this direction — understanding experiential learning in multi-agent systems is foundational for coming real-world multi-agent coordination.

Actionable reframes:
- Frame the work as "experiential learning for multi-agent coordination," not just "MARL baselines comparison."
- Emphasize **autonomous discovery** of coordination (don't hard-code human tactics).
- Use **grounded metrics** (win rate, sample efficiency, wall-clock) over human interpretability.
- Research questions it inspires: how do communication/state methods perform in continual (non-episodic) MARL? Does communication help build better world models of teammates? Does it enable longer-horizon or coordinated exploration?