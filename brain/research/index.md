---
title: Paper & Method Index
description: Navigation hub for the provisional research notes, grouped by method family and topic. Migrated from the Obsidian paper-index.
type: research-note
status: provisional
created: 2026-06-28
tags:
  - research
  - provisional
  - index
  - hub
  - marl
---
# Paper & Method Index

Navigation hub for the provisional research notes in this folder. Each note synthesizes a preserved source in the `external-sources/` layer and is `status: provisional` pending promotion to a canonical `articles/` doc. Migrated from the former Obsidian `rl-knowledge/` vault.

The spine of the collection is the **comm-vs-ctde** question: *when does explicit inter-agent communication outperform state-based coordination (value decomposition), and can they be combined?* Most notes are annotated with their relevance to that question.

## Value Decomposition Methods

- [VDN](./vdn.md) — additive value decomposition for cooperative MARL (2017)
- [QMIX](./qmix.md) — non-linear monotonic value decomposition with state hypernetworks (2018)
- [VBC](./vbc.md) — hybrid value decomposition + selective communication under bandwidth constraints (2019)

## Policy Gradient Methods

- [COMA](./coma.md) — counterfactual multi-agent policy gradients (2018)
- [MAPPO](./mappo.md) — modern on-policy CTDE baseline (2022)
- [CGRPA](./cgrpa-curriculum-learning-marl.md) — curriculum learning + counterfactual group-relative credit assignment (2025)

## Communication Methods

- [CommNet](./commnet.md) — foundational differentiable broadcast communication (2016)
- [VBC](./vbc.md) — hybrid value decomposition + communication with bandwidth constraints (2019)
- [MARL communication survey](./marl-communication-survey.md) — 9-dimensional taxonomy over 41 Comm-MADRL methods (2024)

## Planning & Search

- [Monte Carlo Tree Search](./mcts.md) — decision-time planning via focused simulation (Sutton & Barto Ch. 8.11)
- [MARL Textbook Ch.9](./marl-textbook-ch9-mcts-alphazero-psro.md) — MCTS, AlphaZero, PSRO, AlphaStar for MARL

## Foundation Models

- [MARL-GPT](./marl-gpt.md) — first foundation model for MARL across multiple environments (2026)

## Position Papers & Perspectives

- [The Era of Experience](./the-era-of-experience.md) — Silver & Sutton on the shift to experiential learning (2025)

## LLM Infrastructure (tangential)

- [Recursive Language Models](./recursive-language-models.md) — scaling LLM context to 10M+ tokens via recursive self-querying (2025); low RL relevance

## By Topic

### Credit Assignment
[VDN](./vdn.md) · [QMIX](./qmix.md) · [COMA](./coma.md) · [VBC](./vbc.md) · [CGRPA](./cgrpa-curriculum-learning-marl.md)

### CTDE (Centralized Training, Decentralized Execution)
[QMIX](./qmix.md) · [COMA](./coma.md) · [MAPPO](./mappo.md) · [CommNet](./commnet.md) · [VBC](./vbc.md) · [marl-communication-survey](./marl-communication-survey.md)

### Communication vs State-Based Coordination (the core tension)
[CommNet](./commnet.md) · [QMIX](./qmix.md) · [VBC](./vbc.md) · [marl-communication-survey](./marl-communication-survey.md)

### SMAC Benchmark
[QMIX](./qmix.md) · [COMA](./coma.md) · [MAPPO](./mappo.md) · [VBC](./vbc.md) · [CGRPA](./cgrpa-curriculum-learning-marl.md) · [MARL-GPT](./marl-gpt.md)

### Counterfactual Reasoning
[COMA](./coma.md) · [CGRPA](./cgrpa-curriculum-learning-marl.md)

### Cooperative MARL
[VDN](./vdn.md) · [QMIX](./qmix.md) · [COMA](./coma.md) · [MAPPO](./mappo.md) · [CommNet](./commnet.md) · [VBC](./vbc.md) · [marl-communication-survey](./marl-communication-survey.md)

### Self-Play & Population-Based Training
[MARL Textbook Ch.9](./marl-textbook-ch9-mcts-alphazero-psro.md)

## Sources awaiting a research note

- [SIMA 2](../external-sources/sima2.mdx) — DeepMind generalist embodied agent; preserved but not yet analyzed.

---
*Migrated from `rl-knowledge/_index/paper-index.md`. Total research notes: 13. Preserved sources: 14.*