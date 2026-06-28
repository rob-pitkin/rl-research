---
title: Monte Carlo Tree Search (research note)
description: "Provisional analysis of MCTS: decision-time planning via focused simulation, its RL foundations, and MARL relevance."
type: research-note
status: provisional
sources:
  - external-sources/sutton-barto-mcts-chapter.mdx
created: 2026-04-11
tags:
  - research
  - provisional
  - rl
  - planning
  - decision-time-planning
  - monte-carlo
  - tree-search
  - model-based
---
## Question

What is Monte Carlo Tree Search as a decision-time planning algorithm, how does it relate to RL fundamentals, and where is it relevant to (MA)RL research?

## Sources cited

- [Reinforcement Learning: An Introduction — MCTS Chapter (8.11)](../external-sources/sutton-barto-mcts-chapter.mdx) (Sutton & Barto, 2nd ed.)

## Findings

### Summary

MCTS is a **decision-time planning algorithm** combining rollout methods with incremental tree building to efficiently search large decision spaces. It revolutionized computer Go (amateur → grandmaster, 2005–2015) and was critical to AlphaGo. Unlike traditional planning (global value functions), MCTS builds a **focused search tree** around the current state by iteratively simulating trajectories and accumulating statistics only for promising state-action pairs.

**Core insight**: you don't need to evaluate the entire state space. By successively focusing simulations on high-value trajectory prefixes, MCTS allocates computation to the parts of the tree that matter most for the current decision.

### Four-Phase Iteration

Each iteration: **Selection** (traverse tree via tree policy, e.g. UCB, to a leaf) → **Expansion** (add a child via an unexplored action) → **Simulation** (run a rollout to termination via a simple rollout policy) → **Backup** (propagate the return through visited nodes, updating action-value estimates).

- **Tree policy**: informed policy *inside* the tree (UCB balances explore/exploit using accumulated stats).
- **Rollout policy**: simple/random policy *outside* the tree, must be fast.
- **Tree growth**: memory allocated only to promising prefixes; tree typically discarded after action selection.

### Relation to RL Fundamentals

MCTS is a **rollout algorithm** enhanced with value accumulation: Monte Carlo control (sample-based value estimation), sample (not expected) updates, trajectory sampling (avoids exhaustive DP sweeps), greedy policy improvement (GPI), incremental partial action-value function stored as the tree. Not usually called "learning" (no long-term memory between selections) but uses learning principles per decision.

### Advantages & Trade-offs

**Advantages**: focused search, anytime algorithm, no global approximation, sample-efficient *during search*, scalable to huge state spaces.

**Trade-offs**: requires a fast forward model, decision-time computational cost, performance depends on rollout-policy quality, tree often not reused.

### Connections

- **Extended to MARL**: [MARL textbook Ch.9 — MCTS, AlphaZero, PSRO, AlphaStar](./marl-textbook-ch9-mcts-alphazero-psro.md) (neural MCTS, self-play, population-based training).
- **Neural variants**: AlphaGo / AlphaZero (network provides value + policy priors; trained via self-play).

## Open questions

1. **Model dependency** — requires a forward model (MuZero learns one).
2. **Computational cost** — decision-time planning is slow if the model is expensive.
3. **Rollout-policy quality** — poor rollouts → slow convergence.
4. **Non-reusable computation** — tree often discarded.
5. **Exploration** — may miss distant high-value regions if the tree policy is too greedy.

### Relevance to comm-vs-ctde project

For single-agent RL, MCTS is a powerful baseline when a simulator exists (e.g., game environments) or for large action spaces where Q-learning is intractable. For MARL, multi-agent MCTS (MA-MCTS) is relevant to **model-based MARL** — if the SMAC simulator is available, MCTS with simple rollout could be a decision-time planning baseline against which to ask: does learned value decomposition ([QMIX](./qmix.md)) or learned [communication](./commnet.md) beat planning? Key consideration: does the problem have a cheap forward model? If yes, MCTS may outperform model-free methods with less real interaction.