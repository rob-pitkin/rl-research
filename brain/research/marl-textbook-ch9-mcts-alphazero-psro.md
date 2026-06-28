---
title: MARL Textbook Ch.9 — MCTS, AlphaZero, PSRO, AlphaStar (research note)
description: Provisional analysis of decision-time planning + population-based training in MARL, from MCTS to AlphaStar.
type: research-note
status: provisional
sources:
  - external-sources/marl-textbook-ch9-mcts-alphazero-psro.mdx
created: 2026-04-11
tags:
  - research
  - provisional
  - marl
  - mcts
  - self-play
  - population-based-training
  - alphazero
  - psro
  - zero-sum-games
---
## Question

How do decision-time planning and population-based training scale to MARL — MCTS → AlphaZero (self-play) → PSRO (general-sum) → AlphaStar — and what is relevant to the comm-vs-ctde project?

## Sources cited

- [Multi-Agent Reinforcement Learning (Albrecht, Christianos, Schäfer) — Ch. 9](../external-sources/marl-textbook-ch9-mcts-alphazero-psro.mdx) (2024)

## Findings

### Summary

This chapter covers **decision-time planning and population-based training** in MARL. Core progression: [MCTS](./mcts.md) (single-agent planning) → AlphaZero (self-play MCTS + deep learning for symmetric zero-sum games) → PSRO (population-based training for general-sum games) → AlphaStar (PSRO + deep RL for StarCraft II Grandmaster).

**Key insight**: for games with sparse rewards and huge action spaces, **combining search (MCTS) with learned evaluation functions and population diversity** discovers sophisticated strategies that pure model-free RL struggles to find.

### MCTS for MDPs

k simulations from the current state, each: Selection (UCB tree policy) → Expansion → Simulation (rollout) → Backup (update Q(s,a), N(s,a)). UCB: `a = argmax_a [Q(s,a) + sqrt(2 ln N(s) / N(s,a))]`. The evaluation function `f(s_leaf)` can be heuristic, random rollout, or a **learned neural network** (AlphaZero).

### Self-Play MCTS for Zero-Sum Games

Requires symmetric roles, egocentric observations (state transform `ψ(s)`), and turn-taking. From agent 1's perspective the game becomes an MDP where agent 1 acts every step; apply `ψ(s)` and flip the evaluation sign on opponent turns.

### AlphaZero

Learns `(u, p) = f(s; θ)`: u = predicted outcome, p = action priors guiding the tree policy. Trained on self-play data `{(s_t, π_t, z_T)}` with loss `L = (z-u)² - πᵀ log p + c||θ||²`. UCB-like selection uses priors. **Results**: beats Stockfish (chess), Elmo (shogi), AlphaGo Zero (Go) with the same general algorithm; chess in 9 hours / 44M games at k=800 sims/move.

### Population-Based Training & PSRO

**Loop**: initialize populations → evaluate (run episodes across policy combinations) → modify (add new policies). Generalizes self-play by training against a *distribution* over past policies (reduces overfitting).

**PSRO**: build a meta-game M^k (normal-form, actions = population policies, rewards = empirical returns); a meta-solver computes distributions δ (e.g., Nash) with minimum probability ε per policy; an oracle computes best responses `π'_i ∈ argmax E[U_i(π_i, π_{-i})]` via single-agent RL; grow the population. Converges to a Nash equilibrium of the underlying game (finite, exact oracle). Worked RPS example converges to uniform randomization.

### AlphaStar

StarCraft II (~10²⁶ actions/timestep, partial observability, long horizon, sparse rewards). **League training** (PSRO variant) with main agents, main exploiters, and league exploiters per race; **Prioritized Fictitious Self-Play (PFSP)** weights opponents by win probability (`f_hard(x)=(1-x)^p` focuses on hardest). Initialized via imitation on 971K human replays, then A2C-based RL. Reached **Grandmaster** (top 0.2%) across all three races; 44 days, 32 TPUs. Human data initialization critical.

### Connections

- **Foundation**: [MCTS](./mcts.md) — extended here to multi-agent settings.
- **Model-free MARL contrast**: [MAPPO](./mappo.md), [QMIX](./qmix.md) — no forward model, no self-play search.

## Open questions

- **MCTS for MARL**: needs a fast forward simulator; k sims/action is expensive; self-play assumes turn-taking + symmetric agents.
- **PSRO**: meta-game grows geometrically with population; Nash computation is exponential; oracle may hit local optima.
- **AlphaZero/AlphaStar**: designed for competitive (not cooperative) settings; AlphaStar needs human-data initialization and DeepMind-scale compute.

### Relevance to comm-vs-ctde project

For the (cooperative, model-free) comm-vs-ctde setting these methods are mostly a contrast class, but two angles transfer:
- **Model-based baseline**: with the SMAC simulator, MCTS could provide a decision-time planning baseline — does learned value decomposition ([QMIX](./qmix.md)) or learned [communication](./commnet.md) beat MCTS with simple rollout?
- **Population diversity for exploration**: PSRO's population could explore communication strategies (main agents learn communication, exploiters find counter-strategies) — can population-based training discover emergent protocols single-policy training misses? Also: could the MCTS tree policy include *communication actions*, and how would agents coordinate during simulations?

When NOT to use: no forward model (use model-free MAPPO/QMIX); real-time constraints (MCTS too slow); cooperative MARL with dense rewards (simpler methods suffice).