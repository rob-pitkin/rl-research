---
title: COMA — Counterfactual Multi-Agent Policy Gradients (research note)
description: "Provisional analysis of COMA: centralized critic + counterfactual baseline for credit assignment, vs value decomposition."
type: research-note
status: provisional
sources:
  - external-sources/coma.mdx
created: 2026-04-06
tags:
  - research
  - provisional
  - marl
  - policy-gradient
  - actor-critic
  - credit-assignment
  - ctde
  - counterfactual
---
## Question

How does COMA's counterfactual baseline solve multi-agent credit assignment with a centralized critic, and how does it compare to value-decomposition methods? Relevance to comm-vs-ctde.

## Sources cited

- [COMA — Counterfactual Multi-Agent Policy Gradients](../external-sources/coma.mdx) (Foerster et al., Oxford, AAAI 2018)

## Findings

### Summary

COMA is a multi-agent actor-critic method that uses **decentralized actors** and a **centralized critic** to learn cooperative policies. The key innovation is a **counterfactual baseline** for credit assignment that compares each agent's action to a marginal expectation over that agent's actions while keeping other agents' actions fixed. This enables effective multi-agent credit assignment without requiring extra simulations, reward models, or hand-designed default actions.

**Core Paradigm**: On-policy policy gradient with centralized training, decentralized execution (CTDE).

### Key Contributions

1. **Centralized Critic for Decentralized Policies**: First major work to leverage a centralized critic Q(s, τ, u) conditioning on global state and joint actions to train decentralized actors π^a(u^a|τ^a).
2. **Counterfactual Baseline for Credit Assignment**: agent-specific advantage by marginalizing out each agent's action:
   ```
   A^a(s, τ, u) = Q(s, τ, u) - Σ_u'^a π^a(u'^a|τ^a)Q(s, τ, (u^-a, u'^a))
   ```
   This baseline has **zero expected contribution** to the gradient (like standard baselines) but explicitly addresses credit assignment by comparing actual action to counterfactual alternatives.
3. **Efficient Critic Representation**: outputs Q-values for all |U| actions of agent a conditioned on other agents' actions u^-a as input — counterfactual baseline in a **single forward pass** instead of |U|^n outputs.

### Architecture Details

**Actor** (Decentralized): 128-unit GRU per agent (parameter sharing); input (o^a_t, a, u^a_{t-1}); bounded-softmax action probabilities; conditions only on local history τ^a.

**Critic** (Centralized): feedforward ReLU network; input (u^-a_t, s_t, o^a_t, a, u_{t-1}, h_t); outputs Q for all actions of agent a; single critic shared across agents; trained with TD(λ=0.8) + target networks.

### Counterfactual Baseline vs Alternatives

**Relation to Difference Rewards**: D^a = r(s,u) - r(s,(u^-a, c^a)) needs a simulator / reward model / hand-designed default c^a. COMA uses the learned critic to evaluate counterfactuals — no extra simulation.

**Relation to Aristocrat Utility**: same mathematical form as the COMA advantage, but in value-based methods it creates a recursive policy↔utility dependency. COMA's policy-gradient baseline has zero expected contribution → no self-consistency problem.

**Why not just Q - V?** The central-QV baseline A^a = Q(s,τ,u) - V(s,τ) averages over ALL agents' actions; COMA marginalizes only agent a's action → stronger credit signal.

### Experimental Results: StarCraft Micromanagement

Decentralized benchmark: restricted field of view (= firing range), no macro-actions, invalid actions inflate the effective action space. Scenarios: 3m, 5m (Marines), 5w (Wraiths), 2d_3z (Dragoons + Zealots). Global reward = damage dealt − 0.5×damage taken + 10×kills + 200×win + remaining health.

Mean win % (final 1000 eval episodes):

| Map | IAC-V | IAC-Q | central-V | central-QV | COMA | Heuristic |
|-----|-------|-------|-----------|------------|------|-----------|
| 3m  | 47±3  | 56±6  | 83±3      | 83±5       | **87±3** | 35 |
| 5m  | 63±2  | 58±3  | 67±5      | 71±9       | **81±5** | 66 |
| 5w  | 18±5  | 57±5  | 65±3      | 76±1       | **82±3** | 70 |
| 2d_3z | 27±9 | 19±21 | 36±6     | 39±5       | **47±5** | 63 |

**Key findings**: COMA >> baselines; centralized critics >> decentralized critics; COMA > central-QV (counterfactual baseline matters); best COMA agents approach centralized DQN/GMEZO despite decentralized execution + restricted FOV.

### Ablations

- **IAC** (independent actor-critic, per-agent critic on τ^a only): much worse than centralized critics, especially on heterogeneous 2d_3z.
- **central-V** (V(s) + TD error): good but less stable than COMA, no explicit credit assignment.
- **central-QV** (Q and V, A = Q−V): strictly worse than COMA — counterfactual baseline is crucial.

### Comparison to Value Decomposition Methods

| Aspect | VDN/QMIX | COMA |
|--------|----------|------|
| **Approach** | Value-based (Q-learning) | Policy-based (actor-critic) |
| **Policy** | Implicit (argmax Q) | Explicit stochastic policy |
| **Learning** | Off-policy | On-policy |
| **Credit Assignment** | Value decomposition | Counterfactual baseline |
| **Critic Input** | Q_tot(τ, u) | Q(s, τ, u) |
| **Exploration** | ε-greedy over Q-values | Stochastic policy |
| **Sample Efficiency** | More efficient (replay buffer) | Less efficient (on-policy) |
| **Continuous Actions** | Difficult | Natural extension |

### Connections

- **Contrasts with**: [VDN](./vdn.md), [QMIX](./qmix.md) — value-based vs policy-based credit assignment
- **Related modern baseline**: [MAPPO](./mappo.md) — simpler centralized value often outperforms COMA's counterfactual machinery
- **Builds on**: difference rewards (Wolpert & Tumer 2002), aristocrat utilities
- **Concurrent work**: MADDPG (Lowe et al. 2017) — similar centralized critic for competitive/continuous settings

## Open questions

1. **On-policy sample inefficiency**: cannot use a replay buffer (later work adds stabilized replay).
2. **Centralized critic scalability**: critic input scales with agents/actions (outputs only |U|, not |U|^n).
3. **High variance**: policy gradients are higher variance than value methods even with a baseline.
4. **Multi-agent exploration**: coordinated exploration still hard as agent count grows.
5. **Global state requirement**: critic needs true state s during training (standard CTDE assumption).

### Relevance to comm-vs-ctde project

1. **Alternative to value decomposition**: policy gradients reach performance comparable to VDN/QMIX — a different algorithmic toolbox.
2. **Counterfactual reasoning**: "what if agent a did something different?" is conceptually related to communication (agents sharing info about their counterfactual options).
3. **On-policy nature**: may interact differently with communication than off-policy methods — communication could reduce on-policy variance.
4. **Heterogeneous agents**: 2d_3z results relevant to Stalker/Zealot experiments.
5. **Research angle**: does communication help policy-gradient methods more than value methods? Use the EPyMARL COMA implementation rather than reimplementing.