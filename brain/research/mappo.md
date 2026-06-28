---
title: MAPPO — Multi-Agent PPO (research note)
description: "Provisional analysis of MAPPO: simple on-policy PPO as a strong cooperative MARL baseline, and its role in comm-vs-ctde."
type: research-note
status: provisional
sources:
  - external-sources/mappo.mdx
created: 2026-04-08
tags:
  - research
  - provisional
  - marl
  - ppo
  - on-policy
  - policy-gradient
  - ctde
  - cooperative
  - baseline
---
## Question

Why does simple on-policy PPO become a state-of-the-art cooperative MARL baseline, what implementation factors matter, and how should MAPPO anchor the comm-vs-ctde baselines?

## Sources cited

- [MAPPO — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](../external-sources/mappo.mdx) (Yu et al., Tsinghua / UC Berkeley, NeurIPS 2022)

## Findings

### Summary

MAPPO demonstrates that **PPO, a simple on-policy algorithm, achieves surprisingly strong performance in cooperative MARL**, often matching or exceeding state-of-the-art off-policy methods (QMIX, RODE, QPLEX) in both final returns and sample efficiency. With proper implementation choices (value normalization, agent-specific global states, minimal data reuse, small clipping, large batches), vanilla PPO becomes a competitive baseline across diverse cooperative benchmarks.

**Core paradigm**: on-policy policy gradient with optional centralized training, decentralized execution (CTDE).

### Key Contributions

1. **Empirical demonstration across 4 benchmarks**: MPE (3 cooperative tasks), SMAC (beats QMIX on 18/23 maps), Google Research Football (vastly outperforms QMIX), Hanabi (matches/exceeds VDN and SAD).
2. **Sample efficiency comparable to off-policy methods** — often converges in 10M timesteps (same budget as QMIX/RODE).
3. **Five critical implementation factors**: value normalization, value-function input representation, training-data usage, PPO clipping strength, batch size.
4. **Practical baseline**: no domain-specific tricks, open-source implementation.

### Architecture

- **MAPPO**: decentralized policy πθ(a_i|o_i); centralized value Vφ(s) takes **global state** during training; standard PPO + GAE; value discarded at execution.
- **IPPO**: both policy and value use local observations only (fully decentralized) — surprisingly competitive.
- Design choices: parameter sharing, separate policy/value networks, death masking (SMAC), PopArt normalization.

### Implementation Details That Matter

1. **Value normalization** (PopArt-style) — critical on MPE Spread, stabilizes hard SMAC maps (MMM2, corridor).
2. **Value-function input** — **Agent-Specific (AS)** and **Feature-Pruned (FP)** global states vastly outperform concatenated-local (CL) and environment-provided (EP). Include both global context AND agent-specific features.
3. **Training data usage** — 5–15 epochs, 1–2 mini-batches (vs 30–50 epochs / 32–64 mini-batches in single-agent PPO). Multi-agent non-stationarity punishes data reuse.
4. **PPO clipping** — ε = 0.05–0.2; **0.05 often best** on hard maps (limits policy change → less non-stationarity).
5. **Batch size** — sufficiently large batch critical for performance and sample efficiency.

### Experimental Results

**SMAC** (10M timesteps): MAPPO wins/ties on **18/23 maps** vs QMIX. Hard maps (5m_vs_6m 89%, 3s5z 97%, corridor 100% vs QMIX 76/88/84%); super-hard 27m_vs_30m 94% vs QMIX 39%. IPPO competitive (15/23 vs QMIX).

**GRF**: MAPPO vastly outperforms QMIX (3v1 88% vs 8%; CA-hard 77% vs 3%); beats pretrained TiKick on 4/5 scenarios.

**MPE**: comparable/superior to QMIX and MADDPG on all tasks.

**Hanabi**: 2-player 23.89 avg; 5-player 23.04 avg (exceeds VDN 21.28). Centralized value advantage grows with agent count.

### Comparison to Other Methods

| Aspect | MAPPO | QMIX | COMA | CommNet |
|--------|-------|------|------|---------|
| **Learning** | On-policy | Off-policy | On-policy | On/off-policy |
| **Policy** | Explicit stochastic | Implicit (argmax Q) | Explicit stochastic | Explicit |
| **Credit Assignment** | Centralized value | Value decomposition | Counterfactual baseline | Parameter sharing |
| **Value Input** | Global state | State (mixer) + local obs | Global state + joint actions | N/A |
| **Sample Efficiency** | Medium (no replay) | High (replay) | Low | Medium |
| **SMAC Performance** | 18/23 > QMIX | Strong baseline | Not competitive | Not tested |

### Connections

- **Competes with**: [VDN](./vdn.md), [QMIX](./qmix.md) — centralized VALUE vs value DECOMPOSITION; MAPPO often wins on complex coordination.
- **Vs**: [COMA](./coma.md) — MAPPO's simple centralized V outperforms COMA's counterfactual machinery in practice.
- **Vs communication**: [CommNet](./commnet.md) — no direct comparison in the MAPPO paper (a major gap comm-vs-ctde can fill).

## Open questions

1. **Discrete actions only** — continuous-action performance unstudied.
2. **Cooperative only** — no mixed/competitive scenarios.
3. **Homogeneous agents** — parameter sharing nearly everywhere; heterogeneous teams unclear.
4. **No communication comparison** — doesn't test CommNet/IC3Net/TarMAC.
5. **Implementation sensitivity** — 5+ factors need tuning; not "plug and play."
6. **Theoretical gaps** — why fewer epochs / smaller clipping help is argued empirically, not proven.

### Relevance to comm-vs-ctde project

**MAPPO is essential** — it represents the modern (2022) on-policy "pure CTDE" baseline: global-state coordination without communication, strong and battle-tested.

**Baseline ladder for the project**: IPPO (independent) < MAPPO (centralized value) < [QMIX](./qmix.md) (value decomposition); plus [CommNet](./commnet.md) (pure communication) and a MAPPO+CommNet hybrid (potential contribution).

**Research questions MAPPO sharpens**: When does communication outperform state-based coordination? Is communication redundant when the centralized critic already has full state? Does communication help policy-gradient methods differently than value methods? MAPPO shows **simple methods with proper implementation can be SOTA** — establish it as the on-policy CTDE baseline before claiming communication helps. Use the authors' open-source `on-policy` implementation rather than reimplementing.