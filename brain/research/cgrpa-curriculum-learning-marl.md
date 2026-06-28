---
title: CGRPA — Curriculum Learning + Counterfactual Group Relative Advantage (research note)
description: "Provisional analysis of CGRPA: dynamic curriculum (FlexDiff) + counterfactual group-relative credit assignment for MARL."
type: research-note
status: provisional
sources:
  - external-sources/cgrpa-curriculum-learning-marl.mdx
created: 2026-04-08
tags:
  - research
  - provisional
  - marl
  - curriculum-learning
  - credit-assignment
  - counterfactual
  - non-stationarity
  - ctde
---
## Question

How does CGRPA address "environmental meta-stationarity" (fixed-difficulty opponents) via curriculum learning + counterfactual group-relative credit assignment, and how is it orthogonal to comm-vs-ctde?

## Sources cited

- [CGRPA — Curriculum Learning with Counterfactual Group Relative Policy Advantage for MARL](../external-sources/cgrpa-curriculum-learning-marl.mdx) (Jin, Du, Liu, Kim; arXiv:2506.07548, 2025)

## Findings

### Summary

CGRPA addresses **"environmental meta-stationarity"** — the overlooked problem that most MARL algorithms train against fixed-difficulty opponents, causing overfitting and local optima. It introduces a **dynamic curriculum** that progressively adjusts opponent strength, plus a **counterfactual group relative policy advantage** to stabilize learning during difficulty transitions.

**Paradigm**: CTDE + curriculum learning + enhanced credit assignment.

### Key Contributions

1. **Names the meta-stationarity problem** — fixed difficulty (SMAC's default Level 7) limits generalization.
2. **FlexDiff** — statistical adaptive difficulty scheduler: dual-metric (win rate + reward) evaluation, momentum-driven adjustment buffer, dynamic boundary constraints. Starts at Level 5, adapts up to Level 10.
3. **CGRPA algorithm** — first integration of Group Relative Policy Optimization (GRPO) with [COMA](./coma.md)-style counterfactual policy gradients: `A^CF_i = Q_tot(s,u) - E[Q_tot(s,(u^{-i}, ū_i))] - α D_KL(π_i || π̄_g)` (counterfactual baseline + group alignment). Breaks QMIX monotonicity to enable asymmetric credit.
4. **Extensive SMAC validation** — ~20 maps; QMIX baseline 45–60% → 75–90% on 5m_vs_6m; large gains on Hard/Super-Hard maps.

### Experimental Results

- **Easy** (1c3s5z, 3s5z): 100% win rate, avoids local optima that trap QMIX/VDN; faster convergence.
- **Hard** (2c_vs_64zg +8–14% over EMC; 7s7z/5s10z +20–40% over QMIX). Slower initial convergence (starts easier) but higher final.
- **Super-Hard** (27m_vs_30m ~40%, highest among baselines; 3s5z_vs_3s6z +30% vs QMIX).

**Ablations**: FlexDiff *alone* causes instability (performance collapse at transitions); CGRPA's credit assignment is essential for stable adaptation. λ=0.5 optimal; λ=−1 hurts (the counterfactual term actively improves credit, not just regularizes). Some Super-Hard maps' short episode limits *hide* true performance (agents win at t=220 but limit cuts at 180).

### Connections

- **Base architecture**: [QMIX](./qmix.md) (monotonic mixing) — CGRPA modifies training, plug-and-play.
- **Counterfactual inspiration**: [COMA](./coma.md) — CGRPA = COMA's counterfactual + GRPO's group optimization.
- **Orthogonal to communication**: [VBC](./vbc.md) addresses *when to communicate*; CGRPA addresses *training curriculum* — potential synergy.
- **Modern baseline that could benefit from CL**: [MAPPO](./mappo.md).

## Open questions

1. FlexDiff hyperparameters (window, thresholds) are task-specific, no a-priori method.
2. Scheduler is hand-crafted (not learned) — meta-RL alternative unexplored.
3. SMAC-only evaluation (all combat tasks).
4. Episode-length artifacts underestimate win rate.
5. Heterogeneous coordination below QPLEX on MMM2.
6. **No comparison to communication methods** (CommNet/IC3Net/TarMAC/VBC) — the comm-vs-ctde gap.

### Relevance to comm-vs-ctde project

**Orthogonal but complementary**: the project asks communication vs state-based coordination; CGRPA asks curriculum vs fixed-difficulty training. The key transferable insight is that **training methodology matters as much as algorithm design** — SMAC's default Level-7 opponents may cause overfitting that confounds a fair comm-vs-state comparison.

Concrete extensions:
- Apply FlexDiff curriculum to *both* [QMIX](./qmix.md) and [CommNet](./commnet.md)/[VBC](./vbc.md) baselines for a fair comparison under progressive difficulty.
- Test the hypothesis that **communication becomes more valuable at higher difficulty** (more coordination needed) — plot communication overhead β vs difficulty level.
- Use CGRPA's counterfactual advantage to evaluate per-agent *message* importance (which agents should communicate at which difficulty).