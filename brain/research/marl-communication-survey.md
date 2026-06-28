---
title: MARL Communication Survey (research note)
description: Provisional analysis of the 9-dimensional Comm-MADRL taxonomy and where comm-vs-ctde sits within the design space.
type: research-note
status: provisional
sources:
  - external-sources/marl-communication-survey.mdx
created: 2026-04-08
tags:
  - research
  - provisional
  - marl
  - communication
  - survey
  - emergent-language
  - ctde
---
## Question

What is the systematic design space of communication in MARL (Comm-MADRL), and where does the comm-vs-ctde question — communication vs state-based coordination — sit within it?

## Sources cited

- [A Survey of Multi-Agent Deep RL with Communication](../external-sources/marl-communication-survey.mdx) (Zhu, Dastani, Wang; Utrecht; 2022/2024) — 41 Comm-MADRL methods across 9 dimensions.

## Findings

### Overview

The first systematic taxonomy for **learning tasks with communication** in MADRL. It distinguishes *learning tasks with communication* (solving domain tasks via communication) from *emergent language* (learning symbolic language as the primary goal), and analyzes 41 methods across **9 dimensions**.

### The 9-Dimensional Taxonomy

**Problem settings**
1. **Controlled goals** — Cooperative (36 methods) >> Mixed (7) > Competitive (2).
2. **Communication constraints** — Unconstrained ≈ Bandwidth-limited (VBC, TMC, MAIC, SchedNet, IMAC); Corrupted (DIAL noise, R-MACRL).
3. **Communicatee type** — Other agents (26) > Proxy (8) ≈ Nearby (8).

**Communication processes**
4. **Communication policy** — Predefined (full broadcast: CommNet, TarMAC; partial structure: DGN, VBC) vs Learnable (individual gating: IC3Net, ATOC; global: SchedNet, MAGIC).
5. **Communicated messages** — Existing knowledge (37: observations/history, RNN-encoded) >> Imagined future (4: ATOC intentions, NeurComm policy fingerprints).
6. **Message combination** — Unequally valued (22: attention — TarMAC, MAGIC) > Equally valued (17: mean/sum — CommNet, IC3Net).
7. **Inner integration** — Policy-level (15: CommNet, ATOC, IC3Net) > Both (14: BiCNet, TarMAC) > Value-level (10: DIAL, VBC, TMC, MAIC).

**Training processes**
8. **Learning methods** — Differentiable (25, dominant) > Reinforced (8) > Supervised (5) ≈ Regularized (5: NDQ maximize MI, IMAC minimize MI).
9. **Training schemes** — CTDE + parameter sharing (27, standard) > Decentralized (9) > CTDE individual (6) > CTDE concurrent (2).

### When Does Communication Help?

**Helps**: partial observability, non-stationarity, heterogeneous agents, large state spaces, complex coordination (intentions/plans).

**Struggles / unnecessary**: independent goals (IC3Net agents learn NOT to communicate in competitive modes); **sufficient state information** ([QMIX](./qmix.md) achieves strong performance with state-based coordination, no runtime communication); poorly-managed bandwidth cost.

> [!NOTE]
> **The survey names the comm-vs-ctde tension explicitly**: QMIX's non-linear mixing + state (via hypernetworks) handles complex coordination *without* runtime communication. The central question — when does explicit communication outperform state-based value decomposition? — is identified as underexplored.

### Method Families (selected)

- **Foundational**: DIAL/RIAL (2016), [CommNet](./commnet.md) (2016), BiCNet (2017).
- **Attention-based**: TarMAC (2019, targeted receiving), ATOC (2019, gating + intentions), MAGIC (2021, learned graph).
- **Bandwidth-constrained**: SchedNet, [VBC](./vbc.md), TMC, IMAC, MAIC.
- **Value decomposition + communication**: [VBC](./vbc.md) (first hybrid), NDQ, TMC, MAIC.
- **Emergent language**: GCL, IC (social influence), Bias, DCSS, AE-Comm.

### Benchmarks

Switch/MNIST games, Traffic Junction, **SMAC** (standard modern benchmark; heterogeneous maps 2s3z/3s5z best differentiate methods), predator-prey, Google Research Football.

### Connections

- **State-based CTDE comparators**: [VDN](./vdn.md), [QMIX](./qmix.md).
- **Communication baseline**: [CommNet](./commnet.md).
- **Hybrid exemplar**: [VBC](./vbc.md).
- **Policy-gradient credit assignment**: [COMA](./coma.md) (Gated-ACML pairs COMA-style actor-critic with communication).

## Open questions (research gaps from the survey)

1. **Non-cooperative settings** — trust, deception, strategic manipulation largely unexplored.
2. **Communication constraints** — most works assume unconstrained; asynchrony and fairness rare.
3. **Explainability** — learned messages are "hidden, deep, obscure codes."
4. **Scalability** — most tested on <10 agents; 100+ underexplored.
5. **Communication efficiency metrics are underused** — a concrete opening for the project to contribute.

### Relevance to comm-vs-ctde project

This survey is the **framework paper** for the project: its 9 dimensions let you characterize contributions precisely, and it identifies the exact gap — most methods are purely state-based OR communication-based, with few exploring adaptive switching or principled combination.

Concrete openings it surfaces:
- **Information theory** — what information is redundant between state and messages? Minimum communication given state access? (IMAC-style bottlenecks accounting for state availability.)
- **Adaptive hybrids** — learn *when* to use state vs messages; communication as "correction" to state-based coordination.
- **Evaluation** — report communication-efficiency metrics (underused) alongside win rate; compare to both state-based (QMIX) and communication (CommNet, IC3Net) baselines on SMAC heterogeneous maps.