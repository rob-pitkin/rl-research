---
title: CommNet — Learning Multiagent Communication with Backpropagation (research note)
description: "Provisional analysis of CommNet: differentiable broadcast communication, when it helps, and vs state-based coordination."
type: research-note
status: provisional
sources:
  - external-sources/commnet.mdx
created: 2026-04-08
tags:
  - research
  - provisional
  - marl
  - communication
  - differentiable-communication
  - cooperative
  - ctde
---
## Question

How does CommNet's differentiable broadcast communication work, when does communication help over independent/state-based baselines, and how does it relate to value-decomposition coordination? Relevance to comm-vs-ctde.

## Sources cited

- [CommNet — Learning Multiagent Communication with Backpropagation](../external-sources/commnet.mdx) (Sukhbaatar, Szlam, Fergus; NYU / FAIR; NIPS 2016)

## Findings

### Summary

CommNet introduces a **differentiable continuous communication architecture** for fully cooperative multi-agent tasks. The key innovation is a broadcast communication channel where agents exchange continuous-valued vectors that are averaged and fed back as input at each layer. Because communication is continuous (not discrete), the entire model is differentiable and can be trained end-to-end with backpropagation through the communication channel.

**Paradigm**: CTDE with parameter sharing across agents.

### Core Architecture

For each communication step `i ∈ {0, ..., K}`:

```
h^{i+1}_j = f^i(h^i_j, c^i_j)                    # Agent update
c^{i+1}_j = (1/(J-1)) * Σ_{j'≠j} h^{i+1}_{j'}   # Communication (mean pooling)
```

Full pipeline: **Encoder** `h^0_j = r(s_j)` → **K communication steps** → **Decoder** `a_j ~ q(h^K_j)` (softmax over actions).

When `f^i` is linear + nonlinearity (`h^{i+1}_j = σ(H^i h^i_j + C^i c^i_j)`), the model becomes a feedforward net with a structured, dynamically-sized, permutation-invariant weight matrix.

**Extensions**: local connectivity (communicate only within range), skip connections (input encoding at all steps), temporal recurrence (LSTM modules performed best).

### Training

Policy gradient with a state-specific baseline `b(s,θ)` (α=0.03 baseline weight); also supports supervised learning when action labels exist. **Key advantage**: continuous communication is differentiable, so backprop flows through the channel — no REINFORCE on communication itself.

### Experimental Results

**Lever Pulling** (coordination game): CommNet 99% success (supervised) vs 59% for independent agents. PCA of hidden states shows smooth ordering by agent ID — agents learn to communicate identities.

**Traffic Junction** (partial observability + coordination), failure rate %:

| Module | Independent | Fully-connected | Discrete comm | CommNet |
|--------|-------------|-----------------|---------------|---------|
| MLP    | 20.6±14.1   | 12.5±4.4       | 15.8±9.3     | **2.2±0.6** |
| RNN    | 19.5±4.5    | 34.8±19.7      | 15.2±2.1     | **7.6±1.4** |
| LSTM   | 9.4±5.6     | 4.8±2.4        | 8.4±3.4      | **1.6±1.0** |

With zero visibility (blind driving), CommNet still succeeds ~90%. Communication vectors are mostly "silent" (near zero) — a sparse protocol emerges; clusters correlate with positions requiring coordination.

**Combat** (5v5, heterogeneous, dynamic), win rate %:

| Module | Independent | Fully-connected | Discrete comm | CommNet |
|--------|-------------|-----------------|---------------|---------|
| MLP    | 34.2±1.3    | 17.7±7.1       | 29.1±6.7     | **44.5±13.4** |
| RNN    | 37.3±4.6    | 2.9±1.8        | 33.4±9.4     | **44.4±11.9** |
| LSTM   | 44.3±0.4    | 19.6±4.2       | 46.4±0.7     | **49.5±12.6** |

With m=10 agents/team, CommNet 45.4% vs Independent 30.5%. The fully-connected baseline catastrophically fails (overfits to agent order).

**bAbI QA**: CommNet mean error 7.1% (vs LSTM 36.4%), worse than specialized memory networks (MemN2N 4.2%, DMN+ 2.8%).

### When Does Communication Help?

1. **Partial observability** — sharing observations expands awareness (largest gains under low visibility).
2. **Coordination requirements** — tasks where actions must be coordinated (levers, junction).
3. **Dynamic agent count** — architecture handles varying team size naturally.
4. **Sparse communication suffices** — agents learn to broadcast only when needed.

Communication struggles when local visibility is sufficient, or with discrete (harder to learn) channels.

### Connections

- **Survey context**: [MARL communication survey](./marl-communication-survey.md) features CommNet as a foundational architecture (cooperative, predefined full-broadcast policy, mean-pooling combination, differentiable learning, CTDE).
- **State-based contrast**: [VDN](./vdn.md), [QMIX](./qmix.md) coordinate via value decomposition + global state rather than runtime messages.
- **Hybrid extension**: [VBC](./vbc.md) combines QMIX with communication under bandwidth constraints.
- **Policy gradient pairing**: [COMA](./coma.md) — both on-policy actor-critic + parameter sharing + CTDE.
- **Follow-ups (not yet indexed)**: IC3Net (gating), TarMAC (attention), MAGIC (mixed settings), TMC/MAIC (bandwidth).

## Open questions

1. **No gating** — agents always communicate; IC3Net adds learnable gates.
2. **Broadcast scalability** — mean pooling loses agent-specific info; TarMAC adds attention.
3. **No explicit message semantics** — continuous vectors are hard to interpret (PCA clusters only).
4. **Limited evaluation vs state-based methods** — no comparison to VDN/QMIX value decomposition (**the gap comm-vs-ctde directly addresses**).
5. **Full cooperation assumption** — shared global reward, no mixed/competitive settings.
6. **Sample efficiency** — on-policy policy gradient (8.6M episodes for traffic junction).

### Relevance to comm-vs-ctde project

The comm-vs-ctde project investigates **exactly the gap CommNet leaves open**: CommNet shows communication helps; [QMIX](./qmix.md) shows state-based coordination helps; the open question is *which is better, when, and can they be combined optimally?*

Use CommNet as the **communication baseline** (simple, well-understood, differentiable). Extend by: direct comparison to QMIX on the same SMAC tasks; a hybrid (QMIX value decomposition + CommNet channel); information-theoretic analysis of communication-vs-state redundancy; bandwidth-aware communication building on [VBC](./vbc.md). CommNet is foundational — understand it thoroughly as the communication baseline.