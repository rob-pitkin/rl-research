---
title: VDN — Value-Decomposition Networks (research note)
description: "Provisional analysis of VDN: additive value decomposition for cooperative MARL, and its relevance to comm-vs-ctde."
type: research-note
status: provisional
sources:
  - external-sources/vdn.mdx
created: 2026-04-06
tags:
  - research
  - provisional
  - marl
  - value-decomposition
  - credit-assignment
  - cooperative
---
## Question

What does VDN contribute to cooperative MARL, and how does additive value decomposition enable decentralized execution from centralized training? Relevance to the comm-vs-ctde project.

## Sources cited

- [VDN — Value-Decomposition Networks For Cooperative Multi-Agent Learning](../external-sources/vdn.mdx) (Sunehag et al., DeepMind, 2017)

## Findings

### Summary

VDN introduces a learned additive value decomposition approach for cooperative MARL with a single joint reward. The key insight: decompose the team Q-function into individual agent Q-functions that depend only on local observations, enabling decentralized execution while maintaining centralized training.

**Core Equation**: `Q(h¹, h², ..., hᵈ, a¹, a², ..., aᵈ) ≈ Σᵢ Q̃ᵢ(hⁱ, aⁱ)`

### Key Contributions

1. **Value Decomposition Architecture**: Novel network that learns to decompose team value into agent-wise components through backpropagation from joint reward
2. **Solves Two Critical Problems**:
   - **Lazy Agent Problem**: When one agent learns useful policy, others become discouraged from exploration (their exploration hurts team reward)
   - **Spurious Rewards**: Independent learners can't distinguish teammate actions from environment stochasticity
3. **Centralized Training, Decentralized Execution**: Each agent acting greedily w.r.t. local Q̃ᵢ equivalent to central arbiter maximizing Σ Q̃ᵢ

### Architecture Details

- Based on DQN with enhancements:
  - LSTM for partial observability
  - Dueling architecture (V + Advantage)
  - Multi-step returns with eligibility traces (λ=0.9)
  - Experience replay + target networks
- Each agent: Linear(32) → ReLU → LSTM(32) → ReLU → Dueling layer
- Summation layer at top aggregates individual Q̃ᵢ into joint Q

### Additional Techniques Evaluated

1. **Weight Sharing**: Reduces parameters, enforces agent invariance (helps avoid lazy agent)
2. **Role Information**: 1-hot ID concatenated with observations when specialized roles needed
3. **Information Channels**: Differentiable connections between agent networks
   - Low-level: after first linear layer
   - High-level: after LSTM

### Experimental Results

**Domains** (2-agent, partial observability):
- **Switch**: Coordinate corridor usage to reach opposite ends
- **Fetch**: Synchronize pickup/dropoff cycles
- **Checkers**: Asymmetric rewards (sensitive/insensitive agents)

**Key Finding**: Value decomposition architectures dramatically outperform both:
- Centralized agent with combinatorial action space
- Independent learners with team reward

**Best Configuration**: VDN + weight sharing + role info + low-level communication

### The Learned Decomposition

Figure 6 shows emergent credit assignment in Fetch task:
- Total Q (yellow) anticipates all team rewards
- Individual Q̃₁ (green) and Q̃₂ (purple) **autonomously learn** to attribute rewards to correct agent
- System disambiguates joint rewards without individual reward signals

### Connections

- **Successor**: [QMIX](./qmix.md) — uses non-linear monotonic decomposition
- **Contrasts with**: [COMA](./coma.md) — policy gradient approach to credit assignment
- **Builds on**: credit-assignment and Dec-POMDP foundations
- **Related work**: difference rewards (requires full state), coordination graphs

## Open questions

- Assumes additive decomposition (restrictive for complex coordination)
- Evaluated only on 2-agent scenarios
- Future: non-linear value aggregation, scaling to larger teams

### Relevance to comm-vs-ctde project

This is **foundational** for understanding value decomposition methods in MARL. The "lazy agent problem" and spurious reward issues are critical challenges you'll encounter. VDN's simplicity (linear summation) makes it a good baseline, but its limitations motivate more sophisticated approaches like [QMIX](./qmix.md).

VDN demonstrates that value decomposition enables decentralized execution after centralized training — directly relevant to the communication vs. centralization tradeoffs at the heart of the comm-vs-ctde project.