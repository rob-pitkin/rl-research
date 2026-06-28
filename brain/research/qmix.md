---
title: QMIX — Monotonic Value Function Factorisation (research note)
description: "Provisional analysis of QMIX: non-linear monotonic value decomposition with state hypernetworks, vs VDN and communication."
type: research-note
status: provisional
sources:
  - external-sources/qmix.mdx
created: 2026-04-06
tags:
  - research
  - provisional
  - marl
  - value-decomposition
  - ctde
  - monotonicity
---
## Question

How does QMIX extend VDN with non-linear monotonic value factorisation, and when does state-conditioned mixing beat both VDN and runtime communication? Relevance to comm-vs-ctde.

## Sources cited

- [QMIX — Monotonic Value Function Factorisation](../external-sources/qmix.mdx) (Rashid et al., Oxford, 2018)

## Findings

### Summary

QMIX extends [VDN](./vdn.md) by learning a **non-linear monotonic** combination of per-agent values instead of a simple sum. Key innovation: enforce monotonicity constraint (∂Q_tot/∂Q_a ≥ 0) to guarantee consistency between centralized and decentralized policies, while allowing much richer value function representations than VDN.

**Core Insight**: Full factorization (VDN) isn't necessary. Only need: argmax_u Q_tot(τ,u) = [argmax_u1 Q1, ..., argmax_un Qn]

### Architecture

**Three components**:
1. **Agent Networks**: DRQN for each agent, outputs Q_a(τ^a, u^a)
2. **Mixing Network**: Monotonic feed-forward network that combines Q_a values into Q_tot
   - Weights restricted to be non-negative (enforces monotonicity)
   - Single hidden layer (32 units) with ELU activation
3. **Hypernetworks**: Generate mixing network weights from global state s
   - Allows Q_tot to depend on state in non-monotonic ways
   - Single linear layer + absolute activation → ensures positive weights

```
Q_tot = MixingNet(Q1, Q2, ..., Qn; weights=HyperNet(s))
```

### Why Hypernetworks?

**Problem**: If you pass state s directly through monotonic network, you over-constrain Q_tot's dependence on s

**Solution**: Use s to generate the *weights* of the monotonic network
- Integrates full state information flexibly
- Maintains monotonicity w.r.t. agent Q-values
- Learns state-dependent mixing

### Monotonicity Constraint

**Mathematical**: ∂Q_tot/∂Q_a ≥ 0 for all agents

**Why it works**:
- Ensures global argmax decomposes into individual argmaxes
- Enables tractable off-policy learning (no exponential action space search)
- Guarantees decentralized execution consistency

**Implementation**: Non-negative mixing network weights

### Representational Power

**Can represent**: Any value function factorizable as monotonic combination of agent utilities

**Cannot represent**: Value functions where agent's best action depends on *simultaneous* actions of others

**Example** (2-agent matrix game):
```
     A    B
A  [ 2    1 ]  ← NOT monotonic (A,A better than A,B but B,B better than B,A)
B  [ 1    8 ]     QMIX cannot perfectly represent this
```

But QMIX approximates such functions **much better** than VDN.

### Experiments: StarCraft II Micromanagement

**Tasks** (decentralized control, partial observability): 3m, 5m, 8m (homogeneous Marines); 2s_3z, 3s_5z (Stalkers + Zealots, heterogeneous); 1c_3s_5z (Colossus + Stalkers + Zealots)

**Setup**: each agent controls one unit; reward = damage dealt + kill bonuses; built-in AI disabled; partial obs via sight range.

**Results**:
- **QMIX >> VDN >> IQL** on all maps
- Largest gap on heterogeneous tasks (2s_3z, 3s_5z, 1c_3s_5z)
- VDN plateaus ~20% win rate on 3s_5z; QMIX reaches ~80%

### Ablations

1. **QMIX-NS** (no state in hypernetworks): worse — state conditioning is critical
2. **QMIX-Lin** (linear mixing, no hidden layer): VDN + state-dependent bias; worse on heterogeneous tasks
3. **VDN-S** (VDN + state-dependent term): better than VDN but worse than QMIX; non-linear mixing essential

**Takeaway**: need **both** state information **and** non-linear mixing for complex coordination.

### Learned Behaviors

**8m (homogeneous)**: both VDN and QMIX learn a "semicircle formation" to attack from sides.

**2s_3z / 3s_5z (heterogeneous)**:
- **VDN**: rushes left, attacks when in range (no positioning)
- **QMIX**: Zealots block enemy Zealots (protecting Stalkers from counters), Stalkers fire from safe distance — **emergent tactical coordination** from value decomposition + state

### Comparison to VDN

| Aspect | VDN | QMIX |
|--------|-----|------|
| Decomposition | Linear sum | Non-linear monotonic |
| State info | No | Yes (via hypernetworks) |
| Representational power | Limited | Much richer |
| Heterogeneous agents | Struggles | Excels |
| Complexity | Simpler | More complex architecture |

### Connections

- **Extends**: [VDN](./vdn.md) with non-linear monotonic mixing
- **Compared to**: [COMA](./coma.md) — on-policy actor-critic vs off-policy value-based
- **Successor**: QTRAN (removes monotonicity constraint), QPLEX, WQMIX
- **Hybrid extensions**: [VBC](./vbc.md) combines QMIX with communication

## Open questions

1. **Monotonicity constraint**: cannot represent all value functions (e.g., coordination games where best action depends on others' simultaneous actions)
2. **Scalability**: tested only up to 9 agents
3. **Sample efficiency**: still requires significant training (2M timesteps)
4. **Exploration**: uses standard ε-greedy; no coordinated exploration

### Relevance to comm-vs-ctde project

**Critical paper** for comm-vs-ctde:
- Demonstrates **state information > communication** in many cases (QMIX uses state via hypernetworks, no inter-agent comm during execution)
- Shows value decomposition enables the CTDE paradigm effectively
- Heterogeneous agent results directly relevant to Stalker/Zealot experiments

**Key insight**: non-linear mixing + state information can handle complex coordination without runtime communication. QMIX is likely a strong baseline — the open question is whether a communication architecture can outperform QMIX's state-based coordination.