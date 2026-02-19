# Communication vs CTDE: A Comparative Study

**Project Status**: Literature Review & Design (Feb 2026)
**Start Date**: February 5, 2026
**Environment**: MPE (simple_reference, simple_world_comm)
**Framework**: marlbenchmark/on-policy (MAPPO/VDN implementations)

---

## Research Question

**What is the relative value of centralized training information versus decentralized execution communication?**

Can limited communication during execution compensate for lack of centralized training information? Or are they complementary capabilities that work best together?

---

## Motivation

Most MARL research treats communication and CTDE as separate enhancements:
- CTDE methods (VDN, QMIX, MAPPO) use centralized information during training but no communication at test time
- Communication methods often assume independent learning or don't systematically compare against CTDE

**Gap**: No systematic comparison of the trade-offs between centralized training and decentralized communication.

**Why it matters**:
- Practical: Can communication enable simpler training infrastructure (no centralized state)?
- Theoretical: Understanding when runtime coordination beats offline information sharing
- Design: Knowing whether to invest in CTDE infrastructure or communication mechanisms

---

## Experimental Design

### 2x2 Ablation Study

|                  | No Communication | With Communication |
|------------------|------------------|-------------------|
| **No CTDE (IQL)**| IQL (baseline)   | IQL + comm        |
| **With CTDE**    | VDN              | VDN + comm        |

**All methods use parameter sharing** - architectural choice, not training paradigm. The key distinction remains:
- IQL methods train on local observations only
- VDN methods train with centralized state information

### Hypotheses

**Primary Hypothesis**: VDN+comm > VDN > IQL+comm > IQL
- Each enhancement (CTDE, communication) provides value
- Best performance requires both

**Alternative Hypothesis**: IQL+comm ≈ VDN on certain tasks
- Communication can substitute for CTDE when runtime coordination is critical
- VDN to IQL+comm gap is smaller than expected

**Null Hypothesis**: VDN+comm ≈ VDN
- Communication provides minimal value when centralized training already provides coordination

---

## Environments

### Target Environments (MPE)

1. **simple_reference** - Communication-focused task
2. **simple_world_comm** - World with explicit communication needs
3. **simple_spread** (optional) - Has communication in obs space but unclear action space

**Why MPE**:
- Native communication channels in some scenarios
- On-policy repo already supports MPE
- MAPPO baseline already trained on simple_spread

### Environment Investigation Needed
- [ ] How do simple_reference and simple_world_comm implement communication natively?
- [ ] What's the message format? (Discrete? Continuous?)
- [ ] How are messages integrated into observations and action spaces?
- [ ] Can we use the native communication or do we need custom wrapper?

---

## Communication Design

### Design Decisions

**Message Space**:
- Start with 4-8 discrete messages
- Rationale: Compact enough to interpret, expressive enough to be useful
- Can expand to 16 if needed

**Communication Topology**:
- Start with broadcast (all agents receive the message)
- Future: Add targeted communication if broadcast works
- Rationale: Simpler action space, easier to implement and debug

**Action Space Integration**:
- Separate action heads: agents can move AND communicate simultaneously
- Movement action + communication action per timestep
- Rationale: Most flexible, doesn't force trade-off between moving and talking

**Observation Integration**:
- Concatenate all received messages to observation vector
- Each agent sees messages from all other agents
- Rationale: Simple, matches typical MPE observation structure

**Communication Frequency**:
- Agents can communicate every timestep
- No gating mechanism initially (all messages always sent)
- Future: Add learned gating (IC3Net-style) if interesting

---

## Implementation Plan

### Phase 1: Literature Review (Current)
- [x] Start communication survey paper
- [ ] Finish communication survey paper
- [ ] Investigate MPE communication environments (simple_reference, simple_world_comm)
- [ ] Read value-level communication papers (VBC, NDQ, TMC, MAIC) - understand VDN+comm
- [ ] Read core communication papers (CommNet, IC3Net, TarMAC, RIAL/DIAL)
- [ ] Identify which approach to adapt for experiments

### Phase 2: Baseline Setup
- [ ] Run IQL baseline on chosen MPE environment (no communication)
- [ ] Run VDN baseline on chosen MPE environment (no communication)
- [ ] Verify baselines match expected performance

### Phase 3: Communication Implementation
- [ ] Design communication wrapper for MPE (or use native if available)
- [ ] Implement IQL + comm (independent Q-learning with message passing)
- [ ] Implement VDN + comm (value decomposition with communication)
- [ ] Unit test communication integration (messages sent/received correctly)

### Phase 4: Experiments
- [ ] Run full 2x2 ablation (IQL, IQL+comm, VDN, VDN+comm)
- [ ] Multiple seeds (3-5) for statistical significance
- [ ] Track standard metrics + communication-specific metrics
- [ ] Visualize results on wandb

### Phase 5: Analysis
- [ ] Compare learning curves across all 4 conditions
- [ ] Analyze when/how agents use communication
- [ ] Measure communication efficiency (message diversity, usage patterns)
- [ ] Test hypotheses: Is IQL+comm competitive with VDN?
- [ ] Document findings

---

## Metrics to Track

### Standard RL Metrics
- Episode return (learning curves)
- Success rate (task completion)
- Episode length
- Training time / sample efficiency

### Communication-Specific Metrics
- Message usage frequency (% of timesteps each message is sent)
- Message diversity (entropy over message distribution)
- Temporal message patterns (do certain messages cluster in time?)
- Per-agent communication behavior (do some agents communicate more?)
- Communication necessity: Does ablating communication at test time hurt performance?

### Comparison Metrics
- Performance gap: VDN vs IQL+comm (is communication a substitute for CTDE?)
- Additive value: (VDN+comm - VDN) vs (IQL+comm - IQL) (does comm help more for IQL or VDN?)
- Sample efficiency: Timesteps to reach threshold performance for each method

---

## Wandb Organization

**Project Name**: `communication-vs-ctde`

**Run Naming Convention**: `{method}_{env}_{seed}`
- Examples: `iql_simple-reference_seed1`, `vdn-comm_simple-reference_seed3`

**Tags**:
- Method: `iql`, `iql-comm`, `vdn`, `vdn-comm`
- Environment: `simple-reference`, `simple-world-comm`
- Phase: `baseline`, `communication`, `final`

**Dashboards to Create**:
- Learning curves comparison (all 4 methods overlaid)
- Communication usage patterns (message frequencies over time)
- Sample efficiency comparison (timesteps to threshold)
- Ablation results (bar charts comparing final performance)

---

## Open Questions

1. **Value decomposition + communication**: How exactly do VBC/NDQ/TMC integrate messages into the decomposed Q-function?
2. **Message semantics**: Can we interpret what messages mean? (e.g., "help needed", "target identified")
3. **Communication overhead**: Does communication slow down decision-making in practice?
4. **Generalization**: Do communication protocols learned in training work with new partners?
5. **Scaling**: How does communication effectiveness change with number of agents?

---

## Expected Timeline

- **Week 1-2**: Finish literature review, understand VDN+comm integration
- **Week 3**: Investigate MPE environments, finalize communication design
- **Week 4**: Implement baselines (IQL, VDN without communication)
- **Week 5-6**: Implement communication variants (IQL+comm, VDN+comm)
- **Week 7-8**: Run full experiments, analyze results
- **Week 9**: Write up findings, create visualizations

**Total**: ~2 months

---

## Notes & Observations

### 2026-02-05: Project Kickoff
- Identified research gap while reading communication survey
- Key insight: No systematic comparison of CTDE vs communication trade-offs
- Decided on 2x2 ablation with parameter sharing across all methods
- Prioritized reading VBC/NDQ/TMC/MAIC to understand value decomposition + communication

---

## References

### To Read
- [ ] VBC: Value-based Communication (need citation)
- [ ] NDQ: (need citation)
- [ ] TMC: (need citation)
- [ ] MAIC: (need citation)
- [ ] CommNet: Learning Multiagent Communication with Backpropagation (Sukhbaatar et al., 2016)
- [ ] IC3Net: Learning When to Communicate (Singh et al., 2018)
- [ ] TarMAC: Targeted Multi-Agent Communication (Das et al., 2019)
- [ ] RIAL/DIAL: Learning to Communicate (Foerster et al., 2016)
- [ ] Communication Survey: A survey of multi-agent deep reinforcement learning with communication (arXiv:2203.08975, 2024)

### Key Papers (Already Read)
- VDN: Value-Decomposition Networks (Sunehag et al., 2017)
- QMIX: Monotonic Value Function Factorisation (Rashid et al., 2018)
