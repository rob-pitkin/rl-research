# Counterfactual PPO in MARL - Literature Review

**Goal**: Understand the landscape of counterfactual advantage estimation and PPO-based methods in MARL to determine if combining counterfactual advantages with PPO clipping is novel and valuable.

**Research Question**: Does combining COMA's counterfactual advantage estimation with PPO's trust region optimization improve upon existing methods (MAPPO, COMA) for tasks requiring explicit credit assignment?

---

## PPO (2017): "Proximal Policy Optimization Algorithms"

**Citation**: Schulman et al., 2017 - [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

**Core Idea**:
-

**Key Innovation**:
- Clipped surrogate objective for trust region optimization
-

**Advantages over TRPO**:
-

**Key Results**:
-

**Why it became dominant in single-agent RL**:
-

**Notes**:
-

---

## COMA (2018): "Counterfactual Multi-Agent Policy Gradients"

**Citation**: Foerster et al., 2018 - [arXiv:1705.08926](https://arxiv.org/abs/1705.08926)

**Core Idea**:
-

**Counterfactual Advantage Function**:
- How it's computed:
- Why it helps credit assignment:
-

**Architecture**:
- Centralized critic:
- Decentralized actors:
-

**Key Results**:
-

**Environments Used**:
-

**Limitations**:
- Uses vanilla policy gradients (no trust region)
-

**Notes**:
-

---

## MAPPO (2021): "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"

**Citation**: Yu et al., 2021 - [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)

**Core Idea**:
-

**Architecture**:
- Centralized value function:
- Decentralized PPO actors:
-

**How it handles credit assignment**:
-

**Key Results**:
-

**Why it became SOTA**:
-

**Limitations**:
- No explicit counterfactual baseline
-

**Comparison to COMA**:
-

**Notes**:
-

---

## HAPPO (2021): "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning"

**Citation**: Kuba et al., 2021 - [arXiv:2109.11251](https://arxiv.org/abs/2109.11251)

**Core Idea**:
-

**Key Innovation**:
- Heterogeneous agents with non-shared policies
- Sequential update scheme
-

**How it differs from MAPPO**:
-

**Theoretical Guarantees**:
- Monotonic improvement guarantee:
-

**Credit Assignment Approach**:
-

**Key Results**:
-

**Notes**:
-

---

## PRD-MAPPO (2024): "Assigning Credit with Partial Reward Decoupling in Multi-Agent Proximal Policy Optimization"

**Citation**: Kapoor et al., 2024 - [arXiv:2408.04295](https://arxiv.org/abs/2408.04295)

**Core Idea**:
-

**Partial Reward Decoupling**:
- How it works:
- Attention mechanism:
-

**How it improves credit assignment**:
-

**Key Results**:
-

**Comparison to MAPPO and COMA**:
-

**Is it counterfactual?**:
-

**Notes**:
-

---

## CGRPA (2025): "Curriculum Learning With Counterfactual Group Relative Policy Advantage For Multi-Agent Reinforcement Learning"

**Citation**: arXiv:2506.07548 - [PDF](https://arxiv.org/pdf/2506.07548)

**Core Idea**:
-

**Counterfactual Advantage Function**:
- How it differs from COMA:
- Group relative policy optimization influence:
-

**Curriculum Learning Component**:
-

**Policy Update Mechanism**:
- Does it use PPO clipping?
- Trust region approach:
-

**Key Results**:
-

**Environments Used**:
-

**How close is this to our idea?**:
-

**Notes**:
-

---

## Research Gaps & Potential Contribution

### What exists:
1. **COMA**: Counterfactual advantages + vanilla policy gradients
2. **MAPPO**: Standard advantages + PPO clipping
3. **CGRPA**: Counterfactual advantages + curriculum learning + group relative optimization

### What doesn't exist (or isn't well-explored):
1. Clean combination: Counterfactual advantages (COMA-style) + PPO clipping (vanilla PPO-style)
2. Systematic comparison on diverse environments
3.

### Our potential contribution:
1.
2.
3.

### Key questions to answer:
1. Does CGRPA already do this? If so, how is our approach different?
2. When does explicit counterfactual credit assignment matter vs. implicit (MAPPO)?
3. What environments best demonstrate the benefit?
4.

---

## Environment Selection

### Environments Where Credit Assignment Matters
- **SMAC**: Used by COMA, CGRPA, MAPPO
- **Level-Based Foraging**: Already familiar, explicit cooperation requirements
- **Multi-Agent MuJoCo**: Continuous control, heterogeneous agents
-

### Candidate Environments for Our Work
1. **Level-Based Foraging**
   - Pros: Already set up, understand it well, clear cooperation mechanics
   - Cons: Might be too simple
   - Credit assignment challenge: Different agent levels contribute differently

2. **SMAC (2s3z, 3s5z, etc.)**
   - Pros: Standard benchmark, direct comparison to other work
   - Cons: Many papers already use it
   - Credit assignment challenge: Different unit types, micro-management

3. **[Other environment]**
   - Pros:
   - Cons:
   - Credit assignment challenge:

---

## Potential Experimental Design

### Research Question
Does combining COMA's counterfactual advantage estimation with PPO's clipped objective improve performance and sample efficiency over MAPPO and COMA on tasks requiring explicit credit assignment?

### Hypothesis
Counterfactual PPO (CPPO) will:
1. Outperform COMA due to trust region stability
2. Outperform MAPPO on tasks with clear differential agent contributions
3. Match MAPPO on simpler tasks where implicit credit assignment suffices

### Baselines to Compare
- **COMA**: Counterfactual baseline
- **MAPPO**: Current SOTA
- **IQL**: Independent learning baseline
- **QMIX**: Value decomposition baseline (if discrete actions)
-

### Metrics
- Episode return (mean/std across seeds)
- Sample efficiency (learning curves)
- Credit assignment quality (if measurable)
- Training stability (variance in returns)
-

### Ablations to Consider
1. CPPO vs COMA (isolate effect of PPO clipping)
2. CPPO vs MAPPO (isolate effect of counterfactual baseline)
3. Different environments (where does it help?)
4.

### What to Visualize
- Learning curves (return vs timesteps)
- Advantage estimates over time
- Agent contribution analysis
-

---

## Implementation Plan

### Approach
- Start with EPyMARL's COMA implementation
- Copy to new `cppo.py` and `cppo.yaml` files
- Modify policy update to use PPO clipping instead of vanilla PG
- Keep counterfactual advantage computation identical

### Key Code Changes
1. Policy update: Replace vanilla PG with clipped surrogate objective
2. Add old policy storage for importance sampling ratio
3. Add PPO hyperparameters (clip_epsilon, entropy coefficient, etc.)
4.

### Testing Plan
1. Verify CPPO matches COMA when clip_epsilon → ∞
2. Test on simple environment (2s3z or LBF)
3. Compare to baselines
4.

---

## Notes & Ideas

(Add thoughts, questions, and ideas as you read)

