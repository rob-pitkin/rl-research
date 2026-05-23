# Research Ideas

## Active Ideas
<!-- Ideas currently being explored or designed -->

### 1. Emergent Communication in Cooperative MARL

**Current Status**: Starting literature review (Feb 2026)

**Core Idea**: Study how agents learn to coordinate through communication in cooperative tasks. Investigate emergent communication protocols, when agents choose to communicate, and what information they transmit.

**Research Question**: How does adding communication channels affect multi-agent coordination? What communication protocols emerge? Can we interpret what agents are "saying" to each other?

**Environment**:
- Level-Based Foraging (LBF) - already set up
- Simple cooperative tasks where communication could help

**Technical Approach**:
- Baseline: Agents with no communication
- Experiment 1: Discrete message passing (small vocabulary of tokens)
- Experiment 2: Continuous communication channel
- Analyze when/why agents communicate
- Visualize communication patterns and emergent protocols

**Baselines to Compare**:
- No communication (IQL, VDN)
- CommNet or similar architectures
- Hand-designed communication protocols

**Why It's Interesting**:
- Communication is understudied in MARL
- Interpretability: can visualize what agents communicate
- Practical: understanding coordination without explicit design
- Great for blog posts (visual, intuitive results)

**Compute Feasibility**: ✅ Very feasible - same setup as current work, just add communication layer

**Next Steps**:
1. Implement simple message-passing in EPyMARL
2. Add communication channel to agent architecture
3. Train with/without communication on LBF
4. Visualize communication patterns
5. Write blog post with findings

**Blog Post Ideas**:
- "What Do MARL Agents Say to Each Other? Visualizing Emergent Communication"
- "Do RL Agents Need to Talk? Communication in Cooperative Foraging"

---

### 2. Counterfactual PPO (CPPO) - Combining COMA with PPO Trust Regions

**Status**: COMPLETED - Negative Result (Feb 2026)

**Conclusion**: PPO clipping is fundamentally incompatible with counterfactual advantages. Policy-dependent baselines shift as the policy changes, breaking PPO's assumptions. Extensive testing with 1-step TD, target networks, double Q-learning, and various hyperparameters all resulted in entropy collapse and poor performance. Without PPO clipping, the algorithm reduces to COMA. See `research-notes/cppo-implementation-plan.md` for details.

**Core Idea**: Combine COMA's counterfactual advantage estimation with PPO's clipped objective for better credit assignment AND sample efficiency in cooperative MARL.

**Research Question**: Does combining COMA's explicit counterfactual credit assignment with PPO's trust region optimization outperform existing methods (MAPPO, COMA) on tasks requiring fine-grained credit assignment?

**Motivation**:
- COMA has explicit credit assignment (counterfactual advantages) but uses vanilla policy gradients → less stable/sample-efficient
- MAPPO has PPO stability but uses standard advantages → implicit credit assignment only
- HAPPO/PRD-MAPPO improve on MAPPO but still don't use counterfactual baselines
- CGRPA (2025) combines counterfactual + curriculum learning, but clean COMA+PPO combo isn't well-explored

**Environment**:
- SMACLite (2s3z, 3s5z) - standard benchmark, direct comparison to other work
- Level-Based Foraging - already familiar, clear credit assignment challenges
- Multi-Agent MuJoCo (if time) - continuous control setting

**Technical Approach**:
1. Start with EPyMARL's COMA implementation
2. Keep counterfactual advantage computation identical
3. Replace vanilla policy gradient update with PPO's clipped surrogate objective
4. Add importance sampling ratio and PPO hyperparameters

**Baselines to Compare**:
- COMA (counterfactual + vanilla PG)
- MAPPO (standard advantages + PPO)
- QMIX (value decomposition baseline)
- IQL (independent learning baseline)

**Hypothesis**:
- CPPO > COMA on sample efficiency and stability (PPO benefits)
- CPPO > MAPPO on tasks with clear differential agent contributions (counterfactual benefits)
- CPPO ≈ MAPPO on simpler tasks where implicit credit assignment suffices

**Why It's Interesting**:
- Addresses gap between credit assignment (COMA) and sample efficiency (MAPPO)
- Clean algorithmic contribution (well-motivated combination)
- Systematic analysis of when explicit credit assignment matters
- Recent CGRPA (2025) validates counterfactual+PPO direction, but focuses on curriculum learning

**Compute Feasibility**: ✅ Very feasible - same compute as COMA/MAPPO baselines, SMACLite designed for limited resources

**Novelty Check**:
- Need to read CGRPA paper carefully to understand overlap
- CGRPA uses group relative policy optimization + curriculum learning
- Our approach: vanilla COMA counterfactual + vanilla PPO clipping (simpler, cleaner baseline)

**Next Steps**:
1. Read CGRPA paper to assess novelty and differentiation
2. Read COMA paper for deep understanding of counterfactual advantages
3. Read MAPPO paper for SOTA baseline understanding
4. Fill out literature review template
5. Implement CPPO by modifying EPyMARL's COMA
6. Run experiments on SMACLite and LBF
7. Write blog post with findings

**Blog Post Ideas**:
- "Counterfactual PPO: The Missing Link Between COMA and MAPPO"
- "When Does Explicit Credit Assignment Matter in MARL?"
- "Combining the Best of COMA and PPO for Multi-Agent Credit Assignment"

---

### 3. Transformer-Based Agents in SMACLite with Attention Analysis

**Core Idea**: Replace RNN/LSTM/GRU components in MARL algorithms with transformers, focusing on interpretability - what do agents attend to during cooperation?

**Research Question**: What temporal patterns and agent relationships do transformers learn to attend to in cooperative MARL? Can attention mechanisms reveal emergent cooperation strategies?

**Environment**: SMACLite (lightweight StarCraft II)

**Technical Approach**:
- Implement transformer-based policy networks (last K observations as sequence)
- Could explore cross-agent attention (agents attend to each other's hidden states)
- Visualize attention weights over time and across agents
- Analyze which timesteps/agents get high attention during critical moments

**Baselines**:
- QMIX with GRU
- MAPPO with LSTM
- Standard value-based methods

**Why It's Interesting**:
- Transformers underexplored in MARL vs single-agent RL
- Attention provides interpretability - can visualize what matters
- Potential for better temporal reasoning than RNNs

**Compute Feasibility**: ✅ SMACLite designed for limited compute, small transformer models tractable

**Challenges**:
- Fixed context window (transformer can't handle infinite history)
- Might need to combine with RNN for longer-term memory
- Incremental novelty unless interpretability angle is strong

**Next Steps**:
1. Set up SMACLite environment
2. Implement baseline QMIX/MAPPO with GRU
3. Replace with transformer, match performance
4. Build attention visualization tools
5. Analyze emergent patterns

---

## Potential Ideas
<!-- Ideas to explore further -->

### 3. Zero-Shot Coordination
**Idea**: Train agents independently (self-play or with different training partners), then test if they can cooperate zero-shot with never-seen partners.

**Compute**: ✅ Very feasible - training is parallelizable
**Novelty**: Good - growing area, interesting failure modes
**Angle**: Analyze what policies generalize vs overfit to specific partners

---

### 4. Curriculum Learning for MARL
**Idea**: Design curriculum for progressively harder cooperative scenarios. Study how curriculum design affects final performance and sample efficiency.

**Compute**: ✅ Feasible
**Novelty**: Moderate - curriculum RL is known, but MARL-specific curricula less explored
**Angle**: Focus on emergent complexity - when do agents develop advanced strategies?

---

### 5. Sample Efficiency in MARL
**Idea**: Improve sample efficiency through offline MARL, better experience replay, or auxiliary tasks. Compare different replay strategies.

**Compute**: ✅ Feasible - actually reduces compute needs
**Novelty**: High - sample efficiency is critical but underexplored in MARL
**Angle**: Prioritized experience replay for multi-agent, hindsight experience replay variants

---

### 6. Partial Observability Studies
**Idea**: Systematic study of how observation radius/partial observability affects learning, cooperation, and final strategies. What's the minimum info agents need to cooperate?

**Compute**: ✅ Very feasible
**Novelty**: Moderate - more of an analysis/ablation study
**Value**: High for understanding MARL fundamentals

---

### 8. Emergent Behavior Analysis in Simple Games
**Idea**: Study emergent strategies in simple competitive or mixed-motive games. Focus on analyzing what agents learn rather than SOTA performance.

**Compute**: ✅ Very feasible
**Novelty**: Low for methods, high for insights
**Angle**: Deep dives into specific games, surprising behaviors, failure modes

---

### 9. Population-Based Training for MARL (Small Scale)
**Idea**: Maintain population of agents with different hyperparameters/architectures. Study diversity, overfitting to specific opponents, robustness.

**Compute**: ✅ Feasible at small scale (population of 5-10)
**Novelty**: Moderate - PBT is known but MARL applications interesting
**Angle**: Analyze diversity metrics, opponent overfitting

---

### 10. Adversarial Robustness in Cooperative MARL
**Idea**: How do cooperative agents handle adversarial teammates or opponents? Can we train robust cooperative policies?

**Compute**: ✅ Feasible
**Novelty**: Good - robustness is important but underexplored
**Angle**: Compare training against adversarial vs non-adversarial partners

---

### 11. LoRA-RL: Training SLMs with RL via Low-Rank Adaptation
**Idea**: Fine-tune small language models (SLMs) using reinforcement learning and LoRA to learn policies directly from environment rewards. Inspired by "The Era of Experience" paper - testing whether LLMs can learn from experience rather than just human knowledge.

**Technical Approach**:
- Start with pretrained SLM (Phi-3-mini, Llama 3.2 1B/3B)
- Add LoRA adapters to keep training tractable
- Have LLM interact with simple gridworld via:
  - Tool calls representing actions
  - Or direct action outputs (action_1, action_2, etc.)
- Train LoRA weights using policy gradient methods (PPO) based on environment rewards
- Simple environments first (5x5 gridworld) to validate approach

**Compute**: ⚠️ Moderate - feasible with SLMs and LoRA, but sample inefficiency could be brutal. Needs feasibility check.

**Novelty**: Moderate to Low - core idea explored in prior work, but specific angle might be novel:
- Decision Transformer (Chen et al., 2021) - treats RL as sequence modeling
- GATO (Reed et al., 2022) - multi-task transformer trained with RL
- Various LLM+PPO work (AlpacaFarm, etc.)

**Novel Angles to Explore**:
- How does LoRA compare to full fine-tuning for RL? (efficiency vs performance)
- Can we do multi-task RL with LoRA "task vectors"? (different adapters for different environments)
- What's the minimum model size needed for learning from experience?
- How does linguistic pretraining help/hurt RL learning? (ablation: random init vs pretrained)
- Can we use natural language as part of the action space? (hybrid discrete + text actions)

**Papers to Read**:
- [ ] Decision Transformer (Chen et al., 2021)
- [ ] GATO: A Generalist Agent (Reed et al., 2022)
- [ ] LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- [ ] The Era of Experience paper (already read)
- [ ] Recent LLM+RL papers on Arxiv (search: "language model reinforcement learning LoRA" or "fine-tuning LLM RL")

**Why Parked**:
- Different project from current LLM credit assignment work
- Higher risk due to sample efficiency challenges
- Need to finish current project first to build momentum
- Requires feasibility check before committing (can M1 handle LoRA training?)

**Future Action Items** (when revisiting):
1. Run feasibility check (Week 0): Load SLM + LoRA, measure inference latency, memory usage, training speed
2. Read prior work papers to identify gaps
3. Design minimal viable experiment (simplest environment, smallest model)
4. Consider whether this becomes a single-agent RL project or if there's a MARL angle

---

### 12. Recursive Language Models for Long-Horizon RL
**Idea**: Use recursive language models (LMs that can use Python for memory/context storage) to tackle long-horizon RL tasks where standard LLMs fail due to fixed context windows.

**Core Motivation**: Standard LLMs have fixed context windows and can't remember information across thousands of timesteps. Recursive LMs can use Python to store and retrieve context, enabling memory and planning over extended horizons.

**Target Environment**: Pokemon Red (full game) - iconic, long-horizon task with many intermediate objectives
- Stepping stone: Simpler long-horizon tasks first (e.g., Montezuma's Revenge, long gridworld puzzles)

**Technical Approach**:
- Recursive LM can write/execute Python code for memory management
- Store game state, objectives, progress, strategies in Python data structures
- Retrieve context as needed for decision-making
- Use LM for high-level reasoning, Python for reliable memory

**Why It's Compelling**:
- Addresses fundamental limitation of LLMs in RL (memory/context)
- Long-horizon RL is an unsolved hard problem
- Pokemon Red is recognizable and exciting
- Could work for both single-agent and multi-agent (remembering team strategies)

**Compute**: ⚠️⚠️ High - Pokemon Red is complex, many episodes needed
- Start with simpler environments
- May require Colab or cloud compute

**Novelty**: High - recursive LMs are recent, applying to long-horizon RL is unexplored

**Papers to Read**:
- [ ] Recursive Language Models paper (recursiveLanguageModels.pdf)
- [ ] Pokemon Red RL work (recent progress on this?)
- [ ] Long-horizon RL papers (hierarchical RL, options framework)
- [ ] Memory-augmented agents (DNC, NTM if relevant)

**Challenges**:
- Pokemon Red environment setup (ROM, emulator, action space)
- Credit assignment over thousands of steps
- Compute requirements (many episodes needed)
- Debugging recursive LM behavior
- Defining reward structure for intermediate progress

**Connection to Current Work**:
If recursive LMs can help with credit assignment over episodes (current project), they'd be even more powerful for tasks requiring memory across thousands of steps.

**Why Parked** (for now):
- Higher complexity than current project
- Need to finish LLM credit assignment first
- Requires reading recursive LM paper thoroughly
- Would benefit from experience with LLM+RL integration from current project

**Future Action Items**:
1. Read recursive LM paper in detail
2. Survey Pokemon Red RL work (has anyone succeeded? what are known challenges?)
3. Identify simpler long-horizon task as stepping stone
4. Estimate compute requirements
5. Design memory/context management approach

---

### 13. Efficient and portable MCTS implementation in C++
**Idea** Create an MCTS implementation library that is in C++ with Python bindings, use openspiel as inspiration

### 14. PPO.cpp (like llama.cpp)
**Idea** Implementing PPO from scratch in a single file for practice, in C++, and for efficiency.

### 15. Get better at Helix, use Obsidian to create a knowledge base for the above two ideas

## Parked Ideas
<!-- Interesting but not feasible right now or deprioritized -->

### LLM-Based Credit Assignment in Cooperative MARL

**Status**: PAUSED (Feb 2026) - Hardware constraints (M1 MacBook Pro)

**Why Paused**: LLM inference latency makes training impractical on M1. Even small models (Gemma 1B, Llama 3.2 1B/3B) have significant per-query latency. Training already takes hours; adding episodic LLM queries would make experiments unfeasible for independent research without cloud compute.

**Core Idea**: Use small language models (SLMs) to assign credit in multi-agent environments with shared/common rewards. LLMs can reason about "who contributed what" by analyzing trajectory descriptions.

**Research Question**: Can LLMs provide effective credit assignment signals in cooperative MARL tasks with sparse shared rewards? How does this compare to traditional credit assignment methods?

**Environment**:
- Level-Based Foraging (LBF) - preferred for simplicity
- Multi-agent Particle Environment (MPE) - alternative
- Custom gridworld - fallback option
- Key requirement: deterministic translation from (s,a,r,s') to natural language and back

**Technical Approach**:
- Use local SLMs (Llama 3.2 1B/3B, Phi-3 Mini) to keep compute tractable
- Query LLM episodically (every N episodes or post-hoc) rather than every step
- Convert trajectories to natural language descriptions
- LLM outputs credit scores or natural language explanations
- Use as auxiliary reward signal or for hindsight analysis

**Baselines to Compare**:
- COMA (Counterfactual Multi-Agent Policy Gradients)
- QMIX
- QPLEX
- VDN (Value Decomposition Networks)
- IQL (Independent Q-Learning)

**Why It's Novel**: SIMA 2 uses LLMs for hindsight relabeling in single-agent RL, but multi-agent credit assignment is unexplored. Interpretability is a major benefit.

**Compute Feasibility**: ✅ Small SLMs, simple gridworlds, episodic queries

**Next Steps**:
1. Set up LBF environment and baseline agents
2. Implement trajectory → text translation
3. Test SLM credit assignment on simple scenarios
4. Compare learning curves against baselines

---

### AlphaZero-Style Self-Play for Chess/Board Games
**Why Parked**: High compute requirements even for small games. Great learning project but less novel. Could revisit with better compute or after other projects.

### Transformer + RNN Hybrid Architectures
**Why Parked**: Too incremental without strong motivation. Fold into transformer SMACLite project if needed.

---

## Environments to Explore

### SMACLite (StarCraft Multi-Agent Challenge - Lite)
**What it is**: Lightweight version of SMAC for multi-agent cooperative tasks. Units must coordinate to defeat enemies.

**Why interesting**:
- Designed for resource-constrained research (unlike full SMAC)
- Rich cooperative dynamics (unit positioning, focus fire, kiting)
- Good testbed for credit assignment research
- Well-supported in PyMARL/EPyMARL

**Potential uses**:
- Test LLM credit assignment on more complex scenarios than LBF
- Transformer attention analysis (which units attend to each other)
- Compare credit assignment methods on partial observability

**Setup note**: Already available in EPyMARL, though we disabled import initially. Can re-enable when needed (requires installing libspatialindex: `brew install spatialindex`)

---

### Google Research Football
**What it is**: 3D football (soccer) simulation with both simple and complex action spaces. Supports single-agent and multi-agent scenarios.

**Why interesting**:
- Self-play opportunities (team vs team)
- Observation variety (can use pixels, structured state, or minimap)
- Long-horizon decision making
- Rich emergent behaviors and strategies
- Active research community

**Potential uses**:
- Self-play research (learning team strategies)
- Multi-modal observation experiments (combining vision + structured state)
- Testing recursive LMs for long-horizon play
- Credit assignment in team sports (who contributed to goal?)

**Resources**:
- GitHub: https://github.com/google-research/football
- Supports both single-agent and multi-agent (11v11, 5v5, etc.)
- Kaggle competition was based on this environment

**Setup note**: More complex install (requires C++ compiler, dependencies). Good candidate for Google Colab if needed.

---

## Completed Experiments
<!-- Link to blog posts or experiment results -->
