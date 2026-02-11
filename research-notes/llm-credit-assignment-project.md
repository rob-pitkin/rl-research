# LLM-Based Credit Assignment in Cooperative MARL

**Project Status**: PAUSED (Hardware Constraints)
**Start Date**: January 6, 2026
**Pause Date**: February 2, 2026
**Environment**: Level-Based Foraging (LBF)
**Framework**: PettingZoo + PyTorch

---

## Why Paused

After setting up infrastructure and running baseline experiments, paused due to hardware limitations:

**LLM Inference Latency on M1 MacBook Pro:**
- Even small language models (Gemma 1B, Llama 3.2 1B/3B) have significant inference latency on M1
- Querying LLM episodically would slow down training substantially
- Training 500k timesteps already takes hours; adding LLM queries would make experiments impractical
- Could potentially work with cloud GPU/Colab, but defeats purpose of compute-constrained independent research

**What Was Accomplished:**
- ✅ Literature review (COMA, QMIX, VDN, SIMA 2)
- ✅ LBF environment setup and exploration
- ✅ Trajectory→text translation implementation
- ✅ EPyMARL integration for baselines
- ✅ Multi-seed experiment runner
- ✅ IQL baseline experiments
- ✅ LLM integration design decisions

**Learning:**
- Infrastructure and experimental design skills transferred to other projects
- Deeper understanding of credit assignment methods
- Realistic assessment of compute constraints for LLM-RL integration

**Potential Future Directions:**
- Revisit with cloud compute if available
- Try distilled models or quantized LLMs for lower latency
- Focus on post-hoc analysis rather than training-time credit assignment

---

## Research Question

Can small language models (SLMs) provide effective credit assignment signals in cooperative multi-agent reinforcement learning tasks with sparse shared rewards? How does LLM-based credit assignment compare to traditional methods (QMIX, COMA, VDN, IQL)?

## Hypothesis

SLMs can reason about agent contributions in semantically interpretable environments, providing credit assignment that:
1. Matches or exceeds traditional methods in learning speed/final performance
2. Offers interpretability advantages (natural language explanations)
3. Works even with episodic queries (not requiring per-step computation)

---

## Week 1 Plan: Foundations & Setup

### Goal
By end of week 1, you should have:
- ✅ Solid understanding of prior work
- ✅ LBF environment running with basic agents
- ✅ Clear experimental design
- ✅ Baseline implementation started

---

### Day 1-2: Literature Review (4-6 hours)

**Task 1.1: Core Credit Assignment Papers**
Read and take notes on:
- [x] COMA paper (Foerster et al., 2018) - counterfactual baselines
- [x] QMIX paper (Rashid et al., 2018) - value decomposition
- [x] VDN paper (Sunehag et al., 2017) - simple value decomposition
- [ ] QPLEX paper (Wang et al., 2020) - improved decomposition

**Questions to answer**:
- What credit assignment problem does each solve?
- What are computational costs?
- What environments do they test on?
- What are known failure modes?

**Deliverable**: `literature-review.md` with 1-paragraph summary of each paper

**Note from QMIX**:
"The number of networks required scales linearly with the number of agents."

This is a limitation for QMIX. But how does LLM credit assignment scale?

- If you query the LLM once per episode regardless of agent count, that's constant cost per episode (but still scales with episode length description)
- If you query per-agent, it scales linearly like QMIX
- Trajectory length grows with number of agents → longer text → slower LLM inference

**Comparison Table**:

| Method | Decomposition  | Expressiveness | Implementation | LLM Integration Potential   |
|--------|----------------|----------------|----------------|-----------------------------|
| VDN    | Additive (Σ)   | Low            | Easy           | High (reward decomposition) |
| QMIX   | Monotonic      | Medium         | Complex        | Low (no credit signal)      |
| COMA   | Counterfactual | High           | Medium         | High (augment advantages)   |
| IQL    | None           | N/A            | Very Easy      | Medium (auxiliary rewards)  |

---

**Task 1.2: LLM + RL Papers**
Read and take notes on:
- [x] SIMA 2 paper (you already have this!) - focus on hindsight relabeling section
- [ ] "Language Models as Zero-Shot Planners" (Huang et al., 2022) - if accessible
- [ ] Any recent "LLM for RL" papers from Arxiv (search: "language model reinforcement learning")

**Questions to answer**:
- How do they represent RL state/actions for LLMs?
- What LLM architectures/sizes do they use?
- How often do they query the LLM?
- What are latency/compute costs?

**Deliverable**: Add to `literature-review.md`

---

**Task 1.3: Level-Based Foraging Background**
- [ ] Read LBF paper/documentation
- [ ] Understand state/action space
- [ ] Understand reward structure (cooperative, sparse, etc.)
- [ ] Identify 2-3 specific scenarios to test (e.g., 2 agents, 2 food items, grid size 8x8)

**Deliverable**: Add environment notes to project file

---

### Day 3-4: Environment Setup & Exploration (4-6 hours)

**Task 2.1: Install Dependencies**
- [x] Set up Python environment with `uv` (you're already familiar with this)
- [ ] Install PettingZoo, level-based-foraging, PyTorch
- [ ] Test LBF environment loads and runs

```bash
# Expected commands
uv init llm-credit-assignment
uv add pettingzoo lbforaging torch numpy
```

**Deliverable**: Working environment, basic test script

---

**Task 2.2: Environment Familiarization**
Write a script that:
- [ ] Runs random agents in LBF for 10 episodes
- [ ] Logs state, action, reward, next_state for each step
- [ ] Computes episode lengths, total rewards, success rates
- [ ] Visualizes or renders a few episodes (if LBF supports rendering)

**Key observations to note**:
- What does the state representation look like?
- How sparse are rewards?
- How long are episodes typically?
- What actions are available?

**Deliverable**: `explore_lbf.py` script + observations in project notes

---

**Task 2.3: Design State → Text Translation**
This is critical for your approach. Create a deterministic function that converts:
- State observation → natural language description
- Action → natural language description
- Reward → natural language description

**Example**:
```
State: "Agent 1 at (2,3), Agent 2 at (5,6). Food A at (2,5) [level 2], Food B at (7,7) [level 1]. Agent 1 facing north, Agent 2 facing east."

Action: "Agent 1 moved north. Agent 2 moved east."

Reward: "No food collected. Team reward: 0."
```

**Deliverable**: `trajectory_to_text.py` with translation functions + unit tests

---

### Day 5-6: Experimental Design (3-4 hours)

**Task 3.1: Define Metrics**
Decide what you'll measure:
- [ ] Learning curves (episode reward vs timesteps/episodes)
- [ ] Sample efficiency (timesteps to reach threshold performance)
- [ ] Final performance (average reward over last N episodes)
- [ ] Success rate (% episodes where task completed)
- [ ] Compute cost (wall-clock time, LLM queries per episode)
- [ ] Interpretability metrics (how often LLM credit matches ground truth?)

**Deliverable**: Add "Evaluation Metrics" section to project file

---

**Task 3.2: Design Experiments**
Plan your experiments:

**Experiment 1: Baselines**
- [ ] IQL (Independent Q-Learning) - simplest baseline
- [ ] VDN - simple credit assignment
- [ ] QMIX (if time permits) - more complex

**Experiment 2: LLM Credit Assignment Variants**
- [ ] Post-hoc analysis (query LLM after training, analyze trajectories)
- [ ] Episodic credit (query every N episodes, use as auxiliary reward signal)
- [ ] Compare SLM sizes (1B vs 3B) - does size matter for credit assignment?

**Parameters to decide**:
- Grid size, number of agents, number of food items
- Training budget (number of episodes/timesteps)
- LLM query frequency (every 10 episodes? every 50?)
- Hyperparameters (learning rate, epsilon decay, replay buffer size)

**Deliverable**: Add "Experimental Design" section with clear plan

---

**Task 3.3: Set Up Project Structure**
Organize your codebase:

```
llm-credit-assignment/
├── environments/
│   └── lbf_wrapper.py          # Environment setup
├── agents/
│   ├── iql.py                  # Independent Q-Learning
│   ├── vdn.py                  # Value Decomposition Network
│   └── llm_credit.py           # LLM credit assignment agent
├── llm/
│   ├── trajectory_to_text.py   # State/action translation
│   ├── llm_interface.py        # SLM loading and querying
│   └── prompts.py              # Prompt templates
├── training/
│   ├── train.py                # Main training loop
│   └── evaluate.py             # Evaluation script
├── utils/
│   ├── logger.py               # Logging utilities
│   └── visualization.py        # Plotting, rendering
├── configs/
│   └── experiment_configs.yaml # Hyperparameters
├── experiments/
│   └── results/                # Store results here
└── tests/
    └── test_translation.py     # Unit tests
```

**Deliverable**: Create project directory with this structure

---

### Day 7: Baseline Implementation Kickoff (2-3 hours)

**Task 4.1: Implement IQL Agent**
Start with the simplest baseline:
- [ ] Q-network architecture (MLP with state → Q-values per action)
- [ ] Epsilon-greedy exploration
- [ ] Experience replay buffer
- [ ] Training loop (sample, compute loss, update)

You don't need to finish this in week 1, but get the skeleton set up.

**Deliverable**: `agents/iql.py` with basic structure

---

**Task 4.2: Run Sanity Check**
- [ ] Train IQL agent for 1000 episodes
- [ ] Plot learning curve
- [ ] Verify it learns something (better than random)

**Deliverable**: Basic training working, even if performance isn't good yet

---

## Week 1 Deliverables Checklist

By end of week 1, you should have:

- [ ] `literature-review.md` with summaries of 5-7 key papers
- [ ] LBF environment installed and working
- [ ] `explore_lbf.py` script demonstrating environment
- [ ] `trajectory_to_text.py` with state/action translation
- [ ] Clear experimental design documented in this file
- [ ] Project directory structure created
- [ ] IQL baseline started (even if incomplete)
- [ ] At least one sanity check training run completed

---

## Notes & Observations

### Environment Details
*(Fill in after Day 3)*

### Translation Design Decisions
*(Fill in after Day 4)*

### LLM Integration Design Decisions

**Decision 1: When to Query LLM**
- Query only during training (not evaluation) when reward > 0
- Aligns with CTDE (Centralized Training, Decentralized Execution)
- No point in credit assignment when reward = 0
- Rationale: Credit assignment only matters when there's reward to distribute

**Decision 2: Information Provided to LLM**
- Start with full trajectory using `trajectory_to_text.py`
- Future experiment: Compare full vs summary trajectory
- Full trajectory preserves all information deterministically
- Rationale: Summaries risk losing important details; can optimize later if needed

**Decision 3: LLM Output Format**
- Structured JSON with percentages and reasoning (similar to SIMA 2)
- Format:
  ```json
  {
    "agent_0_credit": <0-100>,
    "agent_1_credit": <0-100>,
    "reasoning": "<brief explanation>"
  }
  ```
- Percentages must sum to 100
- Rationale: Easy parsing, quantitative signal, plus interpretability from reasoning

**Decision 4: How to Use LLM Output**
- Decompose team reward based on LLM credit percentages
- Example: team_reward=1.0, LLM says 60/40 → agent_0 gets 0.6, agent_1 gets 0.4
- Train VDN utilities with decomposed individual rewards
- Rationale: Clean integration with VDN, most direct use of LLM signal

**Decision 5: Error Handling**
- If JSON parsing fails: Retry once with strict prompt ("Output ONLY valid JSON, no other text")
- If retry fails: Fallback to equal split (50/50)
- Log all failures for analysis
- Rationale: Graceful degradation, don't crash training on LLM errors

### Open Questions
- What's the optimal LLM temperature? (0.7 for creativity vs 0.1 for consistency?)
- How to handle stochasticity? (LLM might give different answers for same trajectory)
- Should we cache LLM responses for identical trajectories?
- What's the performance impact of LLM queries? (need to measure latency)

### Risks & Concerns
- LBF might be too simple (agents learn quickly without credit assignment)
- LLM might not provide useful signal (random/noisy credit)
- Translation might lose important information
- Compute cost might be higher than expected

---

## Next Steps (Week 2 Preview)

- Complete baseline implementations (IQL, VDN)
- Set up SLM locally (Llama 3.2 or Phi-3)
- Design and test prompt templates
- Implement LLM credit assignment integration
- Run pilot experiments

---

## Research Log

**2026-01-10**: EPyMARL integration & experiment infrastructure complete! Achievements:
- ✅ Set up Linear project with 9 tasks
- ✅ Reorganized project into clean structure (llm-credit-assignment/ subdirectory)
- ✅ Installed and configured EPyMARL for MARL baselines
- ✅ Integrated LBF with EPyMARL (disabled SMAClite import, works with gymma wrapper)
- ✅ Built multi-seed experiment runner (`experiments/run_epymarl.py`)
- ✅ Ran initial IQL baseline (250k timesteps, 1 seed) - agents learning!
- ✅ Discovered 250k insufficient - learning curves still climbing
- ✅ Designed LLM integration approach (5 key decisions documented)
- ✅ Started rigorous IQL baseline (500k timesteps, 5 seeds) - currently training

**Key Findings**:
- LBF requires 500k+ timesteps for convergence (not 250k)
- EPyMARL plotting infrastructure works great for visualization
- Multi-seed runner enables rigorous experiments

**Files Created**:
- `experiments/run_epymarl.py` - Multi-seed experiment runner with time tracking
- Added Gemma 1B and new environments to ideas.md
- Documented LLM integration design decisions

**LLM Integration Design Finalized**:
- Query during training when reward > 0 (CTDE framework)
- Use full trajectory with `trajectory_to_text.py`
- Structured JSON output with credit percentages + reasoning
- Decompose team reward based on LLM credit
- Graceful error handling (retry → fallback to equal split)

**Next Steps**:
- Complete IQL 5-seed baseline (500k) - currently running
- Plot and analyze IQL results
- Run VDN baseline (5 seeds, 500k)
- Set up Gemma 1B inference
- Implement LLM integration into VDN

**2026-01-09**: Environment setup complete! Achievements:
- ✅ Installed all dependencies (PettingZoo, lbforaging, PyTorch, etc.)
- ✅ Created and ran exploration script - observed 10 episodes with random agents
- ✅ Documented LBF environment characteristics in literature review
- ✅ Implemented state→text translation functions (`trajectory_to_text.py`)

**Key Findings from Environment Exploration**:
- Random agents achieve 0% success rate (shows cooperation is essential)
- Rewards are very sparse (only when food collected)
- State space is clean and interpretable (positions + levels)
- Translation to natural language works well - clear, deterministic mappings

**Files Created**:
- `test_lbf.py` - Quick environment verification
- `explore_lbf.py` - Detailed exploration with statistics
- `trajectory_to_text.py` - State/action/reward → text translation

**2026-01-07**: Literature review in progress.
- ✅ Read and summarized COMA, QMIX, VDN papers
- ✅ Read SIMA 2 paper (LLM as reward function)
- ✅ Identified LLM-credit assignment as novel research gap

**2026-01-06**: Project started. Week 1 plan created. Ready to dive into literature review.
