# Experimental Design: LLM-Based Credit Assignment in MARL

**Date**: 2026-01-09
**Status**: Planning

---

## Research Question

**Primary**: Can small language models (SLMs) provide effective credit assignment signals in cooperative multi-agent RL with sparse shared rewards?

**Secondary Questions**:
1. How does LLM-based credit assignment compare to traditional methods (VDN, IQL) in terms of learning speed and final performance?
2. Does LLM size (1B vs 3B) significantly impact credit assignment quality?
3. Can LLMs provide interpretable explanations for credit assignment?
4. What is the computational tradeoff (LLM queries vs traditional methods)?

---

## 1. Evaluation Metrics

### 1.1 Performance Metrics (Primary)

**Learning Curves**:
- **Episode reward vs episodes**: Plot mean team reward over training
- **Success rate vs episodes**: % of episodes where at least one food item collected
- **Sample efficiency**: Episodes/timesteps to reach 50% success rate threshold

**Final Performance** (last 100 episodes):
- Mean team reward
- Success rate
- Episode length (shorter = more efficient)

**Reporting**: Mean ± std over 5 random seeds

---

### 1.2 Credit Assignment Quality Metrics (Secondary)

These are harder to measure directly, but we can approximate:

**Ground Truth Scenarios**:
Design simple test cases where "correct" credit is obvious:
- **Solo collection**: Only 1 agent moves and collects food → should get 100% credit
- **Equal contribution**: Both agents move same distance to food, collect together → should get ~50% each
- **Follower scenario**: Agent A navigates to food, Agent B just follows → Agent A should get more credit

**Metric**: Agreement between LLM credit assignment and ground truth in these scenarios (% correct)

**Credit Variance**:
- Do agents receive differentiated credit when contributions differ?
- Measure variance in credit assignments across episodes
- Low variance might indicate LLM is just splitting 50/50 blindly

---

### 1.3 Interpretability Metrics

**Qualitative**:
- Read LLM explanations for 10-20 episodes
- Categorize: sensible, nonsensical, ambiguous
- Present examples in blog post

**Quantitative** (optional):
- Length of LLM explanations (tokens)
- Sentiment/reasoning patterns (keyword analysis)

---

### 1.4 Computational Cost Metrics

**Wall-clock time**:
- Total training time (minutes/hours)
- Time per episode
- Time for LLM queries (if applicable)

**LLM-specific**:
- Number of LLM queries
- Average inference latency per query
- Total tokens processed

**Comparison**: Normalize by episodes - "X minutes per 1000 episodes"

---

## 2. Experimental Conditions

### 2.1 Baselines (No LLM)

**Baseline 1: Independent Q-Learning (IQL)**
- Simplest: Each agent learns independently
- Receives full team reward (no credit assignment)
- Expected to perform poorly due to multi-agent credit assignment problem

**Baseline 2: Value Decomposition Network (VDN)**
- Decomposes team Q-value into sum of agent utilities
- Implicit credit assignment through value decomposition
- Expected to perform better than IQL

**Baseline 3 (Stretch): QMIX**
- More sophisticated value decomposition with mixing network
- Higher complexity, better performance expected
- Only if time permits / VDN shows promise

---

### 2.2 LLM Credit Assignment Variants

**Variant 1: Post-Hoc Analysis**
- Train baseline (IQL or VDN) normally
- After training, query LLM to analyze successful episodes
- No feedback to learning - purely for interpretability
- **Goal**: Validate that LLM can reason about credit

**Variant 2: Episodic LLM Credit (Auxiliary Reward)**
- Query LLM every N episodes (e.g., N=10, N=50)
- LLM outputs credit percentages for each agent
- Use as auxiliary reward signal (weighted sum with environment reward)
- **Goal**: Test if LLM credit improves learning

**Variant 3: VDN + LLM Reward Decomposition**
- Use VDN architecture
- When team reward > 0, query LLM: "What % did each agent contribute?"
- Decompose team reward according to LLM percentages
- Train agent utilities with decomposed rewards
- **Goal**: Best of both worlds - VDN structure + LLM credit

**Variant 4: LLM Size Comparison**
- Run Variant 2 or 3 with different SLM sizes:
  - Llama 3.2 1B
  - Llama 3.2 3B
  - Phi-3 Mini 3.8B (if feasible)
- **Goal**: Understand size vs credit quality tradeoff

---

### 2.3 LLM Query Frequency Ablation

For episodic variants, test different query frequencies:
- Every episode (expensive, baseline)
- Every 10 episodes
- Every 50 episodes
- Only successful episodes (reward > 0)

**Goal**: Find optimal cost/benefit tradeoff

---

## 3. Experimental Parameters

### 3.1 Environment Configuration

**LBF Environment**: `Foraging-8x8-2p-2f-coop-v3`
- Grid size: 8x8
- Agents: 2
- Food items: 2
- Cooperative: Yes
- Max episode steps: 50

**Why this configuration?**
- Simple enough for quick experiments (~1-2 hour training)
- Complex enough to require cooperation
- Standard benchmark

---

### 3.2 Training Hyperparameters

**Q-Learning (IQL, VDN)**:
- Learning rate: 0.0005 (Adam optimizer)
- Discount factor (γ): 0.99
- Epsilon (exploration): 1.0 → 0.05 (linear decay over 50k steps)
- Replay buffer size: 10,000 transitions
- Batch size: 64
- Target network update frequency: 1000 steps
- Training episodes: 5,000 (or until convergence)
- Random seeds: 5

**Network Architecture**:
- Input: Observation vector (12 dimensions)
- Hidden layers: [128, 128] with ReLU
- Output: 6 Q-values (one per action)

**VDN-specific**:
- Mixing: Simple sum of agent Q-values
- Shared parameters: No (separate networks per agent)

---

### 3.3 LLM Configuration

**Model**: Llama 3.2 1B (primary), 3B (comparison)
- Local inference using `transformers` library
- Quantization: 8-bit (if needed for memory)
- Temperature: 0.7 (balance determinism and creativity)
- Max tokens: 256 (for credit assignment output)

**Prompt Template** (see Section 4)

**Query Frequency**:
- Primary: Every 10 episodes
- Ablation: {1, 10, 50, success-only}

---

### 3.4 Computational Budget

**Per experiment**:
- Max training time: 2-3 hours on M1 MacBook Pro
- If exceeds budget: reduce to 2,500 episodes or use Google Colab

**Total experiments**:
- Baseline 1 (IQL): 5 seeds = 5 runs
- Baseline 2 (VDN): 5 seeds = 5 runs
- Variant 2 (Episodic LLM): 5 seeds × 1-2 configs = 5-10 runs
- Variant 3 (VDN + LLM): 5 seeds = 5 runs
- Variant 4 (Size comparison): 2 sizes × 3 seeds = 6 runs

**Estimated total**: 26-31 training runs + LLM queries

---

## 4. LLM Prompt Design

### 4.1 Credit Assignment Prompt (Variant 2 & 3)

```
You are analyzing a cooperative foraging task with 2 agents. Your job is to assign credit for the team's performance.

Episode Summary:
{trajectory_text}

Team reward: {total_reward}

Based on this episode, what percentage of credit should each agent receive for the team's outcome? Consider:
- Which agent(s) navigated toward food?
- Which agent(s) successfully loaded food?
- Did agents coordinate effectively?
- Were there wasted actions?

Respond ONLY with a JSON object in this exact format:
{
  "agent_0_credit": <percentage 0-100>,
  "agent_1_credit": <percentage 0-100>,
  "reasoning": "<brief explanation>"
}

The two percentages must sum to 100.
```

**Key design choices**:
- Structured output (JSON) for easy parsing
- Include reasoning for interpretability analysis
- Provide guidance on what to consider
- Emphasize constraint (sum to 100)

---

### 4.2 Post-Hoc Analysis Prompt (Variant 1)

```
You are analyzing a cooperative foraging task. Below is an episode where agents successfully collected food.

{trajectory_text}

Analyze this episode and explain:
1. What did each agent contribute to the team's success?
2. Which agent played a more critical role, and why?
3. Were there any inefficiencies or wasted actions?
4. On a scale of 0-100, how would you rate each agent's contribution?

Provide your analysis in clear, structured paragraphs.
```

**Goal**: Rich qualitative feedback for interpretability

---

## 5. Experimental Protocol

### 5.1 Training Procedure

For each experimental condition:

1. **Initialize**:
   - Set random seed
   - Create environment
   - Initialize agent(s) and networks
   - Create replay buffer
   - Load LLM (if applicable)

2. **Training loop** (for each episode):
   - Reset environment
   - Collect trajectory (states, actions, rewards)
   - Store in replay buffer
   - Sample batch and update Q-networks
   - Log metrics (reward, success, episode length)

   **LLM variants only**:
   - Every N episodes OR when reward > 0:
     - Convert trajectory to text
     - Query LLM for credit assignment
     - Decompose reward OR use as auxiliary signal
     - Log LLM response and latency

3. **Evaluation** (every 100 episodes):
   - Run 10 episodes with greedy policy (ε=0)
   - Log mean reward, success rate

4. **Save**:
   - Final model weights
   - Training logs (CSV or JSON)
   - LLM query logs (if applicable)

---

### 5.2 Evaluation Procedure

After training:

1. **Final evaluation** (100 episodes, greedy policy):
   - Mean team reward ± std
   - Success rate
   - Mean episode length

2. **Ground truth scenarios** (if time permits):
   - Run 20 episodes of each test scenario
   - Query LLM for credit assignment
   - Compute accuracy vs ground truth

3. **Interpretability analysis**:
   - Read LLM reasoning for 10-20 diverse episodes
   - Categorize and summarize

---

## 6. Analysis Plan

### 6.1 Quantitative Analysis

**Learning Curves**:
- Plot episode reward vs episodes for all methods
- Shade ±1 std across seeds
- Identify convergence points

**Performance Comparison**:
- Table: Method | Final Reward | Success Rate | Sample Efficiency | Training Time
- Statistical tests: t-test or Mann-Whitney U for pairwise comparisons

**LLM Ablations**:
- Query frequency vs performance (line plot)
- Model size vs performance (bar chart)

---

### 6.2 Qualitative Analysis

**LLM Reasoning**:
- Extract 5-10 example credit assignments
- Categorize: accurate, plausible but wrong, nonsensical
- Present in blog post with commentary

**Failure Mode Analysis**:
- When does LLM credit assignment fail?
- When do baselines fail?
- Are there systematic patterns?

---

### 6.3 Compute Cost Analysis

**Tradeoff Visualization**:
- Scatter plot: Training time (x-axis) vs Final performance (y-axis)
- Point size = LLM query count
- Shows cost/benefit tradeoff

**Cost Table**:
- Method | Training time | LLM queries | Inference time | Total cost

---

## 7. Success Criteria

### Minimum Viable Results (MVR)

**Required for blog post**:
- ✅ At least 2 baselines (IQL + VDN) trained and evaluated
- ✅ At least 1 LLM variant trained and evaluated
- ✅ Learning curves showing comparison
- ✅ Qualitative analysis of LLM reasoning (10+ examples)

**"Successful" experiment**:
- LLM variant performs comparably OR shows interesting failure modes
- Clear insights about when/why LLM credit helps or doesn't
- Interpretability advantage demonstrated

---

### Stretch Goals

- 🎯 LLM variant outperforms VDN
- 🎯 Complete ablation studies (query frequency, model size)
- 🎯 Ground truth credit assignment accuracy > 80%
- 🎯 QMIX baseline for comparison

---

## 8. Potential Issues & Mitigations

| Issue | Likelihood | Mitigation |
|-------|-----------|------------|
| LLM credit is random/useless | Medium | Focus on interpretability angle; negative results are valuable |
| Training takes too long | Medium | Reduce episodes, use Colab, or simplify environment |
| LLM inference too slow | High | Query less frequently; use smaller model; batch queries |
| VDN/baselines don't learn | Low | Check hyperparameters; verify environment is learnable |
| LLM output parsing fails | Medium | Robust parsing with fallbacks; log failures |
| Too much variance across seeds | Medium | Increase seeds to 10; report median instead of mean |

---

## 9. Timeline Estimate

**Week 2** (Implementation):
- Day 1-2: Implement IQL baseline
- Day 3-4: Implement VDN baseline
- Day 5-7: Implement LLM credit assignment integration

**Week 3** (Experimentation):
- Day 1-3: Run baseline experiments (IQL, VDN)
- Day 4-7: Run LLM variant experiments

**Week 4** (Analysis & Writing):
- Day 1-3: Analyze results, create plots
- Day 4-7: Write blog post

**Total**: ~4 weeks from start to published blog post

---

## 10. Open Questions for Discussion

- [ ] Should we prioritize performance (does it work?) or interpretability (what does LLM say?)
- [ ] If LLM credit is worse than VDN, is that still worth publishing?
- [ ] Should we test on multiple environments or go deep on LBF?
- [ ] What's the minimum LLM query frequency that's still informative?
- [ ] Should we fine-tune the SLM on LBF-specific credit assignment, or use zero-shot?

---

## Notes & Ideas

*Space for additional thoughts during experimentation*

