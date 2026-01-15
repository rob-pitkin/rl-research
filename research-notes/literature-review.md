# Literature Review: LLM-Based Credit Assignment

## Credit Assignment in MARL

### COMA (Counterfactual Multi-Agent Policy Gradients)
**Paper**: Foerster et al., 2018
**Key Idea**:
COMA uses a centralized critic with counterfactual baselines to compute advantage functions that isolate each agent's contribution to the team reward.

**Credit Assignment Method**:
Uses a counterfactual baseline that marginalizes out a single agent's action, while keeping
the other agents' actions fixed. COMA learns a centralized critic that estimates the action-value
for a joint action in a given state, it's conditioned on the central state, s. For each agent, a,
we compute an advantage function that compares the Q-value for the current action $u^a$ with
to a counterfactual baseline that marginalizes out the agent's action, $u^a$, while keeping the
other agents' actions fixed.

$$ A^a(s, \tau, u) = Q(s, \tau, u) - \sum_{u^{-a}} \pi^a(u'^a | \tau^{a})Q(s, \tau, (u^{-a}, u'^a)) $$

In this way, we are able to ask "What is the advantage of agent $a$'s action $u^a$ in state $s$ given the joint action $u$?". A larger advantage value means that the agent's action was better than the counterfactual baseline (the expected value of the agent's action given the fixed joint action $u^{-a}$).

**Computational Cost**:
Requires only a single forward pass of the centralized critic network to calculate the advantage for each
agent due to clever construction of the critic network (takes the rest of the joint action as input,
has one output for each possible action of the agent).

**Environments Tested**:
Used SMAC environment for all experiments.

**Strengths**:
- Uses the CTDE paradigm, which enables leveraging information from other agents' trajectories and state information available at training time.
- Centralized critic and counterfactual baseline enable improved credit assignment and performance when compared with individual agents learning their own policies (IAC).
- Without the counterfactual baseline, it's hard to reason about the credit assignment problem since it's a shared reward.
- It can be used with partial observability (just the observations from individual agents and centralized information).

**Limitations/Failure Modes**:
- Doesn't scale well to very large action spaces or a large number of agents.
- On-policy, so not the most sample efficient.

**Relevance to Our Work**:
- A well-known on-policy approach to the credit assignment problem in MARL.
- The CTDE paradigm could also work with LLMs (LLM is available at train time and can use global state information, not at execution time).


---

### QMIX (Monotonic Value Function Factorisation)
**Paper**: Rashid et al., 2018
**Key Idea**:
QMIX learns a joint action-value function that factorizes the action value function into a monotonic function of each individual agent's utility function via a mixing network. This enables end-to-end learning of the joint action-value function while learning decentralized policies for each agent (greedy with respect to to the agent's individual action value function, like Independent Q-Learning).

The key idea is that by improving on VDNs, QMIX enables SOTA performance on SMAC while using off-policy learning (due to the guarantees provided by VDNs and then relaxed to only require monoticity).

**Credit Assignment Method**:
Similar to Value-Decomposition Networks (VDN), QMIX uses a centralized critic to estimate the action-value for a joint action in a given state, it's conditioned on the action and observation histories of all agents. However, instead of summing the action values of each agent, QMIX uses a mixing network to factorize the action value function into a monotonic function of each individual agent's utility function, enabling a significantly higher representation complexity.

From the paper:
"The value function class representable with QMIX includes any value function that can be factored into a non-linear monotonic combination of the agents’ individual value functions in the fully observable setting. This expands upon the linear monotonic value functions that are representable by VDN."

**Computational Cost**:
Relatively higher computational cost due to each agent having its own action value network and then having to compute the mixing network, which uses multiple hypernetworks to compute the mixing weights conditioned on the current state.

Increased computational cost is what we'd face if we were to add LLMs to reason about credit assignment in MARL. So this is a good comparison point for the computational cost of adding LLMs to credit assignment.

**Environments Tested**:
Also used the SMAC environment for all experiments.

**Strengths**:
- Off-policy, so better sample efficiency than something like COMA.
- Achieves better performance than independent Q-learning and is able to leverage extra state information available at training time.
- Strong theoretical guarantees
- Straightforward and well-known end-to-end learning objective.

**Limitations/Failure Modes**:
- The number of networks required scales linearly with the number of agents. Additionally, the complexity of the mixing network also scales linearly with the number of agents. "QMIX scales linearly with agents, LLMs scale with trajectory complexity."
- Complex to implement.

**Relevance to Our Work**:
- A well known off-policy approach to the credit assignment problem in MARL.
- Off-policy learning is more likely to fit our use-case since using LLMs online for credit assignment would be slow.
- QMIX is harder to integrate with LLMs than actor-critic methods, since there's no explicit advantage/credit signal to augment. Would need to modify the learning objective or use LLMs for replay buffer prioritization.

---

### VDN (Value-Decomposition Networks)
**Paper**: Sunehag et al., 2017
**Key Idea**:
VDN represents a midpoint between independent decentralized learning and fully-centralized learning. VDN uses a centralized action-value function that takes as input, the sum of the individual action values of each agent. This enables end-to-end learning of the joint action-value function while learning decentralized policies for each agent (greedy with respect to to the agent's individual action value function, like Independent Q-Learning).

The key observation was that after training end-to-end, each individual agent acting greedily with respect to their utility function is equivalent to acting greedily with respect to the centralized action value function since it's just the sum of the individual utility values.

**Credit Assignment Method**:
Utilizes a centralized action-value function to estimate the action-value for a joint action in a given state. It's factorized into the sum of the utility values of each agent and optimized in an end-to-end fashion. This enables learning from a single shared team reward but still allows each agent to learn decentralized policies for decentralized execution.

**Computational Cost**:
The computational cost is less than QMIX since it doesn't require a mixing network. So, it's just composed of the action-value (technically utility) networks for each agent. This method is actually much easier to implement than QMIX due to the simplicity of the summation operation for the final joint action-value function.

The VDN paper also mentions using parameter sharing for agent invariance. This would lower the computational cost, but wouldn't be compatible with environments that require specialized policies for agents. They introduce the idea of conditionally invariant agents via parameter sharing by adding a one-hot encoding of the agent ID to the input of the individual utility network, but this doesn't scale well with the number of agents and I'm wary of using one-hot encodings to represent completely different policies in the same network.

**Environments Tested**:
2D Grid worlds called "Switch", "Fetch", and "Checkers". Each environment is partially observable to individual agents with a viewing window of 3x5x5 (RGB).

Agents were required to cooperate to either achieve reward (via sensitive and less sensitive agent reward functions) or navigate environments (a single tunnel or two tunnels of width one).

**Strengths**:
- Beats both individual Q-learning and centralized Q-learning in all environments tested.
- Able to learn from a single shared team reward while still learning decentralized policies for each agent.
- Very simple solution to the credit assignment problem

**Limitations/Failure Modes**:
- Doesn't seem to leverage the extra state information available at training time in the same way as QMIX or COMA.
- More about enabling learning decentralized policies for each agent from a shared team reward rather than taking advantage of the CTDE paradigm
- Cannot represent or capture non-monotonic interactions (QMIX can)

**Relevance to Our Work**:
- This is probably closest to our use case of learning from a single shared team reward while still learning decentralized policies for each agent. We could imagine adding an LLM query to the shared team reward (i.e. "Here's the total reward, what % did each agent contribute") and then train individual DQN agents to learn decentralized policies for each agent.
- Foundation for QMIX, which we could use as a baseline for comparison (i.e. we could use both as baselines).
- We could use more centralized state information for our LLM approach (i.e. describe all agent positions in the environment, or the entire environment state).

---

### QPLEX (Skipped)
**Paper**: Wang et al., 2020
**Key Idea**:

**Credit Assignment Method**:

**Computational Cost**:

**Environments Tested**:

**Strengths**:

**Limitations/Failure Modes**:

**Relevance to Our Work**:

---

## LLM + RL Work

### SIMA 2 (Hindsight Relabeling with LLMs)
**Paper**: [Paper info from your PDF]
**Key Idea**: Use Gemini, SFT, and RL to create an agent capable of interacting in complex 3D environments, including those that are procedurally generated, as well as completing complex tasks in those environments. Tasks are given via multimodal inputs and verified via humans, automatic evaluation from ground truth information (env can verify), and programmatic verification (infra built on top of env to verify, e.g. OCR).

Additionally,

**How LLM is Used**:

**State/Action Representation**:
- For RL with verifiable rewards, used a tuple of the start state, the task to complete, and a verifier function to determine if the task was completed.
- For learning via self-improvement, used entire trajectories of multimodal information.

**LLM Architecture/Size**:
- I think there are at least two instances of Gemini for base SIMA 2. One for reasoning about the task at hand and communicating with the user and another for taking the current context (including the reasoning) and outputting keyboard and mouse actions to take.
- For learning via self-improvement, a Gemini Pro instance is used as both a task creator and a univeral reward function.
- This is a bit ambiguous to me, I would like to nail down the exact architecture of the system and how the input/output loop works.

**Query Frequency**:
- Querying happens every few seconds in the game (at least from what I can gather from the figures in the paper). The initial query is just a task given to the agent in natural language. Then there seems to be internal reasoning/chain-of-thought processing that happens as the agent is completing the task.

**Compute Cost**:
- Relatively high compared to traditional RL methods. Like interacting with an LLM regularly, SIMA 2 requires repeated LLM queries to navigate environments, reason about tasks, and output actions to take.
- Authors used both RL and SFT on a foundational model (even if it was Gemini Flash-lite), which is a lot of compute.

**Results**:
- Incredible progress towards a generalized embodied agent that can navigate complex environments and complete complex tasks. Doubled performance on environments when compared to SIMA 1 and approached human-level performance.
- When using base foundational Gemini models without SFT on actions, even with prompt engineering, performance was much worse.
- In contrast, the models that are trained with SFT or SFT + RL perform worse on coding, math, and STEM benchmarks due to a degredation of reasoning ability and information. Intuitively, this makes sense - the SFT + RL replaces some of the training that was done to create the foundational model in the first place.
- The section on learning from self-improvement is truly remarkable - the SIMA 2 agent is able to take on seemingly infinite tasks due to the Gemini Pro instance that is used as a universal reward function and task setter. I would like more clarity on how the reward function is used and how the self-improvement process works.
- The agent struggles with short-term memory and completing long horizon tasks as well as very fine grained motor control movements (keyboard and mouse).

**Relevance to Our Work**:
- Most directly related - shows LLMs can help with credit/reward, but only single-agent so far
- Uses SFT to train a Gemini instance to output keyboard and mouse actions (we can do something similar with an SLM to output a structured format for assigning credit, e.g. a JSON object with agent IDs and their respective contributions)
- From Claude: "The bridge between the two is: "If LLMs can relabel what goals were achieved (SIMA 2), can they also identify which agents contributed to achieving those goals (your work)?"
- Uses Gemini Pro as a universal reward function for self-improvement. This is along the lines of actually using an LLM to replace a reward function, whereas we just want to use it to assign credit. Gemini provides a rating (0-100) for each trajectory using a rubric as a guide. The score output is then fed back into the SIMA 2 agent for further training (RL?). A score of 50 or greater means the task was completed.

---

### Language Models as Zero-Shot Planners
**Paper**: Huang et al., 2022 (if accessible)
**Key Idea**:

**How LLM is Used**:

**State/Action Representation**:

**Relevance to Our Work**:

---

### Other LLM + RL Papers
*(Add papers found via Arxiv search)*

---

## Level-Based Foraging (LBF)

**Paper/Docs**: https://github.com/semitable/lb-foraging

**Environment Description**:
- Grid size: 8x8 (configurable)
- Number of agents: 2 (configurable)
- Number of food items: 2 (configurable)
- Food levels: 1-2 (randomly assigned)
- Agent capabilities: Level 2-3 (randomly assigned)

**State Space**:
Each agent receives a 12-dimensional observation vector containing:
- Agent 0: (x, y, level)
- Agent 1: (x, y, level)
- Food 0: (x, y, level)
- Food 1: (x, y, level)

All agents observe the full state (positions and levels of all agents and food items).

**Action Space**:
Discrete(6) - each agent can take one of 6 actions:
- 0: NOOP (do nothing)
- 1: MOVE_UP
- 2: MOVE_DOWN
- 3: MOVE_LEFT
- 4: MOVE_RIGHT
- 5: LOAD (attempt to pick up food at current position)

**Reward Structure**:
- Individual rewards: Yes - each agent receives reward when collecting food
- Team rewards: Partially shared - agents can cooperate to collect food
- Sparse: Very sparse - only non-zero when food is successfully collected
- Reward for collecting food: Proportional to food level
- Cooperative mode: In "coop" environments, multiple agents must be adjacent to food to collect it if the food level exceeds a single agent's level

**Cooperation Requirements**:
- Why do agents need to cooperate? In cooperative mode, food with level 2 requires 2 agents (each level 1) or 1 agent (level 2+) to collect. Agents must coordinate to be at the same location simultaneously.
- What happens if agents don't cooperate? Random agents achieve 0% success rate over 10 episodes - no food collected without coordination.

**Prior Work Using LBF**:
- Widely used benchmark for cooperative MARL
- Used in VDN paper and many MARL papers
- Simple enough for quick experiments, complex enough to require cooperation

**Why It's Good for Our Project**:
1. Simple, interpretable state space
2. Clear cooperation requirements
3. Deterministic translation to text possible
4. Computationally lightweight

---

## Key Insights & Gaps

### What We Know
- Credit assignment is fundamental challenge in cooperative MARL
- Traditional methods (COMA, QMIX) work but lack interpretability
- LLMs have strong reasoning capabilities
- LLMs have been used for single-agent RL tasks

### What We Don't Know (Research Gap)
- Can LLMs assign credit in multi-agent settings?
- Do LLMs need per-step information or can they work episodically?
- How does LLM size affect credit assignment quality?
- Can LLM-based credit compete with traditional methods?

### Our Contribution
- First application of LLMs to multi-agent credit assignment
- Comparison of LLM-based vs traditional credit assignment methods
- Analysis of interpretability benefits
- Study of compute tradeoffs (episodic queries vs per-step methods)

---

## Open Questions for Discussion
- [ ] Should we focus on qualitative analysis (interpretability) or quantitative (performance)?
- [ ] What if LLM credit assignment is worse than baselines - is that still valuable?
- [ ] Should we test multiple environments or go deep on LBF?
- [ ] How to evaluate "quality" of credit assignment beyond final performance?
