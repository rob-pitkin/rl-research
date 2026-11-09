# On memory and agents: a unified framework for replay buffers, markdown files, and model weights

*(Bearblog Essay Scaffold — audited against `blog-post-review`)*

---

## Paragraph 1 — The Hook: One Data Structure, Two Silos
* **Anti-Pattern #1 Check (No Meandering Intro):** Open immediately with the core thesis in Sentence 1–2, then ground it in your personal experience.
* **Core Beats:**
  1. **The Thesis Opener:** Ask an LLM engineer what "agent memory" is, and they'll point to markdown files, vector databases, and RAG pipelines. Ask a reinforcement learning researcher, and they'll point to experience replay buffers and neural network weights. Despite the completely different vocabulary, both fields are building the exact same data structure.
  2. **Personal Grounding:** Lately I've been living in both worlds at the same time—building Deep RL agents from scratch (like AlphaZero and AlphaDev) while using LLM coding agents (`CLAUDE.md` / `AGENTS.md`, Honcho) to help me design them.
  3. **The Bridge:** Working on an RL replay buffer and editing an agent's markdown memory file in the same afternoon made it obvious that we can formalize all agentic memory under a single mathematical framework—and once you write that framework down, you immediately notice what both fields are leaving on the table.

---

## Paragraph 2 — Explicit vs. Implicit Memory & The Four Operators
* **Goal:** Introduce the coupled state $\Psi_t = (\theta_t, M_t)$ and the 4 abstract primitives, pairing code intuition with formal notation.
* **Beat 1 (Explicit vs. Implicit State):**
  An agent's memory at time $t$ isn't a single store; it is a coupled state split across two mediums:
  $$\Psi_t = (\theta_t, \, M_t)$$
  - **Implicit memory ($\theta_t$):** Model weights (or human intuition/muscle memory). Reading from $\theta_t$ is $O(1)$ at inference time because it is baked directly into the network's forward pass, but writing to it is slow and risks overwriting old skills (*catastrophic forgetting*, where updating weights for task B accidentally destroys task A).
  - **Explicit memory ($M_t$):** A replay buffer in RL, organized markdown files, or a database. Writing to $M_t$ is an instant $O(1)$ append (`memory.append(e_t)`) with zero interference, but reading from it requires search and doesn't generalize on its own.
* **Beat 2 (The 3-Tuple & 4 Primitives):**
  Every explicit memory $m_i \in M_t$ can be modeled as a tuple of a searchable key $k_i = \phi(e_i)$, a stored payload $v_i$, and a priority/salience score $p_i$:
  $$m_i = (k_i, \, v_i, \, p_i)$$
  Regardless of whether the medium is a PyTorch tensor buffer or a folder of `.md` files, the memory system is defined by four operators:
  1. **Store (`write` / $\mathcal{W}$):** Encode a raw experience $e_t$ and write it to the medium:
     $$\mathcal{W}(M_t, e_t) = M_t \cup \big\{\big(\phi(e_t), \, e_t, \, p_0(e_t)\big)\big\}$$
  2. **Similarity (`sim` / $\mathcal{K}$):** A distance metric $\mathcal{K}(k_a, k_b)$ measuring if memory $A$ is similar or different to memory $B$ (e.g., cosine similarity on text embeddings or latent state distance in RL).
  3. **Recall / Replay (`read` / $P_{\text{recall}}$):** A distribution for pulling past memories back into active compute.
  4. **Consolidate (`update / reflect / delete` / $\mathcal{C}$):** An operator that distills, updates, or prunes memories over time.

---

## Paragraph 3 — "Spiked Information Gain" & The Unified Recall Equation
* **Goal:** Formalize what makes a memory important (Spiked Information Gain) and show how a single Boltzmann recall equation unifies DQN Replay, Prioritized Replay, and LLM RAG.
* **Beat 1 (Spiked Information Gain — Bayesian Surprise):**
  If an agent stores and replays every experience uniformly, it wastes 99% of its compute rehearsing mundane turns where its mental model was already right. What makes a memory worth prioritizing—whether a painful failure ("trauma") or a eureka breakthrough—is **spiked information gain**.
  In information theory, this is *Bayesian surprise*: the KL-divergence measuring how much a single experience $e_t$ forces the agent's prior beliefs $P(\theta \mid \mathcal{H}_{t-1})$ to shift into a new posterior $P(\theta \mid \mathcal{H}_{t-1}, e_t)$:
  $$\mathcal{I}(e_t) = D_{\text{KL}}\Big( P(\theta \mid \mathcal{H}_{t-1}, e_t) \;\Big\|\; P(\theta \mid \mathcal{H}_{t-1}) \Big)$$
* **Beat 2 (The RL Realization — Prediction Error / Advantage):**
  Computing an exact Bayesian posterior over millions of weights is intractable, so RL agents approximate information gain using **prediction error** ($\delta_t$)—the gap between what *actually* happened and what the agent *expected* to happen:
  $$p_t = |\delta_t| + \epsilon = \big| \big(r_t + \gamma V_{\theta_t}(s_{t+1})\big) - V_{\theta_t}(s_t) \big| + \epsilon$$
  When reality matches expectation ($|\delta_t| \approx 0$), information gain is zero. When an agent hits an unexpected wall or stumbles onto an unexpected solution ($|\delta_t| \gg 0$), priority spikes.
* **Beat 3 (The Unified Recall Equation — Two Questions, Two Volume Knobs):**
  How does an agent actually decide which memory to pull up when it is in a situation $q_t$? In plain English, every recall system is just balancing **two questions**:
  1. **Relevance ($\mathcal{K}(q_t, k_i)$):** *"How similar is this memory to what I'm looking at right now?"*
  2. **Surprise ($p_i$):** *"How big of a deal was this memory when it happened?"*
  You can think of $\beta_{\text{sim}}$ and $\beta_{\text{sal}}$ as **two volume knobs (sliders)** that control how much the agent cares about Relevance vs. Surprise, passed through a standard softmax so the probabilities sum to $1$:
  $$P_{\text{recall}}(m_i \mid q_t, M_t) = \frac{\exp\!\Big(\beta_{\text{sim}} \, \mathcal{K}(q_t, k_i) + \beta_{\text{sal}} \log p_i\Big)}{\sum_{j=1}^{|M_t|} \exp\!\Big(\beta_{\text{sim}} \, \mathcal{K}(q_t, k_j) + \beta_{\text{sal}} \log p_j\Big)}$$
  Look at what happens as you turn those two volume knobs:
  - **Both knobs at 0 (`beta_sim = 0, beta_sal = 0`)** $\implies$ **Uniform Experience Replay** (vanilla DQN: pulling memories out of a hat completely at random).
  - **Only the Surprise knob turned up (`beta_sim = 0, beta_sal > 0`)** $\implies$ **Prioritized Experience Replay (PER)** (doesn't filter by current state; just replays its biggest shocks and breakthroughs proportional to $p_i^\alpha$).
  - **Only the Relevance knob turned up (`beta_sim > 0, beta_sal = 0`)** $\implies$ **Standard LLM RAG / Vector Search** (fetches the closest semantic match to the current prompt, completely blind to whether that memory holds an important lesson or mundane noise).
  - **Both knobs turned up (`beta_sim > 0, beta_sal > 0`)** $\implies$ **Full Agentic Recall** (surfaces memories that are both relevant to the problem in front of you *and* high-information-gain).

---

## Paragraph 4 — Consolidation: Backpropagation and Reflection
* **Goal:** Show how the Consolidation operator $\mathcal{C}(\theta_t, M_t)$ bridges Explicit and Implicit memory—unifying neural training (backprop) with agent reflection (`AGENTS.md` / Reanalyze).
* **Beat 1 (The Coupled Consolidation Operator):**
  $$\big(\theta_{t+1}, \, M_{t+1}\big) = \mathcal{C}(\theta_t, \, M_t) = \Big( \mathcal{C}_{\theta}(\theta_t, M_t), \;\; \mathcal{C}_{M}(M_t; \theta_{t+1}) \Big)$$
* **Beat 2 (Pathway 1 — Explicit $\to$ Implicit via Backpropagation $\mathcal{C}_\theta$):**
  $$\theta_{t+1} = \theta_t - \eta \, \mathbb{E}_{m_i \sim P_{\text{recall}}(M_t)} \Big[ w_i \, \nabla_\theta \mathcal{L}(\theta_t; m_i) \Big]$$
  Through this lens, **supervised learning and LLM pre-training are literally memory consolidation**. An LLM's pre-training corpus on disk is a massive explicit memory store $M_t$, and backpropagation is the consolidation operator that compresses terabytes of explicit text trajectories into fixed-size implicit weights $\theta$.
* **Beat 3 (Pathway 2 — Explicit $\to$ Explicit via Reflection & Pruning $\mathcal{C}_M$):**
  When weights $\theta$ are frozen (like an LLM coding agent at inference time) or between gradient updates, consolidation happens *inside* $M_t$:
  $$\mathcal{C}_{M}(M_t; \theta_{t+1}) = \big(\text{Prune} \circ \text{Compress} \circ \text{Re-evaluate}\big)(M_t)$$
  1. **Re-evaluate (`update`):** Revisiting old memories with a smarter model to refresh their labels and surprise scores $p_i' \leftarrow |\delta_i(\theta_{t+1})|$ (in RL, this is **MuZero Reanalyze**).
  2. **Semantic Compression & Eviction (`reflect` / `delete`):** Distilling a cluster of noisy episodic logs $\{m_1, \dots, m_k\}$ into a compact rule $\hat{m}$ in `AGENTS.md` (which acts as a "pseudo-weight" with $p_i = \infty$, permanently pinned in context) and deleting the raw scratch logs.

---

## Paragraph 5 — Putting the Framework to Use: The "Missing Quadrant" & Asymptotic Handoff
* **Goal:** Show how viewing agents through $\Psi_t = (\theta_t, M_t)$ exposes blind spots in existing paradigms and leads in 3 lines of algebra to DeepMind's *Episodic Control* / *Fast & Slow RL*—and its exact LLM analog.
* **Beat 1 (Spotting the Missing Quadrants):**
  Once you map RL and LLM agents onto $\Psi_t = (\theta_t, M_t)$, a glaring asymmetry jumps out:
  - **Standard LLM agents** rely entirely on Explicit Memory ($M_t$) at runtime while keeping their Implicit Memory ($\theta_t$) frozen forever.
  - **Standard RL agents** do the exact opposite: they use Explicit Memory ($M_t$, the replay buffer) *only* during offline training, and ignore it completely at runtime! Even though the replay buffer is sitting right there in RAM, a standard RL agent acts with total hippocampal amnesia at test time—stepping onto the exact same trap 15 episodes in a row while waiting for slow gradient descent to shift its weights $\theta_t$.
* **Beat 2 (Injecting the Unlearned Residual $\delta_t$):**
  What if an RL agent actually used $P_{\text{recall}}$ at runtime? If you naively added fixed rewards or state-value advantages $A(s, a) = G - V(s)$ from the buffer to your policy logits $z_{\theta_t}(s, a)$, you'd eventually **double-count** once the network $\theta_t$ learns. Instead, the buffer should inject *what the implicit weights haven't learned yet*—the **action prediction residual** $\delta_i(\theta_t) = G_i - z_{\theta_t}(s, a)$:
  $$z_{\text{total}}(s, a; \Psi_t) = \underbrace{z_{\theta_t}(s, a)}_{\text{What }\theta_t\text{ Knows}} + \; \lambda \sum_{m_i \in M_t(a)} P_{\text{recall}}(m_i \mid s, M_t) \cdot \underbrace{\big(G_i - z_{\theta_t}(s, a)\big)}_{\text{Explicit Residual }\delta_i(\theta_t)}$$
  *(In code: `logits[a] = net_logits[s, a] + lambda * (recalled_return[s, a] - net_logits[s, a])`)*
* **Beat 3 (The $(1 - \eta)^t$ Consolidation Handoff Proof):**
  Suppose the agent hits a trap with initial shock $\delta_0 = G - z_{\theta_0} = -12$. Each consolidation step updates the implicit weight by $z_{\theta_{t+1}} = z_{\theta_t} + \eta \, \delta_t$, which shrinks the remaining explicit surprise by $\delta_{t+1} = G - (z_{\theta_t} + \eta \delta_t) = (1 - \eta)\delta_t$. Unrolling over $t$ consolidation steps:
  $$\delta_t = (1 - \eta)^t \, \delta_0 \longrightarrow 0, \qquad \Delta z_{\theta_t} = \Big(1 - (1 - \eta)^t\Big) \, \delta_0 \longrightarrow \delta_0$$
  For $\lambda = 1$, their sum $\Delta z_{\theta_t} + \delta_t = \delta_0$ is an **exact conservation law** at every step $t$: at $t = 0^+$, the decision is $100\%$ driven by Explicit Memory ($M_t$), and as $t \to \infty$, the explicit surprise decays to $0$ at the exact rate the Implicit Weights ($\theta_t$) absorb the lesson.
* **Beat 4 (Algebraic Equivalence to DeepMind's *Fast & Slow RL* and LLM Agents):**
  Notice that if you expand $z_{\theta_t} + \lambda(Q_{\text{explicit}} - z_{\theta_t})$, you get:
  $$z_{\text{total}}(s, a) = (1 - \lambda) \, z_{\theta_t}(s, a) + \lambda \, Q_{\text{explicit}}(s, a)$$
  Without setting out to, our 4-operator framework just re-derived the exact mechanism behind DeepMind's **Episodic Control / Ephemeral Value Adjustments (EVA)** (Blundell et al. 2016; Pritzel et al. 2017; Hansen et al. 2018; Botvinick et al. 2019 *Fast and Slow RL*) and **$k$NN-LMs** (Khandelwal et al. 2020)—and shows that when an LLM engineer writes a failure rule into `AGENTS.md` ($\delta_0$ at $t=0^+$) before eventually fine-tuning on session logs ($(1-\eta)^t \to 0$), they are running the exact same hippocampal-to-neocortical handoff:

![Runtime Advantage Injection & Consolidation Handoff](/Users/robpitkin/.gemini/antigravity/brain/74b3c14b-0d1c-4c24-8415-88c2c0e9ffbf/memory_handoff_demo.png)

---

## Paragraph 6 — Why Our Memory Evals Are Broken
* **Goal:** Close with how this mathematical framework exposes the flaw in current agent memory benchmarks.
* **Core Beats:**
  1. Most LLM memory evaluations today are glorified Needle-in-a-Haystack trivia tests (*"What did the user say their dog's name was 50 turns ago?"*).
  2. In terms of our equations, trivia tests only measure flat storage and similarity lookup ($\mathcal{W} + \mathcal{K}$) with $\beta_{\text{sal}} = 0$ and zero consolidation ($\mathcal{C} = \text{Identity}$).
  3. If memory is an active learning loop rather than a passive filing cabinet, a real benchmark has to evaluate the full operator tuple:
     - Does it recall structurally similar past problems ($\mathcal{K}$)?
     - Under a strict capacity limit, does it prioritize spiked information gain ($p_i = |\delta_i|$) and garbage-collect predictable chatter?
     - Most importantly, does consolidation ($\mathcal{C}$) actually hand off explicit surprises into better implicit zero-shot policy decisions on future tasks?
