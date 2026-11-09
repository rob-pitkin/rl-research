---
name: rl-module-codesign
description: >-
  Structured 5-phase pedagogical workflow for co-designing, implementing, visualizing,
  and experimenting with RL algorithms from first principles. Activate this skill
  whenever starting or working through an RL curriculum module (such as Track 1F
  Search, Self-Play & World Models) or designing a new file in an RL project.
---

# RL Module Co-Design & Learning-by-Doing Workflow

Follow this 5-phase progression whenever working through an RL module with Rob. Do not skip phases or batch multiple implementation files into a single turn.

## Phase 1: First-Principles Framing & Constraint Diff
1. **Anchor to Prior Work:** Start by contrasting the new problem with the most recent completed module (e.g., *"In `1F.1` Connect Four, we assumed X. In `1F.2`, X breaks because..."*).
2. **Surface the Core Math/Systems Failure Modes:** Walk through why a naive application of the previous algorithm fails mathematically (e.g., reward scale breaking PUCT, sea of `-1`s killing value gradients, chance nodes diluting visit counts).
3. **Agree on the Minimal Architecture:** Propose a clean 3–5 file structure where each file isolates one core concept.

## Phase 2: Module-by-Module Spec Co-Design (The Pre-Code Gate)
For **one file at a time**:
1. **Draft the Spec (`/plan`):** Include:
   - Exact function/method signatures and tensor shapes (e.g., `(B, C, H, W)`).
   - Worked equations with every variable defined.
   - ASCII or Mermaid diagrams for non-trivial data flow (e.g., residual blocks, tree node transitions, two-stage trajectory labeling).
2. **Pose 1–3 Concrete Design Questions:** Ask Rob to weigh in on real architectural choices (e.g., normalization layer choice, discount factor trade-offs, temperature schedules, buffer eviction).
3. **Verify the Litmus Test:** Only proceed to writing the file once Rob confirms the design and every open question is resolved (*"I understand it well enough that I could code this myself"*).

## Phase 3: Single-File Implementation & Invariant Verification
1. Write **only** the approved module.
2. Run a fast, deterministic smoke test via `uv run` verifying key mathematical invariants:
   - Tensor shapes and legal-action masks (`-inf` logits -> `0.0` probability).
   - Value/reward perspective sign conventions and backup equations.
   - Simulator edge cases (e.g., terminal state queries, batch-size-1 `eval()` mode).

## Phase 4: Generational Curriculum & Interactive Visualization
1. **Interactive Telemetry (`play.py` / Visualizer):** Pair the algorithm with an interactive visualizer (such as `pygame-ce`) that exposes the agent's internal beliefs live (e.g., raw network prior $P(s,a)$ vs. search policy $\pi(a|s)$, value $V(s)$ vs. $Q(s,a)$).
2. **Generation 1 -> Generation 2 Progression:**
   - Train a fast **Generation 1** checkpoint (`< 30s`) to verify end-to-end learning and let Rob interact with/play against it.
   - Diagnose where Gen 1 is weak (e.g., shallow horizon, cold start, opening collapse).
   - Save `gen1.pt` as a frozen benchmark, train **Generation 2** with a deeper budget or curriculum fix, and run a controlled **Arena match (`Gen 2 vs. Gen 1`)** logging win-rate, score %, and trajectory length evolution.

## Phase 5: Theoretical Synthesis & Closeout
1. **Generalize the Lesson:** Discuss how the empirical findings connect to landmark RL papers (AlphaZero, MuZero, Stochastic MuZero, AlphaTensor, AlphaDev, KataGo, AlphaStar) and what breaks next.
2. **Commit, Document, & Sync:**
   - Ask for approval before pushing to GitHub `main`.
   - Post a structured synthesis comment to the active Linear issue and update its status.
   - Log all turns to Honcho.
