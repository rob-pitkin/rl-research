---
name: deepmind-session-sync
description: >-
  Synchronize session context, decisions, and conversation logs with the Honcho
  memory server and the Linear 'Chrome to DeepMind' project. Activate this skill
  at the beginning of a session, when transitioning between Linear issues, or
  when recording curriculum decisions.
---

# DeepMind Track Session & Memory Sync

## 1. Honcho Persistent Memory (Every Turn)
Log every user and assistant turn using `call_mcp_tool` on `honcho` -> `add_messages_to_session`:
- **`workspace_id`**: `"claude_code"`
- **`session_id`**: `"robpitkin-career-planning"`
- **`peers`**: `"robpitkin"` (user) and `"antigravity"` (assistant)

## 2. Linear Canonical Identifiers (`linear-mcp-server`)
- **Team**: `Robpersonal` (key: `ROB`)
- **Project**: `Chrome to DeepMind` (ID: `b027abfa-604b-4494-81e7-9c66d91eb87e`)
- **Living State Document**: `Career context — read this first` (ID: `81b36370-2f34-452e-9963-148ed0851bff`)
  - *Rule:* Edit in place using `save_document` (`patch`) whenever current state or active tracks change.
- **Append-Only History Document**: `Decisions on record` (ID: `c9c17e88-9723-418e-92f6-6e52340d8883`)
  - *Rule:* Append dated entries (`## YYYY-MM-DD — Title`) with **Decision** and **Why** whenever a curriculum or career decision is made.

## 3. Active Track `1F` (Search, Self-Play, & World Models)
- `ROB-137` (`1F.1`): AlphaZero from first principles: Connect Four (`Done`, repo: `rob-pitkin/connect4-alphazero`)
- `ROB-142` (`1F.2`): Single-player AlphaZero: combinatorial discovery, backward curricula, and Q-normalization
- `ROB-143` (`1F.3`): Stochastic MCTS: chance nodes, afterstates, and progressive widening
- `ROB-144` (`1F.4`): MuZero & Stochastic MuZero: planning in latent imagination without a simulator
- `ROB-145` (`1F.5`): The game theory of self-play: distribution collapse, non-transitivity, and league training
