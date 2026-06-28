---
title: Work Log
description: Append-only audit trail of changes to this knowledge base.
---

# Work Log

Append-only audit trail. Add one dated entry per turn that creates, edits, or restructures content. The knowledge-base skill describes what to log and the entry shape.

## 2026-06-28: Migrate Obsidian `rl-knowledge/` vault into `brain/`

- Migrated the full Obsidian wiki KB (`rl-knowledge/`) into this three-layer OpenKnowledge base. Wiki paper/method notes → `research/` (provisional analyses); source PDFs → `external-sources/` (preserved binaries); the old `_index/paper-index.md` → [research/index](./research/index.md) navigation hub.
- **Ingested 14 source PDFs** (copied verbatim from `rl-knowledge/raw/papers/`, each with `sha256` + `bytes` + `source_url` frontmatter and a `![[file.pdf]]` wiki-embed): [vdn](./external-sources/vdn.mdx), [qmix](./external-sources/qmix.mdx), [coma](./external-sources/coma.mdx), [mappo](./external-sources/mappo.mdx), [commnet](./external-sources/commnet.mdx), [vbc](./external-sources/vbc.mdx), [marl-gpt](./external-sources/marl-gpt.mdx), [marl-communication-survey](./external-sources/marl-communication-survey.mdx), [the-era-of-experience](./external-sources/the-era-of-experience.mdx), [recursive-language-models](./external-sources/recursive-language-models.mdx), [cgrpa-curriculum-learning-marl](./external-sources/cgrpa-curriculum-learning-marl.mdx), [sutton-barto-mcts-chapter](./external-sources/sutton-barto-mcts-chapter.mdx), [marl-textbook-ch9-mcts-alphazero-psro](./external-sources/marl-textbook-ch9-mcts-alphazero-psro.mdx), [sima2](./external-sources/sima2.mdx).
- **Created 13 research notes** (faithful migration of the wiki analyses; `[[wiki-links]]` converted to markdown links, each citing its `external-sources/` wrapper, `status: provisional`): [vdn](./research/vdn.md), [qmix](./research/qmix.md), [coma](./research/coma.md), [mappo](./research/mappo.md), [commnet](./research/commnet.md), [vbc](./research/vbc.md), [marl-communication-survey](./research/marl-communication-survey.md), [marl-gpt](./research/marl-gpt.md), [the-era-of-experience](./research/the-era-of-experience.md), [recursive-language-models](./research/recursive-language-models.md), [cgrpa-curriculum-learning-marl](./research/cgrpa-curriculum-learning-marl.md), [mcts](./research/mcts.md), [marl-textbook-ch9-mcts-alphazero-psro](./research/marl-textbook-ch9-mcts-alphazero-psro.md).
- Created the [research/index](./research/index.md) hub (method-family + topic groupings). Verified zero dead links across all research docs.
- Note: `sima2.pdf` was preserved as a source but has no corresponding analysis yet — no wiki note existed for it.
- **Open follow-ups**: verify the `TODO` `source_url`s on [marl-gpt](./external-sources/marl-gpt.mdx) (AAAI 2026) and [sima2](./external-sources/sima2.mdx); decide whether to delete the original `rl-knowledge/` vault now that content is migrated; promote stable notes (e.g. [vbc](./research/vbc.md)) to canonical `articles/` via `consolidate` when ready.
