---
name: blog-post-review
description: >-
  Review, critique, and co-design technical blog posts and Bearblog essays against
  Michael Lynch's 6 software blogging anti-patterns (Refactoring English) while
  preserving Rob's authentic first-person voice. Activate this skill whenever
  scaffolding, drafting, editing, or reviewing a blog post or technical essay.
---

# Technical Blogging & Bearblog Review Skill

Whenever scaffolding, drafting, or reviewing a blog post with Rob, enforce the following **6 Anti-Pattern Guardrails** (adapted from Michael Lynch's *Anti-Patterns in Software Blogging*) and follow the **Co-Writing & Review Protocol** below.

---

## The 6 Software Blogging Anti-Patterns to Eliminate

### 1. The Meandering Intro *(and "Preamble Still Counts as Meandering")*
- **The Anti-Pattern:** Opening with throat-clearing platitudes (*"In recent years, AI agents have grown increasingly capable..."*) or paragraphs of personal backstory before stating the thesis. Adding a meta-disclaimer (*"Before we get to X, some background..."*) still counts as meandering.
- **The Rule:** **Drop the hammer in the first 2–3 sentences.** Lead immediately with the sharpest contrast, surprising claim, or core problem. Weave personal context *in* once the reader already knows why they care.

### 2. "The Reader Knows Everything I Know Except This One Thing"
- **The Anti-Pattern:** Assuming the reader shares your exact current working-memory state—either dropping niche jargon (e.g., *TD-error*, *MuZero Reanalyze*, *PUCT*, *KL-divergence*) without plain-English grounding, or over-explaining basic software engineering concepts.
- **The Rule:** **Calibrate to a smart generalist software engineer.** Assume the reader writes code professionally and understands basic ML/systems, but hasn't spent the past week immersed in your specific subfield. Define every specialized term in one crisp plain-English clause before (or right as) you introduce its formal notation.

### 3. Overreliance on Links
- **The Anti-Pattern:** Using a hyperlink as a crutch instead of explaining a prerequisite concept inline, forcing the reader to bounce across 5 browser tabs just to follow your argument.
- **The Rule:** **Zero-Click Self-Containment.** The essay must make 100% sense if the reader clicks *zero* links. Always summarize the linked concept in a phrase or sentence inline; use links strictly for attribution and optional deep dives.

### 4. The Sequel Injection Bug
- **The Anti-Pattern:** Framing a post as an episode in an ongoing series (*"In my last post, I discussed..."*) or assuming the reader knows your personal project roadmap.
- **The Rule:** **Every post stands alone.** Assume 95% of readers landed on this single URL from Hacker News, search, or a link share with zero prior context about you or your previous posts. If referencing an earlier post, restate its takeaway in a single self-contained clause.

### 5. Excessive Formality
- **The Anti-Pattern:** Slipping into stiff, passive, academic-paper prose (*"We denote the memory tuple as..."*, *"It can be observed that..."*) out of a subconscious belief that formality equals rigor.
- **The Rule:** **Write like you're explaining the idea to a smart peer over coffee.** Preserve Rob's natural first-person voice ("I"), active verbs, and lived engineering anecdotes. When including math or LaTeX, use it sparingly to *sharpen* intuition—never as performative formalism. Prefer readable variable names over dense walls of Greek subscripts.

### 6. Fumbling the Basics of Rendering HTML *(Mobile & Bearblog Overflow)*
- **The Anti-Pattern:** Publishing wide multi-column Markdown tables, unwrapped code blocks, or sprawling single-line display equations (`$$...$$`) that trigger horizontal scrollbars on mobile screens (where over half of readers view Bearblog posts).
- **The Rule:** **Design for a 375px mobile viewport.**
  - Keep display LaTeX equations compact (under ~45–50 characters wide) or break them cleanly across lines.
  - Prefer tight bullet lists over wide 4+ column Markdown tables in published Bearblog posts.
  - Ensure every equation is immediately readable without horizontal scrolling.

---

## Co-Writing & Review Protocol

1. **Protect Rob's Voice (No Ghostwriting Slop):** Never dump a bland, fully AI-written essay unless explicitly asked for a strawman. Co-design scaffolds, critique drafts paragraph-by-paragraph, and let Rob write in his own voice.
2. **Run the 6-Point Audit on Every Draft/Scaffold:** Whenever reviewing a section or full draft, explicitly check:
   - *Hook Check (#1):* Does sentence 1–3 state the core thesis or tension without throat-clearing?
   - *Jargon Calibration (#2):* List every specialized term introduced—is each grounded in plain English on first use?
   - *Zero-Click Test (#3):* If all `<a>` tags were stripped, would any paragraph lose its meaning?
   - *Standalone Test (#4):* Does any sentence assume prior knowledge of Rob's earlier posts or projects?
   - *Voice & Math Check (#5):* Are there any passive/academic phrases or bloated equations that can be simplified into conversational prose?
   - *Mobile Viewport Check (#6):* Will any equation, table, or code block overflow horizontally on a phone screen?
