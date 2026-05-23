# Comm vs CTDE: Project Resumption & Research Plan

This document outlines the strategy for integrating the `cpp-mpe2` environments and advancing the "Communication vs Centralized Training" research.

## 1. Environment Integration (cpp-mpe2)
* **Goal**: Replace the standard `mpe2` (Python) with `cpp-mpe2` (C++) for ~15-50x speedup.
* **Actions**:
    * Install `cpp-mpe2` in editable mode: `uv pip install -e ../cpp-mpe2`
    * Update `envs/mpe_wrapper.py` to import from `cpp_mpe2.simple_reference.simple_reference`.
    * Verify observation/action space alignment and communication masking logic.

## 2. Baseline & Performance Verification
* **Goal**: Establish stable baselines and quantify the iteration speedup.
* **Actions**:
    * Re-run `train_iql_no_comm.sh` and `train_vdn_no_comm.sh`.
    * Compare "Wall-clock time to threshold return" between backends.
    * Use `scripts/run_full_ablation.sh` for multi-seed verification.

## 3. Quantitative Communication Analysis
To provide rigour for future publications/blog posts, we will track these metrics:
* **Message Entropy ($H_{comm}$)**: Measures vocabulary diversity. High entropy indicates agents are using multiple messages; low entropy suggests signal collapse.
* **Communication Sensitivity ($\Delta R$)**: Test-time evaluation by masking communication on a trained model. $\Delta R = R_{full} - R_{masked}$ quantifies reliance.
* **Mutual Information ($I(m; g)$)**: Measure correlation between messages ($m$) and private goals ($g$).

## 4. Mathematical & Theoretical Foundations
Research topics for the upcoming blog post:
* **IGM Principle (Individual-Global-Max)**: Formal consistency between local greedy actions and global optimality (Reference: QMIX/QTRAN).
* **Dec-POMDP Information Flow**: Modeling communication as a reduction in state uncertainty.
* **Bandwidth-Performance Trade-offs**: Theoretical relationship between communication bits and coordination success.

## 5. Network Architecture
* **Paradigm**: Parameter sharing (agent-invariant).
* **Design**: Factored action heads (Movement + Communication) to avoid combinatorial action space explosion.
* **Role Handling**: For asymmetric environments, use one-hot Agent IDs / Role Embeddings rather than separate networks.

## 6. Immediate Next Steps
1. [ ] Install `cpp-mpe2` in development mode.
2. [ ] Update `mpe_wrapper.py` and run a basic test script.
3. [ ] Execute 2-seed baseline runs for IQL/VDN on `simple_reference`.
