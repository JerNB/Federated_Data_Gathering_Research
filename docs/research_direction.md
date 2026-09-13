# Candidate extensions and system confirmation

This file is intentionally secondary. The canonical proposal is
`docs/research_proposal.md`; the direction-ranking and promotion contract is
`docs/exploration_goal.md`. The completed small-sample matrix is
snapshot-calibration evidence, not the primary federated claim.

## Complementary tracks after a direction is promoted

1. **Trainable models:** BPR-MF first, then implicit ALS and item-based methods.
2. **Sampling extensions:** item-stratified, graph-aware, coreset, and recent
   temporal-window sampling.
3. **Data regimes:** additional public datasets and controlled changes in
   sparsity, popularity skew, temporal drift, exposure, and missingness.
4. **Federated-system confirmation:** local-data training, communication cost,
   availability/dropout, a named privacy mechanism, and optimizer controls.
   These are required confirmation evidence; the existing snapshot result does
   not establish them.
5. **Interpretability:** latent-factor or heterogeneity analysis only when it
   answers a declared sampling/generalization question.

## Rules for promotion into the primary study

An extension must have:

- one named scientific question;
- a declared sampling/model/control contract;
- the same full-data reference protocol;
- independent draws or runs sufficient to estimate uncertainty;
- a result artifact under `results/sample_generalization/`;
- a short interpretation stating assumptions and failure cases.

The old federation-first model comparison, oracle-cluster comparison, private
conversation link, and unverified bibliography are intentionally not part of
the active proposal. The verified sources remain in
`docs/literature_sources.md`.
