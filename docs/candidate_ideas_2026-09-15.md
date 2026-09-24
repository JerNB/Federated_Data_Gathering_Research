# Candidate research ideas — 2026-09-15

Status: discussion notes requested by the user; hypotheses only. These ideas
have not been promoted, implemented, or established as novel. They do not
replace the canonical proposal or reinterpret existing experiment results.

## 1. Joint history-budget and feature-group selection

Research question: for a fixed user cohort under declared federated information
boundaries, should the next unit of operational budget be spent on additional
local interaction history or on an additional feature group?

- Horizontal component: users retain their own interaction histories.
- Vertical component: separate feature holders contribute aligned item features.
- Candidate actions: extend local-history budgets, enable a feature group,
  retain the current configuration, or stop acquiring additional information.
- Stopping information acquisition is separate from stopping model training.
- Hypothesis: a joint policy can outperform fixed policies and history-only or
  feature-only adaptation at matched cost. This may fail if sources are redundant
  or decision overhead exceeds their benefit.
- Controls: centralized full-information reference; federated full-information
  reference; federated fixed-budget, history-only, feature-only, and joint policies.
- Count actual communication, compute, and policy diagnostic overhead. A feature
  group and an interaction record are not interchangeable cost units.
- Fit decisions on permitted earlier validation information, never future tests
  or evaluator-only full-information results.
- First feasibility screen: history-size by feature-group quality/cost matrix;
  establish complementary effects before implementing a hybrid controller.

## 2. Attribute-specialist recommendation with late fusion

User proposal: separate branches estimate preferences over genres, actors, or
other attributes, then combine their outputs to recommend complete movies.

- A branch should estimate a user's preference over its attributes, not merely
  classify the attributes of a movie. The preference signal must come from
  permitted user interactions or an explicitly specified collaborative protocol.
- Convert branch outputs into scores for the same candidate movie IDs before
  fusion; preserve soft scores rather than requiring a hard intersection of
  top attribute labels. Calibrate score scales and tune fusion on validation data.
- Begin with genre and available tags; actor/director fields are absent from
  the current MovieLens package. Tag/genome availability at historical cutoffs
  must be checked; full-snapshot derived features are not automatically causal.
- Compare interaction-only, individual specialists, fixed weighted fusion,
  all-branch fusion, and budget-aware conditional branch activation.
- Feature splitting and score fusion alone do not establish vertical FL.
  Specify entity alignment, label ownership, information flows, trust assumptions,
  training protocol, and inference requirements before a federated claim.
- Risks: lost cross-feature interactions, correlated sources, attribute preference
  not implying preference for every matching movie, and candidate retrieval losses.
- Measure training and serving costs separately: local branch compute, messages,
  bytes, latency (including slow parties), caching, and gating overhead.
- Hypothesis extension: invoke another specialist only when expected improvement
  justifies its cost; no novelty or efficiency claim is established yet.

## MovieLens interpretation boundary

Splitting public movie metadata among simulated holders is an algorithmic test
setting, not evidence that these attributes require privacy-preserving federation
in production. A deployment claim requires a justified ownership/access scenario.

## Initial related-work pointers (not a completed novelty review)

- Hybrid Federated Learning: Algorithms and Implementation:
  https://arxiv.org/abs/2012.12420
- LESS-VFL: https://proceedings.mlr.press/v202/castiglia23a.html
- AISS-VFL: https://doi.org/10.1016/j.eswa.2026.132011
- A Federated Multi-View Deep Learning Framework for Privacy-Preserving
  Recommendations: https://arxiv.org/abs/2008.10808
- Compressed-VFL: https://proceedings.mlr.press/v162/castiglia22a.html

These works overlap with hybrid federation, selection, multi-view recommendation,
and communication efficiency. Read and compare their full methods before choosing
a contribution or making a novelty claim.
