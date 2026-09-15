# Documentation index

The repository-level reading order — including which result files to open and in
what sequence — is in the root [`README.md`](../README.md) under "Start here:
reading order". This index covers the documents themselves.

Each document states what it does and does not establish.

## 1. What is claimed

| Document | Role |
| --- | --- |
| [`research_proposal.md`](research_proposal.md) | Canonical proposal: the fixed-cohort local-data-budget question, estimands, controls, evidence phases, and current results. |
| [`exploration_goal.md`](exploration_goal.md) | Decision framework: how one data-gathering direction is promoted, deferred, or rejected. |
| [`direction_assessment.md`](direction_assessment.md) | Executed decision: literature matrix, evidence cards, ranked outcome, real-data result, and model control. |

## 2. What the claim rests on

| Document | Role |
| --- | --- |
| [`literature_sources.md`](literature_sources.md) | Primary-source register S1–S34, with the exact claim each source supports. |
| [`evaluation_protocol.md`](evaluation_protocol.md) | Cutoff and metric protocol: executed v1, proposed v2, and the open decisions. |
| [`candidate_matrix.md`](candidate_matrix.md) | Snapshot-calibration sampler catalog, roles, and interpretation limits. |

## 3. How to reproduce it

| Document | Role |
| --- | --- |
| [`data_management.md`](data_management.md) | Pinned MovieLens package, chunking, and checksum verification. |
| [`experiment_workflow.md`](experiment_workflow.md) | Run procedure, records, and registry conventions. |
| [`research_direction.md`](research_direction.md) | Deferred comparators and the federated-system confirmation track. |

## Experiment contracts and evidence

| Experiment | Contract | Evidence |
| --- | --- | --- |
| Snapshot sampling matrix | `configs/experiments/sample_generalization_v1.json` | `results/explorations/sample_generalization_full/` |
| Fixed-cohort local-data budget | `configs/experiments/fixed_cohort_budget_v1.json` | `results/explorations/fixed_cohort_budget_v1/` |

## Standing rules

- A model is a declared control. Any stochastic recommender must publish its
  same-data seed floor beside the data-policy effect; an effect smaller than
  that floor is not evidence.
- The client cohort is fixed within an episode. Changing which users contribute
  is a different question from changing how much history each user contributes.
- Offline MovieLens replay cannot establish device availability, dropout,
  communication cost, secure aggregation, consent, or federated convergence.
