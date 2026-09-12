# Recommendation research proposal

Status: framing and literature-review stage. The benchmark is not frozen yet.

This document is the canonical Markdown record for the research questions,
priority order, proposal decisions, and open design choices. The local dashboard
presents the same content as an HTML view; it does not replace this file.

## Working objective

Determine whether recommendation methods that look effective on a small,
controlled sample continue to generalize as the available users and
interactions grow. Use a literature-grounded, standardized comparison rather
than treating one dataset, one sample size, or one model family as sufficient
evidence.

The current repository gives us candidate starting points: ALS-style matrix
factorization, clustered/oracle variants, and federated data-gathering or
training strategies. The exact primary task and federated meaning remain open
until the literature review and proposal framing are complete.

## Pinned repository contract and model-choice discrepancy

The repository currently pins `implicit_feedback_ranking`, not ALS, in
`configs/objectives/implicit_mf_bpr.json`. Its objective is BPR pairwise
ranking with positives defined as `rating >= 4.0`; both planned run records
reference `implicit_mf_bpr_v1`. “ALS-style” is therefore a candidate model
family, not the current experiment contract.

The literature review must explicitly decide whether the first comparison is:

- implicit-feedback least-squares ALS versus BPR-SGD;
- BPR-SGD variants only; or
- a deliberately matched ALS/BPR comparison with separate objectives and
  metrics.

This decision changes the objective files under `configs/objectives/`, the
variant and reporting contract in `configs/experiments/`, and the research
framing in `docs/research_direction.md`. Evidence that settles it must come
from the literature decision/reproducibility matrix plus a controlled pilot
showing that the chosen metrics and negative-sampling assumptions support the
small-to-large claim.

The questions below stay confined to those artifacts. Each names the change
boundary and the evidence required before changing it.


## Provisional importance order

This is an execution order, not a claim that later questions are unimportant.

| Priority | Workstream | Why it comes here | Status |
| --- | --- | --- | --- |
| P0 | Literature review + proposal framing | Identify established strategies, defensible claims, available datasets, and feasible baselines before freezing an experiment. | Active |
| P1 | Small-to-large generalization | This is the main scientific question: determine what changes and what remains stable as data scale increases. | Primary objective |
| P2 | Benchmark standardization | Define shared splits, sampling, candidate sets, metrics, compute controls, and reporting so comparisons are interpretable. | Open |
| P3 | Existing-method comparison | Start with ALS/global, clustered methods already in the repository, and federated candidates selected by the review. | Candidate baseline track |
| P4 | Recall versus user perception/utility | Decide how offline ranking quality should be balanced with usefulness, diversity, novelty, coverage, or human perception. | Open |
| P5 | Runtime and data-structure trade-offs | Measure whether optimization helps at current scale without making the method brittle when schema or data structure changes. | Open |
| P6 | Additional datasets | Add external datasets only when they test a defined generalization claim or close a known validity gap. | Open |
| P7 | AI-model plugin track | Consider neural, transformer, or other AI recommenders after the common benchmark interface and classical baselines are stable. | Deferred |
| P8 | Factor analysis | Define whether “挖因子” means latent-factor interpretation, error-factor attribution, observable feature analysis, or another question. | Undefined |

“Federated data gathering” is currently a cross-cutting term rather than a
fixed priority. The review must distinguish privacy-preserving data collection,
federated learning, and a possible combination of both.

## Questions to answer

### Q1. How should recommendation performance be evaluated?

**Current answer:** open. ALS, clustered methods, and federated candidates may
not optimize or expose the same objective.

Resolve:

- Is the first task explicit rating prediction, implicit top-N ranking, or both?
- Which metrics are primary and which are diagnostic?
- What cutoff values (`K`) and candidate-generation rules are required?
- Are calibration, coverage, diversity, novelty, fairness, or robustness part of
a first-class score or a separate report?
- What uncertainty reporting is required across users, clients, and repeated
samples?


### Q2. How do we generalize from a small dataset to a bigger dataset?

**Current answer:** this is the primary scientific question. The scaling unit
is intentionally undecided.

`scaling_unit: Still open`

`generalization_success: Literature-defined`

Resolve through the literature review:

- Should the size ladder grow users, interactions, clients, or more than one?
- Should each size use the same dataset and split contract?
- What must remain stable: method ordering, absolute quality, quality/cost
curves, or all three?
- How many scale points and repetitions are enough to separate sampling noise
from a real trend?
- What would count as failure to generalize?

### Q3. How should the benchmark be standardized?

Resolve:

- Dataset version and license.
- User, interaction, time, and client sampling rules.
- Train/validation/test construction and leakage controls.
- Candidate catalogs and support thresholds.
- Model capacity, negative sampling, tuning budget, hardware, and stopping
rules.
- Runtime, memory, communication, and reproducibility reporting.

### Q4. How should recall be balanced with perception and user utility?

Resolve whether “perception” means human preference, usefulness, satisfaction,
explanation quality, or a measurable proxy. Keep ranking accuracy and user
utility as separate dimensions until the review supports a defensible combined
score.

### Q5. Are additional datasets necessary?

Use the literature review to decide whether MovieLens is sufficient for the
small-to-large claim. Add datasets only if they introduce a required shift,
such as different sparsity, temporal behavior, client structure, item cold
start, or privacy setting. Each added dataset must have a stated role rather
than serving as an unbounded collection.

### Q6. Should AI, transformer, or other model plugins be included?

**Current answer:** not in the first frozen comparison. First determine from
the literature whether a transformer or other AI model tests the same research
claim and can fit the benchmark's compute and data constraints.

If included later, require one narrow adapter contract for:

- training data and local/client boundaries;
- candidate scoring;
- evaluation outputs;
- runtime and memory accounting;
- model and checkpoint provenance.

### Q7. What should the literature review produce?

The review should happen before the benchmark is frozen and in parallel with
proposal refinement. Use a combined decision and reproducibility matrix for
each relevant strategy:

- source and research question;
- recommendation task and objective;
- dataset scale, sampling, and split;
- model or federated protocol;
- evaluation metrics and generalization evidence;
- runtime, memory, communication, and hardware;
- privacy, fairness, or robustness assumptions;
- code, data, license, and implementation availability;
- reported result and limitation;
- reusable idea and implication for this proposal.

### Q8. How long should a run take?

Measure end-to-end wall time separately for data preparation, training,
evaluation, communication, and reporting. Report hardware and scale with every
measurement. Do not set a target runtime until comparable literature evidence
and a first baseline implementation exist.

### Q9. Should algorithms be optimized for the current data structure?

Treat this as a controlled trade-off:

- establish a simple, data-structure-agnostic baseline;
- profile before optimizing;
- isolate optimizations behind explicit adapters or stages;
- repeat the same benchmark after changing schema, sparsity, or scale;
- report where an optimization stops generalizing.

An optimization is not a research improvement if it only works for one
serialization, schema, or dataset layout.

### Q10. What does “federated data gathering” mean here?

**Current answer:** open. The proposal must separate at least:

- privacy-preserving collection or aggregation of information;
- federated learning where raw data remains at clients;
- a combined collection-plus-training protocol.

For each interpretation, state the client boundary, trust model, information
shared, privacy threat, communication budget, and downstream recommendation
task.

## Confined decision matrix

Each question changes only the named artifact boundary. The literature review
and controlled evidence must settle the change before it is pinned.

| Question | Artifact boundary | Evidence required |
| --- | --- | --- |
| Q1 evaluation | `configs/objectives/`, `configs/experiments/`, `docs/research_direction.md` | Literature-standard objective/metrics plus a pilot that separates candidate methods. |
| Q2 generalization | Scale ladder in `configs/experiments/`; interpretation in `docs/research_direction.md` | Literature across scale plus repeated runs separating noise from trend. |
| Q3 benchmark | Splits, catalogs, metrics, capacity, and reporting in `configs/experiments/` | Reproducibility details and local leakage/contract validation. |
| Q4 perception | Utility/perception outputs in `configs/experiments/`; rationale in `docs/research_direction.md` | Validated user-utility measures or defensible proxies. |
| Q5 factors | Factor-analysis outputs in `configs/experiments/`; interpretation in `docs/research_direction.md` | A literature-supported factor definition that changes a decision. |
| Q6 datasets | Dataset roles in `configs/experiments/`; validity rationale in `docs/research_direction.md` | A specific distribution, sparsity, temporal, client, or privacy shift. |
| Q7 AI plugins | New objective/variant in `configs/objectives/` and `configs/experiments/` | Same-claim literature evidence plus a resource-bounded pilot against BPR. |
| Q8 literature review | Review criteria and proposal rationale in `docs/research_direction.md` | Completed decision/reproducibility matrix with source, objective, scale, and limitations. |
| Q9 runtime | Timing and hardware fields in `configs/experiments/`; interpretation in `docs/research_direction.md` | Comparable end-to-end measurements across the scale ladder. |
| Q10 optimization | Explicit baseline/optimization stages in `configs/experiments/` | Profiling and repeat runs across schema, sparsity, and scale. |
| Q11 federated meaning | Client, shared-information, and communication fields in `configs/experiments/`; definition in `docs/research_direction.md` | Protocol evidence distinguishing collection, federated learning, or both. |

## Immediate next steps


1. Run the combined literature and reproducibility review.
2. Refine this proposal from the review findings.
3. Define the exact small-to-large generalization claim and scaling unit.
4. Freeze the benchmark contract only after steps 1–3.
5. Select the smallest defensible set of ALS, clustered, and federated baselines.
6. Defer transformer plugins and factor analysis until their roles are explicit.
