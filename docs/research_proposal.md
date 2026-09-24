# Privacy-compatible local-data budgeting for federated recommendation: proposal v3

Status: research design. The executed full-snapshot matrix is a calibration
study; it is neither a federated deployment nor an external-validity result.

Canonical proposal: `docs/research_proposal.md`
Candidate matrix and status labels: `docs/candidate_matrix.md`
Executed full-data result: `results/explorations/sample_generalization_full/report.md`
Source register: `docs/literature_sources.md`
Dashboard: the local dashboard renders this document and the executed figures.

## 1. Core question

Federated recommendation keeps raw interaction histories on client devices, but
training cannot use every eligible client and every local event on every round.
Availability, communication, device limits, privacy mechanisms, and
heterogeneous client histories make “use all data” infeasible and sometimes
undesirable.

This study asks:

```text
Can a privacy-compatible controller use an initial, fixed client cohort's
permitted pilot diagnostics to choose a local-data plan and a stopping budget
that preserves a declared recommendation decision relative to using all
permitted history from that same cohort?
```

For an evaluation episode `e`, let:

- `C_e` be the fixed eligible client cohort, selected before collection;
- `D_full(C_e)` be every permitted collection-window event from those clients;
- `D_g,b(C_e)` be local events retained or used under data plan `g` at cost
  budget `b`;
- `A` be a fixed recommendation-training procedure;
- `T_e` be later held-out events from the same cohort; and
- `Pi` be the controller, which observes only permitted pilot diagnostics `Z`
  and returns `stop` or a next `(g, b)` action.

The historical full data is evaluator-only ground truth for a backtest. It is
not uploaded in a real federated deployment. The primary comparison is:


```text
A(D_g,b(C_e)) evaluated on T_e  versus  A(D_full(C_e)) evaluated on T_e
```

Here “gathering” means authorizing, retaining, or scheduling the use of
existing permitted local events; it does not mean inducing users to create new
events or uploading their histories. A policy that changes consent or user
behavior requires a prospective randomized study, not this historical replay.

### Federation topology: horizontal only

This study is **horizontal (cross-device) federated recommendation**: one user
is one client, every client holds the same feature schema
`(user, item, rating, timestamp)`, and clients are partitioned by *sample*, not
by feature. The local-data budget is therefore a per-client row budget.

**Vertical federated learning is explicitly out of scope for the current
claim.** In a vertical setting the same users are shared across parties that
each hold *different features*—for example a ratings platform, a tagging
service, and a content-metadata provider—so the open problems change:

| Concern | Horizontal (this study) | Vertical (not studied) |
| --- | --- | --- |
| Partition | by user/sample | by feature/party |
| Budget unit | local interactions per client | shared feature blocks and aligned entities per party |
| Prerequisite | client eligibility and availability | private entity alignment across parties |
| Label location | every client has its own interactions | labels usually sit with one party only |
| Per-round traffic | model updates per device | intermediate representations per aligned batch |
| Main leakage risk | update inversion | intermediate-representation and alignment leakage |

A vertical variant is constructible from this snapshot, because `ratings.csv`,
`tags.csv`, and `genome-scores.csv` share `movieId` and `userId` while carrying
different feature blocks. That is recorded as a separate future track with its
own estimand, alignment protocol, and cost model. It is **not** evidence for or
against the horizontal local-data-budget claim, and results from the two
topologies must never be pooled.

The executed replay is a centralized *emulator* of the horizontal data
partition. It fixes who contributes and varies how much each contributes; it
does not yet run federated rounds.

The existing snapshot comparison—fixed-protocol recommendation result
`R_a(D_N)` versus `R_a(S_s(D_N, n, r))`—remains a calibration study. It
measures which sampling mechanisms can distort a full-data result before the
client-history controller is introduced.

## 2. Federated data-gathering pain points and scoped contribution

| Pain point | Why ordinary small-data sampling is insufficient | Proposed response |
| --- | --- | --- |
| **Availability and participation bias** | Devices eligible to train can correlate with time zone, network access, device class, and therefore local data. Selected clients may also drop out. | Fix the eligible cohort for the primary comparison; report invited, available, completed, and covered-client counts; require minimum coverage by declared client strata. |
| **Non-IID and unbalanced local histories** | More interactions or more clients can alter aggregation bias, variance, and subgroup outcomes; more heterogeneous data is not automatically better. | Allocate and assess budget by client-history and coverage strata, not only global interaction count; report worst-stratum quality and decision stability. |
| **Communication and device cost** | Rounds consume bandwidth, battery, computation, and wall-clock time; full local histories can be excessive. | Make cost a first-class budget: interactions processed, participating clients, bytes, rounds, local compute, and time. The controller must beat fixed policies at the same declared cost. |
| **Limited local storage and streaming events** | Devices may need to decide online which local events to retain; random retention can distort the stored distribution and harm convergence. | Assess simple history caps first, then assess local relevance/retention selectors only under an explicit compute, metadata, and privacy contract. |
| **Privacy-constrained observability** | A server cannot inspect raw client histories; gradients can leak information; secure aggregation exposes aggregates rather than individual updates. | The controller may use only predeclared client-side or securely aggregated pilot diagnostics. It broadcasts a data plan; it does not centralize histories or require per-client raw-data scores. |
| **Optimization and deployment gaps** | A centralized sample result does not establish FL convergence, dropout behavior, or secure-aggregation compatibility. | First use an offline client-history backtest; then confirm surviving policies in a federated simulator with availability, dropout, communication, and a named privacy mechanism. |

The proposed first direction is a **coverage-constrained local-data budget
controller**: decide whether more local history is worth gathering or using for
a fixed eligible cohort, while allowing `unknown / gather more` when pilot
coverage or uncertainty is inadequate.

Two direct antecedents require a narrower claim than “no related method.”
FLRD dynamically selects locally relevant training examples to improve a global
model; it relies on a private selector and server validation feedback [S27].
ODE selects streaming local records for limited device storage using local
gradients and cross-client coordination metadata [S26]. Neither paper targets
a predeclared recommendation-decision fidelity rule, false-accept control, or
the comparison against all permitted history from a fixed federated-recommender
cohort. They are assessment directions and baselines, not ignored prior work.

User-governed FedRec data contribution is a separate privacy model: users may
upload varying proportions of data to a server [S28]. Oort, clustered sampling,
and ProxyRL-FRS decide which clients participate; FALE selects unlabeled
instances for labels; Data-Genie selects a centralized recommender sampling
scheme [S19, S23–S25, S29]. The study makes no novelty claim before the
direction assessment records overlap, feasibility, and empirical differences.

## 3. Generalizability claims and measurements

“Generalizability” is not one score. Every claim must name a source condition,
a target condition, a recommendation outcome, and the unit that varies. For an
algorithm `a`, condition `c`, and outcome `M`, report the signed effect
`Δ(a,c) = M(a,c) - M(a, reference)` plus its uncertainty; do not collapse
different targets into one ranking.

| Claim | Target condition | Effect and required measurements | Status |
| --- | --- | --- | --- |
| **Federated local-data sufficiency** | A declared data plan and cost budget for a fixed eligible cohort, compared with all permitted history from that cohort. | Future-window NDCG@10/Recall@10 effect; decision fidelity; false-accept rate; client and stratum coverage; interaction, device, byte, round, compute, and time cost. | **Planned primary claim** |
| **Sampling-design robustness** (sample-to-reference fidelity) | A declared training-sample method and fraction versus the same snapshot's full training set. | For each model: NDCG@10 and Recall@10 effect; draw uncertainty; panel support; model-order/top-choice agreement when selecting among models; time and memory. | **Executed calibration** |
| **Model-family robustness** | Whether the local-data effect holds for different recommender families. | Distribution of the local-data effect across models, conclusion agreement, and model-specific cost. This is not a ranking of models. | Planned confirmation |
| **Temporal and availability robustness** | Later interaction windows and declared device-availability/dropout regimes. | Future-window metric effects, completion-weighted client coverage, user/item churn, and drift. | Planned |
| **Population/subgroup robustness** | Declared activity, long-tail, client-resource, or preference groups. | Per-group metric effects, worst-group result, between-group gap, support, and group coverage. | Planned |
| **External-domain robustness** | Independently collected datasets or domains. | The same within-dataset effect per dataset, its distribution across datasets, and dataset-level heterogeneity. Raw NDCG values are not pooled across domains. | Not executed |
| **Federated-system confirmation** | A named FL optimizer, privacy mechanism, and system model under the same data plans. | Global and per-client metric effects, convergence, completed rounds, communication, privacy cost, and failure rate. | Not executed |

The current result supports only the sampling-design calibration row. Its
estimand, for a fixed model, split, and evaluation panel, is:

```text
Δ(a, s, f) = mean over draws r of [M(a, S_s(D_N, f, r)) - M(a, D_N)]
```

The planned primary estimand for episode `e` is:

```text
Δ_e(g, b) = M(A(D_g,b(C_e)), T_e) - M(A(D_full(C_e)), T_e).
```

The controller makes a false acceptance when it declares a budget adequate but
the later full-reference comparison violates a predeclared metric tolerance,
decision rule, or coverage requirement. The chronological test target measures
held-out prediction performance, but one snapshot and one cutoff do not
estimate temporal, client, external-domain, or federated-system robustness.

### Metric contract and structural ceilings

> **Under revision.** `docs/evaluation_protocol.md` proposes protocol v2
> (deep cutoffs, `Precision@K` instead of the ad-hoc `HitRate`, `MRR` dropped)
> on the evidence of S30–S34. The rules below describe **executed** protocol v1
> and remain in force until that draft is accepted.

Metric choice is a control, not a presentation detail. Top-`K` retrieval metrics
have different ceilings, and those ceilings vary systematically with exactly the
user property this study manipulates—how much history a client has.

| Metric | Definition at `K` | Ceiling behaviour |
| --- | --- | --- |
| `NDCG@K` | `DCG@K / IDCG` with `IDCG` over `min(K, R)` | Reaches 1.0 for any `R`; **not** capped. Primary outcome. |
| `HitRate@K` | `hits / min(K, R)` | Reaches 1.0 for any `R`; cap-aware retrieval check. Secondary outcome. |
| `Recall@K` | `hits / R` | Capped at `K / R`. A user with 50 future positives cannot exceed 0.20 at `K = 10`. |
| `Precision@K` | `hits / K` | Capped at `R / K`. A user with 3 future positives cannot exceed 0.30 at `K = 10`. |

Here `R` is the size of a user's relevant set after the fixed seen-item
exclusion. The consequence is a real confound, not a cosmetic one: heavy users
have large `R`, so their `Recall@K` ceiling is low, and heavy users are also the
clients whose histories a per-client cap truncates most. A raw-recall comparison
across caps therefore mixes a data effect with a metric-ceiling effect.

Rules currently in force (protocol v1):

- Report `NDCG@10` as primary and `HitRate@10` as the retrieval secondary. Raw
  `Recall@10` may be reported for continuity, never alone.
- Publish the ceiling audit beside the results: relevant-set size distribution,
  the share of users above `K`, and the mean and minimum recall ceiling.
- Never compare `Recall@K` or `Precision@K` across groups whose relevant-set
  sizes differ without stating both ceilings.
- Use algorithm ordering only when the decision is specifically “which model
  should be selected.” Practical tolerances must be declared for that decision;
  there is no universal acceptable NDCG error.
- Also report uncertainty over the natural unit (clients, episodes, time
  windows, groups, datasets, or sample draws), coverage, and resource cost.

Known weaknesses of v1, measured or sourced:

- `K = 10` is the shallowest depth studied in S30 and is the **least robust and
  least discriminative**; deeper cutoffs near 100 dominate it on both axes while
  rarely changing system order. The value 10 was never justified here.
- `HitRate@K` equals `Precision@K` whenever `R >= K`, which covers 62.5% of the
  evaluated users, so it adds a non-standard name rather than a new measurement.
- The relevant set is “items later rated at least 4”, an observation artifact
  under missing-not-at-random feedback (S32); every metric dividing by `R`
  inherits that bias.
- No popularity-bias treatment is applied, so averaged accuracy may diverge from
  unbiased accuracy (S33, S34).

Not yet measured, and therefore not claimed: MRR, catalog coverage of the
recommendations themselves, per-activity-group metric breakdowns, and any
beyond-accuracy objective such as diversity or novelty.

## 4. Calibration estimand and controller controls

The full snapshot is the calibration reference population, not an oracle for
other domains. For every snapshot sample cell, report the mean, standard
deviation, and interval across independent draws. A single draw is descriptive
only.

### Controller observability and decision controls

- Select `C_e` using only information available before each collection episode.
- The controller receives only `Z`: predeclared client-side or securely
  aggregated diagnostics, plus permissible system telemetry. It never receives
  raw histories, raw gradients, the full-reference data, or future test events.
- A data plan is broadcast and executed locally. The primary plan family uses
  client-history caps and aggregate coverage quotas, not individual
  utility-ranked raw-data inspection. Any client stratum used by a quota must
  be computable locally and released only through the named privacy contract.
- Keep the candidate models, metric rule, training budget, and client cohort
  fixed between `D_g,b(C_e)` and `D_full(C_e)`.
- Match policies on an explicit cost vector. Interaction count alone is
  insufficient when participating-client, byte, or round costs differ.
- Tune `Pi` only on earlier episodes. Freeze it before later episodes. Report
  its false-accept and abstention rates with client-level uncertainty.

### Fixed controls

- MovieLens `ml-latest`, snapshot version `2023-07-20`.
- Positive proxy: `rating >= 4.0`; this is implicit-style, not click data.
- One chronological 80/10/10 split made once from the full snapshot.
- Samples are drawn from the full training pool only.
- The full validation/test target is frozen. Sample-native re-splitting is a
  sensitivity analysis, not the primary comparison.
- Primary ranking always excludes the full-reference training positives for the
  evaluated user. Using the sampled training positives instead is logged as a
  separate sample-native sensitivity, never silently mixed into the headline
  metric.
- Report both the fixed evaluation-panel size and the number of users with
  sampled training support. A shrinking supported-user count is a coverage
  failure or a conditional result, not evidence of metric generalization.
- Same candidate catalog, metric definition, ranking cutoff, model code,
  hyperparameters, initialization schedule, negative-sampling policy, and
  stopping/update budget for every cell.
- No tuning on the frozen test target. If tuning is needed, use the validation
  portion inside every cell with the same budget.
- Ten independent sample draws per `(sample size, scheme)` cell for the first
  local matrix. Add draws only under a declared precision rule; never interpret
  a one-draw cell as evidence.

The first implementation uses deterministic popularity controls and a fixed
item-item cosine baseline because they can be run over the full snapshot locally
and isolate sampling effects before a trainable model is introduced. The pinned
BPR-MF objective remains the first trainable model candidate, but its full
matrix-factorization runner is a later execution layer rather than a claim that
the current exploratory run already trained BPR.

## 5. Snapshot-calibration sampling matrix

The following candidates are calibration conditions, not deployable federated
data plans. In particular, user sampling changes the contributing population
and therefore cannot answer the fixed-cohort local-history question by itself.

The candidates answer different questions and must not be treated as equivalent.
All schemes use the same target interaction-budget ladder and record realized
users, items, interactions, density, activity quantiles, and item-popularity
quantiles.

| ID | Sampling operator | What it tests | Primary status |
| --- | --- | --- | --- |
| `uniform_user` | Sample users without replacement; retain each selected user's full pre-split history | Smaller user population with intact user histories | Primary |
| `activity_stratified_user` | Sample users within full-data activity strata | Whether preserving user-activity composition improves transfer | Primary |
| `uniform_interaction` | Sample training interactions uniformly | Raw-row compute reduction; can damage user coverage | Primary |
| `within_user_history` | Sample interactions within each user's training history | Compute reduction while preserving user coverage | Primary |
| `item_stratified` | Sample interactions while preserving item-popularity strata | Whether item-tail coverage controls transfer | Secondary |
| `recent_window` | Restrict to a recent temporal window | Distribution shift / recency stress test, not representative sampling | Stress test |

The initial matrix should begin with the first four methods. `item_stratified`
and `recent_window` enter after the sampler and reporting format are stable.
The project must keep a uniform random baseline even if a structured sampler
looks better; otherwise improvement cannot be attributed to structure.

### Planned federated local-data plans

Every plan retains the same eligible cohort and uses chronological local events
unless an offline-only upper-bound baseline is explicitly labeled otherwise.

| ID | Data plan | What it tests |
| --- | --- | --- |
| `equal_history_cap` | Each completed client uses its next `k` eligible local events. | A simple, privacy-compatible per-client history budget. |
| `activity_capped` | A bounded activity-proportional cap with a minimum per covered stratum. | Whether extra history from active clients helps without erasing low-activity coverage. |
| `coverage_quota` | Broadcast quotas across declared activity, long-tail, or resource strata; randomize within an eligible stratum. | Whether representative coverage is safer than an aggregate-only budget. |
| `adaptive_budget` | `Pi` extends one of the above plans only when pilot uncertainty or coverage fails. | The primary stop-or-gather decision. |
| `offline_random_history` | Randomly retain past local events at the same cost. | A non-deployable upper bound on what arbitrary retention could achieve. |

Measure the planned cost axis as a vector, then set a primary operational budget
before each experiment: completed clients, local interactions processed,
uplink/downlink bytes, rounds, local compute, and wall-clock time.

### Direction assessment gate

Before implementing a new data plan, record its action, allowed diagnostics,
privacy disclosure, primary cost, reference condition, and fatal assumption.
Evaluate it against the fixed-cohort full-history reference only when its action
can be replayed without accessing unavailable future data. The initial screen
compares:

1. `coverage_constrained_budget` — simple caps and aggregate coverage with a
   stop-or-gather safety decision; the recommended first direction;
2. `local_relevance_selection` — FLRD-like client-side relevance filtering;
3. `online_retention_selection` — ODE-like streaming storage selection; and
4. `participant_or_contribution_policy` — client selection or voluntary raw-data
   sharing, recorded as an adjacent question rather than pooled into the first
   estimand.

Promote one direction only if it improves the declared safety–cost frontier
under a privacy-feasible observability contract. The completed feasibility
screen in `docs/direction_assessment.md` promotes
`coverage_constrained_budget` to the first implementation, defers FLRD- and
ODE-like selectors, and excludes participant/contribution policies from this
estimand. This is not an empirical performance result.

### Sample-size axis

Use training-interaction fractions as the primary cost axis:

```text
1%, 2.5%, 5%, 10%, 25%, 50%, 100%
```

Report the realized user and item counts beside every result. A user-count
ladder is a secondary analysis because user sampling and interaction sampling
otherwise compare different effective budgets.

### Independent draws

For each active `(fraction, scheme)` cell, use ten independent seeds generated
from a stable key such as:

```text
sha256(sample_generalization_v1:scheme:fraction:replicate)
```

Draws may overlap in rows, but their sampling decisions must be independent.
The current one-seed probes (`20260909`) are retained as historical diagnostics,
not as evidence for a generalization claim.

## 6. Evidence phases

### Phase 0 — problem and estimand freeze

Declare the recommendation decision, client cohort eligibility rule, allowed
diagnostics, data-plan family, cost vector, privacy mechanism for later
confirmation, safety tolerance, and acceptable false-accept bound. Complete the
direction assessment gate before implementing a learned selector. Learning
curves support estimating the effect of more data, but their shapes are diverse
and no universal curve model is justified [S16].

### Phase 1 — full-snapshot calibration

Run the existing candidate samplers over MovieLens without and then with cheap
recommendation models. Generate coverage summaries, support failures, metric
effects, algorithm-order changes, and cost. This phase identifies which
mechanisms cannot safely approximate the snapshot; it is not a federated claim.

### Phase 2 — pseudo-prospective fixed-cohort emulator

Create multiple historical episodes. Select `C_e` before a collection window,
retain every permitted event for `D_full(C_e)`, and replay each client-history
data plan chronologically. Reserve later events from the same cohort as `T_e`.
First compare the simple coverage controller with only those direct-selection
directions whose diagnostics can be computed from information available at that
time. The controller may receive only pilot diagnostics and increments already
collected by its own plan. Repeat over client-cohort draws, temporal episodes,
and plan randomizations. Bootstrap clients, not correlated interactions.

### Phase 3 — controller backtest

Calibrate `Pi` on earlier episodes, then lock it. On later episodes compare
`adaptive_budget` against the fixed plans at matched cost. The primary safety
outcome is the false-accept rate; the primary utility outcome is minimum cost
subject to the fidelity and coverage constraints. An abstention is correct when
the controller lacks enough evidence to accept a budget.

### Phase 4 — federated-system confirmation

Run the policies that survive Phase 3 in a federated simulator. Hold cohort,
model, and local-data plan constant while measuring optimizer convergence,
client availability, dropout, completed rounds, bytes, local compute, and a
named privacy mechanism such as secure aggregation. A centralized pass alone
cannot establish these effects.

### Phase 5 — external-data and assumption checks

MovieLens backtests establish internal evidence only. Re-run the locked protocol
on independently collected datasets or constructed regimes that vary sparsity,
long-tail behavior, temporal drift, exposure, missingness, and availability.
Use leave-one-dataset-out evaluation for any claim that the controller transfers.

## 7. Analysis rules

For every algorithm, episode, and data-plan cell, produce raw metrics against
the cost vector with client-level uncertainty, paired effects versus the
full-cohort reference, cutoff sensitivity, stratified breakdowns, weighted and
unweighted values, coverage counts, and resource cost. A positive
local-data-sufficiency claim additionally requires the predeclared safety,
coverage, cost, and federated-confirmation conditions to hold.

## 8. Current evidence

The tracked snapshot contains 33,832,162 ratings from 330,975 users. The
sampling-calibration matrix in `results/explorations/sample_generalization_full/`
remains the prior evidence layer: it shows that sampling mechanism, support, and
coverage change a full-data result, with item-item cosine far more sensitive
than global popularity controls.

`results/explorations/fixed_cohort_budget_v2/` holds the executed protocol-v2
replay. The cohort is the entire eligible pool of 16,084 users, of which 4,100
have future positives; the frozen reference uses 2,205,099 training
interactions, 117,092 of them collection-window events the policies control.
Metrics are reported at cutoffs 10, 20, 50, and 100 with 100 as the primary
depth, alongside self-normalized propensity-weighted values and stratified
breakdowns.

Full-reference NDCG@100 is 0.1095 popularity, 0.1097 rating-weighted
popularity, 0.1228 item-item cosine, and 0.1459 three-seed implicit ALS. Under
the deterministic probe, **every tested cap now shows a significant deficit
against full history**: -0.0058 [-0.0065, -0.0051] at cap 1, shrinking
monotonically to -0.0012 [-0.0015, -0.0009] at cap 50. Propensity-weighted
deltas agree in sign and shrink in the same order, so the effect is not an
exposure artifact.

This reverses the protocol-v1 null result, which was measured at cutoff 10 with
520 evaluated users and found no cap whose interval excluded zero. S30
identifies cutoff 10 as the least discriminative depth studied; the null did not
survive a deeper cutoff and a census cohort. The practical magnitude stays
small: capping at one collection event per client keeps 5,089 of 117,092
collection interactions and costs about 4.7% of NDCG@100 relative to the
full-history reference.

The tail-reserve policy remains unpromoted. Four of 48 cells show a strictly
positive paired interval, all implicit ALS at caps 10 and 20 with effects near
0.0003; every deterministic-probe cell is negative or null. ALS is the strongest
model at this depth but still fails its own control: no ALS cell exceeds the
0.0203 same-data seed floor.

These results are in-reference calibration and fixed-cohort replay evidence,
not evidence that a federated data plan is safe. Varying total per-client
history, repeating over time windows, and federated-system confirmation remain
planned evidence.

## 9. Models and system boundaries

The model is a controlled probe of the data-plan claim, and the model itself is
a declared control rather than a free choice.

- **Executed calibration:** popularity, rating-weighted popularity, and a
  fixed-support item-item cosine baseline.
- **Primary personalized probe:** deterministic item-item cosine over a fixed
  full-reference support set. It has no random state, so a budget effect is not
  confounded with training noise, and it is the strongest model measured here.
- **Secondary personalized probe:** implicit ALS [S1], averaged over a common
  seed set for every condition.
- **Mandatory model-noise control:** every stochastic recommender must publish
  its same-data seed floor—per-seed mean metric spread and mean per-user
  absolute metric difference—next to the data-policy effect. A data-budget
  claim is invalid when the effect is smaller than that floor.
- **Next trainable confirmation:** the repository's BPR-MF ranking objective
  [S2, S15], under the same seed-floor rule.
- **Federated-system control:** compare a named baseline such as FedAvg with a
  heterogeneity control such as FedProx only after the same local-data plan has
  passed the fixed-cohort emulator. This separates data-plan effects from
  optimizer effects [S7, S8].
- **Privacy boundary:** raw data locality is not a privacy proof. The
  confirmation must name the secure-aggregation or other protection mechanism,
  its threshold, and its added communication/availability constraints [S12,
  S22].

Active labeling, individual utility scoring from raw logs, incentives, and
clustered personalization are outside the first estimand. Local relevance and
online-retention selection are controlled comparators only when their
observability, compute, and privacy contracts are specified.

## 10. References used here

- [S1] Implicit-feedback ALS and missing-data semantics.
- [S2] Bayesian Personalized Ranking.
- [S3–S5] Recommendation splitting, evaluation, and leakage controls.
- [S6] MovieLens context and limitations.
- [S7–S9] Federated communication, heterogeneous optimization, and client drift.
- [S12] Federated recommendation privacy boundary.
- [S16] Learning-curve estimands and shape uncertainty.
- [S17] Repeated training-set choices and full-scale extrapolation.
- [S18] Cross-validation estimands and variance limitations.
- [S19] Recommendation-dataset sampling and ranking preservation.
- [S20] Multi-dataset recommendation scaling evidence.
- [S21] Aggregation–heterogeneity trade-off.
- [S22] Production FL availability, bias, system, and secure-aggregation constraints.
- [S23–S24] Participant selection and representative client sampling.
- [S25] Privacy-constrained federated active data selection.
- [S26] Streaming limited-storage local-data retention.
- [S27] Dynamic private relevance selection for federated training data.
- [S28] User-governed federated-recommender contribution boundary.
- [S29] Recommender-specific client selection comparator.


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
| P8 | Factor analysis | Define whether "factor mining" means latent-factor interpretation, error-factor attribution, observable feature analysis, or another question. | Undefined |

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

See `docs/literature_sources.md` for the complete evidence register.
