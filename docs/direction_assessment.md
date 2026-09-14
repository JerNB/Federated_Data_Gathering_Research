# Federated local-data direction assessment

Status: **completed feasibility screen and one executed real-data replay with a
four-model control.** The tested item-tail reserve policy is rejected, and the
deterministic personalized probe shows no measurable mean-quality penalty from
capping collection-window history on this cohort.

## Review limitation

The source review is primary-source-backed, but it is **not** an independent
multi-agent review. After the requested retry, the GPT reviewer again reached
its two-minute runtime limit without output; the requested Anthropic agent had
no configured model or credentials. Earlier librarian/reviewer runs likewise
returned no usable review. This report records direct source inspection and
must not be represented as independently reviewed.

This report runs the decision framework in `docs/exploration_goal.md`. It first
identified a replayable direction, then executed its bounded MovieLens
comparison. It does not establish federated deployment or external validity.

## Evidence matrix

| Direction | Primary antecedent | What the antecedent optimizes | Non-transferable assumption | Assessment role |
| --- | --- | --- | --- | --- |
| Coverage-constrained local-data budget | No exact direct antecedent; combines recommender sampling sensitivity [S19], learning-curve caution [S16], and representative client sampling [S24]. | A new safety-and-cost question: when to stop using additional local history. | Aggregate activity/coverage diagnostics can be obtained without raw-history exposure; real availability must later be simulated. | **Tested as an item-tail reserve cap; not promoted by the executed replay.** |
| Local relevance selection | FLRD [S27]. | Global model quality after locally filtering irrelevant/noisy examples. | Private selector and server validation feedback are feasible and do not leak an impermissible target-distribution signal. | Defer as a controlled comparator. |
| Online retention selection | ODE [S26]. | FL convergence and accuracy under streaming, storage-limited records. | Per-event gradients, local storage state, and cross-client coordination metadata fit the device/privacy budget. | Defer as a controlled comparator. |
| Participant selection | Oort [S23], Clustered Sampling [S24], ProxyRL-FRS [S29]. | Which devices participate, based on speed, utility, clustering, or learned contribution. | Device availability, system speed, and participation signals exist and are measured. | Exclude from the fixed-cohort estimand; retain as an adjacent later study. |
| User-governed contribution | UGFedRec [S28]. | User-specific privacy/data-sharing trade-off. | Users' consent and willingness to contribute can be observed or randomized. | Excludes itself from the no-raw-upload question; defer. |
| Federated active labeling | FALE [S25]. | Which unlabeled examples receive queried labels. | Label-query cost and oracle responses exist. | Out of scope for implicit-feedback MovieLens replay. |

`S*` identifiers resolve in `docs/literature_sources.md`, which contains the
primary-source URLs and scope notes.

## Dataset feasibility facts

The pinned MovieLens `ml-latest` snapshot supports a chronological,
client-history backtest:

- `ratings.csv` supplies anonymized stable `userId`, `movieId`, explicit
  `rating`, and UTC `timestamp` fields.
- The manifest records 33,832,162 ratings from 330,975 users through
  2023-07-20.
- The existing calibration already fixes `rating >= 4.0`, chronological
  per-user partitions, fixed candidate controls, and NDCG@10/Recall@10.

It does **not** supply device capability, charging/network eligibility,
communication bytes, client dropout, consent, privacy preferences, secure
aggregation output, or genuine federated rounds. Those values cannot be
estimated from MovieLens and must remain modeled assumptions until a
federated-system confirmation.

## Evidence cards

### 1. Coverage-constrained local-data budget — promoted

| Field | Assessment |
| --- | --- |
| Action and estimand | Keep the eligible cohort `C_e` fixed. Broadcast chronological per-client history caps and aggregate coverage quotas. Increase a budget only when the controller cannot safely stop. Compare model result on `D_g,b(C_e)` with all permitted collection-window history `D_full(C_e)` on later events from the same cohort. |
| Permitted diagnostics | Client-side history count, timestamp/order, coarse locally computed activity band, completed-client count, aggregate stratum coverage, and local validation stability. A server must not receive raw histories or individual utility scores. |
| Privacy/system contract | Historical screen assumes only local computation plus aggregate count release. Production confirmation must name the aggregation threshold, telemetry disclosure, availability/dropout model, bytes, rounds, and local compute. |
| Replay validity | **Pass.** Rating timestamps permit chronological replay; the cohort can be selected from the pilot window only. The test window remains hidden from the controller. |
| Safety and utility | Predeclare a practical NDCG@10 tolerance, model-selection rule, maximum false-accept bound, required activity/item-tail coverage, and one primary cost. Report Recall@10, abstention, and client-level uncertainty. |
| Fatal assumption | Aggregate coverage and stability diagnostics are enough to decide data sufficiency. A real deployment may have availability bias that this snapshot cannot reveal. |
| Decision | **Promote.** This is the only direction that matches the fixed-cohort, no-raw-upload question and can be replayed from available data without adding an unobserved model. |

### 2. Local relevance selection — deferred comparator

| Field | Assessment |
| --- | --- |
| Action and estimand | Each client filters or weights local records with an FLRD-like relevance selector before local training. Compare at a fixed local-record budget. |
| Permitted diagnostics | Per-record local features/labels, selector state, and server validation feedback. The feedback channel must be specified before evaluation. |
| Privacy/system contract | FLRD's private selector does not by itself prove that the feedback signal, selector updates, and metadata satisfy this project's aggregate-only controller boundary. |
| Replay validity | **Conditional.** MovieLens can replay labels and timestamps, but an online selector needs a trainable recommender/FL loop and a validation signal available before the future test period. |
| Safety and utility | Must separately prove that selection improves the safety–cost frontier, not merely the training metric. |
| Fatal assumption | A server validation set represents the target deployment without leaking future distribution information. |
| Decision | **Defer.** It is a strong comparator after the simple controller passes, but it adds selector and feedback confounding to the first experiment. |

### 3. Online retention selection — deferred comparator

| Field | Assessment |
| --- | --- |
| Action and estimand | As ratings stream in, each client accepts/evicts records under a storage cap using an ODE-like local value rule. |
| Permitted diagnostics | Per-event model gradients, local storage state, and ODE-style rough cross-client coordination metadata. |
| Privacy/system contract | Per-event gradient computation and any cross-client metadata release need an explicit compute and privacy accounting. Secure aggregation may not expose the metadata ODE needs. |
| Replay validity | **Conditional.** Timestamps support streaming replay, but the repository has no executed per-event-gradient federated recommender loop. |
| Safety and utility | Must be compared with an equal chronological cap at equal storage, compute, and communication cost. |
| Fatal assumption | Event-gradient value is stable enough to guide retention despite non-IID client data and a recommender objective. |
| Decision | **Defer.** Directly relevant to storage, but its observability and compute requirements make it unsuitable as the first claim. |

### 4. Participant selection — excluded from primary estimand

| Field | Assessment |
| --- | --- |
| Action and estimand | Select a changing subset of devices for a training round. |
| Permitted diagnostics | Device speed/availability and client utility or cluster information. |
| Replay validity | **Fail for the current question.** MovieLens lacks availability, dropout, device, and communication observations; changing participants also changes `C_e`. |
| Fatal assumption | The observed participants represent the target population. |
| Decision | **Adjacent later study.** Never compare its result as if it were a same-cohort local-history budget. |

### 5. User-governed contribution — deferred boundary case

| Field | Assessment |
| --- | --- |
| Action and estimand | A user chooses how much data to share, potentially uploading raw data. |
| Permitted diagnostics | Consent, privacy preference, contribution willingness, and a user-specific privacy budget. |
| Replay validity | **Fail.** The MovieLens snapshot has none of these variables and cannot simulate changed consent or behavior. |
| Fatal assumption | Historical interaction data predicts data-sharing willingness. |
| Decision | **Defer.** This is a different privacy/consent research program. |

## Ranked decision after real-data replay

1. **Do not promote the tested tail-reserve coverage policy.** It materially
   raised tail-share but had no tested cell with a strictly positive paired
   95% interval for lower absolute NDCG error than the equal chronological cap.
2. **Retain equal chronological history caps as the empirical baseline.** They
   establish a reproducible fixed-cohort reference curve, not a stop-controller.
3. **Defer local relevance selection.** It remains the closest record-quality
   comparator, but requires a specified private selector and non-leaking
   validation contract.
4. **Defer online retention selection.** It remains storage-relevant but needs
   per-event-gradient compute and metadata/privacy design unavailable here.
5. **Exclude participant selection and user-governed contribution from this
   estimand.** They change the client population or consent/privacy model.

The negative result is evidence about this policy, cohort, models, and time
episode—not evidence that every coverage-aware local-data policy fails.

## Tested experiment contract

### Hypothesis

For a fixed eligible cohort, a coverage-constrained chronological local-history
policy can meet a predeclared recommendation-decision fidelity and coverage
requirement at lower primary cost than an equal-history-cap policy, without a
higher false-accept rate than the predeclared limit.

### Episode construction

1. Partition interactions by **global chronological cutoffs** into pilot,
   collection, and future-test windows. Do not use the current per-user
   80/10/10 calibration split as a federated proxy.
2. Choose `C_e` only from pilot-window observations: users with at least a
   predeclared minimum number of positive pilot interactions.
3. Retain every collection-window event from `C_e` only for evaluator-side
   `D_full(C_e)`.
4. Replay data plans chronologically over the collection window. The controller
   observes only its pilot diagnostics and the increments its plan has retained.
5. Evaluate all trained candidates on future-test positives from the same
   original cohort. Report the no-future-event and non-completion rates instead
   of silently dropping them.

### Initial policies

- **Reference:** all permitted collection-window history from `C_e`.
- **Required baseline:** equal chronological history cap per completed client.
- **Tested coverage candidate:** chronological caps with a 50% reserve for
  items at or below the pilot 20th-percentile item-count ceiling; randomize no
  client membership.
- **Not tested:** uniform random historical retention, FLRD, ODE, participant
  selection, and user-governed contribution.

### Locked controls before implementation

The following must be set before inspecting Phase-2 outcomes, not calibrated on
their future-test results:

- practical NDCG@10 non-inferiority tolerance;
- model-selection tie/no-decision rule;
- maximum false-accept bound and confidence procedure;
- pilot eligibility minimum and coarse activity/item-tail strata;
- primary cost: begin with completed-client local interactions, then report
  proxy bytes/rounds/compute separately as unvalidated system estimates;
- model, training budget, candidate catalog, and held-out evaluation protocol.

No numerical tolerance is selected in this assessment because MovieLens has no
product-risk definition for an acceptable recommendation loss. Picking one from
the future-test result would invalidate the safety claim.

### Falsification rule

Reject a tested policy if, across locked episodes and client-level uncertainty
intervals, it either:

- has no lower-cost fidelity advantage over the equal chronological cap;
- loses required client/item-tail coverage; or
- consumes at least as much primary cost while producing no better safety or
  fidelity outcome.

## Executed real-data result

`configs/experiments/fixed_cohort_budget_v1.json` and
`results/explorations/fixed_cohort_budget_v1/` implement the contract on the
pinned 33,832,162-rating MovieLens snapshot.

- The deterministic cohort had 2,000 users selected from 16,084 users with at
  least five positive pilot events and a positive event in the final 365 pilot
  days—criteria known before collection.
- Global positive-rating windows ended on 2020-08-23 for pilot and 2021-10-02
  for collection. The frozen reference used 270,880 training interactions,
  including 15,537 collection-window interactions.
- 520 cohort users had a future positive after fixed seen-item exclusion;
  1,480 had none and are reported rather than silently removed.
- The tail reserve increased selected tail share—for cap 1, 0.627 versus
  0.177—but was worse at cap 1 for both deterministic controls. No cap/model
  cell had a strictly positive paired 95% interval for lower absolute NDCG
  error, so the tested tail-reserve policy is rejected for this episode.

The equal cap at 5 retained 2,735 of 15,537 collection-window interactions
(82.4% fewer) with mean absolute NDCG errors of 0.0023 for popularity and
0.0012 for rating-weighted popularity. The configured 0.005 tolerance is
exploratory, so this is a fidelity observation, not a validated safe-stop
decision.

## Recommender-model control

The replay treats the recommendation model as an explicit control, because a
global popularity ranking is nearly identical for every user and cannot stand
in for a personalized recommender.

| Model | Type | Full-reference NDCG@10 | Same-data noise floor |
| --- | --- | ---: | --- |
| popularity | deterministic global | 0.0901 | none |
| rating-weighted popularity | deterministic global | 0.0912 | none |
| item-item cosine | deterministic personalized | **0.1155** | none |
| implicit ALS | stochastic personalized | 0.1020 | 0.0419 per-user; 0.0101 mean spread |

- **Item-item cosine is the primary probe.** It is the strongest model here and
  has no random state, so any difference between caps is attributable to the
  data policy. It scores 2,416 items with full-reference support >= 20 and a
  top-100 neighborhood, with the support set fixed across every policy.
- **ALS is demoted to a secondary probe.** On identical full-reference data,
  seed changes alone move mean NDCG@10 across 0.0965–0.1066 and produce a
  0.0419 mean per-user absolute difference. A separate stability probe found
  0.55 mean top-10 list agreement between two seeds at 32 factors and 0.31 at
  64 factors. Every ALS policy-versus-full per-user error (0.0157–0.0270) lies
  below that floor, so the ALS rows cannot separate a data effect from
  initialization noise. ALS metrics are therefore averaged over a common seed
  set and published with the floor.

## What the deterministic probe shows

With item-item cosine, the tested caps do not measurably change mean future
quality on this cohort:

| Cap | Collection rows kept | Item-item NDCG@10 | Paired 95% CI vs full | Per-user abs. error |
| ---: | ---: | ---: | --- | ---: |
| 1 | 643 of 15,537 | 0.1142 | [-0.0049, 0.0024] | 0.0156 |
| 5 | 2,735 | 0.1146 | [-0.0039, 0.0024] | 0.0135 |
| 20 | 7,656 | 0.1147 | [-0.0039, 0.0023] | 0.0122 |
| 50 | 11,762 | 0.1164 | [-0.0013, 0.0034] | 0.0071 |

- No cap's paired interval excludes zero, including a cap of one event per
  client, which keeps 4.1% of collection-window interactions.
- Per-user absolute error still falls monotonically with the cap
  (0.0156 to 0.0071), so individual recommendations keep changing even while the
  cohort mean is stable. Mean stability and per-user stability are different
  claims.
- The deterministic global controls are *more* mean-sensitive at small caps
  (popularity CIs exclude zero at caps 1, 2 and 10), because they aggregate
  every retained event into one ranking.
- The tail reserve is now significantly worse for the personalized probe at
  caps 5 and 50 (improvement -0.0016 [-0.0033, -0.0001] and -0.0009 [-0.0016,
  -0.0001]), strengthening its rejection.

**Design consequence.** The collection window supplies only 15,537 of 270,880
training interactions (5.7%), so this episode cannot resolve a large budget
effect: the pilot history dominates every model. The next experiment must vary
total per-client history, including pilot history, rather than only the
collection increment.

## Metric-ceiling audit

Top-`K` metrics do not share a ceiling, and the ceiling moves with the user
property this study manipulates. Measured on the 520 evaluated users:

| Quantity | Value |
| --- | ---: |
| Users with more than 10 future positives | 325 of 520 (62.5%) |
| Relevant-set size: mean / median / p90 / max | 29.0 / 16 / 69 / 562 |
| Mean `Recall@10` ceiling | 0.628 |
| Minimum `Recall@10` ceiling | 0.018 |

A user with 562 future positives cannot exceed `10/562 = 1.8%` recall no matter
how good the model is. Raw recall therefore understates retrieval quality by
more than a factor of two at the reference:

| Model | Raw `Recall@10` | Cap-aware `HitRate@10` | `NDCG@10` |
| --- | ---: | ---: | ---: |
| popularity | 0.0417 | 0.0883 | 0.0901 |
| rating-weighted popularity | 0.0415 | 0.0879 | 0.0912 |
| item-item cosine | 0.0477 | 0.1103 | 0.1155 |
| implicit ALS | 0.0415 | 0.0959 | 0.1020 |

This is a confound, not a display issue: heavy users have the lowest recall
ceilings and are also the clients a per-client cap truncates most, so a
raw-recall comparison across caps mixes a data effect with a ceiling effect.
`NDCG@10` and `HitRate@10` both normalize by `min(K, R)` and are unaffected.

The replay reports `NDCG@10` as primary, cap-aware `HitRate@10` as the retrieval
secondary, raw `Recall@10` only for continuity, and publishes the ceiling audit
in `reference_artifact.json`. MRR, recommendation-side catalog coverage, and
per-activity-group breakdowns are declared and not yet measured.

## Federation topology

Everything executed here is **horizontal** federated recommendation emulated
centrally: one user is one client, all clients share the schema
`(user, item, rating, timestamp)`, and the budget unit is local rows per client.

**Vertical** federated learning—same users, different feature blocks held by
different parties—is a different problem with different prerequisites: private
entity alignment, labels usually held by one party, and per-batch intermediate
representations instead of per-device updates. It is not tested, and no result
here transfers to it. A vertical variant is constructible from this snapshot
because `ratings.csv`, `tags.csv`, and `genome-scores.csv` share identifiers
while carrying different features; it is tracked separately in
`docs/research_direction.md`.

## Next action

Keep `equal_chronological_cap` as the control policy and item-item cosine as
the primary probe. Do not implement the rejected tail-reserve policy further.
Re-run the design with total per-client history as the budget axis, repeated
cohorts, and multiple time windows before stating any sufficiency rule; report
per-user error beside mean error. Federated-system confirmation remains
unavailable.
