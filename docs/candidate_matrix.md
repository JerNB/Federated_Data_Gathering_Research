# Candidate exploration matrix

Status: executed exploratory matrix on the pinned MovieLens snapshot. The full run is
`results/explorations/sample_generalization_full/`; its machine-readable summary is
`candidate_summary.json` and its concise report is `report.md`.

## Scope before selection

This matrix measures **sampling-design robustness** (sample-to-reference
fidelity): for a fixed model, split, and panel, does changing the training
sampling method or fraction change the reported recommendation result relative
to full training data? It estimates that effect across repeated draws. Sampling
method and model are experimental factors inside this claim; neither is itself
a type of generalizability.

For each sampled model, compare NDCG/Recall with the same model's full-data
result, then report draw uncertainty, panel support, and resource cost. `top
choice` and Kendall τ add a separate model-selection check when that is the
narrow decision. The full framework, including temporal, subgroup, external,
and federated claims, is in `docs/research_proposal.md`.

## Conditional result for the current model-ranking task

If the sole task is **choosing among the three executed models** from a 50%
sample, `uniform_interaction` is the better primary sampler: all 2,000 panel
users are supported, the top model is preserved in every draw, and full-order
agreement is τ `0.867`. Use `within_user_history` as the cross-check: it also
supports all users and has slightly lower item-item error (`0.0032` versus
`0.0035`), but lower full-order agreement (τ `0.600`).

Within the three executed models, item-item cosine is first on the frozen
full-data reference (NDCG@10 `0.0616`) versus popularity (`0.0391`) and
rating-weighted popularity (`0.0390`). This is not a recommender-family
decision: BPR, ALS, neural, and federated models have not been evaluated under
the same protocol.

`uniform_user` and `activity_stratified_user` fail this narrow collaborative
model-ranking task at 50%: they support only about half the panel and have
item-item errors `0.0269` and `0.0276`.


## Decision rule

A candidate is **primary** when it changes the sampling estimand while keeping the
reference, split, catalog, metrics, and model controls fixed. It is a **convergence
check** when its fraction is above 10%. A candidate is **secondary/stress** when it
changes a scientifically relevant axis but is not part of the headline matrix. A
candidate is **deferred** when the repository does not yet have an executable,
resource-bounded implementation; it is documented but not presented as evidence.

Every primary cell in the executed matrix uses the same full-data reference, the
same panel and test targets, the same candidate catalog, the same chronological
split, the same positive rule, the same cutoff, and deterministic seeds. The only
headline change is how the training positives are sampled.

## Executed sampling candidates

| ID | Area | Role | What changes | What stays fixed | Full-data evidence |
| --- | --- | --- | --- | --- | --- |
| `uniform_user` | user-level | primary | Uniformly selects eligible users, then takes the prescribed within-user training sample. | Full reference, catalog, panel, test target, model, metrics, seeds, fractions, 10 draws. | 7 fractions × 10 draws (fraction 1.0 reused) × 3 models. |
| `activity_stratified_user` | user heterogeneity | primary | Samples users across training-activity strata so high- and low-activity users are represented. | Same controls as above. | 7 fractions × 10 draws (fraction 1.0 reused) × 3 models. |
| `uniform_interaction` | interaction-level | primary | Samples positive interactions uniformly, retaining their associated users. | Same controls as above; full-reference exclusion remains headline frame. | 7 fractions × 10 draws (fraction 1.0 reused) × 3 models. |
| `within_user_history` | longitudinal/user history | primary | Samples each eligible user's training history, preserving user participation while varying observed history. | Same controls as above. | 7 fractions × 10 draws (fraction 1.0 reused) × 3 models. |

Fractions are `0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.0`. The first four
form the near-independent ladder; the last three are convergence checks with a
finite-population correction. The full fraction is evaluated once per scheme and
must reproduce the reference exactly.

## Executed model candidates

| ID | Area | Role | Definition | Why included |
| --- | --- | --- | --- | --- |
| `popularity` | global control | primary control | Positive-training item count, deterministic item-ID tie break. | Strong low-variance baseline; separates sampling noise from model instability. |
| `rating_weighted_popularity` | global control | primary control | Smoothed positive rating mass divided by smoothed positive count. | Tests whether rating intensity changes the sampling response beyond counts. |
| `item_item_cosine` | collaborative | primary collaborative baseline | Fixed-support item-item cosine from training user-item incidence; query-profile scores exclude the user's full-reference positives. | Sampling-sensitive collaborative baseline without a learned factor model; exposes support and exclusion effects. |

The item-item support threshold is 20 full-reference positive interactions and
unsupported candidates receive zero score rather than being removed. The run
uses `item_item_top_k=100` and an exclusion-aware score limit of 1,253.

## Secondary and stress candidates

These are documented to keep exploration broad; they are not mixed into the
headline evidence until their controls are frozen and an executable run record
exists.

| Candidate | Area | Expected question | Current status / required control |
| --- | --- | --- | --- |
| `item_stratified` | item coverage | Does preserving long-tail item mass change sample-to-reference convergence? | Secondary sampler; define item strata from training only and report item-coverage shift. |
| `recent_window` | time/exposure | Does a recent interaction window trade temporal relevance for reference fidelity? | Stress sampler; pin window boundaries and compare against chronological test targets. |
| `cluster_stratified_user` | federated clients | Does client/community composition explain sampling error? | Requires a declared client/cluster map fit from training only. |
| `local_client_history` | federated/local sparsity | How much does per-client history preservation matter under local data limits? | Requires communication and local-update accounting. |
| `long_tail_boost` | coverage/fairness | Can sample design improve tail coverage without invalidating the estimand? | Requires exposure-aware diagnostics; do not fold into headline NDCG. |
| `cold_start_holdout` | robustness | Does the conclusion survive unseen-item or unseen-user stress? | Requires a separate catalog and a predeclared zero-support policy. |

## Deferred model candidates

| Candidate | Area | Reason deferred | Gate before execution |
| --- | --- | --- | --- |
| BPR-MF | learned collaborative | Pairwise training and negative-sampling budget are not yet matched to the fixed controls. | Implement deterministic negative sampling, fixed epochs/updates, capacity, and full resource logging. |
| implicit ALS | learned collaborative | Objective and capacity are not directly comparable to the current item-item baseline. | Freeze confidence weighting, iterations, factors, regularization, and catalog. |
| neural/two-tower | representation learning | Compute, tuning, and checkpoint provenance would add a new resource regime. | Same candidate catalog, training budget, seed, and evaluation API. |
| sequence transformer | temporal modeling | Requires a separate sequence task and much larger model budget. | Predeclare sequence length, causal split, training budget, and cold-start policy. |
| federated FedMF/FedProx/SCAFFOLD | federated optimization | Full-data centralized evidence must establish the sampling signal before communication effects are interpreted. | Pin clients, rounds, local steps, aggregation, privacy/communication budget, and capacity. |

Deferred does not mean rejected; it prevents an unpriced model change from being
mistaken for a sampling result.

## Constant-control contract

- Dataset: MovieLens `ml-latest`, snapshot `2023-07-20`, manifest and chunk hashes pinned.
- Raw size: 33,832,162 ratings, 330,975 users, 83,239 rating-bearing items.
- Positive rule: `rating >= 4.0`, applied after one chronological 80/10/10 partition per user.
- Training positives: 13,653,758; evaluation panel: 2,000 eligible users with positive test targets.
- Candidate catalog: all full-snapshot items; unsupported item-item candidates score zero.
- Metrics: NDCG@10, absolute/reference error, sample-native minus fixed-frame gap, coverage, Kendall rank agreement, top-choice agreement, sample rows, wall time, and peak RSS.
- Repetitions: 10 for fractions below 1.0; full fraction reused once per scheme and checked for zero error.
- Reference: built once and reused; cache identity includes the dataset manifest, panel size, seed, support threshold, cutoff, and item-item score limit.

## Observed resource envelope

The final full run reused the frozen reference/training cache. Per-cell
evaluation time was `0.10–33.68 s`; replicate-pass time was `22.2–55.5 s`;
peak resident memory was `3,303–4,195 MiB` across draw rows. The initial
uncached full-reference build was measured separately at about 96.8 s during
the real-data smoke/full preparation. These are local hardware measurements,
not cross-machine performance claims. The memory envelope is a release gate
for any deferred learned or federated model.

## Evidence boundary

This is an in-reference approximation study. It does not claim independent
external generalization: the target is fixed full-snapshot test behavior for the
same panel and catalog. Native/fixed divergence is a protocol-bias diagnostic,
not a second test set. A conclusion requires metric tolerance, stable ordering,
coverage checks, and agreement across more than one sampling scheme.
