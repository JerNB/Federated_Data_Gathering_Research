# Evaluation protocol (draft v2 — proposed, not executed)

Status: **design document awaiting approval.** The executed results still use
protocol v1 (`cutoff = 10`, primary `NDCG@10`). Nothing here has been run.

## 1. Why v1 is questionable

| v1 choice | Problem | Source |
| --- | --- | --- |
| `cutoff = 10` only | Shallow cutoffs are the least robust to sparsity and popularity bias and have the weakest discriminative power. The choice was never justified. | S30 |
| `Recall@10` as retrieval secondary | Structurally capped at `K/R`; 62.5% of our evaluated users exceed `K`. Recall also has weak discriminative power (DP 7.0 vs nDCG 1.4 on MovieLens 1M). | S30, measured locally |
| Ad-hoc `HitRate@10 = hits / min(K, R)` | Not a standard metric. For `R >= K` it is exactly `Precision@K`, so it silently renames precision and invites misreading. | — |
| Relevant set = "later rated >= 4" | An observation artifact under missing-not-at-random feedback, not the user's true interest set. Any metric dividing by `R` inherits that bias. | S32 |
| No popularity-bias treatment | Averaged accuracy can diverge from unbiased accuracy; popularity may be signal or artifact. | S33, S34 |

What v1 got right, and must not change:

- **AllItems ranking over the full rating-bearing catalog, with no negative
  sampling.** Sampled metrics are inconsistent with exact metrics and can
  reverse system order (S31). This is also the protocol used in S30.
- **`NDCG@K` as the primary outcome.** It is the most discriminative metric
  studied in S30 and is normalized by `min(K, R)`, so it is not capped.
- **A fixed per-user exclusion set**, identical across policies.

## 2. Proposed protocol v2

### Cutoffs

Report every metric at `K` in `{10, 20, 50, 100}`.

- `K = 100` is the **primary evaluation depth**, following S30: deeper cutoffs
  are more robust and more discriminative, and system order rarely changes
  across cutoffs (Kendall tau mostly above 0.9).
- `K = 10` is retained as the **product-display depth**, reported for continuity
  and for readers who care about a ten-slot surface.
- Evaluation depth and display depth are different decisions. Using `K = 100`
  offline does not claim that a deployment shows 100 items.

### Metrics

| Role | Metric | Justification |
| --- | --- | --- |
| Primary | `NDCG@K` | Highest discriminative power; normalized by `min(K, R)`; not capped. |
| Robust secondary | `Precision@K` | Most robust to sparsity and popularity bias in S30. |
| Coverage secondary | `Recall@K` | Reported for interpretability only; never a decision metric; always accompanied by its `K/R` ceiling. |
| Diagnostic | Relevant-set-size distribution and ceiling audit | Makes the `Recall`/`Precision` ceilings explicit per cutoff. |
| Dropped | `HitRate = hits / min(K, R)` | Redundant with `Precision@K` when `R >= K`; keep the standard name instead. |
| Dropped | `MRR` | Least robust metric in S30 and weakly discriminative. |

### Decision rule

A data-budget difference counts only when it holds on `NDCG@K` at the primary
depth with a paired confidence interval excluding zero, and is not contradicted
at the other cutoffs. Cutoff disagreement is itself a reportable result.

### Stratified reporting

Averaged accuracy hides where a policy fails (S34). Report the primary metric
additionally for:

- item-popularity strata (head / mid / tail candidates);
- user-activity strata (light / medium / heavy local history);
- users whose `R` is below and above the cutoff.

### Optional debiasing layer (later phase)

Inverse-propensity or self-normalized-propensity weighting with a declared
popularity-based exposure model (S33). Adopt only with the propensity model
written into the experiment contract, and always report the unweighted result
beside it.

## 3. What this changes for the current claim

The executed conclusion — that per-client caps show no significant mean-quality
penalty under the deterministic item-item probe — was measured at `K = 10`,
which S30 identifies as the **least discriminative** depth. A null result at a
shallow cutoff is therefore weak evidence of equivalence: part of the
"no difference" may be low discriminative power rather than true equivalence.

Protocol v2 is expected to change confidence, not direction. It must be run
before any sufficiency statement is made.

## 4. Implementation cost

- `score_limit` must cover `100 + |seen|` per user rather than `10 + |seen|`.
- The item-item neighborhood size must be at least the primary cutoff.
- Metric accumulation can snapshot all cutoffs in a single pass over each
  ranking, so the added cost is roughly linear in the deepest cutoff.
- Expected runtime increase for the fixed-cohort replay: moderate, dominated by
  the unchanged ALS training cost.

## 5. Open questions to settle before running

1. Primary depth: `K = 100` as S30 recommends, or `K = 50` as a compromise given
   that our evaluated cohort is 520 users rather than thousands?
2. Do we add popularity/activity stratification now, or after the cutoff
   sensitivity result?
3. Do we adopt propensity weighting at all for this study, given that it
   introduces an exposure model we cannot validate on MovieLens?
