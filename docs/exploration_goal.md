# Federated local-data direction assessment goal

Status: **completed feasibility screen and one real-data replay with a
four-model control.** The tested item-tail reserve policy was rejected; the
deterministic item-item cosine probe and the equal-history-cap control are
retained. Results are in `docs/direction_assessment.md`. Neither the snapshot
matrix nor this replay is federated-system evidence.

## Decision

Select **at most one** primary direction for implementation under this question:

```text
For a fixed eligible cohort, can a privacy-compatible policy decide how much
existing permitted local history to retain or use, and when to stop, while
preserving a declared recommender decision at lower operational cost than using
all permitted history?
```

A direction may be rejected or deferred. “No direction is ready” is a valid
outcome.

“Gathering” here means retaining or scheduling existing local events. It does
not include inducing new user behavior, changing consent, or uploading raw
histories. Those interventions need a prospective study.

## Candidate directions

| Rank to assess | Direction | Operational action | Why it is distinct | Initial disposition |
| --- | --- | --- | --- | --- |
| 1 | **Coverage-constrained budget controller** | Broadcast chronological per-client caps and aggregate coverage quotas; extend budget only when the safety rule fails. | Targets the stated sufficiency and stop-or-gather decision without individual raw-data inspection. | **Recommended primary screen** |
| 2 | **Local relevance selection** | Each client filters or weights training events using a local relevance selector, as in FLRD. | Targets which records are useful, not only how many; may need server validation feedback. | Comparator; must pass a privacy/feedback feasibility gate. |
| 3 | **Online retention selection** | Each device retains streaming records under a storage cap, as in ODE. | Directly targets limited local storage; may require per-event gradients and cross-client metadata. | Comparator; must pass compute, storage, and metadata gates. |
| 4 | **Participant selection** | Select clients/round participants for utility, speed, or item-tail coverage. | Changes the contributing cohort and aggregation path rather than local-history sufficiency. | Adjacent experiment; never pool its result with directions 1–3. |
| 5 | **User-governed contribution/privacy policy** | Users choose whether or how much data to share under a different privacy model. | Changes consent and may upload data to a server. Historical MovieLens replay cannot identify willingness or behavioral effects. | Boundary case; defer without a prospective privacy/consent study. |

## Assessment protocol

Each direction receives one evidence card before implementation:

1. **Action and estimand** — what the policy changes, the fixed cohort, the
   all-permitted-history reference, and the future held-out outcome.
2. **Observable inputs** — exact client-side, aggregated, or public diagnostics.
   Mark every per-client score, gradient, label count, and validation signal that
   would be exposed or transmitted.
3. **Privacy and system contract** — raw-data rule, secure-aggregation or other
   protection, disclosure threshold, client availability, dropout, device
   compute, storage, bytes, and rounds.
4. **Replay validity** — confirm that the policy can be replayed chronologically
   from available logs without future information or behavior-changing actions.
5. **Safety and utility rule** — predeclare metric/decision tolerance, required
   client and subgroup coverage, maximum false-accept rate, and primary cost.
6. **Fatal assumption** — one assumption whose failure invalidates the result,
   with a measurement or an explicit limitation.

Reject a direction at the evidence-card stage if it needs raw server inspection,
a hidden target-distribution validation set, unavailable future data, an
unmodeled consent intervention, or a cost/benefit measure that MovieLens cannot
replay.

## Empirical promotion rule

For directions that pass the evidence card:

1. Construct fixed-cohort, chronological historical episodes.
2. Compare the direction with equal-history-cap, coverage-quota, and
   full-permitted-history references at matched cost.
3. Evaluate future NDCG@10, Recall@10, declared model decision, false acceptance,
   abstention, completed-client coverage, worst-stratum outcome, bytes, rounds,
   local compute, and wall-clock time.
4. Bootstrap clients and repeat over time windows, cohort draws, and policy
   randomness. Never treat interactions from one client as independent samples.
5. Promote only a direction that improves the predeclared safety–cost frontier.
   Confirm that survivor in a federated simulator before making a federated
   claim.

## Deliverables

- A literature matrix linking each candidate direction to a primary source and
  identifying its non-transferable assumptions.
- One evidence card per direction.
- A ranked decision report: promote, comparator, defer, or reject.
- An experiment contract only for the promoted direction; do not implement all
  candidate methods merely because they are documented.
- Machine-readable results, a short assumptions/failure report, and dashboard
  status once an experiment is actually executed.

## Non-goals

This assessment does not claim a universal sampling fraction, prove privacy from
data locality, optimize a federated optimizer, compare arbitrary recommender
families, or establish external-domain transfer from MovieLens alone.
