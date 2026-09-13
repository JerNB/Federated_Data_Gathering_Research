# Fixed-cohort local-data-budget replay

Status: real MovieLens offline chronological replay; not a federated-system, privacy, or external-generalization result.

## Design

- Fixed deterministic cohort: 2,000 of 16,084 pilot-eligible users.
- Global positive-rating windows: pilot through 2020-08-23T23:59:59+00:00; collection through 2021-10-02T23:59:59+00:00; future test thereafter.
- Full reference training interactions: 270,880; evaluated users with future positives: 520.
- Pilot item-tail ceiling: <= 1 positive pilot interactions.
- Policies differ only in collection-window local-record retention; evaluation excludes the same full pre-test seen set for every policy.

## Policy evidence

| Policy | Cap | Model | Mean NDCG@10 | Mean abs. NDCG error | 95% CI of NDCG delta vs full | Mean Recall@10 | Collection rows | Tail share | Tail-vs-equal abs.-error improvement (95% CI) |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| equal_chronological_cap | 1 | popularity | 0.0883 | 0.0035 | [-0.0031, -0.0005] | 0.0405 | 643 | 0.177 | -0.0011 [-0.0019, -0.0005] |
| equal_chronological_cap | 1 | rating_weighted_popularity | 0.0885 | 0.0036 | [-0.0047, -0.0012] | 0.0405 | 643 | 0.177 | -0.0006 [-0.0013, -0.0001] |
| tail_reserve_cap | 1 | popularity | 0.0878 | 0.0047 | [-0.0037, -0.0008] | 0.0403 | 643 | 0.627 | -0.0011 [-0.0019, -0.0005] |
| tail_reserve_cap | 1 | rating_weighted_popularity | 0.0882 | 0.0042 | [-0.0056, -0.0012] | 0.0407 | 643 | 0.627 | -0.0006 [-0.0013, -0.0001] |
| equal_chronological_cap | 2 | popularity | 0.0889 | 0.0024 | [-0.0024, -0.0003] | 0.0410 | 1,223 | 0.163 | -0.0007 [-0.0013, -0.0002] |
| equal_chronological_cap | 2 | rating_weighted_popularity | 0.0889 | 0.0035 | [-0.0044, -0.0008] | 0.0410 | 1,223 | 0.163 | 0.0000 [-0.0005, 0.0005] |
| tail_reserve_cap | 2 | popularity | 0.0885 | 0.0031 | [-0.0029, -0.0006] | 0.0403 | 1,223 | 0.357 | -0.0007 [-0.0013, -0.0002] |
| tail_reserve_cap | 2 | rating_weighted_popularity | 0.0886 | 0.0035 | [-0.0046, -0.0010] | 0.0405 | 1,223 | 0.357 | 0.0000 [-0.0005, 0.0005] |
| equal_chronological_cap | 5 | popularity | 0.0894 | 0.0023 | [-0.0018, 0.0004] | 0.0409 | 2,735 | 0.150 | 0.0001 [-0.0001, 0.0004] |
| equal_chronological_cap | 5 | rating_weighted_popularity | 0.0906 | 0.0012 | [-0.0013, 0.0001] | 0.0411 | 2,735 | 0.150 | 0.0000 [-0.0001, 0.0001] |
| tail_reserve_cap | 5 | popularity | 0.0892 | 0.0021 | [-0.0020, 0.0002] | 0.0409 | 2,735 | 0.341 | 0.0001 [-0.0001, 0.0004] |
| tail_reserve_cap | 5 | rating_weighted_popularity | 0.0906 | 0.0012 | [-0.0013, 0.0001] | 0.0411 | 2,735 | 0.341 | 0.0000 [-0.0001, 0.0001] |
| equal_chronological_cap | 10 | popularity | 0.0901 | 0.0013 | [-0.0009, 0.0007] | 0.0415 | 4,763 | 0.163 | -0.0005 [-0.0010, -0.0002] |
| equal_chronological_cap | 10 | rating_weighted_popularity | 0.0907 | 0.0011 | [-0.0012, 0.0001] | 0.0413 | 4,763 | 0.163 | -0.0002 [-0.0005, -0.0001] |
| tail_reserve_cap | 10 | popularity | 0.0899 | 0.0018 | [-0.0012, 0.0007] | 0.0415 | 4,763 | 0.276 | -0.0005 [-0.0010, -0.0002] |
| tail_reserve_cap | 10 | rating_weighted_popularity | 0.0908 | 0.0013 | [-0.0012, 0.0002] | 0.0413 | 4,763 | 0.276 | -0.0002 [-0.0005, -0.0001] |
| equal_chronological_cap | 20 | popularity | 0.0897 | 0.0007 | [-0.0010, 0.0001] | 0.0413 | 7,656 | 0.172 | -0.0002 [-0.0006, 0.0003] |
| equal_chronological_cap | 20 | rating_weighted_popularity | 0.0910 | 0.0010 | [-0.0008, 0.0004] | 0.0413 | 7,656 | 0.172 | -0.0000 [-0.0001, 0.0000] |
| tail_reserve_cap | 20 | popularity | 0.0899 | 0.0008 | [-0.0009, 0.0003] | 0.0415 | 7,656 | 0.239 | -0.0002 [-0.0006, 0.0003] |
| tail_reserve_cap | 20 | rating_weighted_popularity | 0.0910 | 0.0011 | [-0.0009, 0.0004] | 0.0413 | 7,656 | 0.239 | -0.0000 [-0.0001, 0.0000] |
| equal_chronological_cap | 50 | popularity | 0.0899 | 0.0006 | [-0.0008, 0.0002] | 0.0416 | 11,762 | 0.179 | 0.0000 [0.0000, 0.0000] |
| equal_chronological_cap | 50 | rating_weighted_popularity | 0.0913 | 0.0004 | [-0.0003, 0.0005] | 0.0415 | 11,762 | 0.179 | -0.0001 [-0.0005, 0.0000] |
| tail_reserve_cap | 50 | popularity | 0.0899 | 0.0006 | [-0.0008, 0.0002] | 0.0416 | 11,762 | 0.200 | 0.0000 [0.0000, 0.0000] |
| tail_reserve_cap | 50 | rating_weighted_popularity | 0.0912 | 0.0005 | [-0.0005, 0.0005] | 0.0414 | 11,762 | 0.200 | -0.0001 [-0.0005, 0.0000] |

## Interpretation boundary

- A positive tail-vs-equal improvement means the tail-reserve policy had lower mean per-user absolute NDCG error than the equal chronological cap at the same cap.
- No tested tail-reserve cell has a strictly positive paired 95% interval for lower absolute NDCG error than the equal chronological cap.
- The configured 0.005 mean-absolute-NDCG tolerance is exploratory, not a product-risk threshold or a validated stop-controller rule.
- The replay validates policy-specific sample-to-reference behavior on this cohort and time partition only. It does not validate client availability, dropout, communication, secure aggregation, local compute, consent, or federated convergence.
