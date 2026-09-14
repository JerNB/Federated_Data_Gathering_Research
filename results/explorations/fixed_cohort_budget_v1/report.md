# Fixed-cohort local-data-budget replay

Status: real MovieLens offline chronological replay; not a federated-system, privacy, or external-generalization result.

## Design

- Fixed deterministic cohort: 2,000 of 16,084 pilot-eligible users.
- Global positive-rating windows: pilot through 2020-08-23T23:59:59+00:00; collection through 2021-10-02T23:59:59+00:00; future test thereafter.
- Full reference training interactions: 270,880; evaluated users with future positives: 520.
- Pilot item-tail ceiling: <= 1 positive pilot interactions.
- Models: popularity, rating_weighted_popularity, item_item_cosine, implicit_als. Item-item cosine is the deterministic primary personalized probe over 2,416 items with support >= 20; implicit ALS is the stochastic secondary probe.
- Full-reference NDCG@10: popularity 0.0901; rating_weighted_popularity 0.0912; item_item_cosine 0.1155; implicit_als 0.1020.
- ALS control: per-user metrics are averaged over seeds [20261003, 20261004, 20261005]; the same-data seed floor is 0.0419 mean per-user absolute NDCG difference with a 0.0101 mean-NDCG spread.
- Metric ceilings: 325 of 520 evaluated users (62.5%) have more than 10 future positives, so their Recall@10 is structurally capped; the mean recall ceiling is 0.628 (minimum 0.018). NDCG@10 and the cap-aware HitRate@10 divide by min(K, |relevant|) instead.
- Policies differ only in collection-window local-record retention; evaluation excludes the same full pre-test seen set for every policy.

## Policy evidence

| Policy | Cap | Model | Mean NDCG@10 | Mean abs. NDCG error | 95% CI of NDCG delta vs full | Mean Recall@10 | Mean HitRate@10 | Collection rows | Tail share | Tail-vs-equal abs.-error improvement (95% CI) |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| equal_chronological_cap | 1 | popularity | 0.0883 | 0.0035 | [-0.0031, -0.0005] | 0.0405 | 0.0859 | 643 | 0.177 | -0.0011 [-0.0019, -0.0005] |
| equal_chronological_cap | 1 | rating_weighted_popularity | 0.0885 | 0.0036 | [-0.0047, -0.0012] | 0.0405 | 0.0859 | 643 | 0.177 | -0.0006 [-0.0013, -0.0001] |
| equal_chronological_cap | 1 | item_item_cosine | 0.1142 | 0.0156 | [-0.0049, 0.0024] | 0.0465 | 0.1097 | 643 | 0.177 | 0.0002 [-0.0009, 0.0016] |
| equal_chronological_cap | 1 | implicit_als | 0.1038 | 0.0267 | [-0.0025, 0.0068] | 0.0435 | 0.1025 | 643 | 0.177 | -0.0003 [-0.0018, 0.0010] |
| tail_reserve_cap | 1 | popularity | 0.0878 | 0.0047 | [-0.0037, -0.0008] | 0.0403 | 0.0857 | 643 | 0.627 | -0.0011 [-0.0019, -0.0005] |
| tail_reserve_cap | 1 | rating_weighted_popularity | 0.0882 | 0.0042 | [-0.0056, -0.0012] | 0.0407 | 0.0859 | 643 | 0.627 | -0.0006 [-0.0013, -0.0001] |
| tail_reserve_cap | 1 | item_item_cosine | 0.1132 | 0.0154 | [-0.0058, 0.0011] | 0.0462 | 0.1087 | 643 | 0.627 | 0.0002 [-0.0009, 0.0016] |
| tail_reserve_cap | 1 | implicit_als | 0.1029 | 0.0270 | [-0.0035, 0.0059] | 0.0438 | 0.1019 | 643 | 0.627 | -0.0003 [-0.0018, 0.0010] |
| equal_chronological_cap | 2 | popularity | 0.0889 | 0.0024 | [-0.0024, -0.0003] | 0.0410 | 0.0868 | 1,223 | 0.163 | -0.0007 [-0.0013, -0.0002] |
| equal_chronological_cap | 2 | rating_weighted_popularity | 0.0889 | 0.0035 | [-0.0044, -0.0008] | 0.0410 | 0.0865 | 1,223 | 0.163 | 0.0000 [-0.0005, 0.0005] |
| equal_chronological_cap | 2 | item_item_cosine | 0.1143 | 0.0159 | [-0.0047, 0.0023] | 0.0464 | 0.1093 | 1,223 | 0.163 | 0.0005 [-0.0004, 0.0013] |
| equal_chronological_cap | 2 | implicit_als | 0.1048 | 0.0264 | [-0.0015, 0.0074] | 0.0438 | 0.1028 | 1,223 | 0.163 | 0.0003 [-0.0009, 0.0016] |
| tail_reserve_cap | 2 | popularity | 0.0885 | 0.0031 | [-0.0029, -0.0006] | 0.0403 | 0.0861 | 1,223 | 0.357 | -0.0007 [-0.0013, -0.0002] |
| tail_reserve_cap | 2 | rating_weighted_popularity | 0.0886 | 0.0035 | [-0.0046, -0.0010] | 0.0405 | 0.0861 | 1,223 | 0.357 | 0.0000 [-0.0005, 0.0005] |
| tail_reserve_cap | 2 | item_item_cosine | 0.1140 | 0.0154 | [-0.0050, 0.0019] | 0.0462 | 0.1087 | 1,223 | 0.357 | 0.0005 [-0.0004, 0.0013] |
| tail_reserve_cap | 2 | implicit_als | 0.1029 | 0.0262 | [-0.0033, 0.0056] | 0.0430 | 0.1010 | 1,223 | 0.357 | 0.0003 [-0.0009, 0.0016] |
| equal_chronological_cap | 5 | popularity | 0.0894 | 0.0023 | [-0.0018, 0.0004] | 0.0409 | 0.0871 | 2,735 | 0.150 | 0.0001 [-0.0001, 0.0004] |
| equal_chronological_cap | 5 | rating_weighted_popularity | 0.0906 | 0.0012 | [-0.0013, 0.0001] | 0.0411 | 0.0871 | 2,735 | 0.150 | 0.0000 [-0.0001, 0.0001] |
| equal_chronological_cap | 5 | item_item_cosine | 0.1146 | 0.0135 | [-0.0039, 0.0024] | 0.0467 | 0.1092 | 2,735 | 0.150 | -0.0016 [-0.0033, -0.0001] |
| equal_chronological_cap | 5 | implicit_als | 0.1031 | 0.0250 | [-0.0027, 0.0052] | 0.0441 | 0.1015 | 2,735 | 0.150 | 0.0002 [-0.0013, 0.0017] |
| tail_reserve_cap | 5 | popularity | 0.0892 | 0.0021 | [-0.0020, 0.0002] | 0.0409 | 0.0869 | 2,735 | 0.341 | 0.0001 [-0.0001, 0.0004] |
| tail_reserve_cap | 5 | rating_weighted_popularity | 0.0906 | 0.0012 | [-0.0013, 0.0001] | 0.0411 | 0.0871 | 2,735 | 0.341 | 0.0000 [-0.0001, 0.0001] |
| tail_reserve_cap | 5 | item_item_cosine | 0.1140 | 0.0152 | [-0.0048, 0.0022] | 0.0452 | 0.1082 | 2,735 | 0.341 | -0.0016 [-0.0033, -0.0001] |
| tail_reserve_cap | 5 | implicit_als | 0.1037 | 0.0248 | [-0.0021, 0.0061] | 0.0441 | 0.1030 | 2,735 | 0.341 | 0.0002 [-0.0013, 0.0017] |
| equal_chronological_cap | 10 | popularity | 0.0901 | 0.0013 | [-0.0009, 0.0007] | 0.0415 | 0.0879 | 4,763 | 0.163 | -0.0005 [-0.0010, -0.0002] |
| equal_chronological_cap | 10 | rating_weighted_popularity | 0.0907 | 0.0011 | [-0.0012, 0.0001] | 0.0413 | 0.0873 | 4,763 | 0.163 | -0.0002 [-0.0005, -0.0001] |
| equal_chronological_cap | 10 | item_item_cosine | 0.1140 | 0.0136 | [-0.0046, 0.0019] | 0.0468 | 0.1094 | 4,763 | 0.163 | 0.0001 [-0.0013, 0.0016] |
| equal_chronological_cap | 10 | implicit_als | 0.1035 | 0.0234 | [-0.0027, 0.0056] | 0.0434 | 0.1019 | 4,763 | 0.163 | 0.0003 [-0.0011, 0.0018] |
| tail_reserve_cap | 10 | popularity | 0.0899 | 0.0018 | [-0.0012, 0.0007] | 0.0415 | 0.0877 | 4,763 | 0.276 | -0.0005 [-0.0010, -0.0002] |
| tail_reserve_cap | 10 | rating_weighted_popularity | 0.0908 | 0.0013 | [-0.0012, 0.0002] | 0.0413 | 0.0873 | 4,763 | 0.276 | -0.0002 [-0.0005, -0.0001] |
| tail_reserve_cap | 10 | item_item_cosine | 0.1139 | 0.0135 | [-0.0048, 0.0018] | 0.0466 | 0.1087 | 4,763 | 0.276 | 0.0001 [-0.0013, 0.0016] |
| tail_reserve_cap | 10 | implicit_als | 0.1034 | 0.0231 | [-0.0028, 0.0054] | 0.0434 | 0.1015 | 4,763 | 0.276 | 0.0003 [-0.0011, 0.0018] |
| equal_chronological_cap | 20 | popularity | 0.0897 | 0.0007 | [-0.0010, 0.0001] | 0.0413 | 0.0877 | 7,656 | 0.172 | -0.0002 [-0.0006, 0.0003] |
| equal_chronological_cap | 20 | rating_weighted_popularity | 0.0910 | 0.0010 | [-0.0008, 0.0004] | 0.0413 | 0.0875 | 7,656 | 0.172 | -0.0000 [-0.0001, 0.0000] |
| equal_chronological_cap | 20 | item_item_cosine | 0.1147 | 0.0122 | [-0.0039, 0.0023] | 0.0463 | 0.1085 | 7,656 | 0.172 | -0.0004 [-0.0014, 0.0006] |
| equal_chronological_cap | 20 | implicit_als | 0.1041 | 0.0207 | [-0.0013, 0.0059] | 0.0438 | 0.1020 | 7,656 | 0.172 | 0.0004 [-0.0012, 0.0020] |
| tail_reserve_cap | 20 | popularity | 0.0899 | 0.0008 | [-0.0009, 0.0003] | 0.0415 | 0.0879 | 7,656 | 0.239 | -0.0002 [-0.0006, 0.0003] |
| tail_reserve_cap | 20 | rating_weighted_popularity | 0.0910 | 0.0011 | [-0.0009, 0.0004] | 0.0413 | 0.0875 | 7,656 | 0.239 | -0.0000 [-0.0001, 0.0000] |
| tail_reserve_cap | 20 | item_item_cosine | 0.1137 | 0.0126 | [-0.0049, 0.0015] | 0.0458 | 0.1076 | 7,656 | 0.239 | -0.0004 [-0.0014, 0.0006] |
| tail_reserve_cap | 20 | implicit_als | 0.1046 | 0.0203 | [-0.0005, 0.0060] | 0.0444 | 0.1025 | 7,656 | 0.239 | 0.0004 [-0.0012, 0.0020] |
| equal_chronological_cap | 50 | popularity | 0.0899 | 0.0006 | [-0.0008, 0.0002] | 0.0416 | 0.0879 | 11,762 | 0.179 | 0.0000 [0.0000, 0.0000] |
| equal_chronological_cap | 50 | rating_weighted_popularity | 0.0913 | 0.0004 | [-0.0003, 0.0005] | 0.0415 | 0.0881 | 11,762 | 0.179 | -0.0001 [-0.0005, 0.0000] |
| equal_chronological_cap | 50 | item_item_cosine | 0.1164 | 0.0071 | [-0.0013, 0.0034] | 0.0458 | 0.1090 | 11,762 | 0.179 | -0.0009 [-0.0016, -0.0001] |
| equal_chronological_cap | 50 | implicit_als | 0.1049 | 0.0157 | [-0.0003, 0.0059] | 0.0442 | 0.1023 | 11,762 | 0.179 | -0.0004 [-0.0016, 0.0007] |
| tail_reserve_cap | 50 | popularity | 0.0899 | 0.0006 | [-0.0008, 0.0002] | 0.0416 | 0.0879 | 11,762 | 0.200 | 0.0000 [0.0000, 0.0000] |
| tail_reserve_cap | 50 | rating_weighted_popularity | 0.0912 | 0.0005 | [-0.0005, 0.0005] | 0.0414 | 0.0879 | 11,762 | 0.200 | -0.0001 [-0.0005, 0.0000] |
| tail_reserve_cap | 50 | item_item_cosine | 0.1163 | 0.0079 | [-0.0016, 0.0033] | 0.0459 | 0.1094 | 11,762 | 0.200 | -0.0009 [-0.0016, -0.0001] |
| tail_reserve_cap | 50 | implicit_als | 0.1047 | 0.0161 | [-0.0003, 0.0057] | 0.0438 | 0.1017 | 11,762 | 0.200 | -0.0004 [-0.0016, 0.0007] |

## Interpretation boundary

- A positive tail-vs-equal improvement means the tail-reserve policy had lower mean per-user absolute NDCG error than the equal chronological cap at the same cap.
- No tested tail-reserve cell has a strictly positive paired 95% interval for lower absolute NDCG error than the equal chronological cap.
- Item-item cosine has zero model-noise floor, yet no cell's paired 95% NDCG interval against full history excludes zero: the tested caps do not measurably change this personalized model.
- No ALS cell exceeds the same-data seed-variance floor, so per-user ALS differences here are not separable from optimizer initialization noise.
- Mean NDCG stability and per-user stability are different claims; report both.
- The configured 0.005 mean-absolute-NDCG tolerance is exploratory, not a product-risk threshold or a validated stop-controller rule.
- The replay validates policy-specific sample-to-reference behavior on this cohort and time partition only. It does not validate client availability, dropout, communication, secure aggregation, local compute, consent, or federated convergence.
