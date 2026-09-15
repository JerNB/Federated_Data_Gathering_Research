# Fixed-cohort local-data-budget replay (protocol v2)

Status: real MovieLens offline chronological replay; not a federated-system, privacy, or external-generalization result.

## Design

- Fixed deterministic cohort: 16,084 of 16,084 pilot-eligible users; 4,100 have future positives.
- Global positive-rating windows: pilot through 2020-08-23T23:59:59+00:00; collection through 2021-10-02T23:59:59+00:00; future test thereafter.
- Full reference training interactions: 2,205,099, of which 117,092 are collection-window events the policies control.
- Cutoffs [10, 20, 50, 100] with primary depth 100; the display depth is 10.
- Models: item-item cosine is the deterministic primary probe over 7,912 items with support >= 20; implicit ALS is the stochastic secondary probe.
- Full-reference NDCG@100: popularity 0.1095; rating_weighted_popularity 0.1097; item_item_cosine 0.1228; implicit_als 0.1459.
- ALS control: per-user metrics are averaged over seeds [20261003, 20261004, 20261005]; the same-data seed floor is 0.0203 mean per-user absolute NDCG difference with a 0.0005 mean-NDCG spread.
- Propensity weighting: item popularity power law, p_i proportional to n_i^((gamma+1)/2), normalized to a maximum of 1, gamma 0.5, floor 0.01, maximum inverse weight 100.0. The exposure model is an assumption, so weighted values never replace unweighted ones.
- Metric ceilings: 166 of 4,100 users (4.0%) have more than 100 future positives; mean recall ceiling 0.987, minimum 0.096.
- Policies differ only in collection-window local-record retention; evaluation excludes the same full pre-test seen set for every policy.

## Policy evidence at the primary cutoff

| Policy | Cap | Model | NDCG@100 | Weighted NDCG@100 | Abs. NDCG error | 95% CI of NDCG delta | Precision@100 | Recall@100 | Collection rows | Tail share | Tail-vs-equal improvement (95% CI) |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| equal_chronological_cap | 1 | popularity | 0.1079 | 0.0171 | 0.0027 | [-0.0021, -0.0013] | 0.0348 | 0.1672 | 5,089 | 0.116 | -0.0001 [-0.0001, -0.0000] |
| equal_chronological_cap | 1 | rating_weighted_popularity | 0.1083 | 0.0172 | 0.0025 | [-0.0017, -0.0011] | 0.0348 | 0.1675 | 5,089 | 0.116 | -0.0001 [-0.0001, -0.0000] |
| equal_chronological_cap | 1 | item_item_cosine | 0.1170 | 0.0207 | 0.0106 | [-0.0065, -0.0051] | 0.0384 | 0.1719 | 5,089 | 0.116 | -0.0001 [-0.0003, -0.0000] |
| equal_chronological_cap | 1 | implicit_als | 0.1390 | 0.0280 | 0.0162 | [-0.0079, -0.0061] | 0.0469 | 0.2160 | 5,089 | 0.116 | -0.0001 [-0.0003, 0.0001] |
| tail_reserve_cap | 1 | popularity | 0.1079 | 0.0171 | 0.0028 | [-0.0021, -0.0013] | 0.0348 | 0.1671 | 5,089 | 0.562 | -0.0001 [-0.0001, -0.0000] |
| tail_reserve_cap | 1 | rating_weighted_popularity | 0.1083 | 0.0172 | 0.0025 | [-0.0018, -0.0011] | 0.0348 | 0.1674 | 5,089 | 0.562 | -0.0001 [-0.0001, -0.0000] |
| tail_reserve_cap | 1 | item_item_cosine | 0.1167 | 0.0207 | 0.0107 | [-0.0068, -0.0053] | 0.0383 | 0.1716 | 5,089 | 0.562 | -0.0001 [-0.0003, -0.0000] |
| tail_reserve_cap | 1 | implicit_als | 0.1388 | 0.0279 | 0.0163 | [-0.0081, -0.0063] | 0.0469 | 0.2157 | 5,089 | 0.562 | -0.0001 [-0.0003, 0.0001] |
| equal_chronological_cap | 2 | popularity | 0.1083 | 0.0171 | 0.0021 | [-0.0016, -0.0010] | 0.0348 | 0.1671 | 9,715 | 0.111 | -0.0001 [-0.0002, -0.0001] |
| equal_chronological_cap | 2 | rating_weighted_popularity | 0.1085 | 0.0172 | 0.0023 | [-0.0016, -0.0010] | 0.0348 | 0.1678 | 9,715 | 0.111 | -0.0001 [-0.0002, -0.0000] |
| equal_chronological_cap | 2 | item_item_cosine | 0.1176 | 0.0208 | 0.0102 | [-0.0060, -0.0045] | 0.0385 | 0.1732 | 9,715 | 0.111 | -0.0002 [-0.0003, 0.0000] |
| equal_chronological_cap | 2 | implicit_als | 0.1394 | 0.0281 | 0.0156 | [-0.0074, -0.0057] | 0.0470 | 0.2168 | 9,715 | 0.111 | -0.0000 [-0.0002, 0.0001] |
| tail_reserve_cap | 2 | popularity | 0.1082 | 0.0171 | 0.0022 | [-0.0017, -0.0010] | 0.0348 | 0.1672 | 9,715 | 0.307 | -0.0001 [-0.0002, -0.0001] |
| tail_reserve_cap | 2 | rating_weighted_popularity | 0.1084 | 0.0172 | 0.0024 | [-0.0016, -0.0010] | 0.0348 | 0.1676 | 9,715 | 0.307 | -0.0001 [-0.0002, -0.0000] |
| tail_reserve_cap | 2 | item_item_cosine | 0.1174 | 0.0209 | 0.0103 | [-0.0061, -0.0046] | 0.0385 | 0.1729 | 9,715 | 0.307 | -0.0002 [-0.0003, 0.0000] |
| tail_reserve_cap | 2 | implicit_als | 0.1393 | 0.0280 | 0.0156 | [-0.0076, -0.0058] | 0.0470 | 0.2167 | 9,715 | 0.307 | -0.0000 [-0.0002, 0.0001] |
| equal_chronological_cap | 5 | popularity | 0.1086 | 0.0172 | 0.0017 | [-0.0012, -0.0007] | 0.0348 | 0.1676 | 21,734 | 0.110 | -0.0001 [-0.0001, -0.0000] |
| equal_chronological_cap | 5 | rating_weighted_popularity | 0.1090 | 0.0174 | 0.0014 | [-0.0009, -0.0005] | 0.0349 | 0.1680 | 21,734 | 0.110 | -0.0001 [-0.0002, -0.0001] |
| equal_chronological_cap | 5 | item_item_cosine | 0.1183 | 0.0210 | 0.0089 | [-0.0052, -0.0039] | 0.0388 | 0.1752 | 21,734 | 0.110 | -0.0002 [-0.0005, -0.0000] |
| equal_chronological_cap | 5 | implicit_als | 0.1408 | 0.0284 | 0.0141 | [-0.0060, -0.0043] | 0.0474 | 0.2190 | 21,734 | 0.110 | 0.0002 [-0.0001, 0.0004] |
| tail_reserve_cap | 5 | popularity | 0.1086 | 0.0172 | 0.0017 | [-0.0013, -0.0008] | 0.0348 | 0.1675 | 21,734 | 0.286 | -0.0001 [-0.0001, -0.0000] |
| tail_reserve_cap | 5 | rating_weighted_popularity | 0.1090 | 0.0174 | 0.0015 | [-0.0009, -0.0006] | 0.0348 | 0.1679 | 21,734 | 0.286 | -0.0001 [-0.0002, -0.0001] |
| tail_reserve_cap | 5 | item_item_cosine | 0.1179 | 0.0209 | 0.0092 | [-0.0055, -0.0042] | 0.0386 | 0.1742 | 21,734 | 0.286 | -0.0002 [-0.0005, -0.0000] |
| tail_reserve_cap | 5 | implicit_als | 0.1409 | 0.0284 | 0.0139 | [-0.0059, -0.0042] | 0.0475 | 0.2192 | 21,734 | 0.286 | 0.0002 [-0.0001, 0.0004] |
| equal_chronological_cap | 10 | popularity | 0.1089 | 0.0173 | 0.0012 | [-0.0007, -0.0005] | 0.0348 | 0.1678 | 37,346 | 0.113 | -0.0001 [-0.0001, -0.0000] |
| equal_chronological_cap | 10 | rating_weighted_popularity | 0.1092 | 0.0174 | 0.0011 | [-0.0007, -0.0004] | 0.0349 | 0.1684 | 37,346 | 0.113 | -0.0000 [-0.0001, 0.0000] |
| equal_chronological_cap | 10 | item_item_cosine | 0.1192 | 0.0212 | 0.0075 | [-0.0042, -0.0030] | 0.0390 | 0.1771 | 37,346 | 0.113 | -0.0002 [-0.0004, -0.0000] |
| equal_chronological_cap | 10 | implicit_als | 0.1422 | 0.0288 | 0.0125 | [-0.0046, -0.0030] | 0.0478 | 0.2211 | 37,346 | 0.113 | 0.0003 [0.0000, 0.0005] |
| tail_reserve_cap | 10 | popularity | 0.1089 | 0.0173 | 0.0012 | [-0.0008, -0.0005] | 0.0348 | 0.1677 | 37,346 | 0.221 | -0.0001 [-0.0001, -0.0000] |
| tail_reserve_cap | 10 | rating_weighted_popularity | 0.1091 | 0.0174 | 0.0011 | [-0.0007, -0.0004] | 0.0349 | 0.1683 | 37,346 | 0.221 | -0.0000 [-0.0001, 0.0000] |
| tail_reserve_cap | 10 | item_item_cosine | 0.1190 | 0.0212 | 0.0077 | [-0.0044, -0.0032] | 0.0390 | 0.1766 | 37,346 | 0.221 | -0.0002 [-0.0004, -0.0000] |
| tail_reserve_cap | 10 | implicit_als | 0.1423 | 0.0289 | 0.0122 | [-0.0044, -0.0029] | 0.0478 | 0.2215 | 37,346 | 0.221 | 0.0003 [0.0000, 0.0005] |
| equal_chronological_cap | 20 | popularity | 0.1092 | 0.0173 | 0.0008 | [-0.0005, -0.0003] | 0.0349 | 0.1680 | 58,752 | 0.117 | 0.0000 [-0.0000, 0.0001] |
| equal_chronological_cap | 20 | rating_weighted_popularity | 0.1095 | 0.0175 | 0.0007 | [-0.0004, -0.0001] | 0.0349 | 0.1688 | 58,752 | 0.117 | -0.0000 [-0.0001, 0.0001] |
| equal_chronological_cap | 20 | item_item_cosine | 0.1204 | 0.0215 | 0.0056 | [-0.0029, -0.0019] | 0.0393 | 0.1792 | 58,752 | 0.117 | -0.0003 [-0.0005, -0.0001] |
| equal_chronological_cap | 20 | implicit_als | 0.1437 | 0.0292 | 0.0104 | [-0.0029, -0.0016] | 0.0483 | 0.2230 | 58,752 | 0.117 | 0.0004 [0.0001, 0.0006] |
| tail_reserve_cap | 20 | popularity | 0.1092 | 0.0173 | 0.0008 | [-0.0005, -0.0002] | 0.0349 | 0.1681 | 58,752 | 0.182 | 0.0000 [-0.0000, 0.0001] |
| tail_reserve_cap | 20 | rating_weighted_popularity | 0.1094 | 0.0174 | 0.0007 | [-0.0004, -0.0002] | 0.0349 | 0.1684 | 58,752 | 0.182 | -0.0000 [-0.0001, 0.0001] |
| tail_reserve_cap | 20 | item_item_cosine | 0.1202 | 0.0214 | 0.0059 | [-0.0031, -0.0020] | 0.0393 | 0.1786 | 58,752 | 0.182 | -0.0003 [-0.0005, -0.0001] |
| tail_reserve_cap | 20 | implicit_als | 0.1435 | 0.0292 | 0.0100 | [-0.0032, -0.0018] | 0.0483 | 0.2230 | 58,752 | 0.182 | 0.0004 [0.0001, 0.0006] |
| equal_chronological_cap | 50 | popularity | 0.1093 | 0.0174 | 0.0006 | [-0.0004, -0.0002] | 0.0349 | 0.1682 | 89,039 | 0.119 | -0.0000 [-0.0000, 0.0000] |
| equal_chronological_cap | 50 | rating_weighted_popularity | 0.1096 | 0.0175 | 0.0003 | [-0.0002, -0.0000] | 0.0349 | 0.1690 | 89,039 | 0.119 | -0.0000 [-0.0001, 0.0000] |
| equal_chronological_cap | 50 | item_item_cosine | 0.1216 | 0.0219 | 0.0031 | [-0.0015, -0.0009] | 0.0397 | 0.1816 | 89,039 | 0.119 | -0.0001 [-0.0003, -0.0000] |
| equal_chronological_cap | 50 | implicit_als | 0.1452 | 0.0297 | 0.0063 | [-0.0012, -0.0003] | 0.0490 | 0.2256 | 89,039 | 0.119 | 0.0001 [-0.0001, 0.0003] |
| tail_reserve_cap | 50 | popularity | 0.1093 | 0.0174 | 0.0006 | [-0.0004, -0.0002] | 0.0349 | 0.1682 | 89,039 | 0.142 | -0.0000 [-0.0000, 0.0000] |
| tail_reserve_cap | 50 | rating_weighted_popularity | 0.1096 | 0.0175 | 0.0004 | [-0.0002, -0.0000] | 0.0349 | 0.1688 | 89,039 | 0.142 | -0.0000 [-0.0001, 0.0000] |
| tail_reserve_cap | 50 | item_item_cosine | 0.1216 | 0.0218 | 0.0033 | [-0.0015, -0.0009] | 0.0396 | 0.1812 | 89,039 | 0.142 | -0.0001 [-0.0003, -0.0000] |
| tail_reserve_cap | 50 | implicit_als | 0.1451 | 0.0297 | 0.0062 | [-0.0012, -0.0004] | 0.0490 | 0.2254 | 89,039 | 0.142 | 0.0001 [-0.0001, 0.0003] |

## Cutoff sensitivity (equal chronological cap)

| Cap | Model | NDCG@10 | NDCG@20 | NDCG@50 | NDCG@100 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | popularity | 0.0851 | 0.0836 | 0.0915 | 0.1079 |
| 1 | rating_weighted_popularity | 0.0855 | 0.0839 | 0.0920 | 0.1083 |
| 1 | item_item_cosine | 0.1015 | 0.0949 | 0.1009 | 0.1170 |
| 1 | implicit_als | 0.1044 | 0.1043 | 0.1179 | 0.1390 |
| 2 | popularity | 0.0857 | 0.0840 | 0.0919 | 0.1083 |
| 2 | rating_weighted_popularity | 0.0855 | 0.0839 | 0.0920 | 0.1085 |
| 2 | item_item_cosine | 0.1019 | 0.0951 | 0.1012 | 0.1176 |
| 2 | implicit_als | 0.1047 | 0.1047 | 0.1183 | 0.1394 |
| 5 | popularity | 0.0858 | 0.0844 | 0.0923 | 0.1086 |
| 5 | rating_weighted_popularity | 0.0864 | 0.0846 | 0.0927 | 0.1090 |
| 5 | item_item_cosine | 0.1019 | 0.0954 | 0.1017 | 0.1183 |
| 5 | implicit_als | 0.1053 | 0.1053 | 0.1194 | 0.1408 |
| 10 | popularity | 0.0863 | 0.0848 | 0.0926 | 0.1089 |
| 10 | rating_weighted_popularity | 0.0865 | 0.0848 | 0.0929 | 0.1092 |
| 10 | item_item_cosine | 0.1022 | 0.0957 | 0.1023 | 0.1192 |
| 10 | implicit_als | 0.1062 | 0.1061 | 0.1206 | 0.1422 |
| 20 | popularity | 0.0865 | 0.0849 | 0.0929 | 0.1092 |
| 20 | rating_weighted_popularity | 0.0869 | 0.0849 | 0.0931 | 0.1095 |
| 20 | item_item_cosine | 0.1031 | 0.0966 | 0.1030 | 0.1204 |
| 20 | implicit_als | 0.1077 | 0.1074 | 0.1219 | 0.1437 |
| 50 | popularity | 0.0866 | 0.0851 | 0.0931 | 0.1093 |
| 50 | rating_weighted_popularity | 0.0871 | 0.0851 | 0.0933 | 0.1096 |
| 50 | item_item_cosine | 0.1039 | 0.0975 | 0.1041 | 0.1216 |
| 50 | implicit_als | 0.1081 | 0.1083 | 0.1228 | 0.1452 |

## Stratified evidence at the primary cutoff

| Policy | Cap | Model | NDCG light | NDCG medium | NDCG heavy | hits head | hits mid | hits tail |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| equal_chronological_cap | 1 | popularity | 0.1354 | 0.0980 | 0.0902 | 14,258 | 0 | 0 |
| equal_chronological_cap | 1 | rating_weighted_popularity | 0.1356 | 0.0984 | 0.0908 | 14,277 | 0 | 0 |
| equal_chronological_cap | 1 | item_item_cosine | 0.1553 | 0.1105 | 0.0850 | 15,665 | 71 | 0 |
| equal_chronological_cap | 1 | implicit_als | 0.1815 | 0.1370 | 0.0983 | 19,126 | 97 | 0 |
| tail_reserve_cap | 1 | popularity | 0.1353 | 0.0980 | 0.0902 | 14,254 | 0 | 0 |
| tail_reserve_cap | 1 | rating_weighted_popularity | 0.1356 | 0.0984 | 0.0908 | 14,275 | 0 | 0 |
| tail_reserve_cap | 1 | item_item_cosine | 0.1548 | 0.1103 | 0.0849 | 15,635 | 72 | 0 |
| tail_reserve_cap | 1 | implicit_als | 0.1809 | 0.1370 | 0.0983 | 19,120 | 100 | 0 |
| equal_chronological_cap | 2 | popularity | 0.1356 | 0.0983 | 0.0908 | 14,266 | 0 | 0 |
| equal_chronological_cap | 2 | rating_weighted_popularity | 0.1358 | 0.0985 | 0.0909 | 14,288 | 0 | 0 |
| equal_chronological_cap | 2 | item_item_cosine | 0.1564 | 0.1106 | 0.0854 | 15,724 | 70 | 0 |
| equal_chronological_cap | 2 | implicit_als | 0.1820 | 0.1375 | 0.0985 | 19,162 | 99 | 0 |
| tail_reserve_cap | 2 | popularity | 0.1355 | 0.0983 | 0.0907 | 14,260 | 0 | 0 |
| tail_reserve_cap | 2 | rating_weighted_popularity | 0.1357 | 0.0984 | 0.0909 | 14,283 | 0 | 0 |
| tail_reserve_cap | 2 | item_item_cosine | 0.1561 | 0.1105 | 0.0853 | 15,703 | 71 | 0 |
| tail_reserve_cap | 2 | implicit_als | 0.1817 | 0.1374 | 0.0985 | 19,140 | 98 | 0 |
| equal_chronological_cap | 5 | popularity | 0.1358 | 0.0987 | 0.0912 | 14,277 | 0 | 0 |
| equal_chronological_cap | 5 | rating_weighted_popularity | 0.1361 | 0.0992 | 0.0917 | 14,290 | 0 | 0 |
| equal_chronological_cap | 5 | item_item_cosine | 0.1566 | 0.1118 | 0.0862 | 15,833 | 67 | 0 |
| equal_chronological_cap | 5 | implicit_als | 0.1842 | 0.1387 | 0.0993 | 19,327 | 99 | 0 |
| tail_reserve_cap | 5 | popularity | 0.1358 | 0.0987 | 0.0911 | 14,277 | 0 | 0 |
| tail_reserve_cap | 5 | rating_weighted_popularity | 0.1360 | 0.0990 | 0.0917 | 14,286 | 0 | 0 |
| tail_reserve_cap | 5 | item_item_cosine | 0.1565 | 0.1113 | 0.0857 | 15,778 | 67 | 0 |
| tail_reserve_cap | 5 | implicit_als | 0.1840 | 0.1388 | 0.0995 | 19,317 | 99 | 0 |
| equal_chronological_cap | 10 | popularity | 0.1359 | 0.0991 | 0.0917 | 14,285 | 0 | 0 |
| equal_chronological_cap | 10 | rating_weighted_popularity | 0.1362 | 0.0994 | 0.0918 | 14,297 | 0 | 0 |
| equal_chronological_cap | 10 | item_item_cosine | 0.1576 | 0.1128 | 0.0870 | 15,934 | 66 | 0 |
| equal_chronological_cap | 10 | implicit_als | 0.1858 | 0.1402 | 0.1003 | 19,520 | 98 | 0 |
| tail_reserve_cap | 10 | popularity | 0.1359 | 0.0989 | 0.0916 | 14,285 | 0 | 0 |
| tail_reserve_cap | 10 | rating_weighted_popularity | 0.1362 | 0.0993 | 0.0918 | 14,291 | 0 | 0 |
| tail_reserve_cap | 10 | item_item_cosine | 0.1577 | 0.1124 | 0.0867 | 15,916 | 66 | 0 |
| tail_reserve_cap | 10 | implicit_als | 0.1858 | 0.1401 | 0.1007 | 19,520 | 99 | 0 |
| equal_chronological_cap | 20 | popularity | 0.1361 | 0.0992 | 0.0920 | 14,300 | 0 | 0 |
| equal_chronological_cap | 20 | rating_weighted_popularity | 0.1365 | 0.0996 | 0.0922 | 14,314 | 0 | 0 |
| equal_chronological_cap | 20 | item_item_cosine | 0.1590 | 0.1142 | 0.0877 | 16,048 | 70 | 0 |
| equal_chronological_cap | 20 | implicit_als | 0.1881 | 0.1413 | 0.1015 | 19,691 | 104 | 0 |
| tail_reserve_cap | 20 | popularity | 0.1361 | 0.0992 | 0.0920 | 14,309 | 0 | 0 |
| tail_reserve_cap | 20 | rating_weighted_popularity | 0.1363 | 0.0996 | 0.0922 | 14,311 | 0 | 0 |
| tail_reserve_cap | 20 | item_item_cosine | 0.1589 | 0.1140 | 0.0876 | 16,032 | 69 | 0 |
| tail_reserve_cap | 20 | implicit_als | 0.1872 | 0.1413 | 0.1018 | 19,701 | 102 | 0 |
| equal_chronological_cap | 50 | popularity | 0.1362 | 0.0993 | 0.0921 | 14,304 | 0 | 0 |
| equal_chronological_cap | 50 | rating_weighted_popularity | 0.1365 | 0.0997 | 0.0925 | 14,328 | 0 | 0 |
| equal_chronological_cap | 50 | item_item_cosine | 0.1602 | 0.1155 | 0.0889 | 16,188 | 69 | 0 |
| equal_chronological_cap | 50 | implicit_als | 0.1904 | 0.1421 | 0.1028 | 20,000 | 102 | 0 |
| tail_reserve_cap | 50 | popularity | 0.1362 | 0.0993 | 0.0921 | 14,305 | 0 | 0 |
| tail_reserve_cap | 50 | rating_weighted_popularity | 0.1364 | 0.0997 | 0.0926 | 14,329 | 0 | 0 |
| tail_reserve_cap | 50 | item_item_cosine | 0.1602 | 0.1154 | 0.0888 | 16,171 | 70 | 0 |
| tail_reserve_cap | 50 | implicit_als | 0.1900 | 0.1421 | 0.1031 | 20,000 | 106 | 0 |

## Interpretation boundary

- A positive tail-vs-equal improvement means the tail-reserve policy had lower mean per-user absolute NDCG error than the equal chronological cap at the same cap.
- At least one tested tail-reserve cell has a strictly positive paired 95% interval; inspect the complete grid before promoting it.
- Item-item cosine has zero model-noise floor and separates the budget in 12 of 12 cells, where the paired 95% NDCG interval against full history excludes zero.
- No ALS cell exceeds the same-data seed-variance floor, so per-user ALS differences here are not separable from optimizer initialization noise.
- Mean NDCG stability and per-user stability are different claims; report both.
- The configured 0.005 mean-absolute-NDCG tolerance is exploratory, not a product-risk threshold or a validated stop-controller rule.
- The replay validates policy-specific sample-to-reference behavior on this cohort and time partition only. It does not validate client availability, dropout, communication, secure aggregation, local compute, consent, or federated convergence.
