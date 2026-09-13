# Sample-generalization local run

Status: exploratory full-snapshot evidence; no external-data claim.

## Design

- The full-data reference is computed once, persisted in `reference_artifact.json`, and reused for every candidate.
- MovieLens `ml-latest` 2023-07-20: 33,832,162 raw ratings, 330,975 users, 83,239 items.
- Positive training interactions: 13,653,758; fixed evaluation panel: 2,000 users.
- Schemes: uniform_user, activity_stratified_user, uniform_interaction, within_user_history; models: popularity, rating_weighted_popularity, item_item_cosine.
- Fractions: 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1; draws per cell: 10.
- Fractions <= 10% are the primary near-independent ladder; larger fractions are convergence checks with finite-population correction.
- Headline metrics use full-reference training-positive exclusion. Sample-native exclusion is a separate sensitivity frame.
- Item-item cosine uses fixed full-reference support >= 20; unsupported candidates receive zero score rather than being removed.

## Aggregate evidence

| Scheme | Fraction | Role | Model | Mean NDCG | Abs. error | Native-fixed gap | Support | Kendall tau | Top choice | Sample rows |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activity_stratified_user | 0.01 | primary_independence | item_item_cosine | 0.0075 | 0.0541 | -0.0004 | 20.0 / 2000.0 | -0.533 | 0.000 | 137673 |
| activity_stratified_user | 0.01 | primary_independence | popularity | 0.0389 | 0.0003 | -0.0167 | 20.0 / 2000.0 | -0.533 | 0.000 | 137673 |
| activity_stratified_user | 0.01 | primary_independence | rating_weighted_popularity | 0.0388 | 0.0002 | -0.0167 | 20.0 / 2000.0 | -0.533 | 0.000 | 137673 |
| activity_stratified_user | 0.025 | primary_independence | item_item_cosine | 0.0083 | 0.0533 | -0.0004 | 50.1 / 2000.0 | -0.667 | 0.000 | 338857 |
| activity_stratified_user | 0.025 | primary_independence | popularity | 0.0390 | 0.0002 | -0.0164 | 50.1 / 2000.0 | -0.667 | 0.000 | 338857 |
| activity_stratified_user | 0.025 | primary_independence | rating_weighted_popularity | 0.0390 | 0.0001 | -0.0166 | 50.1 / 2000.0 | -0.667 | 0.000 | 338857 |
| activity_stratified_user | 0.05 | primary_independence | item_item_cosine | 0.0095 | 0.0521 | -0.0004 | 97.9 / 2000.0 | -0.533 | 0.000 | 680480 |
| activity_stratified_user | 0.05 | primary_independence | popularity | 0.0391 | 0.0002 | -0.0162 | 97.9 / 2000.0 | -0.533 | 0.000 | 680480 |
| activity_stratified_user | 0.05 | primary_independence | rating_weighted_popularity | 0.0390 | 0.0001 | -0.0162 | 97.9 / 2000.0 | -0.533 | 0.000 | 680480 |
| activity_stratified_user | 0.1 | primary_independence | item_item_cosine | 0.0120 | 0.0496 | -0.0004 | 198.2 / 2000.0 | -0.600 | 0.000 | 1368204 |
| activity_stratified_user | 0.1 | primary_independence | popularity | 0.0390 | 0.0001 | -0.0154 | 198.2 / 2000.0 | -0.600 | 0.000 | 1368204 |
| activity_stratified_user | 0.1 | primary_independence | rating_weighted_popularity | 0.0390 | 0.0001 | -0.0154 | 198.2 / 2000.0 | -0.600 | 0.000 | 1368204 |
| activity_stratified_user | 0.25 | convergence_check | item_item_cosine | 0.0196 | 0.0420 | -0.0004 | 495.7 / 2000.0 | -0.533 | 0.000 | 3423945 |
| activity_stratified_user | 0.25 | convergence_check | popularity | 0.0391 | 0.0001 | -0.0130 | 495.7 / 2000.0 | -0.533 | 0.000 | 3423945 |
| activity_stratified_user | 0.25 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | -0.0130 | 495.7 / 2000.0 | -0.533 | 0.000 | 3423945 |
| activity_stratified_user | 0.5 | convergence_check | item_item_cosine | 0.0340 | 0.0276 | -0.0002 | 1008.9 / 2000.0 | -0.600 | 0.000 | 6828793 |
| activity_stratified_user | 0.5 | convergence_check | popularity | 0.0390 | 0.0000 | -0.0083 | 1008.9 / 2000.0 | -0.600 | 0.000 | 6828793 |
| activity_stratified_user | 0.5 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | -0.0083 | 1008.9 / 2000.0 | -0.600 | 0.000 | 6828793 |
| activity_stratified_user | 1 | convergence_check | item_item_cosine | 0.0616 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| activity_stratified_user | 1 | convergence_check | popularity | 0.0391 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| activity_stratified_user | 1 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| uniform_interaction | 0.01 | primary_independence | item_item_cosine | 0.0045 | 0.0571 | -0.0003 | 966.3 / 2000.0 | -0.600 | 0.000 | 136403 |
| uniform_interaction | 0.01 | primary_independence | popularity | 0.0391 | 0.0002 | -0.0169 | 966.3 / 2000.0 | -0.600 | 0.000 | 136403 |
| uniform_interaction | 0.01 | primary_independence | rating_weighted_popularity | 0.0391 | 0.0002 | -0.0170 | 966.3 / 2000.0 | -0.600 | 0.000 | 136403 |
| uniform_interaction | 0.025 | primary_independence | item_item_cosine | 0.0026 | 0.0590 | -0.0001 | 1477.0 / 2000.0 | -0.800 | 0.000 | 341267 |
| uniform_interaction | 0.025 | primary_independence | popularity | 0.0390 | 0.0002 | -0.0168 | 1477.0 / 2000.0 | -0.800 | 0.000 | 341267 |
| uniform_interaction | 0.025 | primary_independence | rating_weighted_popularity | 0.0391 | 0.0003 | -0.0168 | 1477.0 / 2000.0 | -0.800 | 0.000 | 341267 |
| uniform_interaction | 0.05 | primary_independence | item_item_cosine | 0.0056 | 0.0560 | -0.0015 | 1789.8 / 2000.0 | -0.467 | 0.000 | 682758 |
| uniform_interaction | 0.05 | primary_independence | popularity | 0.0392 | 0.0002 | -0.0167 | 1789.8 / 2000.0 | -0.467 | 0.000 | 682758 |
| uniform_interaction | 0.05 | primary_independence | rating_weighted_popularity | 0.0391 | 0.0001 | -0.0167 | 1789.8 / 2000.0 | -0.467 | 0.000 | 682758 |
| uniform_interaction | 0.1 | primary_independence | item_item_cosine | 0.0218 | 0.0397 | -0.0098 | 1954.8 / 2000.0 | -0.600 | 0.000 | 1365617 |
| uniform_interaction | 0.1 | primary_independence | popularity | 0.0391 | 0.0001 | -0.0162 | 1954.8 / 2000.0 | -0.600 | 0.000 | 1365617 |
| uniform_interaction | 0.1 | primary_independence | rating_weighted_popularity | 0.0391 | 0.0001 | -0.0162 | 1954.8 / 2000.0 | -0.600 | 0.000 | 1365617 |
| uniform_interaction | 0.25 | convergence_check | item_item_cosine | 0.0479 | 0.0136 | -0.0238 | 1999.1 / 2000.0 | 0.867 | 1.000 | 3413627 |
| uniform_interaction | 0.25 | convergence_check | popularity | 0.0391 | 0.0001 | -0.0146 | 1999.1 / 2000.0 | 0.867 | 1.000 | 3413627 |
| uniform_interaction | 0.25 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | -0.0145 | 1999.1 / 2000.0 | 0.867 | 1.000 | 3413627 |
| uniform_interaction | 0.5 | convergence_check | item_item_cosine | 0.0581 | 0.0035 | -0.0243 | 2000.0 / 2000.0 | 0.867 | 1.000 | 6827256 |
| uniform_interaction | 0.5 | convergence_check | popularity | 0.0391 | 0.0000 | -0.0113 | 2000.0 / 2000.0 | 0.867 | 1.000 | 6827256 |
| uniform_interaction | 0.5 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | -0.0112 | 2000.0 / 2000.0 | 0.867 | 1.000 | 6827256 |
| uniform_interaction | 1 | convergence_check | item_item_cosine | 0.0616 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| uniform_interaction | 1 | convergence_check | popularity | 0.0391 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| uniform_interaction | 1 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| uniform_user | 0.01 | primary_independence | item_item_cosine | 0.0074 | 0.0541 | -0.0004 | 19.2 / 2000.0 | -0.667 | 0.000 | 135495 |
| uniform_user | 0.01 | primary_independence | popularity | 0.0392 | 0.0003 | -0.0169 | 19.2 / 2000.0 | -0.667 | 0.000 | 135495 |
| uniform_user | 0.01 | primary_independence | rating_weighted_popularity | 0.0392 | 0.0003 | -0.0169 | 19.2 / 2000.0 | -0.667 | 0.000 | 135495 |
| uniform_user | 0.025 | primary_independence | item_item_cosine | 0.0083 | 0.0533 | -0.0004 | 47.0 / 2000.0 | -0.667 | 0.000 | 340119 |
| uniform_user | 0.025 | primary_independence | popularity | 0.0391 | 0.0003 | -0.0167 | 47.0 / 2000.0 | -0.667 | 0.000 | 340119 |
| uniform_user | 0.025 | primary_independence | rating_weighted_popularity | 0.0392 | 0.0002 | -0.0167 | 47.0 / 2000.0 | -0.667 | 0.000 | 340119 |
| uniform_user | 0.05 | primary_independence | item_item_cosine | 0.0094 | 0.0522 | -0.0004 | 93.8 / 2000.0 | -0.800 | 0.000 | 680681 |
| uniform_user | 0.05 | primary_independence | popularity | 0.0391 | 0.0002 | -0.0163 | 93.8 / 2000.0 | -0.800 | 0.000 | 680681 |
| uniform_user | 0.05 | primary_independence | rating_weighted_popularity | 0.0392 | 0.0002 | -0.0164 | 93.8 / 2000.0 | -0.800 | 0.000 | 680681 |
| uniform_user | 0.1 | primary_independence | item_item_cosine | 0.0120 | 0.0496 | -0.0004 | 197.2 / 2000.0 | -0.467 | 0.000 | 1366010 |
| uniform_user | 0.1 | primary_independence | popularity | 0.0392 | 0.0001 | -0.0154 | 197.2 / 2000.0 | -0.467 | 0.000 | 1366010 |
| uniform_user | 0.1 | primary_independence | rating_weighted_popularity | 0.0391 | 0.0001 | -0.0154 | 197.2 / 2000.0 | -0.467 | 0.000 | 1366010 |
| uniform_user | 0.25 | convergence_check | item_item_cosine | 0.0205 | 0.0410 | -0.0003 | 496.6 / 2000.0 | -0.667 | 0.000 | 3410433 |
| uniform_user | 0.25 | convergence_check | popularity | 0.0390 | 0.0000 | -0.0128 | 496.6 / 2000.0 | -0.667 | 0.000 | 3410433 |
| uniform_user | 0.25 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | -0.0129 | 496.6 / 2000.0 | -0.667 | 0.000 | 3410433 |
| uniform_user | 0.5 | convergence_check | item_item_cosine | 0.0346 | 0.0269 | -0.0002 | 1002.7 / 2000.0 | -0.600 | 0.000 | 6829603 |
| uniform_user | 0.5 | convergence_check | popularity | 0.0390 | 0.0001 | -0.0082 | 1002.7 / 2000.0 | -0.600 | 0.000 | 6829603 |
| uniform_user | 0.5 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | -0.0082 | 1002.7 / 2000.0 | -0.600 | 0.000 | 6829603 |
| uniform_user | 1 | convergence_check | item_item_cosine | 0.0616 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| uniform_user | 1 | convergence_check | popularity | 0.0391 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| uniform_user | 1 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| within_user_history | 0.01 | primary_independence | item_item_cosine | 0.0011 | 0.0605 | -0.0001 | 2000.0 / 2000.0 | -0.800 | 0.000 | 375509 |
| within_user_history | 0.01 | primary_independence | popularity | 0.0363 | 0.0027 | -0.0148 | 2000.0 / 2000.0 | -0.800 | 0.000 | 375509 |
| within_user_history | 0.01 | primary_independence | rating_weighted_popularity | 0.0365 | 0.0026 | -0.0149 | 2000.0 / 2000.0 | -0.800 | 0.000 | 375509 |
| within_user_history | 0.025 | primary_independence | item_item_cosine | 0.0005 | 0.0611 | -0.0000 | 2000.0 / 2000.0 | -0.867 | 0.000 | 546478 |
| within_user_history | 0.025 | primary_independence | popularity | 0.0368 | 0.0022 | -0.0149 | 2000.0 / 2000.0 | -0.867 | 0.000 | 546478 |
| within_user_history | 0.025 | primary_independence | rating_weighted_popularity | 0.0370 | 0.0020 | -0.0152 | 2000.0 / 2000.0 | -0.867 | 0.000 | 546478 |
| within_user_history | 0.05 | primary_independence | item_item_cosine | 0.0035 | 0.0581 | -0.0008 | 2000.0 / 2000.0 | -1.000 | 0.000 | 862576 |
| within_user_history | 0.05 | primary_independence | popularity | 0.0373 | 0.0017 | -0.0151 | 2000.0 / 2000.0 | -1.000 | 0.000 | 862576 |
| within_user_history | 0.05 | primary_independence | rating_weighted_popularity | 0.0377 | 0.0013 | -0.0154 | 2000.0 / 2000.0 | -1.000 | 0.000 | 862576 |
| within_user_history | 0.1 | primary_independence | item_item_cosine | 0.0216 | 0.0400 | -0.0092 | 2000.0 / 2000.0 | -0.867 | 0.000 | 1519926 |
| within_user_history | 0.1 | primary_independence | popularity | 0.0385 | 0.0006 | -0.0157 | 2000.0 / 2000.0 | -0.867 | 0.000 | 1519926 |
| within_user_history | 0.1 | primary_independence | rating_weighted_popularity | 0.0387 | 0.0004 | -0.0159 | 2000.0 / 2000.0 | -0.867 | 0.000 | 1519926 |
| within_user_history | 0.25 | convergence_check | item_item_cosine | 0.0485 | 0.0131 | -0.0241 | 2000.0 / 2000.0 | 0.533 | 1.000 | 3534484 |
| within_user_history | 0.25 | convergence_check | popularity | 0.0390 | 0.0001 | -0.0145 | 2000.0 / 2000.0 | 0.533 | 1.000 | 3534484 |
| within_user_history | 0.25 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0001 | -0.0145 | 2000.0 / 2000.0 | 0.533 | 1.000 | 3534484 |
| within_user_history | 0.5 | convergence_check | item_item_cosine | 0.0584 | 0.0032 | -0.0243 | 2000.0 / 2000.0 | 0.600 | 1.000 | 6906787 |
| within_user_history | 0.5 | convergence_check | popularity | 0.0390 | 0.0001 | -0.0113 | 2000.0 / 2000.0 | 0.600 | 1.000 | 6906787 |
| within_user_history | 0.5 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | -0.0113 | 2000.0 / 2000.0 | 0.600 | 1.000 | 6906787 |
| within_user_history | 1 | convergence_check | item_item_cosine | 0.0616 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| within_user_history | 1 | convergence_check | popularity | 0.0391 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |
| within_user_history | 1 | convergence_check | rating_weighted_popularity | 0.0390 | 0.0000 | 0.0000 | 2000.0 / 2000.0 | 1.000 | 1.000 | 13653758 |

## Resource evidence

- Reference build: 0.0s; reference cache hit: True.
- Per-cell evaluation time: 0.10–33.68s; replicate pass time: 22.2–55.5s.
- Peak resident set size across draw rows: 3303–4195 MiB.
- These resource figures describe this local run and hardware; they are not a cross-machine performance claim.

## Interpretation

- Fixed-frame error is an in-reference approximation comparison, not an independent external-generalization test.
- Native/fixed divergence measures exclusion-set protocol bias.
- Support is essential for per-user models; it is diagnostic rather than a failure condition for global item-statistic controls.
- A positive approximation claim requires metric tolerance, stable ordering, coverage/temporal checks, and replication by an independent scheme.
- If the full reference is unaffordable, rerun with a declared largest-affordable reference (for example 50k users); conclusions then describe convergence to that reference and extrapolation beyond it.

## Figures

