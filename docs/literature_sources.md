# Literature source register

Accessed: 2026-09-12

This register records sources used by the proposal. It separates executed
snapshot-calibration evidence from sources that justify the planned
federated local-data controller and its system-confirmation boundary.

## Primary-study sources

### S1 — Hu, Koren, and Volinsky (2008), Collaborative Filtering for Implicit Feedback Datasets

- Primary source: https://yifanhu.net/PUB/cf.pdf
- DOI: https://doi.org/10.1109/ICDM.2008.22
- Evidence used: observed and missing feedback have different meanings; a
  confidence-weighted implicit objective is distinct from explicit-rating
  reconstruction.
- Proposal consequence: thresholded MovieLens ratings are an implicit-style
  proxy, not naturally observed clicks; ALS remains a later sensitivity track.

### S2 — Rendle et al. (2009), BPR: Bayesian Personalized Ranking from Implicit Feedback

- Primary source: https://arxiv.org/abs/1205.2618
- Published context: UAI 2009.
- Evidence used: personalized top-N ranking from implicit feedback requires a
  ranking-oriented objective; BPR is a pairwise criterion.
- Proposal consequence: the first recommendation implementation keeps the
  tracked BPR contract and records all sampling and optimization controls.

### S3 — Meng et al. (2020), Exploring Data Splitting Strategies for the Evaluation of Recommendation Models

- Primary source: https://arxiv.org/abs/2007.13237
- Evidence used: split choices can change recommender performance and model
  ordering.
- Proposal consequence: the full-data reference split is frozen before sample
  comparisons; sample-native splits are sensitivity results.

### S4 — Zhao et al. (2020), Revisiting Alternative Experimental Settings for Evaluating Top-N Item Recommendation Algorithms

- Primary source: https://arxiv.org/abs/2010.04484
- Evidence used: splitting, sampled evaluation, and domain choices affect
  top-N conclusions.
- Proposal consequence: full eligible-catalog ranking is primary; sampled
  negative evaluation is not treated as interchangeable.

### S5 — Ji et al. (2023), A Critical Study on Data Leakage in Recommender System Offline Evaluation

- Primary source: https://arxiv.org/abs/2010.11060
- Published version: https://doi.org/10.1145/3569930
- Evidence used: timeline leakage can inflate accuracy and change relative model
  ordering.
- Proposal consequence: sampling uses training data only, and evaluation keeps
  a fixed chronological reference target.

### S6 — Harper and Konstan (2015), The MovieLens Datasets: History and Context

- Primary source: https://doi.org/10.1145/2827872
- Evidence used: MovieLens is a long-running controlled research benchmark with
  documented context and limitations.
- Proposal consequence: MovieLens supports an internal resampling study, not by
  itself a claim about production click or consumption data.

### S14 — Dong, Li, and Schnabel (2023), When Newer is Not Better

- Primary source: https://arxiv.org/abs/2305.01801
- Published context: SIGIR 2023.
- Evidence used: traditional and neural recommenders can differ by criterion and
  dataset; no model family dominates every evaluation.
- Proposal consequence: start with one controlled classical objective, then test
  algorithm-family robustness as a later result layer.

### S15 — Milogradskii et al. (2024), Revisiting BPR: A Replicability Study

- Primary source: https://arxiv.org/abs/2409.14217
- DOI: https://doi.org/10.1145/3640457.3688073
- Evidence used: BPR implementations and hyperparameters materially affect
  results.
- Proposal consequence: the algorithm protocol is frozen across sample cells;
  seeds, negative sampling, regularization, and update budgets are recorded.

### S16 — Viering and Loog (2023), The Shape of Learning Curves: a Review

- Primary source: https://arxiv.org/abs/2103.10948
- DOI: https://doi.org/10.1109/TPAMI.2022.3220744
- Evidence used: learning curves describe expected generalization performance as
  training size changes; they can help forecast the value of more data and
  reduce training cost, but their shapes vary and no universal curve is valid.
- Proposal consequence: estimate an empirical sample-size curve with uncertainty;
  do not assume monotonicity, power-law shape, or a universal threshold.

### S17 — Cortes et al. (1993), Learning Curves: Asymptotic Values and Rate of Convergence

- Primary source: https://proceedings.neurips.cc/paper_files/paper/1993/file/1aa48fc4880bb0c9b8a3bf979d3b917e-Paper.pdf
- Evidence used: expected test and training errors are functions of training-set
  size; repeated choices of training sets at intermediate sizes can support
  extrapolation toward a large database and avoid expensive full runs.
- Proposal consequence: use repeated independent draws at each size; a single
  trajectory cannot separate sampling variation from systematic bias.

### S18 — Bates, Hastie, and Tibshirani (2023), Cross-validation: what does it estimate and how well does it do it?

- Primary source: https://arxiv.org/abs/2104.00673
- DOI: https://doi.org/10.1080/01621459.2023.2197686
- Evidence used: cross-validation estimates average prediction error over other
  training sets, not necessarily the exact fitted model; ordinary fold-based
  variance estimates can be too small because fold results are correlated.
- Proposal consequence: k-fold cross-validation may support tuning or a
  sensitivity check, but repeated independent subsampling is the primary
  sample-size design.

### S19 — Sachdeva, Wu, and McAuley (2022), On Sampling Collaborative Filtering Datasets

- Primary source: https://arxiv.org/abs/2201.04768
- DOI: https://doi.org/10.1145/3488560.3498439
- Evidence used: user, interaction, and graph sampling schemes can change
  recommendation performance and algorithm ranking; preserving relative model
  ordering is a useful but nontrivial sampling target.
- Proposal consequence: compare declared sampling schemes, report both metric
  error and ranking/conclusion stability, and inspect user/item sparsity and
  long-tail coverage.

### S20 — Abdou (2026), The Unreasonable Effectiveness of Data for Recommender Systems

- Primary source: https://arxiv.org/abs/2604.06420
- Evidence used: a multi-dataset study uses absolute stratified-user sampling
  across many sizes and measures NDCG@10; performance often changes with data
  size, but behavior varies by dataset and algorithm.
- Proposal consequence: this supports the study motivation and a learning-curve
  design, not a universal claim; the present study adds independent replicates
  and explicit conclusion-stability tests.

## Federated data-gathering and system sources

These sources justify the planned federated local-data-sufficiency study. They
do not turn the executed central snapshot matrix into federated evidence.

### S7 — McMahan et al. (2017), Communication-Efficient Learning of Deep Networks from Decentralized Data

- Primary source: https://proceedings.mlr.press/v54/mcmahan17a.html
- Proposal use: defines federated local data and server aggregation under
  communication and non-IID constraints.

### S8 — Li et al. (2020), Federated Optimization in Heterogeneous Networks

- Primary source: https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
- Proposal use: FedProx is a named heterogeneity control for later
  federated-system confirmation.

### S9 — Karimireddy et al. (2020), SCAFFOLD

- Primary source: https://proceedings.mlr.press/v119/karimireddy20a.html
- Proposal use: control variates are a candidate control for client drift after
  the data-plan effect is established.

### S10 — Ghosh et al. (2020), An Efficient Framework for Clustered Federated Learning

- Primary source: https://proceedings.neurips.cc/paper_files/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html
- Proposal use: IFCA is a candidate learned-routing method outside the first
  local-data-budget claim.

### S11 — Ammad-ud-din et al. (2019), Federated Collaborative Filtering for Privacy-Preserving Personalized Recommendation System

- Primary source: https://arxiv.org/abs/1901.09888
- Proposal use: federated recommendation is a separate local-data and
  communication confirmation experiment.

### S12 — Chai et al. (2019), Secure Federated Matrix Factorization

- Primary source: https://arxiv.org/abs/1906.05108
- Proposal use: raw-data locality alone does not establish privacy; gradients
  can leak information and require a named protection mechanism.

### S13 — Wang and Chang (2020), Federated Matrix Factorization: Algorithm Design and Application to Data Clustering

- Primary source: https://arxiv.org/abs/2002.04930
- Readable mirror: https://ar5iv.labs.arxiv.org/html/2002.04930
- Proposal use: recommender-specific federated matrix-factorization
  optimization is distinct from generic FedAvg and the data-plan estimand.

### S21 — Zhao, Wang, and Lin (2023), The Aggregation–Heterogeneity Trade-off in Federated Learning

- Primary source: https://proceedings.mlr.press/v195/zhao23b.html
- Proposal use: adding more heterogeneous data is not automatically beneficial;
  aggregation can trade variance against heterogeneity-induced bias.

### S22 — Bonawitz et al. (2019), Towards Federated Learning at Scale: System Design

- Primary source: https://arxiv.org/abs/1902.01046
- Proposal use: production FL eligibility, device availability, dropout,
  resource limits, and secure aggregation can create participation bias and
  constrain which diagnostics and data plans are feasible.

### S23 — Lai et al. (2021), Oort: Efficient Federated Learning via Guided Participant Selection

- Primary source: https://www.usenix.org/conference/osdi21/presentation/lai
- Proposal use: participant selection can trade data utility and device speed
  while enforcing a declared participant-data distribution. This is a
  round-participation baseline, not a per-client local-history budget.

### S24 — Fraboni et al. (2021), Clustered Sampling

- Primary source: https://proceedings.mlr.press/v139/fraboni21a.html
- Proposal use: client clustering by sample size or model similarity can improve
  representativeness and aggregation variance under non-IID, unbalanced data.
  It motivates coverage-stratified baselines, not raw-history inspection.

### S25 — Tang et al. (2025), Efficient Heterogeneity-Aware Federated Active Data Selection

- Primary source: https://proceedings.mlr.press/v267/tang25i.html
- Proposal use: privacy limits cross-client data-query information. FALE studies
  label-query selection for federated regression, so it establishes a related
  observability constraint rather than a recommender-local-history solution.

### S26 — Gong et al. (2024), ODE: An Online Data Selection Framework for Federated Learning With Limited Storage

- Primary source: https://doi.org/10.1109/TNET.2024.3365534
- Open paper: https://zhengzhenzhe220.github.io/papers/TON24.pdf
- Proposal use: direct antecedent for online local retention under streaming,
  storage-constrained FL. ODE values per-event gradients and coordinates
  cross-client storage metadata to improve convergence and final accuracy; it
  does not test recommendation-decision sufficiency or false-accept control.

### S27 — Nagalapatti, Mittal, and Narayanam (2022), Is Your Data Relevant?

- Primary source: https://ojs.aaai.org/index.php/AAAI/article/view/20755
- Proposal use: direct antecedent for dynamic client-side training-example
  selection. FLRD uses a private relevance selector and server validation
  feedback; it targets global model quality under irrelevant/noisy data, not a
  fixed-cohort local-history budget or a privacy-aggregate-only controller.

### S28 — Qu et al. (2024), Towards Personalized Privacy: User-Governed Data Contribution for Federated Recommendation

- Primary source: https://arxiv.org/abs/2401.17630
- Proposal use: a federated-recommendation boundary case where users choose
  whether and how much data to upload to the server. It changes the no-raw-data
  privacy model, so it is an adjacent contribution-policy direction.

### S29 — Qu et al. (2025), Proxy Model-Guided Reinforcement Learning for Client Selection in Federated Recommendation

- Primary source: https://arxiv.org/abs/2508.10401
- Status: under review at the cited preprint.
- Proposal use: a recommender-specific participant-selection comparator that
  couples client contribution estimates with training policy. It changes which
  clients participate rather than establishing local-history sufficiency.

## Offline evaluation-protocol sources

These sources govern cutoff choice, metric selection, and known offline
evaluation biases. They are read as method constraints, not as results.

### S30 — Valcarce, Bellogín, Parapar, and Castells (2018), On the Robustness and Discriminative Power of IR Metrics for Top-N Recommendation

- Primary source: https://doi.org/10.1145/3240323.3240347
- Open draft: https://www.dc.fi.udc.es/~dvalcarce/pubs/valcarce-etal-recsys2018.pdf
- Measured claims (MovieLens 1M, LibraryThing, BeerAdvocate; 21 recommenders;
  AllItems protocol; relevance threshold 4):
  - System rankings correlate strongly across cutoffs 5–100 (Kendall tau mostly
    above 0.9; lowest observed 0.76 between `@5` and `@100` on MovieLens), so
    the cutoff rarely reverses "which system is better".
  - Deeper cutoffs (about 100) are more robust to both sparsity and popularity
    bias and have higher discriminative power than shallow cutoffs of 5–10.
  - Discriminative-power score (lower is better, cutoff 100) on MovieLens 1M:
    nDCG 1.4, precision 2.6, MAP 2.8, recall 7.0, infAP 8.4, bpref 9.9,
    MRR 15.5.
  - Precision is the most robust metric; nDCG is the most discriminative; MRR,
    bpref, and infAP perform poorly for recommendation.
- Proposal use: justifies reporting deep cutoffs, using nDCG as primary and
  precision as the robust secondary, and treating recall and MRR as weak
  primaries. A shallow cutoff is a product-display choice, not an evaluation
  requirement.

### S31 — Krichene and Rendle (2020), On Sampled Metrics for Item Recommendation

- Primary source: https://doi.org/10.1145/3394486.3403226
- Open paper: http://walid.krichene.net/papers/KDD-sampled-metrics.pdf
- Claim: metrics computed against a sampled subset of irrelevant items are
  inconsistent with their exact counterparts and can reverse system order;
  as the sample shrinks, metrics degenerate toward AUC. Corrections reduce but
  do not remove the inconsistency; avoid sampling when exact evaluation is
  affordable.
- Proposal use: confirms the executed AllItems protocol—ranking the full
  rating-bearing catalog with no negative sampling—is the correct choice, and
  forbids switching to sampled evaluation for speed.

### S32 — Steck (2013), Evaluation of Recommendations: Rating-Prediction and Ranking

- Primary source: https://doi.org/10.1145/2507157.2507160
- Claim: the decisive difference between rating-prediction and ranking
  evaluation is the data each uses, not the metric. Observed ratings are
  missing not at random, so evaluating only on observed items answers a small
  and biased part of the task.
- Proposal use: the relevant set in this study is "items the user later rated
  at least 4", which is an observation artifact rather than the user's true
  interest set. Any metric that divides by that set inherits the bias.

### S33 — Yang et al. (2018), Unbiased Offline Recommender Evaluation for Missing-Not-At-Random Implicit Feedback

- Primary source: https://doi.org/10.1145/3240323.3240355
- Claim: average-over-all offline evaluation of implicit feedback is biased
  toward popular items; inverse-propensity weighting with a popularity-based
  exposure model reduces that bias, at the cost of estimator variance.
- Proposal use: a candidate debiasing layer for a later phase. It requires a
  declared propensity model, so it is not adopted silently.

### S34 — Cañamares and Castells (2018), Should I Follow the Crowd?

- Primary source: https://doi.org/10.1145/3209978.3210014
- Claim: popularity can be either a genuine effectiveness signal or an
  evaluation artifact depending on how rating, discovery, and relevance
  interact; offline accuracy can diverge from unbiased accuracy.
- Proposal use: motivates reporting tail-restricted and popularity-stratified
  results rather than a single averaged accuracy number.
