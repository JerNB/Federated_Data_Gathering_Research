# Research direction

## Recommended question

In cross device federated implicit feedback matrix factorization, with user factors kept local, how do between community preference separation, within user multi interest overlap, item support overlap, and local data sparsity determine when capacity matched cluster conditioned item models outperform one shared item model? How much of the oracle specialization gain can federated compatible hard or soft routing recover?

Use **empirical crossover point** instead of **specialization threshold**. The result may depend on sample size, model capacity, number of specialists, routing method, and heterogeneity structure. A valid result can show that a crossover does not occur.

## Core model

Treat each MovieLens user as one client. A shared matrix factorization model can use a local user factor and a shared item factor:

```text
score(user, item) = local_user_factor(user) dot shared_item_factor(item)
```

Hard specialization replaces the shared item factor with a cluster specific item factor. Soft specialization combines several cluster specific scores with user specific routing weights.

The experiment must state ownership for user factors, item factors, biases, and optimizer state. A global model already personalizes through local user factors. The study therefore measures the value of multiple shared item spaces beyond that existing personalization.

## Main scientific idea

The useful variable is the structure of heterogeneity, not only its average amount.

Two client populations can have the same Jensen Shannon divergence and produce different results when one contains clear preference communities, another contains users with several interests, and another contains popularity skew or exposure differences. Random partition noise and differences concentrated in rare items create further cases. These structures can favor different collaboration strategies.

Recommendation adds three important effects:

1. Clients share an item universe with uneven item coverage.
2. Interactions are sparse and missing interactions are not reliable negative labels.
3. A dissimilar client can still provide useful information about items that another client has not observed.

The study should separate exploitable preference structure from exposure, popularity, and data collection effects.

## Recommended study design

1. **Federation unit**

   Use one user per client for the initial cross device setting. Keep any alternative client construction as a separate experiment.

2. **Controlled factors**

   Vary between community separation, within user multi interest overlap, item support overlap, popularity skew, exposure bias, and local sample size as separate factors where possible.

3. **Partition conditions**

   Compare random grouping, interpretable grouping, oracle grouping, centralized training only grouping, federated compatible hard routing, and soft routing. Label the information available to each condition.

4. **Baselines**

   Include global matrix factorization, local training, global plus local adaptation, regularized local models, clustered models, and soft mixture models. Use FedProx or SCAFFOLD when optimization stability needs separate control.

5. **Capacity and optimization controls**

   Report equal total parameter capacity and equal per model capacity. Match communication, computation, update count, initialization, tuning budget, and candidate catalog. A specialist should not win because it received more training or an easier catalog.

6. **Data splits**

   Prefer user level chronological splits when the claim concerns future recommendation. Keep routing and cluster construction within training information. Mark any oracle use clearly.

7. **Evaluation**

   Rank against the same eligible catalog for every method. Prefer full catalog evaluation for the primary result. Report macro user utility, interaction weighted utility, worst group utility, and long tail utility. Include uncertainty intervals and the difference between specialist and global performance.

8. **Capacity of the claim**

   Start with a controlled empirical study. A method that estimates routing from privacy permitted signals can follow after the oracle to feasible gap is measured.

## Execution order

The data and experiment registry comes before model implementation. The source snapshot is frozen through `data/dataset_manifest.json`. A canonical working package, experiment configuration, objective formula, run record, and milestone report are separate tracked objects.

The first comparison contains exactly two model variants:

1. Global matrix factorization with local user factors and one shared item factor matrix.
2. Capacity matched oracle clustered matrix factorization with local user factors and one item factor matrix per declared cluster.

The pilot uses a fixed deterministic target of 5,000 users. User eligibility and item support are computed from pilot training data only. The initial oracle setting uses four clusters. An item enters the pilot catalog only when every cluster has at least 20 training interactions for that item. The pilot requires at least 1,000 eligible items.

The support report records selected users, ratings, eligible items, per cluster users and ratings, minimum support, median support, support percentiles, and items below the threshold. If the support requirement fails, increase the selected user population or reduce the cluster count before interpreting a model difference. A small gain or loss from an under supported catalog is a data power result.

The sweep uses the fixed selected users and the support filtered training catalog, with the same catalog for both variants. Full population and full catalog confirmation starts after the protocol and hyperparameters are frozen. Later model variants, including deployable routing, follow only after the oracle to global difference is measured under these controls.

Use model variant, comparison, condition, milestone, run, and result table as the standard terms for experiment planning.

## Important risks

1. **Global matrix factorization is already personalized.** Local user factors may absorb much of the preference variation. The specialist component must be defined precisely.
2. **Ratings mix preference and exposure.** A genre histogram does not identify true preference without assumptions about what users could see and chose to rate.
3. **Optimization can imitate representation failure.** A poorly tuned global optimizer can make specialization look useful. Stabilize training before interpreting a gain.
4. **Specialization fragments evidence.** Small clusters, sparse items, routing errors, and extra parameters can increase variance.
5. **Latent factors are non identifiable.** Independently trained factor matrices can rotate or reflect. Averaging them requires a shared training alignment or an explicit alignment method.
6. **Evaluation can create leakage.** Test interactions, future behavior, or a restricted specialist catalog can reveal the answer.
7. **Federation does not by itself guarantee privacy.** State whether the project uses locality only, secure aggregation, differential privacy, or a separate privacy analysis.
8. **A single scalar threshold may not exist.** Treat the crossover as conditional on the experimental path and report cases with no crossing.

## Recommended contribution

The moderate contribution is the strongest starting point:

> Equal numerical heterogeneity does not imply equal value from specialization. Community separability, user multi interest, item support overlap, local sparsity, capacity, and routing quality determine the useful degree of sharing.

A conservative version compares global and oracle clustered matrix factorization across controlled preference structures. An ambitious version learns privacy safe routing and adapts among global, neighborhood, clustered, and soft mixture collaboration. Build the controlled study first.

## Open decisions

1. Which parameters remain local, global, or cluster specific in the first model?
2. Which preference generator creates the controlled client populations?
3. Which routing signals are allowed for each feasible method?
4. Which capacity comparison is primary?
5. Which split, catalog, negative sampling rule, and primary metric define success?
6. What result would falsify the premise, including a strong global baseline that remains competitive across realistic regimes?

## First implementation artifacts

1. A source dataset manifest and canonical package contract.
2. A model ownership specification.
3. A training only split and support selection specification.
4. A synthetic heterogeneity generator with separate control axes.
5. A first comparison configuration for global and oracle clustered models.
6. A run record schema with formulas, dataset versions, metrics, and artifacts.
7. An evaluation report with macro user and worst group results.
8. A short privacy and license statement.

## Selected references

1. [Conversation archive for this project](https://chatgpt.com/s/cx_6a9f72cef5ac8191b31ec6c5c5bbf5ee)
2. [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html)
3. [FedProx](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)
4. [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html)
5. [IFCA](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html)
6. [Ditto](https://proceedings.mlr.press/v139/li21h.html)
7. [FedRep](https://proceedings.mlr.press/v139/collins21a)
8. [Federated recommendation and FedMF](https://arxiv.org/abs/1906.05108)
9. [FedCA](https://arxiv.org/abs/2406.03933)
10. [Aggregation heterogeneity trade off](https://proceedings.mlr.press/v195/zhao23b.html)
11. [MovieLens dataset paper](https://doi.org/10.1145/2827872)
