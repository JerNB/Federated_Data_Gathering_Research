# Next hypotheses worth testing (plain version)

Chinese version: [`04_next_hypotheses_zh.md`](04_next_hypotheses_zh.md). Source: 5 subagents (3 literature scans, 1 scan of what this repo did not test, 1 cross-field analogies).
**Status: candidates only, none has been run.** Every numeric prediction is an inference; "no prior work found" comes from search summaries, not an exhaustive survey.

## 1. New storyline: from "how many people" to "who to ask, how many records, how often, when to stop"

The results so far read as description. Together these hypotheses can become a **toolkit**: each gives a rule or a number you can look up.
The old negative results (hybrid fails, more people is not more accurate) become the motivation for why such rules are needed.

## 2. Candidates (in my recommended order)

### 1. One number is enough: how many people chose the boundary item (almost no compute)
- **Idea:** accuracy depends not on N but on **h = number of sampled people who chose the item ranked 100th**. Redraw all 11 datasets against h and the curves should collapse onto one line.
- **Prediction:** overlap@100 reaches 0.8 at h of about 15-30; required people N ≈ h / (choice rate of the 100th item).
- **Why not obvious:** so far the answer was "depends on the dataset", with only 0.54 correlation to the Zipf slope. A collapse turns "it depends" into one number.
- **Analogy:** TV ratings use a "±10% relative error" rule, which is also about counts, not headcount.
- **Falsifier:** after rescaling to h, the spread across datasets is not 2x tighter than at fixed N, or leave-one-dataset-out predictions miss by more than 2x in over half the cells.

### 2. Ask light users first, not heavy users
- **Idea:** with k ≤ 5 records per person, sampling people with probability 1/(history length) beats uniform sampling, because heavy users have niche tastes and their records land on unpopular items.
- **Hints already in this repo's tables:** lastfm-1k at N=100: 0.481 vs 0.352. Counter-example: amazon-musical-instruments. At k = all the direction reverses.
- **Prediction:** the best sampling exponent a* is negative for small k, crosses zero near k = 20-50, positive at k = all; its sign matches the sign of the correlation between history length and head share.
- **Falsifier:** no better than uniform at k=5 in half the datasets, or signs do not match.

### 3. Why do recency methods recover only 2%? Most of the gap is newborn items
- **Idea:** a large part of the later list is items that did not exist at the start. No reweighting of the old sample can see them; only asking again can.
- **Prediction:** newborn items carry at least 40% of the head mass of the later list in Amazon, MovieLens and KuaiRand, under 10% in Gowalla and Last.fm (venues, artists); removing them cuts the gap at least in half.
- **Value:** explains our central finding directly, and shows why repair is limited in principle.
- **Falsifier:** removing newborn items cuts the gap by less than 20%.

### 4. Half-life calculator: how often to re-ask, how much to down-weight old data
- **Idea:** the best half-life h* is proportional to the list's own staleness time τ (0.5-2x), and longer for smaller samples.
- **Value:** turns the 0.66 ceiling into "how much a correct half-life recovers". Needs rolling time origins (we have one).
- **Falsifier:** h*/τ varies more than 4x across datasets, or a tuned half-life beats the fixed one by under 0.005.

### 5. Detect staleness early with a 50-person probe
- **Idea:** ask 50 people, compare their recent list with the current list, and tell whether it has gone stale (AUC ≥ 0.8), saving 30%+ of participants over fixed-interval refresh.
- **Falsifier:** AUC below 0.65, or no participants saved.
- **Risk:** self-assessment failed before (X14, X16). This asks a different question, but it may fail too.

### 6. When to stop: split the sample in two halves and check they agree
- **Idea:** overlap between two halves predicts the gap to the true list (error ≈ 0.35 × (1 − half-overlap)), giving a "stop when the halves agree" rule.
- **Falsifier:** calibration slope outside [0.25, 0.5], or the target is met in under 85% of stopped runs.
- **Note:** related to X14, which showed we cannot tell which draw is better; this asks how large one draw's error is, which may or may not hold.

### 7. A volunteer-bias ceiling: more volunteers stop helping (federated opt-in)
- **Idea:** if participation correlates with taste, the list plateaus however many join; then 200 randomly chosen people can beat 10,000 volunteers.
- **Value:** explains why HT weighting barely helped, and gives "past this number, stop recruiting and change who you recruit".
- **Falsifier:** activity-biased samples show no plateau up to N=10,000.

### 8. Under privacy, there is an optimal number of records per person
- **Idea:** without privacy, more people and more records tie. With differential privacy, noise grows with the records each person gives, so a clear best k* appears (k* = 1 under local privacy).
- **Value:** breaks the C1 tie and gives federated learning a design rule.
- **Falsifier:** at equal total records, a mechanism prefers k > 1 in most cells.

### 9. Cheaper ideas
- **Same panel or fresh people each round?** Rotating-panel theory predicts 20-50% lower error on estimated change, at a small cost in level.
- **Select crowd vs diverse crowd:** invite people who agreed most with the earlier list; may win at small N, gone by N ≥ 1000.
- **KuaiRand random-exposure log:** popularity under recommendation vs under random display (needs a re-download; disk is tight, so low priority).
- **Zero-compute re-analyses from existing results:** training records vs profile length; minimum history worth personalizing; whether lifting the 1,500-candidate cap changes N*; how much an oracle-tuned model lowers N*.

## 3. My recommendation

Do **1, 3, 2** first (mostly existing results and counts, about 1-3 CPU hours), then **4, 7, 8**. Leave 5 and 6 for last because they carry more risk.
Each gets a pre-registration with falsifiers before it runs, with fans and memory still capped.

## 4. Which directions are "hot" nearby

- **Data minimization and privacy** (federated analytics, heavy hitters, local DP): hypotheses 7, 8.
- **When to retrain or refresh recommenders:** hypotheses 3, 4, 5.
- **Popularity bias:** hypotheses 2, 7.
- This is the subagents' judgment from search summaries; nobody counted papers.

## 5. Main prior work the agents opened

- Ji et al. 2020/2022: recent-window popularity beats all-history popularity; time leakage in offline evaluation.
- Verachtert et al. 2023: retail barely goes stale within days, news within hours.
- Meng 2018: selection bias in big data and "effective sample size".
- Salganik et al. 2006: MusicLab.
- Spearman-Brown formula and the Kish design effect are standard tools, so hypotheses 4 and 6 are applications to popularity lists, not new theory.
