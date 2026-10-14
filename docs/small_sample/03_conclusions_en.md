# Conclusions (plain version): can you build a reliable "most popular" list by asking only a few people?

Chinese version: [`03_conclusions_zh.md`](03_conclusions_zh.md). Full numbers and figures: [`02_results_en.md`](02_results_en.md).
Pre-registration (predictions written before the runs): [`01_preregistration_en.md`](01_preregistration_en.md).

What we did: 11 public datasets (movies, music, products, check-ins), 117,069 small experiments, plus 1,220 repeats with new random seeds.

---

## 1. Eight terms you need

1. **Popularity list**: sort everything by how many times it was chosen. The simplest recommender.
2. **Small sample**: ask only **N people at random**, and see only **k records** from each. Real case: in federated
   learning, only a few people agree to share data.
3. **Two "right answers"** (we test both):
   - **A: the whole community's list at that time.** Question: does a few people's list look like everyone's list?
   - **B: what people chose afterwards.** Question: can this list predict the future?
4. **NDCG@100**: a score from 0 to 1 for how closely our top 100 matches the right answer, with top places counting more.
   1 = identical. 0.88 = very similar, but the first few places may differ.
5. **Personalized model**: looks at each person's own history and recommends different things to each person.
   Smarter than a popularity list, but needs more data.
6. **Hybrid**: part popularity list, part personalized. Our idea was "let the sample decide the mix".
7. **Crossover N\***: how many people you need before personalization starts to beat the popularity list.
   A smaller N\* means personalization helps sooner.
8. **Drift**: tastes change over time. A list made today is a little stale tomorrow.

---

## 2. The one-page answer

> **A small sample is enough for a popularity list. The real problem is that the list goes stale.**
> A list from a few hundred people already looks a lot like everyone's list at that moment. But once the moment has
> passed, the gap between the list and the future is far larger than the gap caused by asking only a few hundred people.

Before running, we wrote down 3 predictions. **Result: one inconclusive, one overturned, one half right.**
That is useful: why these three plausible ideas fail is itself a finding.

---

## 3. The three predictions and what happened

### 1. "More people beats longer histories" (C1): inconclusive

- **Idea:** with a fixed budget (say 10,000 records), ask 1,000 people with 10 records each, or 100 people with 100 each?
  We guessed the first is better.
- **Result:** accuracy is almost a tie (the first wins in 61% of 46 settings; average gap 0.0005, negligible).
- **Interesting part:** the "more people" version is **more stable**. Repeat the draw many times and the list wobbles
  less (in 39 of 46 settings).
- **In one line:** more people buys stability, not accuracy.

### 2. "Let the sample choose the mix" (C2): overturned

- **Idea:** hide some people's records as a test, use the test score to pick the mix of popularity and personalized, and
  the result should never be worse than the better of the two.
- **Result:** it fails. On average it is slightly worse, and better in only 1 of 7 datasets.
- **Why:** the test has almost no skill. It is right 68% of the time, but **always guessing "use popularity" with no test
  is already right 67%**. It also overstates how much personalization helps.
- **In one line:** choosing the mix inside a small sample is as good as not choosing.

### 3. "A crossover N\* exists" (C3): half right

- **Idea:** past some sample size, personalization beats popularity, and the threshold drops as histories get longer.
- **Result:** a crossover does exist, but **it is not one universal number**:
  - some datasets (ml-latest, amazon-software, kuairand-pure) need only about 20 people;
  - others need 100 (ml-1m) or 1,000 (gowalla);
  - two smaller Amazon datasets show none up to 2,000.
  - "Longer histories lower the threshold" was not observed.
- **Note:** the 20-person level comes from a stage we **added afterwards**, once we saw our smallest planned size (100)
  was too coarse. Label it as such.
- **In one line:** the threshold depends on the dataset, not on history length.

---

## 4. New findings worth a paper (exploratory: not predicted in advance, seen in the data)

### 4. Time costs more accuracy than a small sample does

- **Plain version:** like guessing next month's bestsellers from last month's chart.
- **Data:** the gap between an earlier list and a later list is 0.66 points; the gap from asking only 1,000 people is
  0.003. True in 10 of 10 datasets we could judge.
- **Repair:** using only recent records, or weighting new records more, does help (wins 179 of 223 settings), but
  recovers only about **2%** of the gap.
- **Honest limit:** 0.66 is a **ceiling** computed with the future answer in hand; no real method can reach it.

### 5. The popularity list tops out early; the personalized model keeps improving

- **Plain version:** a popularity list is "read the chart", which a few hundred people reveal; personalization is
  "know each person", which keeps improving with more people.
- **Data:** growing from 100 people to the largest sample (up to 4,000), popularity improves 1.22 times, personalized 1.94 times.
- **Limit:** nothing was tested beyond 4,000 people.

### 6. The mathematically "unbiased" estimate does worse in practice

- **Plain version:** "unbiased" = right on average but jumpy each time; "biased" = slightly off but steady. With small
  samples, **steady beats exact**.
- **Data:** in 50 settings, the unbiased method beat plain counting in only 3.

### 7. The usual "rank error bar" is too optimistic

- **Plain version:** we say "this item ranks 80th, give or take: 60 to 100".
- **Data:** the common bootstrap range actually contains the true rank only 85% of the time. A stricter method
  (Mogstad et al.) always did (reaching 95%) but its range is **3 times wider**.
- **In one line:** to rank from a small sample, accept wide ranges or don't trust the exact rank.

### 8. What privacy costs

- **Plain version:** differential privacy adds noise so nobody can tell whether one person took part.
- **Data:** with a trusted center adding the noise (central), at privacy strength ε ≥ 2 and 2,000+ people, list quality
  stays at 86%. With each person adding noise on their own phone (local), you need **about 50 times** more people.
- **In one line:** local privacy is expensive, and it is paid for with headcount.

### 9. Special subgroups: we guessed backwards

- **Idea:** the more a subgroup differs from the crowd, the more it needs its own data.
- **Result:** the more unusual the subgroup, the **sooner** its own sample beats the crowd list (correlation −0.44).
  After removing the effect of series that never found a crossover, only −0.20 is left, which is weak.

---

## 5. How to tell this negative result

1. All three predictions were written and committed before the runs (pre-registration, 15 dated amendments), not picked afterwards.
2. Two failures were judged by failure lines set in advance, and we found why.
3. The by-products give a usable diagnosis: stop worrying about how many people to ask; worry about how fast the list expires.

Suggested title: *"Small samples are enough for popularity. The list going stale is the problem."*

## 6. Do not say

- "N\* is X" (there is no single number).
- "More people always beats longer histories."
- "The hybrid is safe."
- "We solved staleness" (only about 2% recovered).
- Most of the 50 replicated "surprise hunt" results are small-sample artifacts; do not call them important.
- The reviews were by subagents from the same provider, not an outside audit.

## 7. Next steps, by value

1. **Build a "when to ask again" rule**, turning the 0.66 ceiling into a measurable loss.
2. Redo C2 and C3 with a different, non-overlapping test group, to confirm they are not a quirk of one group.
3. Remove the "choose among only 1,500 candidates" limit (on some datasets it covers only 6-37% of real choices).
4. Two things we wanted to test (X4, X13) cannot be tested with the current data.

## 8. Reading order

1. `00_start_here_en.md` → 2. `01_preregistration_en.md` → 3. `02_results_en.md` → 4. this file
5. Figures (`results/small_sample_popularity/04_figures/`): `F2-C1_contrasts.png`, `F2-C2a_hybrid_vs_pure.png`,
   `F2-C3a_crossover_table.png`, `an_explore_a_x10_drift_vs_sampling.png`, `an_explore_b_x7_recency.png`, `an_curves_two_speeds.png`
