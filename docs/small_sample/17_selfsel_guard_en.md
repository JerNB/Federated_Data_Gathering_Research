# 17. Can the selector be fixed for 30 people and for monthly rounds? (pre-registered test on a third window)

*Pre-registered: the plan `configs/selfsel_guard_registry.json` (11 hypotheses G1–G11, thresholds, decision rules) was committed in `cece971` before any cell of the third window was run. Verdicts are computed by `scripts/ssp/an_guard.py`. Anything marked "post hoc" was not in the plan. Tables: `results/small_sample_popularity/03_tables/guard/`.*

## 0. Summary

1. **Registered outcome.** All three registered decisions came out positive; G8 and G11 failed; the word "deployable" is not allowed (item 5).
2. **What was tested.** The *history-seeded selector* gives the plain selector a head start: for every log it takes the mean self-score that each memory earned in the log's *other* 156-week windows (mostly *later* in time than the test window, item 5) and counts it as 38 pseudo-rounds that fade like the normal weights. On a fresh third window of 24 logs (18 Amazon; 142 cells) its regret fell below the plain selector's by 0.0074 at 30 people and monthly rounds (interval −0.0100 to −0.0049; 0.0104 → 0.0030) and by 0.0019 at 30 people and weekly rounds (−0.0027 to −0.0011; 0.0048 → 0.0030).
3. **From 100 people up it is not worse on average** (non-inferiority met): weekly −0.0006 (upper bound −0.0003), monthly −0.0007 (upper bound +0.0015). The weekly gain is small but present at every group size; the monthly gain fades with group size: −0.0074, −0.0022, −0.0010 and +0.0002 at 30, 100, 300 and 1,000 people (post hoc). **That average hides a split (post hoc): the gains are on the 18 Amazon logs. On the 6 other logs there is no gain at 30 people, and at monthly rounds from 100 people the seeded selector is worse than the plain one by 0.0053 (+0.0005 to +0.0102).**
4. **At 30 people most of the gain does not need the log's own history (post hoc).** A prior built from the self-scores of *other* logs at the same group size and round length (no truth, no own history) gives −0.0017 (weekly) and −0.0060 (monthly), against −0.0019 and −0.0074 with the log's own windows; the differences (+0.0002 and +0.0013) are within noise. From 100 people up the other-log prior gives no gain (+0.0004, +0.0008), while the log's own history still gains a little (−0.0006, −0.0007).
5. **Time order: this tests stability across eras, not forecasting.** The history windows lie later than the test window for 21 of the 24 logs; only 3 logs have both history windows earlier, and 3 logs support no claim (the seeded selector is better there in all four regions). The registered gate for the word "deployable" (at least 8 such logs) stayed closed.
6. **Freezing the memory is not better than seeding** (G4 met; it can only show "not better"): a memory frozen from the same history is worse than the seeded selector by 0.0025, 0.0017, 0.0018 and 0.0031 in the four regions (at 30 people and monthly rounds the two are not separable: interval −0.0008 to +0.0052). The seeded selector also beats a fixed memory chosen with the truth of the other windows from 100 people up (+0.0015 and +0.0032; post hoc). G8 (the oracle memory from other windows beats the plain selector at weekly rounds, as K12 of doc 15 suggested) was not met: +0.0010, interval −0.0005 to +0.0028; the K12 reversal was not confirmed.
7. **Self-scores choose a fixed memory close to the truth-chosen one, on average** (G9 met: within 0.0004 in every region; the intervals reach ±0.0018, so "within 0.001" is not shown).
8. **A table prior with a safety margin helps a new log in one region only**: 30 people and monthly rounds (−0.0046, interval −0.0079 to −0.0015; G5), and it costs nothing measurable from 100 people up (+0.0001 and −0.0005; G6). It needs the truth of other logs. The other-log self-score prior of item 4 does as well or better there (−0.0060) without any truth.
9. **Fixed-share weights do nothing** (G7, a registered null: within ±0.0003). **More weight on the history is not better** (G11 not met): 104 pseudo-rounds are worse than 38 by 0.0015 (+0.0005 to +0.0027) at monthly rounds from 100 people.
10. **The in-sample development numbers shrank on fresh data:** G1 −0.0101 → −0.0074, G2 −0.0033 → −0.0019, G5 −0.0087 → −0.0046.

## 1. What was done

1. **A third window.** For each log, the densest 156 weeks that overlap neither the window of the weekly study (w1) nor the confirmation window (w2). 24 of the 27 logs qualified: steam has no third window; amazon-musical-instruments and amazon-industrial-and-scientific had too few rounds with enough truth rows (coverage 0.44 and 0.38, rule: at least 0.70). 18 of the 24 are Amazon logs. A log joins a group size when its median weekly number of active people is at least 3 times the group size.
2. **142 cells**: 71 at 7-day rounds and 71 at 28-day rounds (30 people: 24 logs; 100: 24; 300: 17; 1,000: 6 logs, each at both round lengths); 8 random draws; the same engine as docs 13–16.
3. **The rules.** *Plain* is the selector of docs 14–15. *Warm* (the history-seeded selector) adds the history as 38 pseudo-rounds (*warm104*: 104). *Own-self* freezes the memory with the best mean self-score in the log's other windows. *Margin-table* keeps the table's memory (the one that was best on average in the other logs' w1 and w2 cells) unless another memory's self-score is better by more than 0.005. *Fixed share* is a standard tracking-the-best-expert update (η = 3, α = 0.05; the common variant that mixes with the uniform distribution; doc 90, section 6). *Own-oracle* is a fixed memory chosen with the TRUE list quality in the other windows (a reference, not deployable). *Same-w3 table* is a reference table fitted on the other logs' third windows. Regret = the best fixed memory of the cell, chosen with the truth, minus the rule's list quality.
4. **Regions** (the log is the unit; a log's cells in a region are averaged first): 30 people weekly (n30_7d), 30 people monthly (n30_28d), 100 or more people weekly (n100plus_7d), 100 or more people monthly (n100plus_28d).
5. **Pre-registration.** The parameters were tuned on the development windows w1 and w2 (370 stored cells, `tune_guard.py`); those in-sample numbers are optimistic and are not results. A same-provider critic found that the first draft compared a fixed arm that saw two windows of self-scores with a plain selector that saw only the third window, which is unfair; the plan was rebuilt around the *warm* control, which sees the same evidence. A second round (a code reviewer, a prober with planted synthetic worlds, two more critic passes) found no defect that could change a verdict and led to a confidence-interval test for G4, a minimum number of logs for G10, and the closed gate for the word "deployable".
6. **Run.** 142 cells on 2026-10-16 (07:11–07:16), no failed cell, no cell dropped; the registered files are unchanged since the freeze commit.

## 2. Results (registered hypotheses)

Differences are paired per log (24 logs; 95% bootstrap interval over logs, seed 7). Negative means the first rule has lower regret (is better).

| ID | Registered claim (plain words) | Observed | Verdict |
|---|---|---|---|
| G1 | 30 people, monthly: seeded selector beats plain by ≥ 0.006 (interval below 0) | −0.0074 (−0.0100 to −0.0049); better in 22 of 24 logs | MET |
| G2 | 30 people, weekly: seeded selector beats plain by ≥ 0.0015 (interval below 0) | −0.0019 (−0.0027 to −0.0011); 19 of 24 logs | MET |
| G3 | 100+ people: not worse than plain (interval upper bound < +0.001 weekly, < +0.002 monthly) | weekly −0.0006 (upper −0.0003); monthly −0.0007 (upper +0.0015) | MET |
| G4 | a memory frozen from the same history does not beat the seeded selector by 0.001 or more (interval lower bound > −0.001 in every region) | fixed minus seeded: +0.0025, +0.0017, +0.0018, +0.0031; lower bounds +0.0002, −0.0008, +0.0005, +0.0014 | MET |
| G5 | new log, 30 people, monthly: the table guard beats plain by ≥ 0.004 | −0.0046 (−0.0079 to −0.0015); 17 of 24 logs | MET |
| G6 | the table guard costs at most +0.001 weekly and +0.002 monthly from 100 people | +0.0001 (upper 0.0007); −0.0005 (upper 0.0011) | MET |
| G7 | null: fixed share does not beat plain by 0.0005 or more in any region | +0.0003, −0.0001, −0.0002, −0.0000 | MET |
| G8 | replication of K12: the oracle memory from other eras beats plain at weekly rounds (≤ −0.0005, interval below 0) | +0.0010 (−0.0005 to +0.0028) | NOT MET |
| G9 | self-scores pick the fixed memory within 0.001 of the oracle in every region | +0.0004, −0.0000, +0.0003, −0.0001 | MET |
| G10 | weekly: at most 10% of cells with seeded regret above 0.01 and none above 0.03 | 0 of 71; largest 0.0065 | MET |
| G11 | 104 pseudo-rounds are no worse than 38 (≤ +0.0005 in every region) | −0.0000, −0.0001, +0.0002, **+0.0015** | NOT MET |

Decisions: seeding is an improvement over plain (G1, G2, G3): **yes**. Freezing is not better than seeding (G4): **yes**. The table guard is a cold-start fix for 30 people at monthly rounds (G5, G6): **yes**.

Mean regret over logs by rule and region (for the record):

| Region | Plain | Seeded (38) | Seeded (104) | Own-self | Own-oracle | Margin-table | Fixed share | "2 rounds" |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 people, weekly | 0.0048 | 0.0030 | 0.0029 | 0.0055 | 0.0051 | 0.0052 | 0.0051 | 0.2041 |
| 30 people, monthly | 0.0104 | 0.0030 | 0.0029 | 0.0047 | 0.0047 | 0.0058 | 0.0102 | 0.1177 |
| 100+ people, weekly | 0.0012 | 0.0007 | 0.0009 | 0.0024 | 0.0022 | 0.0013 | 0.0011 | 0.1085 |
| 100+ people, monthly | 0.0043 | 0.0036 | 0.0050 | 0.0067 | 0.0067 | 0.0038 | 0.0043 | 0.0800 |

## 3. Interpretation

1. **Seeding helps where the sample is small, by supplying evidence about which memory suits this kind of data.** At 30 people and monthly rounds the plain selector has 39 rounds of 30-person samples to learn from and its regret is 0.0104 (8 times the weekly 100+ level). With a prior it falls to 0.0030, a 71% cut. At weekly rounds the absolute gain is smaller because the plain selector has about 150 rounds to learn from.
2. **The prior does not have to come from the log itself (post hoc).** At 30 people, a prior made of other logs' self-scores (same group size and round length, same group of logs when at least 3 exist) recovers most of the gain: −0.0017 and −0.0060 against −0.0019 and −0.0074 for the log's own windows (other-log minus own: +0.0002, interval −0.0002 to +0.0006, and +0.0013, interval −0.0004 to +0.0032). So the reading "the log's own eras carry the information" is not what the data show at 30 people: a generic prior about which memory suits a sample of that size is enough, and a server that has run other logs already has one without any ground truth. From 100 people up it is the other way round: the other-log prior gives nothing (+0.0004, +0.0008) and the own history still adds a little (−0.0006, −0.0007; other-log minus own +0.0009 weekly, interval +0.0005 to +0.0014). The seeded selector's gain over plain is therefore (a) evidence about what suits the sample size at 30 people and (b) log-specific evidence from 100 people up.
3. **Freezing is not better than seeding, and seeding is better than freezing from 100 people up.** The memory frozen from the history is worse than plain from 100 people up (+0.0012 weekly, +0.0024 monthly; point estimates, the intervals include 0), about equal at 30 people weekly (+0.0006), and better than plain only at 30 people monthly (−0.0057). It is worse than the seeded selector everywhere (G4), although at 30 people monthly the two are not separable (interval −0.0008 to +0.0052). A seeded selector is also better than a fixed memory chosen *with the truth of the other windows* from 100 people up (+0.0015 weekly, +0.0032 monthly; intervals above 0; post hoc); this comparison is with the truth of the other windows, not the truth of the cell. At 30 people monthly the oracle memory from other windows beats the plain selector by 0.0056 (post hoc, outside G8); at weekly rounds (G8) it does not (+0.0010, interval −0.0005 to +0.0028). Doc 15's K12 reversal was therefore not confirmed.
4. **Self-scores choose a fixed memory about as well as the truth does, on average** (G9). The arm with the best mean self-score in the other windows is within 0.0004 of the arm with the best true list quality there; the intervals reach ±0.0018. A server with no ground truth can tell which memory has been working for a log, but a fixed memory is not a good rule from 100 people up (point 3).
5. **The table prior helps a new log in one region, and needs truth.** At 30 people and monthly rounds it beats plain by 0.0046 (17 of 24 logs better) and is still worse than the seeded selector (0.0058 against 0.0030 regret); at weekly rounds it does nothing at 30 people and costs nothing from 100 people up. It needs other logs with known truth, so it is a research table, not something a server can build for itself. G5 passed by 0.0006.
6. **Fixed share does nothing here** (G7), with the settings tuned on the development windows (η = 3, α = 0.05); the self-scores of neighbouring memories differ by too little for the weights to move. This is a statement about these settings, not about the method in general.
7. **Do not put more weight on the history** (G11). At monthly rounds from 100 people, 104 pseudo-rounds are worse than 38 by 0.0015, and with 104 the non-inferiority test G3 would not have passed (interval upper bound +0.0042). The registered improvement is shown for 38 pseudo-rounds only.
8. **Domains differ (post hoc).** On the 18 Amazon logs the seeded selector is better in all four regions (−0.0027, −0.0099, −0.0006, −0.0027); on the 6 other logs it is neither better nor clearly different at 30 people (+0.0004, +0.0001) and it is worse at monthly rounds from 100 people (+0.0053, interval +0.0005 to +0.0102). The registered G3 is an average over the 24 logs, where 18 Amazon logs outweigh 6 others; it hides that harm. A likely reading is that when a log's best memory changes between eras the prior points the wrong way. That reading was not tested: doc 15 found the best memory only a little less stable on the other logs (the same half-life in 73% of cells against 78% on Amazon, 7-day rounds), and six logs are few.
9. **The development gains shrank on fresh data** by 27%, 43% and 47% (G1, G2, G5): the usual shrinkage of tuning in-sample. The registered thresholds were set with that margin and still passed.
10. **Time order (post hoc).** The history windows lie *later* in time than the third window for 21 of the 24 logs; only 3 logs (amazon-baby-products, amazon-video-games, ml-latest) have both history windows strictly earlier. On those 3 the seeded selector is better than plain in all four regions (−0.0022, −0.0052, −0.0004, −0.0028), but 3 logs support no claim, and the gate in the registry (at least 8 causal logs in every region) is closed. In development (370 cells, in-sample) the gain was at least as large when the history was earlier as when it was later (30 people monthly: −0.0121 earlier against −0.0082 later), which is consistent with stationarity but not a test of forecasting.

## 4. Post-hoc extras (not registered)

Written by `scripts/ssp/explore_guard_extra.py`; tables `posthoc_warm_by_n.csv`, `posthoc_warm_by_group.csv`, `posthoc_warm_vs_oracle.csv`, `posthoc_warm_other_log.csv`, `posthoc_causal.csv`.

1. **Seeded minus plain by group size (24 / 24 / 17 / 6 logs; share of logs where seeding is better).**

   | Round | 30 people | 100 | 300 | 1,000 |
   |---|---:|---:|---:|---:|
   | weekly | −0.0019 (−0.0027 to −0.0011); 79% | −0.0007 (−0.0010 to −0.0004); 83% | −0.0004 (−0.0007 to −0.0001); 88% | −0.0005 (−0.0009 to −0.0002); 6 of 6 |
   | monthly | −0.0074 (−0.0100 to −0.0049); 92% | −0.0022 (−0.0044 to +0.0004); 75% | −0.0010 (−0.0021 to +0.0002); 65% | +0.0002 (−0.0016 to +0.0023); 67% |

2. **Seeded minus plain by log group.**

   | Region | 18 Amazon logs | 6 other logs |
   |---|---:|---:|
   | 30 people, weekly | −0.0027 (−0.0034 to −0.0018) | +0.0004 (−0.0001 to +0.0011) |
   | 30 people, monthly | −0.0099 (−0.0125 to −0.0076) | +0.0001 (−0.0017 to +0.0026) |
   | 100+ people, weekly | −0.0006 (−0.0009 to −0.0003) | −0.0004 (−0.0008 to +0.0001) |
   | 100+ people, monthly | −0.0027 (−0.0038 to −0.0016) | **+0.0053 (+0.0005 to +0.0102)** |

3. **Oracle fixed memory minus seeded selector (positive: seeded is better; the oracle is chosen with the truth of the other windows).** 30 people weekly +0.0022 (−0.00003 to +0.0047); 30 people monthly +0.0017 (−0.0008 to +0.0053); 100+ weekly +0.0015 (+0.0003 to +0.0030); 100+ monthly +0.0032 (+0.0014 to +0.0052).
4. **Reference table of the same window.** The seeded selector beats a table fitted on the other logs' third windows by 0.0099 at 30 people weekly (−0.0149 to −0.0050) and 0.0104 at 100+ people weekly (−0.0150 to −0.0063); at monthly rounds the differences include 0 (−0.0013 and −0.0056).
5. **Other-log prior against own-history prior.** `warm_other` seeds the selector with the mean self-score per arm over the other logs' w1 and w2 cells at the same group size and round length (same group of logs when at least 3 exist), 38 pseudo-rounds, no truth and no data from the log itself.

   | Region | other-log prior minus plain | own history minus plain | other-log minus own history |
   |---|---:|---:|---:|
   | 30 people, weekly | −0.0017 (−0.0026 to −0.0008) | −0.0019 (−0.0027 to −0.0011) | +0.0002 (−0.0002 to +0.0006) |
   | 30 people, monthly | −0.0060 (−0.0089 to −0.0032) | −0.0074 (−0.0100 to −0.0049) | +0.0013 (−0.0004 to +0.0032) |
   | 100+ people, weekly | +0.0004 (−0.0001 to +0.0009) | −0.0006 (−0.0008 to −0.0003) | +0.0009 (+0.0005 to +0.0014) |
   | 100+ people, monthly | +0.0008 (−0.0012 to +0.0027) | −0.0007 (−0.0026 to +0.0015) | +0.0015 (−0.0013 to +0.0042) |

## 5. Limits and disclosures

1. **Not an independent sample of the world.** The same 24 of the 27 logs as before, windows next to the earlier ones, simulated random reporters; intervals over logs overstate independence. The six non-Amazon logs make the group findings fragile. **Erratum:** the registry's caveat text says "20 are Amazon"; the 24-log set has 18 Amazon logs and 6 others (the file is hash-frozen and was not edited).
2. **"Deployable" is not claimed.** The history windows are other eras of the same log, mostly *later* than the cell; this tests whether the best memory is stable across eras, not forecasting from the past. A causal version needs logs with three windows in time order; only 3 qualify, so the registered gate stayed closed. Collecting more logs would need a download, which needs your approval (doc 90).
3. **The development numbers were in-sample**, known when the thresholds were set, including the split by time order.
4. **Some tests are weak by construction.** G6, G7, G9 and G11 use means without a confidence bound; for G9 the intervals reach ±0.0018, so "almost as well" is not shown to be within 0.001. G10 is over cells, not logs. G4 can only show "not better", and in the region n30_28d its interval lower bound (−0.0008) is close to its limit and the two rules are not separable.
5. **Before the freeze,** a probe (not shown here) computed which logs have a third window, the coverage of truth rows and the time order of the windows, and decided the 24-log set; the development numbers and the split by time order were known when the thresholds were set. The registry's literature note (fixed share, forecast combination) was background only; forecast combination was not tested.
6. **Fixed share is the uniform-mixing variant**, not the exact update of Herbster & Warmuth 1998 (doc 90, section 6, item 4).
7. **The group split and the other-log prior are post hoc,** found after the verdicts, on the same 24 logs; the six non-Amazon logs, in particular, are too few for a firm statement.
8. **Reviews are same-provider subagents, not an outside audit.** An Opus reviewer rebuilt all 142 cells and every rule (plain, two, warm, warm104, ownself, own_oracle, table_only, margin_table, fixedshare) from the raw rows without importing the project code and matched every per-cell regret (to 5·10⁻⁷, the CSV rounding) and every paired mean and bootstrap interval exactly; it found no leakage of third-window data into any prior, confirmed 140 cells with two history windows and 2 with one (ml-latest at 300 people), recomputed the 3 causal logs, and checked the freeze and the stored hashes. The analysis was re-run once more and its tables are byte-identical. A second Opus critic read this document against the registry and the tables and asked for the wording changes now in sections 0 and 3: "other eras, mostly later" instead of "past", the group split and the time-order caveat in the summary, G4 and G8 stated as "not better" and "not confirmed", and the other-log prior check of section 4.5.
9. **No new material** beyond the repository's own data and code (doc 90).

## 6. What comes next

1. **Recipe (for simulated use; not a deployment claim):** at 30 people, seed the selector with the self-scores of other logs at the same group size and round length (or, with a log's own earlier windows, with its own); keep 38 pseudo-rounds. From 100 people up the plain selector is already good; on the 18 Amazon logs the log's own history gains 0.0006 (weekly) and 0.0027 (monthly) on top of it, on the 6 other logs it loses 0.0053 at monthly rounds. The table prior (needs truth) is only worth it at 30 people and monthly rounds when no self-score history exists. Untested: earlier-only history beyond 3 logs; the 6 non-Amazon logs at 30 people (no gain) and at monthly rounds from 100 people (a loss).
2. **Programme P** (planned): cut the price of privacy by bounding each person's contribution (doc 16, section 6).
3. **Programme V** (planned): correct volunteer bias with inverse-probability weights (doc 16, section 6).
4. **A causal test of the seeded selector** needs logs with at least three 156-week windows in time order; only 3 of the 24 qualify.
5. **Forecast combination** (blending the lists of the memories, weighted by recent self-scores) was left for its own registry.

## 7. Files

1. Plan and code: `configs/selfsel_guard_registry.json`, `scripts/ssp/window3_core.py`, `run_guard.py`, `guard_core.py`, `tune_guard.py`, `an_guard.py`; post hoc: `explore_guard_extra.py`.
2. Tables: `results/small_sample_popularity/03_tables/guard/` (`g_verdicts.json` with the file hashes, `g_cells.csv`, `g_summary.csv`, `g_paired.csv`, `g_dev_grid.csv` (development, in-sample); post hoc: `posthoc_causal.csv`, `posthoc_warm_by_n.csv`, `posthoc_warm_by_group.csv`, `posthoc_warm_vs_oracle.csv`, `posthoc_warm_other_log.csv`).
3. Raw rows (not committed): `results/small_sample_popularity/02_runs/guard.jsonl` (142 rows), `guard_plan.json`.
4. Rerun: `nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/run_guard.py --workers 3`, then `.venv/bin/python3 -W ignore scripts/ssp/an_guard.py` and `explore_guard_extra.py`.
