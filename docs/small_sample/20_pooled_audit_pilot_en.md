# 20. If the audit's people also enter the list, are volunteers worth recruiting? (development pilot, not a test)

Chinese version: [`20_pooled_audit_pilot_zh.md`](20_pooled_audit_pilot_zh.md). Follows doc 19 (section 4, point 9 and section 6, point 2). **Status: development stage on window w1. Nothing was registered and nothing here is a verdict; the numbers decide whether, and how, a registered test is written (section 5).** Same-provider reviews only; no download, install or execution beyond the repository's own code.

## 0. Summary

1. **Pooling closes most of the equal-head-count gap.** In doc 19 the audit of 50 random people only scored the lists, and 100 heavy volunteers fell short of 150 random reporters. If the audit's people also enter the list (pooled with the volunteers), the shortfall on window w1 (27 logs) falls from 0.0241 to **0.0037** for the lists a server can choose (audit-selected weights and memory: −0.0037, 95% interval −0.0068 to −0.0009; positive in 8 of 27 logs, within −0.005 in 18 of 27). With the best fixed arm chosen in hindsight the two designs tie (+0.0005, −0.0026 to +0.0033). Pooling adds +0.0204 (+0.0162 to +0.0254; 27 of 27 logs) over the audit-scoring-only design.
2. **Where the gain comes from.** A pooled list without weights is 0.0103 to 0.0114 below 150 random reporters; weights and memory chosen on the audit add +0.0077 (+0.0039 to +0.0118; 22 of 27 logs).
3. **On window w1 the break-even cost of a volunteer rises from about 0.6 to about 0.9 of a random reporter.** Break-even = the cost of one volunteer, in units of one random reporter, at which volunteers plus the audit equal an all-random sample of the same total cost (an audit person costs the same as a random reporter). On the line through 100 and 150 random reporters (27 logs): 0.615 (0.529 to 0.692) for the audit-scoring-only design and **0.941 (0.895 to 0.986)** pooled. With more volunteers (22 logs, random reporters up to 400): the pooled list beats 150 random reporters by +0.0173 at a cost ratio of 0.67 (150 volunteers; 21 of 22 logs), +0.0310 at 0.50 (200; 22 of 22) and +0.0484 at 0.33 (300; 22 of 22); the break-even is 0.89 to 0.97 (interpolated, audit-selected; lowest at 300 volunteers).
4. **Reading (development window, simulated heavy volunteers).** If a volunteer costs less than about 0.85 of a random reporter and a random audit of 50 is affordable, recruiting heavy volunteers and pooling the audit beat asking only random people in this simulation; at equal cost per head the two are about even. It is not shown for light volunteers, for volunteers who differ in taste, or on a window that was not used to choose the design.

## 1. What was done

1. **Question.** Doc 19, section 4, point 9 (post hoc): 100 heavy volunteers plus a random audit of 50, the audit only scoring the lists, score 0.626 against 0.649 for 150 random reporters (−0.0203, 0 of 23 logs positive); a pooled list was not tested. Here the audit's people also enter the list.
2. **The pooled list** (`vpool:a=1,w=B,na=50`, `scripts/ssp/pool_core.py`, patched into the engine without editing `deep_core.py` or `weight_core.py`). Each round: the volunteers of `vpps:a=1` (the same people as the programme-V arms: chance of reporting proportional to the person's rows that week, an oracle rule), their reports weighted by rows^(−B); plus the 50 random active people of the engine's audit sample (the same people, unweighted counts; the sampler rebuilds them from the seed, so its `na` must equal the cell's `audit_n`, one constant `AUDIT` = 50 in `tune_pool.py`). The round's share is λ · share(volunteers) + (1 − λ) · share(audit), λ = NV / (NV + 50) (the head-count share, 100/150 at 100 volunteers; fixed in advance, not tuned). The weight exponent B is on programme V's grid (−1, −0.5, 0, 0.5, 1) and is chosen, together with the memory (the days arms), by the audit's self-score as in doc 19.
3. **Order.** The engine scores a list against the next round's sample before the list is updated with it, so the audit people of round t score the list built up to t − 1 and enter it only afterwards; no list is scored on people it contains. The truth is the full next round, for every arm alike.
4. **Cells.** Window w1 (the weekly-study windows: the densest 156 weeks of each log; the test windows of programmes P and V are not used), 7-day rounds, the 27 logs of stage a at 100 people, the 8 draws of the stored cell. 100 heavy volunteers + audit of 50 in all 27 logs; with the grid, 150, 200 and 300 volunteers (audit still 50) and 200, 300 and 400 random reporters where the log's median weekly active people are at least three times the number of people (22 logs have every cell, median weekly active ≥ 1,200). 205 cells in all, 49 arms in a pooled cell.
5. **Comparators** (values as in docs 14–19: the best fixed arm in hindsight, the reporter selector, the audit selector; steady steps, mean over draws, the log is the unit): 100 and 150 random reporters (the reporter selector of docs 14–15), the audit-scoring-only design of programme V (read from its development cells, same volunteers) and the pooled list. "Audit-selected" means weights and memory chosen without the truth.
6. **Checks.** A design critic (Opus, same provider) read the code before the results were seen and found no leakage; it asked for the overlap of the two draws, the weight-free pooled value, splits by log family and size, and a go rule (all applied). `scripts/ssp/check_pool.py` compares the pooled share with independently built volunteer and audit samples (480 rounds on each of two logs, no mismatch); the std arms of every pooled cell equal the random cell of that size exactly (largest difference 0); every table was re-run twice with byte-identical output (the first 16 hex digits of each sha256 are in section 6).

## 2. Results

Mean over the 27 logs (a log is the unit). Selector values are those a server could reach without the truth.

| Design | Selector | Best fixed arm in hindsight |
|---|---:|---:|
| 100 random reporters | 0.5993 | 0.6018 |
| 150 random reporters | 0.6307 | 0.6322 |
| 100 heavy volunteers, uncorrected (reporter selector) | 0.5872 | |
| 100 heavy volunteers + audit of 50 that only scores (programme V) | 0.6065 | 0.6106 |
| pooled, no weights (B = 0; reporter selector; audit selector) | 0.6204; 0.6193 | 0.6225 |
| pooled, audit-selected weights and memory | **0.6270** | **0.6326** |

Paired differences (mean over logs; 95% bootstrap interval over logs, seed 7; logs positive; logs within −0.005):

| Contrast | Selector | Hindsight |
|---|---|---|
| pooled − 150 random reporters | −0.0037 (−0.0068 to −0.0009); 8 of 27; 18 of 27 | +0.0005 (−0.0026 to +0.0033); 17 of 27; 22 of 27 |
| pooled − 100 random reporters | +0.0277 (+0.0225 to +0.0328); 27 of 27 | +0.0308 (+0.0253 to +0.0363); 27 of 27 |
| pooled − audit scores only | +0.0204 (+0.0162 to +0.0254); 27 of 27 | |
| audit scores only − 150 random reporters | −0.0241 (−0.0312 to −0.0181); 0 of 27 | |
| pooled, no weights − 150 random reporters | −0.0103 (−0.0173 to −0.0041); 8 of 27 (reporter selector); −0.0114 (−0.0180 to −0.0055); 5 of 27 (audit selector) | |
| audit-selected − no weights, pooled | +0.0077 (+0.0039 to +0.0118); 22 of 27 | |

By log family and size (pooled − 150 random reporters; pooled − audit scores only): 20 Amazon logs −0.0042 (−0.0081 to −0.0006) and +0.0233; 7 other logs −0.0022 (−0.0060 to +0.0011) and +0.0123; 14 logs with at least 4,550 median weekly active people −0.0041 (−0.0090 to +0.0004) and +0.0258; 13 logs below −0.0033 (−0.0071 to −0.0002) and +0.0146. No steady round of any log has 150 or fewer active people (no census round).

More volunteers and the break-even (22 logs; the random-reporter curve of the selector value is 0.6029, 0.6365, 0.6559, 0.6825, 0.6972 at 100, 150, 200, 300, 400 people):

| Volunteers (+ audit of 50) | Cost ratio at the budget of 150 random reporters | Pooled − 150 random reporters | Logs positive | Break-even cost ratio (audit-selected; hindsight) |
|---:|---:|---|---:|---|
| 100 | 1.00 | −0.0030 (−0.0062 to +0.0000) | 8 of 22 | 0.955 (0.909 to 1.000); 1.044 |
| 150 | 0.67 | +0.0173 (+0.0135 to +0.0210) | 21 of 22 | 0.964 (0.907 to 1.023); 1.076 |
| 200 | 0.50 | +0.0310 (+0.0252 to +0.0366) | 22 of 22 | 0.969 (0.913 to 1.021); 1.065 |
| 300 | 0.33 | +0.0484 (+0.0399 to +0.0560) | 22 of 22 | 0.888 (0.844 to 0.934); 1.042 |

The two draws are independent, so a person can be both a volunteer and an audit person in a round: at 100 volunteers 2.17 people per round on average over the 27 logs (largest 8.08), so the head count is about 148, not 150. The overlap grows with the number of volunteers (a reviewer's replay of one log, beeradvocate: 3.37 people at 100, 5.19 at 150 and 10.30 at 300 volunteers).

## 3. Interpretation

1. **Why pooling matters.** In doc 19's design the 50 audit people spend their reports on scoring and none enters a list, so 100 volunteers compete with 150 random reporters while carrying 50 random reporters' worth of cost. Pooled, the 150 heads all contribute; the volunteers send more rows per head (about 3 times as many, 8.1 against 2.7 rows per reporter at 100 people on w3, doc 19 section 4, point 4; a reviewer counted 8.35 against 2.51 on w1), the audit people are unbiased, and the mixture lands within 0.004 of 150 random reporters.
2. **Weights and memory still help on top of pooling.** The pooled list without weights is 0.010 to 0.011 short; the audit's choice of weight exponent and memory recovers +0.0077 of it (22 of 27 logs). So the audit has two jobs: it supplies unbiased people and it picks the correction.
3. **Equal cost per head is about even; cheaper volunteers win.** At a cost ratio of 1 the pooled design is 0.003 to 0.004 short (the interval ends at 0 for the 22 logs and below 0 for the 27) and ties in hindsight. Each tested increase in volunteers (150, 200, 300), bought at a lower cost per head, adds value: +0.017, +0.031, +0.048 against 150 random reporters at 0.67, 0.50 and 0.33 of the cost. The break-even is 0.89 to 0.97 of a random reporter's cost on interpolation (hindsight 1.04 to 1.08).
4. **Compared with doc 19.** On the same two-point interpolation the audit-scoring-only design breaks even at 0.615, as doc 19 found on w3 (about 0.6). The pooled list moves it to 0.941.

## 4. Limits and disclosures

1. **A development look, not a test.** Window w1 is where programme V's weights were tuned; the audit-scoring-only numbers are V's development cells. In doc 19 the test gains were 60–66% of the development gains for the audit-weighted selector; the pooled-minus-V difference and the break-even may shrink on a window that was not used to choose the design. No threshold was registered, and the go rule (mean ≥ 0 and at least two thirds of the logs within −0.005) is not met on the mean for 100 volunteers (−0.0037) and met on the share (18 of 27).
2. **Looked at in steps.** The 54 base cells (100 volunteers, 150 random) were run and read before the grid (150, 200, 300 volunteers; 200, 300, 400 random reporters) and before the overlap, family, size and no-weights contrasts were added; the pooling rule (λ) and the weight grid were fixed before any cell was run.
3. **Best case for volunteers.** The heavy rule is an oracle (the chance of reporting uses the same week's rows) and the weight family rows^(−B) is nearly exactly right for it (doc 19, section 5, point 4). Light volunteers, volunteers who differ in taste, a feasible rule (previous week's rows) and person-weighted popularity were not run.
4. **The cost model.** Cost per head; an audit person costs the same as a random reporter; a volunteer's cost does not depend on the rows sent; the audit is random and every audit person answers (no non-response). Nothing here says what a volunteer costs in practice.
5. **Interpolation.** The break-even reads a random head count off the mean curve by linear interpolation between 100, 150, 200, 300 and 400 people. The curve is concave, so the interpolation overstates the random head count that matches a given value and with it the break-even cost ratio, in favour of volunteers; the two-point 0.941 is overstated the same way. The 22-log grid is a different set of logs from the 27 of the base contrasts (median weekly active ≥ 1,200 against all 27); its equal-head-count difference is −0.0030, the base −0.0037. At 300 volunteers the chances of being drawn saturate (doc 19, section 5, point 4).
6. **Overlap.** A person can be drawn as a volunteer and as an audit person in the same round (2.17 people per round on average at 100 volunteers, largest 8.08, more with more volunteers); the pooled list counts such a person twice. This works against the pooled list and is part of the design.
7. **`rows_per_round` of a pooled arm is meaningless** (the round's share sums to 1); the rows per head quoted in section 3 come from doc 19 and a reviewer's count.
8. **Logs and windows.** The same 27 logs as docs 13–19 (20 Amazon, 7 other); intervals over logs treat logs as independent and overstate independence. 8 draws per cell.
9. **Same-provider reviews, not an outside audit.** Before the results, the design critic (section 1, point 6). After the results, a first Opus reviewer stalled (no output in one hour, the runtime limit) and the review was run again as two smaller ones: one re-derived every per-log value, paired difference and interval, the 22-log grid, the random curve and the break-even ratios from the raw rows with a separate implementation (all matched to within rounding); the other re-derived that the std arms of every pooled cell equal the random cell of the same size (and the programme-V cells at 100 volunteers) exactly, checked the seeds, the order and the normalisation in `pool_core.py`, and replayed the two draws with the same seeds for 5 logs (the overlap table matched exactly). Their wording points are applied here (the break-even range and the 0.85, the volunteer counts tested, the direction of the interpolation bias, the growth of the overlap, λ for the grid cells, the saved check, the window label); the overlap replay also showed that `overlap()` counted the last round, which the cells never draw (2.18 → 2.17, 8.09 → 8.08; fixed).
10. **A governor with the higher limits** of `configs/ssp_quiet.json` was used (5 workers); it decides only when workers run. **No new material** (doc 90); the ledger check passed.

## 5. What next: a registered test

From the critic and from the numbers above (a proposal; nothing is registered):

1. **Window.** One not yet seen for these arms. w2 has not been used for any weighted or pooled arm (doc 16 used it for plain volunteers, doc 18 for privacy); w3 was used for programme V's test and its post hoc equal-cost cells.
2. **Primary comparison.** Audit-selected pooled list minus 150 random reporters with the selector of docs 14–15, paired by log, at cost ratios 1, 2/3, 1/2 and 1/3 (100, 150, 200, 300 volunteers, audit 50); thresholds taken from section 2 at about half the development differences, a margin for equal cost per head, and the log as the unit.
3. **Secondary.** Pooled minus the audit-scoring-only design; weights against no weights; λ chosen by the audit on a fixed grid (1/2, 2/3, 4/5, 1; λ = 1 is the audit-scoring-only design); light volunteers and random volunteers (a safety check) pooled; a feasible volunteer rule.
4. **Report.** The distinct head count, splits by log family and size, every planned contrast, the break-even with its interval, and the logs positive.

## 6. Files

1. Code: `scripts/ssp/pool_core.py` (the `vpool` sampler, a patch on top of `weight_core.py`), `scripts/ssp/tune_pool.py` (cells, report, overlap), `scripts/ssp/check_pool.py` (the equality check of the pooled share); engine and V code unchanged.
2. Tables in `results/small_sample_popularity/03_tables/volunteer/`: `pilot_pool_per_log.csv` (27 logs, every value; sha256 `faf463755b5122d8…`), `pilot_pool_grid.csv` (22 logs, pooled values and the random curve; `8bdeabe53d689d77…`), `pilot_pool_summary.csv` (every contrast and break-even above; `d59184a72349b245…`), `pilot_pool_overlap.csv` (`380f82563e0d0881…`). Raw rows (not committed): `results/small_sample_popularity/02_runs/pool_dev.jsonl` (205 rows).
3. Rerun: `.venv/bin/python3 -W ignore scripts/ssp/check_pool.py`; `nice -n 10 caffeinate -i .venv/bin/python3 -W ignore scripts/ssp/tune_pool.py --run --grid --workers 5`, then `.venv/bin/python3 -W ignore scripts/ssp/tune_pool.py --report --grid` and `--overlap`; the programme-V development rows `02_runs/vol_dev.jsonl` and the weekly rows `weekly.jsonl` are inputs.
