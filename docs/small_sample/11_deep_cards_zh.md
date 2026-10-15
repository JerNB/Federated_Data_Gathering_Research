# 11. 深度研究计划：假设卡片（自动生成）

每个假设一张卡片，由 `scripts/ssp/an_deep.py` 根据冻结的注册表和运行结果生成。表格见 `results/small_sample_popularity/03_tables/deep/`。

**完整性：** 所有阶段的所有计划格子都已完成。

## 成绩单

| 编号 | 假设 | 结果 |
| --- | --- | --- |
| D01 | 带阻尼的趋势项 | 没有差别；预测不成立 |
| D02 | 截断单轮尖峰 | 无法判断；预测不成立 |
| D03 | 倒数排名聚合 | 变差；预测成立 |
| D04 | 对新物品更快遗忘 | 变差；预测成立 |
| D05 | 更细的半衰期网格 | 有小幅下降；预测成立 |
| D06 | 榜单跳变时加快遗忘 | 没有差别；预测成立 |
| D07 | 多个半衰期的集成 | 变差；预测不成立 |
| D08 | 让榜单保持稳定 | 有小幅提升；预测不成立 |
| D09 | 为上升物品预留名额 | 没有差别；预测成立 |
| D10 | 去年同一季节 | 有提升；预测成立 |
| D11 | 轮换小组 | 没有差别；预测成立 |
| D12 | 按活跃度比例抽人 | 有提升；预测成立；已确认 |
| D13 | 按轮的样本量加权 | 有提升；预测成立；已确认 |
| D14 | 样本能给定期普查再加分吗 | 变差；预测成立 |
| D15 | 样本追平定期普查所需的人数 | 预测不成立 |
| D16 | 中心化差分隐私的代价表 | 预测不成立 |
| D17 | 本地差分隐私的代价表 | 预测成立 |
| D18 | 在计数阈值下能发布多长的榜单 | 预测不成立 |
| D19 | 等效年龄（重复检验） | 预测不成立 |
| D20 | 最佳半衰期随榜单长度的变化 | 预测成立 |
| D21 | 达到上限 95% 所需的人数 | 无法检验 |
| D22 | 冷启动需要几轮 | 预测不成立 |
| D23 | 糟糕的轮次有多糟 | 预测不成立 |
| D24 | 服务器自己的评分能反映真实质量吗 | 预测成立 |
| D25 | 榜单变动幅度能说明榜单有多旧吗 | 预测成立 |
| D26 | 半样本一致度作为质量指标 | 预测不成立 |
| D27 | 最佳半衰期（按天）有多平坦 | 预测不成立 |
| D28 | 非 Amazon 日志上日历半衰期稳定吗 | 预测成立 |
| D29 | 其他类型日志上的自调半衰期 | 预测不成立 |
| D30 | 结论是否依赖评价指标 | 预测不成立 |

## 卡片

### D01. 带阻尼的趋势项（A combine rounds）

- **问题。** Holt 阻尼趋势平滑（份额的水平加上带阻尼的趋势）给出的榜单比衰减累加更好。
- **事先注册的预测（英文原文）。** No gain: |mean diff| <= 0.005 for the damped primary at n=100; undamped (p=1.0) loses at least 0.02.
- **文献依据（英文原文）。** Gardner 2006: damped trend hard to beat on average, but a trend term is a smoothed difference of noisy levels (variance); local drift slope median 0.72 and H2 (rising items) lost 0.039
- **结果。** 没有差别；预测不成立

主规则 `holt:a=0.3,b=0.1,p=0.9` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0022（95% 区间 -0.0071 到 +0.0027）；62% 的日志更好。Amazon 日志 +0.0023，其他日志 -0.0145。通俗结论：没有差别。事先注册的检验：未确认（发现用日志 -0.0030，没有通过错误发现率筛选；留出日志 -0.0017）。 预测检查：damped -0.0022 (upper bound +0.0027); undamped -0.00328; rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 4 个方向相同。

### D02. 截断单轮尖峰（A combine rounds）

- **问题。** 把每轮中每个物品的计数截断在 3 条再计算份额，榜单更好。
- **事先注册的预测（英文原文）。** No difference: |mean diff| < 0.003.
- **文献依据（英文原文）。** Gardner 2006 s4.1 (trimmed smoothing helped on daily supermarket series); our caps of 30 rows lost 0.0006, so little heavy-tail damage is left
- **结果。** 无法判断；预测不成立

主规则 `clip:c=3` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0161（95% 区间 -0.0353 到 +0.0031）；35% 的日志更好。Amazon 日志 -0.0008，其他日志 -0.0575。通俗结论：无法判断。事先注册的检验：未确认（发现用日志 -0.0212，没有通过错误发现率筛选；留出日志 -0.0123）。 预测检查：mean diff -0.0161, 95% interval -0.0353 to +0.0031 (needs inside +/-0.003); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 7 个方向相同。

### D03. 倒数排名聚合（A combine rounds）

- **问题。** 把每轮的倒数排名分数 1/(60+排名) 按半衰期 2 累加，比累加份额得到更好的榜单。
- **事先注册的预测（英文原文）。** No gain: the 95% interval of the mean diff excludes +0.01. (A one-log smoke test first showed +0.043; the design review found that it came from breaking ties by item id, which follows item age; with random tie-breaks the same smoke shows -0.026, so the prediction was reset to no gain before the real run.)
- **文献依据（英文原文）。** Cormack et al. 2009 RRF k=60 beat score fusion by 4-5% MAP in retrieval; Renda and Straccia 2003 found score sums beat Borda in metasearch
- **结果。** 变差；预测成立

主规则 `rrf:k=60` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0218（95% 区间 -0.0285 到 -0.0150）；8% 的日志更好。Amazon 日志 -0.0212，其他日志 -0.0232。通俗结论：变差。事先注册的检验：未确认（发现用日志 -0.0209，没有通过错误发现率筛选；留出日志 -0.0221）。 预测检查：mean -0.0218, upper 95% bound -0.0150 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 7 个方向相同。

### D04. 对新物品更快遗忘（A combine rounds）

- **问题。** 首次出现不到 4 轮的物品用半衰期 1，更老的物品用半衰期 4，比所有物品用同一半衰期更好。
- **事先注册的预测（英文原文）。** No gain (likely a loss): mean diff <= +0.005.
- **文献依据（英文原文）。** [INFERENCE] earlier tournament killed every re-weighting of the baseline; item-age mixing splices two score scales
- **结果。** 变差；预测成立

主规则 `twospeed:hf=1,hs=4,age=4` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0595（95% 区间 -0.0782 到 -0.0408）；8% 的日志更好。Amazon 日志 -0.0629，其他日志 -0.0503。通俗结论：变差。事先注册的检验：未确认（发现用日志 -0.0635，没有通过错误发现率筛选；留出日志 -0.0566）。 预测检查：mean -0.0595 (needs <= +0.005), upper 95% bound -0.0408 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 7 个方向相同。

### D05. 更细的半衰期网格（A combine rounds）

- **问题。** 半衰期 3 轮比基线半衰期 2 轮更好（早先的网格每档差 2 倍，真正的最优值可能落在两档之间）。
- **事先注册的预测（英文原文）。** No gain: the 95% interval of the mean diff excludes +0.01 (the optimum is flat around 2 rounds). Replaces an empirical-Bayes arm that the design review showed is algebraically a longer half-life.
- **文献依据（英文原文）。** Exponential smoothing practice: a flat optimum; earlier tracking study found best half-life 2 in 13 of 14 Amazon logs at the factor-2 grid
- **结果。** 有小幅下降；预测成立

主规则 `decay:h=3` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0087（95% 区间 -0.0174 到 -0.0000）；31% 的日志更好。Amazon 日志 -0.0138，其他日志 +0.0051。通俗结论：有小幅下降。事先注册的检验：未确认（发现用日志 -0.0108，没有通过错误发现率筛选；留出日志 -0.0072）。 预测检查：mean -0.0087, upper 95% bound -0.0000 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 5 个方向相同。

### D06. 榜单跳变时加快遗忘（A combine rounds）

- **问题。** 当某一轮份额与追踪器的差异超过平常变化的 1.5 倍时，把追踪器的记忆减半，榜单更好。
- **事先注册的预测（英文原文）。** No gain: mean diff <= +0.005 (the detector rarely fires under slow drift).
- **文献依据（英文原文）。** ADWIN (Bifet and Gavalda 2007); Gama et al. 2014 survey: detectors pay off under abrupt change, our measured drift slope is 0.2-1.26 (gradual)
- **结果。** 没有差别；预测成立

主规则 `drift:z=1.5,f=0.5` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 +0.0008（95% 区间 -0.0001 到 +0.0018）；12% 的日志更好。Amazon 日志 +0.0007，其他日志 +0.0010。通俗结论：没有差别。事先注册的检验：未确认（发现用日志 +0.0000，没有通过错误发现率筛选；留出日志 +0.0014）。 预测检查：mean +0.0008 (needs <= +0.005), upper 95% bound +0.0018 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 7 个方向相同。

### D07. 多个半衰期的集成（A combine rounds）

- **问题。** 把半衰期 1、2、4 的归一化衰减榜单相加，比只用半衰期 2 更好。
- **事先注册的预测（英文原文）。** No gain on Amazon logs (mean diff <= +0.005); a gain of at least +0.005 on the non-Amazon logs where the best half-life is not 2.
- **文献依据（英文原文）。** Brown 2008 / forecast-combination literature: combining smoothers helps mainly when the best parameter is unknown
- **结果。** 变差；预测不成立

主规则 `ens` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0103（95% 区间 -0.0142 到 -0.0065）；12% 的日志更好。Amazon 日志 -0.0112，其他日志 -0.0080。通俗结论：变差。事先注册的检验：未确认（发现用日志 -0.0102，没有通过错误发现率筛选；留出日志 -0.0105）。 预测检查：Amazon -0.0112; other -0.0080; rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 7 个方向相同。

### D08. 让榜单保持稳定（A combine rounds）

- **问题。** 给上一期榜单上的物品加 10% 的分数加成，使榜单变动至少减少 30%，而 NDCG 损失小于 0.005。
- **事先注册的预测（英文原文）。** Churn falls by >= 30% (relative) and NDCG diff >= -0.005.
- **文献依据（英文原文）。** Hysteresis / stability regularisation in top-k maintenance; exact null on NDCG alone is expected
- **结果。** 有小幅提升；预测不成立

主规则 `hyst:d=0.1` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 +0.0024（95% 区间 +0.0009 到 +0.0038）；69% 的日志更好。Amazon 日志 +0.0022，其他日志 +0.0028。通俗结论：有小幅提升。 预测检查：discovery: churn cut 0.142, NDCG lower bound -0.0006; holdout: churn cut 0.109, NDCG lower bound +0.0007 (needs cut >= 0.30 and lower bound >= -0.005 on both); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 4 个方向相同。

### D09. 为上升物品预留名额（A combine rounds）

- **问题。** 把榜单最后 10 个名额换成近期份额（半衰期 1）最超出旧份额（半衰期 4）的物品，榜单更好。
- **事先注册的预测（英文原文）。** No gain: mean diff <= +0.005 (reserved slots cost ranks 91-100 more than rising items earn).
- **文献依据（英文原文）。** Bandit-style exploration slots; tournament round 1 (H2) found extra weight for rising items cost 0.039
- **结果。** 没有差别；预测成立

主规则 `reserve:r=10` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0009（95% 区间 -0.0024 到 +0.0006）；38% 的日志更好。Amazon 日志 -0.0004，其他日志 -0.0022。通俗结论：没有差别。事先注册的检验：未确认（发现用日志 -0.0000，没有通过错误发现率筛选；留出日志 -0.0015）。 预测检查：mean -0.0009 (needs <= +0.005), upper 95% bound +0.0006 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 5 个方向相同。

### D10. 去年同一季节（A combine rounds）

- **问题。** 在 90 轮（每轮约 3 个月）下，加入一年前同一季节的份额（权重 0.5），榜单更好。季节项在不到一半稳态步中起作用的格子不运行。
- **事先注册的预测（英文原文）。** Gain at 90 rounds: mean diff >= +0.01 on Amazon logs; no gain on non-Amazon logs. (Earlier year-ago-list test: year-ago list alone was worse in 16 of 16 categories; here it is only an added term.)
- **文献依据（英文原文）。** Seasonal naive / Holt-Winters; our S1 test (year-ago list alone, 0 of 16 better); Amazon December peaks
- **结果。** 有提升；预测成立

主规则 `seas:g=0.5` 对比 `decay:h=2`（NDCG@100，越高越好）：28 份日志的平均变化 +0.0142（95% 区间 +0.0056 到 +0.0228）；68% 的日志更好。Amazon 日志 +0.0179，其他日志 +0.0048。通俗结论：有提升。事先注册的检验：未确认（发现用日志 +0.0110，没有通过错误发现率筛选；留出日志 +0.0165）。 预测检查：Amazon +0.0179 (holdout Amazon +0.0197); other +0.0048; rows per round x1.00 and people x1.00 of the reference。在其他 5 个（人数、轮长）组合中，有 1 个方向相同。

### D11. 轮换小组（B whom to ask）

- **问题。** 保留上一轮仍然活跃的 75% 的汇报者（轮换率 0.25，空缺用新人补足），比完全更新的小组（轮换率 1.0，同一个抽样器）榜单更好。
- **事先注册的预测（英文原文）。** No gain, and a loss that grows with overlap: the 95% interval of the diff excludes +0.01; rho=0.1 worse than rho=0.25 in at least 60% of logs.
- **文献依据（英文原文）。** Cantwell 2008, Kish 1999: gain depends on within-unit correlation rho (about 0 for item-level popularity on review logs); plain decayed sum: fresh wins when drift is slow
- **结果。** 没有差别；预测成立

主规则 `decay:h=2@panel:rho=0.25` 对比 `decay:h=2@panel:rho=1.0`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0032（95% 区间 -0.0074 到 +0.0010）；46% 的日志更好。Amazon 日志 -0.0017，其他日志 -0.0073。通俗结论：没有差别。事先注册的检验：未确认（发现用日志 -0.0018，没有通过错误发现率筛选；留出日志 -0.0042）。 预测检查：rho=0.25 diff -0.0032 (upper bound +0.0010); rho=0.1 worse than 0.25 in 65% of logs; rows per round x1.08 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 7 个方向相同。

### D12. 按活跃度比例抽人（B whom to ask）

- **问题。** 按本轮行数成比例地抽 n 个汇报者（这是不可行的“先知”做法：服务器在提问前并不知道），比均匀抽样榜单更好；可行的变体使用上一轮的行数。
- **事先注册的预测（英文原文）。** Oracle arm: gain of at least +0.01 at n=100, but with about 1.8 times more rows collected per round (a one-log smoke showed +0.067 with 1.8x rows); the feasible arm (last round's rows) is secondary and expected to show no gain. Rows per round are reported next to every difference.
- **文献依据（英文原文）。** Hansen-Hurwitz / Horvitz-Thompson: with p proportional to size and an uncorrected pooled count the estimate is size-biased, but the target (row counts) is itself dominated by heavy users; the oracle also collects more rows per person asked (reported as rows per round)
- **结果。** 有提升；预测成立；已确认

主规则 `decay:h=2@pps` 对比 `decay:h=2`（NDCG@100，越高越好）：26 份日志的平均变化 +0.0361（95% 区间 +0.0213 到 +0.0509）；88% 的日志更好。Amazon 日志 +0.0476，其他日志 +0.0049。通俗结论：有提升。事先注册的检验：确认（发现用日志 +0.0355，留出日志 +0.0344，干净的重复检验集 +0.0142，6 份日志）。 预测检查：holdout label GAIN (holdout mean +0.0344); all-log label GAIN; rows per round x2.73 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 5 个方向相同。

### D13. 按轮的样本量加权（B whom to ask）

- **问题。** 当各轮汇报者人数不同时，按人数给各轮加权，比等权更好。
- **事先注册的预测（英文原文）。** Gain: mean diff >= +0.01 against equal weighting under the same variable round sizes (set after a one-log smoke showed +0.033).
- **文献依据（英文原文）。** Inverse-variance weighting (standard); variance of a round's share vector is proportional to 1/n_w
- **结果。** 有提升；预测成立；已确认

主规则 `ivw:g=1@varn` 对比 `decay:h=2@varn`（NDCG@100，越高越好）：26 份日志的平均变化 +0.0458（95% 区间 +0.0373 到 +0.0543）；100% 的日志更好。Amazon 日志 +0.0486，其他日志 +0.0381。通俗结论：有提升。事先注册的检验：确认（发现用日志 +0.0422，留出日志 +0.0467，干净的重复检验集 +0.0419，6 份日志）。 预测检查：holdout label GAIN (holdout mean +0.0467); all-log label GAIN; rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 7 个方向相同。

### D14. 样本能给定期普查再加分吗（G census plus samples）

- **问题。** 如果每 4 轮有一次全量统计，再加上每轮 n=100 的样本（普查的权重是一轮样本的 8 倍），比只用普查更好。
- **事先注册的预测（英文原文）。** No gain at n=100 (mean diff <= +0.005 against the stale census list); a gain at n=1000 would be reported as exploratory.
- **文献依据（英文原文）。** Composite / benchmark estimators need the sample to carry information the benchmark lacks; at n=100 a 4-round-old full count already beats the sample (eq. age 4.4 rounds)
- **结果。** 变差；预测成立

主规则 `cens:m=4,w=8` 对比 `censold:m=4`（NDCG@100，越高越好）：26 份日志的平均变化 -0.0435（95% 区间 -0.0696 到 -0.0175）；35% 的日志更好。Amazon 日志 -0.0633，其他日志 +0.0101。通俗结论：变差。事先注册的检验：未确认（发现用日志 -0.0346，没有通过错误发现率筛选；留出日志 -0.0501）。 预测检查：mean -0.0435 (needs <= +0.005), upper 95% bound -0.0175 (a gain of 0.01 is ruled out); rows per round x1.00 and people x1.00 of the reference。在其他 7 个（人数、轮长）组合中，有 4 个方向相同。

### D15. 样本追平定期普查所需的人数（G census plus samples）

- **问题。** 样本追踪器追平每 4 轮或 8 轮一次全量统计所需的每轮人数。
- **事先注册的预测（英文原文）。** Break-even n (first n on the grid with NDCG(sampled) >= NDCG(censold m=4)) is above 300 in at least 70% of eligible logs.
- **文献依据（英文原文）。** [INFERENCE] equivalent age 4.4 rounds at n=100 (tracking study)
- **结果。** 预测不成立

在 24 个日志中，把抽样追踪器与每 4 轮一次的全量统计（普查）比较。抽样榜单第一次不差于普查所需的每轮人数，在 54% 的日志里高于 300（或在最多 3000 的网格内始终达不到）；追平所需人数的中位数是 >3000（对每 8 轮一次的普查：8% 高于 300，中位数 300）。预测要求至少 70% 高于 300，因此不成立。

### D16. 中心化差分隐私的代价表（C privacy）

- **问题。** 中心化差分隐私（每人最多 10 条）要使 NDCG 损失小于 0.05，每轮需要多少人。
- **事先注册的预测（英文原文）。** At n=10000 (90 rounds), eps=2 per round loses < 0.05 against the non-private baseline in at least 50% of eligible logs; at n=1000, eps=8 does NOT reach < 0.05 in at least 50% of logs (any round length). Also reported: noise-only loss against the capped, noise-free reference (eps=1e6) on the same people.
- **文献依据（英文原文）。** Balle-Wang analytic Gaussian (sigma 3.7/2.0/1.08 at eps 1/2/4 for L2 sensitivity 1); Desfontaines et al. / Korolova: thresholds need about 12 users at eps=1; amplification by subsampling not applied here (conservative)
- **结果。** 预测不成立

中心化隐私（对计数加一次噪声，每人最多 10 条）与同样但无噪声的榜单比较。每轮 10000 人、隐私水平 eps=2 时，9 个日志中有 11% 的损失低于 0.05 NDCG；每轮 1000 人、eps=8 时，36 个日志中有 17% 的损失低于 0.05。预测要求前者至少 50%、后者不足 50%，因此不成立。

### D17. 本地差分隐私的代价表（C privacy）

- **问题。** 本地差分隐私（优化的一元编码，每人一条）在 n=10000 时即使 eps=4 也损失超过 0.1 NDCG。
- **事先注册的预测（英文原文）。** At n=10000 (90 rounds), eps=4 loses > 0.1 against the non-private baseline in at least 50% of eligible logs. Also reported: loss against the one-row-per-person noise-free reference.
- **文献依据（英文原文）。** Wang et al. 2017: Var = 4 n e^eps/(e^eps-1)^2 per count (std 0.19 of share at eps=1, n=100 per LitPrivacy); local DP needs about 50-100x more people in our main study (X9)
- **结果。** 预测成立

本地隐私（每个人自己对汇报加噪声，服务器看不到真实计数）与同样但无噪声的榜单比较。每轮 10000 人、隐私水平 eps=4 时，9 个日志中有 89% 的榜单损失超过 0.1 NDCG（平均损失 0.493，中位数 0.582）。预测要求至少 50% 的日志，因此成立。

### D18. 在计数阈值下能发布多长的榜单（C privacy）

- **问题。** 每个发布的物品要求累计至少 5 次汇报时，每轮 100 人的服务器只能发布不足 50 个物品。
- **事先注册的预测（英文原文）。** At n=100, t=5: median list length (len100 metric) < 50 and precision of the published items (prec100) >= 0.5.
- **文献依据（英文原文）。** LitPrivacy: deployed thresholds 10-50 users (Chrome k=10-50, TrieHH theta=10-17); publishable items <= n*c/tau
- **结果。** 预测不成立

每个发布的物品要求累计至少 5 次汇报。每轮 100 人时，发布榜单的中位长度是 0.5 个物品（完整榜单应为 100 个），其中 22% 的物品确实在真实前 100 名里（26 个日志的中位数）。能发满 100 个物品的最小规模：网格内（最多 3000）没有。预测要求长度少于 50 且准确率至少 50%，因此不成立。

### D19. 等效年龄（重复检验）（D planning）

- **问题。** 每轮 100 人的榜单，相当于一份 3 到 6 轮前的全量统计，在新类别和新的非 Amazon 日志上同样如此。
- **事先注册的预测（英文原文）。** Median equivalent age at n=100 over fresh and new logs lies in [3, 6] rounds.
- **文献依据（英文原文）。** Tracking study (equivalent age 1.3-8, median 4.4 on Amazon)
- **结果。** 预测不成立

“等效年龄”指：全量统计要旧到什么程度，才和抽样榜单一样好；即全量统计的 NDCG 等于抽样榜单 NDCG 时的滞后轮数（在 0 到 8 轮之间插值，并以 0 和 8 为上下限）。每轮 100 人时，7 个新类别和新的非 Amazon 日志的等效年龄中位数为 1.97 轮（1 个被截在 0，0 个被截在 8；全部日志的中位数 3.31）。预测要求中位数在 3 到 6 之间，因此不成立。

### D20. 最佳半衰期随榜单长度的变化（D planning）

- **问题。** 前 10 名榜单的最佳半衰期短于前 300 名榜单。
- **事先注册的预测（英文原文）。** In at least 60% of logs the best half-life for k=10 is <= the best half-life for k=300 (ties count as met), and strictly shorter in at least 25%.
- **文献依据（英文原文）。** [INFERENCE] head items turn over faster than the tail
- **结果。** 预测成立

对每个日志，分别找出每轮 100 人时使前 10 名榜单和前 300 名榜单最好的半衰期（旧汇报衰减的速度）。在 26 个日志中，100% 的日志里前 10 名的最佳半衰期不长于前 300 名，42% 的日志里严格更短。预测要求分别至少 60% 和 25%，因此成立。

### D21. 达到上限 95% 所需的人数（D planning）

- **问题。** 即使每轮 3000 人，也达不到用同样半衰期的全量统计所能达到的 95%，在多数日志里如此。
- **事先注册的预测（英文原文）。** For k=100, NDCG(baseline, n=3000) < 0.95 x NDCG(fulldecay h=2) in at least 70% of logs eligible at n=3000.
- **文献依据（英文原文）。** Dyer-Owen style bounds / main study: drift not sampling dominates; equivalent age 4.4 at n=100
- **结果。** 无法检验

无法检验：n=3000 时只有 4 个日志可用，至少需要 6 个。

### D22. 冷启动需要几轮（D planning）

- **问题。** 半衰期为 2 时，榜单在 3 轮内达到稳态 NDCG 的 90%（新追踪器 3 轮后就有用）。
- **事先注册的预测（英文原文）。** Median round at which mean NDCG (over draws) first reaches 90% of the steady mean is <= 3 for h=2 at n=100 (logs whose first scored round is already above 90% are left-censored and excluded). The h=inf row is reported but decides nothing (its list is not a steady state).
- **文献依据（英文原文）。** Exponential smoothing: effective memory 1/(1-2^(-1/h)) = 3.4 rounds at h=2
- **结果。** 预测不成立

冷启动轮数是指榜单的平均 NDCG 第一次达到稳态水平（第 8 轮及以后的平均）90% 的那一轮，条件为每轮 100 人。半衰期为 2 时中位轮数是 6.0（18 个日志）；不遗忘（半衰期无穷大）时是 0.0（12 个日志）；第 3 轮之后首次评估时就已达 90% 的日志无法观察冷启动，被排除（分别 8 个和 14 个）。预测要求半衰期 2 不超过 3 轮、不遗忘至少 4 轮，因此不成立。

### D23. 糟糕的轮次有多糟（D planning）

- **问题。** n=300 时，第 10 百分位的那一轮与平均轮相差不超过 0.10；n=30 时相差超过 0.15。
- **事先注册的预测（英文原文）。** Median over logs of (mean - 10th percentile of per-round NDCG over steps >= 8 and draws): <= 0.10 at n=300 and >= 0.15 at n=30.
- **文献依据（英文原文）。** [INFERENCE] sampling noise shrinks as 1/sqrt(n); round-to-round shifts are part of drift
- **结果。** 预测不成立

“糟糕的轮次”指 NDCG 处于最差 10% 的那些轮（第 8 轮及以后，合并所有重复），我们度量它比平均轮低多少。各日志差距的中位数：每轮 300 人时为 0.121 NDCG（24 个日志），每轮 30 人时为 0.091（28 个日志）。预测要求 n=300 时不超过 0.10、n=30 时至少 0.15，因此不成立。

### D24. 服务器自己的评分能反映真实质量吗（E self-monitoring）

- **问题。** 已发布榜单对下一轮自身样本的 NDCG（不需要真值）与真实 NDCG 的秩相关在各日志、轮长、n 之间不低于 0.7。
- **事先注册的预测（英文原文）。** MEDIAN over n values (those with at least 12 cells) of the within-n Spearman correlation across (log, n_win) cells between steady mean selfnd and steady mean nd100 of the baseline is >= 0.7; the pooled correlation across n is also reported (it is inflated because both rise with n).
- **文献依据（英文原文）。** LitMonitoring: the next-round sample gives an unbiased unnormalised DCG (prequential scheme, Gama et al. 2013); the sample-ideal normaliser is biased upward
- **结果。** 预测成立

在 245 个单元中，我们把服务器对自己榜单的评分（用下一轮的汇报者样本打分，不需要真值）与真实 NDCG@100（0 到 1，越高越接近真实前 100）比较。秩相关系数（1 表示排序完全一致，0 表示无关）为 0.95（对 4 个 n 值分别计算后取中位数；不分 n 合并时为 0.94，因为两种评分都随 n 上升而被抬高），门槛为 0.7，因此预测成立。自评分平均是真实分数的 0.41 倍，这个比值与 n 的秩相关为 0.66。

### D25. 榜单变动幅度能说明榜单有多旧吗（E self-monitoring）

- **问题。** 榜单在相邻两轮之间的变动幅度，与它落后于全量统计的程度（全量衰减榜单 NDCG 减基线 NDCG）的秩相关不低于 0.5。
- **事先注册的预测（英文原文）。** Not met expected: |Spearman| < 0.5 across (log, n_win) cells at n=100 (literature: churn measures noise and inertia, not staleness).
- **文献依据（英文原文）。** Webber et al. 2010 (RBO between successive lists); Garg et al. 2022 (accuracy without labels is as hard as learning); decayed sums mechanically lower churn at longer half-lives
- **结果。** 预测成立

在 n=100 的 124 个单元中，我们比较了榜单相邻两轮之间的变动（前 100 名被替换的比例）与榜单落后于全量统计的程度（全量榜单 NDCG 减基线 NDCG）。两者的秩相关系数为 0.72；命题要求绝对值不低于 0.5，因此结果为成立（本研究事先预期不成立）。与最佳天数半衰期随轮数变化的幅度相比，29 个日志上的相关系数为 -0.22。

### D26. 半样本一致度作为质量指标（E self-monitoring）

- **问题。** 由随机两半汇报者各自建出的前 100 名榜单的重合度，与真实 NDCG 的秩相关不低于 0.7（n 不小于 300 时）。
- **事先注册的预测（英文原文）。** MEDIAN over n in {300,1000,3000} (those with at least 12 cells) of the within-n Spearman(agree, nd100) across logs is >= 0.7; at n=30 the within-n correlation is below 0.5.
- **文献依据（英文原文）。** LitMonitoring: split-half is the noise floor; Mogstad et al. 2022 rank confidence sets; informative only when n >= about 400
- **结果。** 预测不成立

半样本一致度是用随机分成的两半汇报者各自建出的前 100 名榜单的重合程度。它与真实 NDCG@100 的秩相关（1 表示排序一致）在 n 取 300、1000、3000 且日志数不少于 12 的 1 个取值上取中位数为 0.70（门槛 0.7），在 n=30 的 28 个单元上为 0.76（须低于 0.5）。预测不成立。

### D27. 最佳半衰期（按天）有多平坦（F calendar）

- **问题。** 在 Amazon 日志上，对于每轮 3 到 9 个月的情形，300 到 700 天之间的任意半衰期都与天数网格中的最佳值相差不超过 0.01 NDCG。
- **事先注册的预测（英文原文）。** For at least 80% of Amazon (log, n_win in {30,45,60,90}) cells the NDCG regret of each of 300, 400, 534, 700 days is <= 0.01 relative to the best of the days values. Reported separately for the clean set (holdout_fresh and new logs) because the anchor numbers come from older logs.
- **文献依据（英文原文）。** Our tournament rounds 3 (400 days +0.057, 534 days +0.054, 700 days +0.031 at 90 rounds on holdout)
- **结果。** 预测不成立

在 79 个 Amazon 单元（20 个日志，每轮 3 到 9 个月）上，我们计算了每个天数半衰期比八个候选值中的最佳者低多少 NDCG@100（遗憾值）。遗憾值不超过 0.01 的单元比例为：300 天 0.35，400 天 0.63，534 天 0.73，700 天 0.44；每一项都须达到 0.8，因此预测不成立。

### D28. 非 Amazon 日志上日历半衰期稳定吗（F calendar）

- **问题。** 在每个非 Amazon 日志上，最佳半衰期（按天）在 45、60、90 轮之间变化不超过 2 倍。
- **事先注册的预测（英文原文）。** At least 60% of eligible non-Amazon logs have max/min best days <= 2 across n_win 45, 60, 90 (decay:days arms only). A log is censored (left out, counted) when its best days is at the edge of the grid (3 or 1500 days) at any of the three widths or when all days arms are within 0.001 NDCG of each other.
- **文献依据（英文原文）。** Earlier calendar check (11 of 18 logs same in days; Amazon 494-587; Gowalla 83); [INFERENCE] growing platforms prefer shorter spans
- **结果。** 预测成立

对 8 个非 Amazon 日志，我们分别找出 45、60、90 轮时最佳的半衰期（按天），再用最大值除以最小值。有 8 个日志（1.00）的比值不超过 2 倍，门槛为 0.6，因此预测成立。比值中位数为 1.25，最大为 2.00；旧日志 1.00，新日志 1.00。

### D29. 其他类型日志上的自调半衰期（F calendar）

- **问题。** 在非 Amazon 日志上，用自己样本选半衰期的服务器，在 60% 的情形里追回最佳固定半衰期相对半衰期 2 的至少一半收益，且从不差于 0.01；在 Amazon 日志上则有至少 30% 的情形差于 0.01。
- **事先注册的预测（英文原文）。** Non-Amazon (log, n_win) cells at n=100: share with (selfsel - base) >= 0.5 x (best fixed - base) is >= 0.6 among cells where best fixed - base >= 0.01, and no cell has selfsel - base < -0.01; Amazon cells: share with selfsel - base < -0.01 is >= 0.3.
- **文献依据（英文原文）。** Tracking study PD (Other stratum 9 of 12 cells recovered half, 0 harmful; Amazon harmful in 51% of 59 cells)
- **结果。** 预测不成立

在非 Amazon 日志上，最佳固定半衰期比半衰期 2 好 0.01 NDCG 以上的单元有 23 个，自调服务器在其中 13 个（0.57；门槛 0.6）追回了至少一半的收益；在 40 个非 Amazon 单元中，比半衰期 2 差 0.01 以上的有 2 个（须为 0）。在 Amazon 日志上，84 个单元中有 38 个（0.45；门槛 0.3）差这么多。预测不成立。

### D30. 结论是否依赖评价指标（H metric choice）

- **问题。** 三十多个聚合方式按平均差排序，在 recall@100、NDCG@10、NDCG@30、NDCG@300 下与 NDCG@100 下一致（秩相关不低于 0.9），且没有任何增益或损失的结论反号。
- **事先注册的预测（英文原文）。** Among agg arms other than the oracle and census arms (fulldecay, cens, censold) and with |mean diff in NDCG@100| <= 0.02, the Spearman between mean diffs (arm minus baseline, n=100, 30 rounds, all logs) under NDCG@100 and under each other metric is >= 0.9; number of arms whose sign flips with |diff| > 0.005 under NDCG@100 is 0.
- **文献依据（英文原文）。** LitMonitoring: NDCG@k vs recall@k vs RBO agree on large effects, can disagree on small ones
- **结果。** 预测不成立

我们在 n=100、29 个日志上，按各聚合方式相对基线的平均 NDCG@100 差值给 23 个方式排序，并用 NDCG@10、NDCG@30、NDCG@300 和 recall@100 重复。它们与 NDCG@100 排序的秩相关（1 表示排序一致）分别为 nd10 0.74，nd30 0.90，nd300 0.97，rec100 0.97，均须不低于 0.9。NDCG@100 差值绝对值大于 0.005 而符号反转的方式有 1 个（须为 0），因此预测不成立。
