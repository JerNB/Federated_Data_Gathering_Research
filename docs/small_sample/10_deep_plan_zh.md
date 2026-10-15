# 10. 深度研究计划：方案（运行前写好并提交）

## 一、为什么做这一轮

锦标赛的第 1 到 3 轮（文档 08、09）淘汰了大多数“重新给各轮加权”的想法，找到了一条有用的规则（按天数而不是按轮数遗忘）。这一轮深入问 30 个问题，按使用者要做的决定分组：怎样合并各轮、问谁、隐私的代价、需要多少人和多少轮、服务器能对自己测量什么、怎么设半衰期，以及结论是否依赖评价指标。每个问题在运行前都登记了要超过的数字。

## 二、先做的调研（五组文献扫描，由同一提供方的调研代理完成，每个说法都在来源处核对）

- **合并各轮。** 阻尼趋势平滑是预测里难以超越的默认方法（Gardner 2006），但漂移缓慢时趋势项只会增加方差。倒数排名融合在检索里胜过分数融合（Cormack 等 2009）；在元搜索里分数相加胜过 Borda（Renda 与 Straccia 2003）。曝光相同时对计数做收缩不会改变排序。变化检测（ADWIN）在突变时有用，而我们测得的漂移是渐变的。
- **问谁。** 轮换小组只在同一个人的回答跨轮重复时才有帮助（Cantwell 2008，Kish 1999）；评论日志上的物品热度几乎没有重复。按活跃度成比例抽人再直接累加计数会有“按规模偏倚”（Hansen-Hurwitz）。各轮人数不同时，标准做法是按方差倒数加权。
- **隐私。** epsilon 不小于 1 时应使用解析高斯机制（Balle 与 Wang 2018），而不是经典公式；衰减累加是对各轮加噪直方图的后处理，所以不需要树机制；本地差分隐私每个计数的误差是 4 n e^eps/(e^eps-1)^2（Wang 等 2017）；实际部署的最小计数约为 10 到 50 个用户。
- **自我监测。** 没有来源表明相邻两期榜单之间的变动能度量榜单有多旧。下一轮自己的样本可以给出 DCG 的无偏代理。半样本一致度只是噪声下限，只有 n 大时才有信息。
- **额外数据。** 又下载了六份公开日志（三个 Goodreads 类别、Steam 评论、BeerAdvocate、RateBeer）来扩充非 Amazon 一侧；全部日志现在共 30 份（13 份发现用，17 份留出）。

## 三、每个问题怎么检验

- **引擎。** `scripts/ssp/deep_core.py`：约 20 种建榜规则和 7 种抽汇报者的方式，在完全相同的人上运行；指标有 NDCG@10、30、100、300、recall@100、榜单变动、发布榜单长度、半样本一致度，以及服务器自己的评分。
- **配对问题（D01 到 D14）。** 对每份日志：新规则与参照规则在 100 人、30 轮时的差，对各次抽样取平均。发现用日志：单侧 t 检验，对“声称有提升”的各项做 10% 的 Benjamini-Hochberg 筛选，效应至少 0.01。只有留出日志也显示至少 0.01 且 p 小于 0.05 除以被选中的项数，才算“确认”。每个问题还要根据全部日志的 95% 区间给出通俗标签，并统计在多少个其他（人数、轮长）组合中方向相同。
- **计算类问题（D15 到 D30）。** 事先写好数值预测；答案按所述数字判为“成立”或“不成立”，并附完整表格。
- **深度。** 剂量响应的各个版本（例如 3 种上限、5 种隐私预算、8 种以天计的半衰期）、30 轮和 60 轮、30 到 3000 人、16 次随机抽样、分层（Amazon、其他、新类别）。

| 阶段 | 每轮人数 | 轮数 | 抽样次数 |
| --- | --- | --- | --- |
| agg | 30, 100, 300, 1000 | 30, 60 | 16 |
| seas | 100, 300 | 45, 60, 90 | 16 |
| plan | 10, 30, 100, 300, 1000, 3000 | 30 | 16 |
| cal | 100 | 20, 30, 45, 60, 90 | 16 |
| dp | 1000, 3000, 10000 | 30, 90 | 12 |

## 四、30 个假设和事先登记的预测（预测原文为英文）

| 编号 | 类别 | 想法 | 问题 | 登记的预测 |
| --- | --- | --- | --- | --- |
| D01 | A combine rounds | 带阻尼的趋势项 | Holt 阻尼趋势平滑（份额的水平加上带阻尼的趋势）给出的榜单比衰减累加更好。 | No gain: |mean diff| <= 0.005 for the damped primary at n=100; undamped (p=1.0) loses at least 0.02. |
| D02 | A combine rounds | 截断单轮尖峰 | 把每轮中每个物品的计数截断在 3 条再计算份额，榜单更好。 | No difference: |mean diff| < 0.003. |
| D03 | A combine rounds | 倒数排名聚合 | 把每轮的倒数排名分数 1/(60+排名) 按半衰期 2 累加，比累加份额得到更好的榜单。 | No gain: the 95% interval of the mean diff excludes +0.01. (A one-log smoke test first showed +0.043; the design review found that it came from breaking ties by item id, which follows item age; with random tie-breaks the same smoke shows -0.026, so the prediction was reset to no gain before the real run.) |
| D04 | A combine rounds | 对新物品更快遗忘 | 首次出现不到 4 轮的物品用半衰期 1，更老的物品用半衰期 4，比所有物品用同一半衰期更好。 | No gain (likely a loss): mean diff <= +0.005. |
| D05 | A combine rounds | 更细的半衰期网格 | 半衰期 3 轮比基线半衰期 2 轮更好（早先的网格每档差 2 倍，真正的最优值可能落在两档之间）。 | No gain: the 95% interval of the mean diff excludes +0.01 (the optimum is flat around 2 rounds). Replaces an empirical-Bayes arm that the design review showed is algebraically a longer half-life. |
| D06 | A combine rounds | 榜单跳变时加快遗忘 | 当某一轮份额与追踪器的差异超过平常变化的 1.5 倍时，把追踪器的记忆减半，榜单更好。 | No gain: mean diff <= +0.005 (the detector rarely fires under slow drift). |
| D07 | A combine rounds | 多个半衰期的集成 | 把半衰期 1、2、4 的归一化衰减榜单相加，比只用半衰期 2 更好。 | No gain on Amazon logs (mean diff <= +0.005); a gain of at least +0.005 on the non-Amazon logs where the best half-life is not 2. |
| D08 | A combine rounds | 让榜单保持稳定 | 给上一期榜单上的物品加 10% 的分数加成，使榜单变动至少减少 30%，而 NDCG 损失小于 0.005。 | Churn falls by >= 30% (relative) and NDCG diff >= -0.005. |
| D09 | A combine rounds | 为上升物品预留名额 | 把榜单最后 10 个名额换成近期份额（半衰期 1）最超出旧份额（半衰期 4）的物品，榜单更好。 | No gain: mean diff <= +0.005 (reserved slots cost ranks 91-100 more than rising items earn). |
| D10 | A combine rounds | 去年同一季节 | 在 90 轮（每轮约 3 个月）下，加入一年前同一季节的份额（权重 0.5），榜单更好。季节项在不到一半稳态步中起作用的格子不运行。 | Gain at 90 rounds: mean diff >= +0.01 on Amazon logs; no gain on non-Amazon logs. (Earlier year-ago-list test: year-ago list alone was worse in 16 of 16 categories; here it is only an added term.) |
| D11 | B whom to ask | 轮换小组 | 保留上一轮仍然活跃的 75% 的汇报者（轮换率 0.25，空缺用新人补足），比完全更新的小组（轮换率 1.0，同一个抽样器）榜单更好。 | No gain, and a loss that grows with overlap: the 95% interval of the diff excludes +0.01; rho=0.1 worse than rho=0.25 in at least 60% of logs. |
| D12 | B whom to ask | 按活跃度比例抽人 | 按本轮行数成比例地抽 n 个汇报者（这是不可行的“先知”做法：服务器在提问前并不知道），比均匀抽样榜单更好；可行的变体使用上一轮的行数。 | Oracle arm: gain of at least +0.01 at n=100, but with about 1.8 times more rows collected per round (a one-log smoke showed +0.067 with 1.8x rows); the feasible arm (last round's rows) is secondary and expected to show no gain. Rows per round are reported next to every difference. |
| D13 | B whom to ask | 按轮的样本量加权 | 当各轮汇报者人数不同时，按人数给各轮加权，比等权更好。 | Gain: mean diff >= +0.01 against equal weighting under the same variable round sizes (set after a one-log smoke showed +0.033). |
| D14 | G census plus samples | 样本能给定期普查再加分吗 | 如果每 4 轮有一次全量统计，再加上每轮 n=100 的样本（普查的权重是一轮样本的 8 倍），比只用普查更好。 | No gain at n=100 (mean diff <= +0.005 against the stale census list); a gain at n=1000 would be reported as exploratory. |
| D15 | G census plus samples | 样本追平定期普查所需的人数 | 样本追踪器追平每 4 轮或 8 轮一次全量统计所需的每轮人数。 | Break-even n (first n on the grid with NDCG(sampled) >= NDCG(censold m=4)) is above 300 in at least 70% of eligible logs. |
| D16 | C privacy | 中心化差分隐私的代价表 | 中心化差分隐私（每人最多 10 条）要使 NDCG 损失小于 0.05，每轮需要多少人。 | At n=10000 (90 rounds), eps=2 per round loses < 0.05 against the non-private baseline in at least 50% of eligible logs; at n=1000, eps=8 does NOT reach < 0.05 in at least 50% of logs (any round length). Also reported: noise-only loss against the capped, noise-free reference (eps=1e6) on the same people. |
| D17 | C privacy | 本地差分隐私的代价表 | 本地差分隐私（优化的一元编码，每人一条）在 n=10000 时即使 eps=4 也损失超过 0.1 NDCG。 | At n=10000 (90 rounds), eps=4 loses > 0.1 against the non-private baseline in at least 50% of eligible logs. Also reported: loss against the one-row-per-person noise-free reference. |
| D18 | C privacy | 在计数阈值下能发布多长的榜单 | 每个发布的物品要求累计至少 5 次汇报时，每轮 100 人的服务器只能发布不足 50 个物品。 | At n=100, t=5: median list length (len100 metric) < 50 and precision of the published items (prec100) >= 0.5. |
| D19 | D planning | 等效年龄（重复检验） | 每轮 100 人的榜单，相当于一份 3 到 6 轮前的全量统计，在新类别和新的非 Amazon 日志上同样如此。 | Median equivalent age at n=100 over fresh and new logs lies in [3, 6] rounds. |
| D20 | D planning | 最佳半衰期随榜单长度的变化 | 前 10 名榜单的最佳半衰期短于前 300 名榜单。 | In at least 60% of logs the best half-life for k=10 is <= the best half-life for k=300 (ties count as met), and strictly shorter in at least 25%. |
| D21 | D planning | 达到上限 95% 所需的人数 | 即使每轮 3000 人，也达不到用同样半衰期的全量统计所能达到的 95%，在多数日志里如此。 | For k=100, NDCG(baseline, n=3000) < 0.95 x NDCG(fulldecay h=2) in at least 70% of logs eligible at n=3000. |
| D22 | D planning | 冷启动需要几轮 | 半衰期为 2 时，榜单在 3 轮内达到稳态 NDCG 的 90%（新追踪器 3 轮后就有用）。 | Median round at which mean NDCG (over draws) first reaches 90% of the steady mean is <= 3 for h=2 at n=100 (logs whose first scored round is already above 90% are left-censored and excluded). The h=inf row is reported but decides nothing (its list is not a steady state). |
| D23 | D planning | 糟糕的轮次有多糟 | n=300 时，第 10 百分位的那一轮与平均轮相差不超过 0.10；n=30 时相差超过 0.15。 | Median over logs of (mean - 10th percentile of per-round NDCG over steps >= 8 and draws): <= 0.10 at n=300 and >= 0.15 at n=30. |
| D24 | E self-monitoring | 服务器自己的评分能反映真实质量吗 | 已发布榜单对下一轮自身样本的 NDCG（不需要真值）与真实 NDCG 的秩相关在各日志、轮长、n 之间不低于 0.7。 | MEDIAN over n values (those with at least 12 cells) of the within-n Spearman correlation across (log, n_win) cells between steady mean selfnd and steady mean nd100 of the baseline is >= 0.7; the pooled correlation across n is also reported (it is inflated because both rise with n). |
| D25 | E self-monitoring | 榜单变动幅度能说明榜单有多旧吗 | 榜单在相邻两轮之间的变动幅度，与它落后于全量统计的程度（全量衰减榜单 NDCG 减基线 NDCG）的秩相关不低于 0.5。 | Not met expected: |Spearman| < 0.5 across (log, n_win) cells at n=100 (literature: churn measures noise and inertia, not staleness). |
| D26 | E self-monitoring | 半样本一致度作为质量指标 | 由随机两半汇报者各自建出的前 100 名榜单的重合度，与真实 NDCG 的秩相关不低于 0.7（n 不小于 300 时）。 | MEDIAN over n in {300,1000,3000} (those with at least 12 cells) of the within-n Spearman(agree, nd100) across logs is >= 0.7; at n=30 the within-n correlation is below 0.5. |
| D27 | F calendar | 最佳半衰期（按天）有多平坦 | 在 Amazon 日志上，对于每轮 3 到 9 个月的情形，300 到 700 天之间的任意半衰期都与天数网格中的最佳值相差不超过 0.01 NDCG。 | For at least 80% of Amazon (log, n_win in {30,45,60,90}) cells the NDCG regret of each of 300, 400, 534, 700 days is <= 0.01 relative to the best of the days values. Reported separately for the clean set (holdout_fresh and new logs) because the anchor numbers come from older logs. |
| D28 | F calendar | 非 Amazon 日志上日历半衰期稳定吗 | 在每个非 Amazon 日志上，最佳半衰期（按天）在 45、60、90 轮之间变化不超过 2 倍。 | At least 60% of eligible non-Amazon logs have max/min best days <= 2 across n_win 45, 60, 90 (decay:days arms only). A log is censored (left out, counted) when its best days is at the edge of the grid (3 or 1500 days) at any of the three widths or when all days arms are within 0.001 NDCG of each other. |
| D29 | F calendar | 其他类型日志上的自调半衰期 | 在非 Amazon 日志上，用自己样本选半衰期的服务器，在 60% 的情形里追回最佳固定半衰期相对半衰期 2 的至少一半收益，且从不差于 0.01；在 Amazon 日志上则有至少 30% 的情形差于 0.01。 | Non-Amazon (log, n_win) cells at n=100: share with (selfsel - base) >= 0.5 x (best fixed - base) is >= 0.6 among cells where best fixed - base >= 0.01, and no cell has selfsel - base < -0.01; Amazon cells: share with selfsel - base < -0.01 is >= 0.3. |
| D30 | H metric choice | 结论是否依赖评价指标 | 三十多个聚合方式按平均差排序，在 recall@100、NDCG@10、NDCG@30、NDCG@300 下与 NDCG@100 下一致（秩相关不低于 0.9），且没有任何增益或损失的结论反号。 | Among agg arms other than the oracle and census arms (fulldecay, cens, censold) and with |mean diff in NDCG@100| <= 0.02, the Spearman between mean diffs (arm minus baseline, n=100, 30 rounds, all logs) under NDCG@100 and under each other metric is >= 0.9; number of arms whose sign flips with |diff| > 0.005 under NDCG@100 is 0. |

## 五、披露

- 搭建引擎时，看过一次在一份小日志（乐器类，2 次抽样，100 人）上的试跑结果。其中倒数排名聚合 +0.043、按活跃度抽人 +0.048、可变轮长加权 +0.033，而过期的全量统计榜单远远领先样本。D03、D12、D13、D15 的预测是在看到这些之后写的，注册表里已标明。
- 复用了早先的日志。有九份留出日志在早先的研究里决定了基线半衰期；只有五个新的 Amazon 类别和六份新的非 Amazon 日志是完全全新的。
- 有几个想法在早先几轮里已经以别的形式被淘汰（上升物品、混合、滑动窗口）；这里只在机制不同时才重新测试。
- 审查者是同一提供方的子代理，不是外部审计。

## 六、文件

`configs/deep_registry.json`（已冻结）、`scripts/ssp/deep_core.py`、`run_deep.py`、`an_deep.py`、`an_deep_paired.py`、`an_deep_calc1.py`、`an_deep_calc2.py`、`an_deep_lib.py`、`run_deep_night.sh`、`prep_extra.py`、`datasets/fetch_extra.sh`。结果：`results/small_sample_popularity/03_tables/deep/`，卡片见文档 11。
