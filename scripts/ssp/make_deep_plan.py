"""Writes docs/small_sample/10_deep_plan_{en,zh}.md from configs/deep_registry.json plus the fixed narrative below."""
import json

reg = json.load(open("configs/deep_registry.json"))
H = reg["hypotheses"]
ST = reg["stages"]
sp = reg["split"]
n_disc = len(sp["discovery"]["amazon"] + sp["discovery"]["other"] + sp["new_discovery"])
n_hold = len(sp["holdout_old"]["amazon"] + sp["holdout_old"]["other"] + sp["holdout_fresh"] + sp["new_holdout"])

rows_en = "\n".join(f"| {h['id']} | {h['family']} | {h['title_en']} | {h['statement_en']} | {h['prediction']['text']} |" for h in H)
rows_zh = "\n".join(f"| {h['id']} | {h['family']} | {h['title_zh']} | {h['statement_zh']} | {h['prediction']['text']} |" for h in H)
stage_en = "\n".join(f"| {k} | {', '.join(map(str, v['n']))} | {', '.join(map(str, v['n_win']))} | {v['draws']} |" for k, v in ST.items())

EN = f"""# 10. Deep programme: plan (written and committed before the run)

## 1. Why this round

Rounds 1 to 3 of the tournament (documents 08 and 09) killed most ideas for re-weighting the rounds and found one useful rule (forget in days, not in rounds). This round asks 30 questions in depth, grouped by what a practitioner has to decide: how to combine rounds, whom to ask, what privacy costs, how many people and rounds are needed, what the server can measure about itself, how to set the half-life, and whether conclusions depend on the metric. Every question is registered with a number to beat before anything runs.

## 2. Research done first (five scans, same-provider research agents; each claim checked at its source by the agent)

- **Combining rounds.** Damped-trend smoothing is the hard-to-beat default for forecasting (Gardner 2006), but a trend term adds variance when drift is slow. Reciprocal-rank fusion (Cormack et al. 2009) beat score fusion in retrieval; score sums beat Borda in metasearch (Renda and Straccia 2003). Shrinkage of counts with equal exposure cannot change a ranking. Change detectors (ADWIN) pay off under abrupt change; our measured drift is gradual.
- **Whom to ask.** Rotating panels help only when a person's answers repeat across rounds (Cantwell 2008, Kish 1999); item popularity on review logs has near-zero repeat. Asking people in proportion to their activity with an uncorrected pooled count is size-biased (Hansen-Hurwitz). Inverse-variance weighting is the standard answer when round sizes differ.
- **Privacy.** Use the analytic Gaussian mechanism (Balle and Wang 2018) instead of the classical formula for epsilon of 1 or more; a decayed sum is post-processing of per-round noisy histograms, so no tree mechanism is needed; local-DP error per count is 4 n e^eps/(e^eps-1)^2 (Wang et al. 2017); deployed minimum counts are about 10 to 50 users.
- **Self-monitoring.** No source shows that churn between successive lists measures staleness. The next round's own sample gives an unbiased proxy for DCG. Split-half agreement is a noise floor and only informative at large n.
- **Extra data.** Six more public logs (three Goodreads genres, Steam reviews, BeerAdvocate, RateBeer) were downloaded to widen the non-Amazon side; the whole set is now {n_disc + n_hold} logs ({n_disc} discovery, {n_hold} holdout).

## 3. How each question is tested

- **Engine.** `scripts/ssp/deep_core.py`: about 20 list-building rules and 7 ways of choosing reporters run on identical people; metrics are NDCG at 10, 30, 100 and 300, recall@100, list churn, published list length, split-half agreement and the server's own self-score.
- **Paired questions (D01 to D14).** Per log: difference between the new rule and its reference at 100 people and 30 rounds, averaged over draws. Discovery logs: one-sided t test, Benjamini-Hochberg filter at 10% across the claims of a gain, effect of at least 0.01. A claim is CONFIRMED only if the holdout logs also show at least 0.01 with p below 0.05 divided by the number of claims selected. Every question also gets a plain label from the 95% interval over all logs and a count of how many other (people, rounds) combinations point the same way.
- **Calculator questions (D15 to D30).** A numeric prediction written in advance; the answer is MET or NOT MET on the stated numbers, with full tables.
- **Depth.** Dose-response arms (for example three cap sizes, five privacy budgets, eight half-lives in days), 30 and 60 rounds, 30 to 3000 people, 16 random draws, strata (Amazon, other, new categories).

| Stage | People per round | Round counts | Draws |
| --- | --- | --- | --- |
{stage_en}

## 4. The 30 hypotheses and their registered predictions

| # | Family | Idea | Question | Registered prediction |
| --- | --- | --- | --- | --- |
{rows_en}

## 5. Disclosures

- While building the engine, one smoke run on one small log (musical instruments, 2 draws, 100 people) was looked at. It showed reciprocal-rank aggregation +0.043, activity-proportional sampling +0.048 and variable-round weighting +0.033, and a stale census list far ahead of samples. The predictions for D03, D12, D13 and D15 were written after seeing it; they are marked in the registry.
- Earlier logs are reused. Nine holdout logs chose the baseline half-life in an earlier study; only the five new Amazon categories and the six new non-Amazon logs are fully fresh.
- Several ideas were in earlier rounds already killed in a different form (rising items, blends, sliding windows); here they are tested again only where the mechanism is different.
- The reviewers are same-provider subagents, not an outside audit.

## 6. Files

`configs/deep_registry.json` (frozen), `scripts/ssp/deep_core.py`, `run_deep.py`, `an_deep.py`, `an_deep_paired.py`, `an_deep_calc1.py`, `an_deep_calc2.py`, `an_deep_lib.py`, `run_deep_night.sh`, `prep_extra.py`, `datasets/fetch_extra.sh`. Results: `results/small_sample_popularity/03_tables/deep/`, cards in document 11.
"""

ZH = f"""# 10. 深度研究计划：方案（运行前写好并提交）

## 一、为什么做这一轮

锦标赛的第 1 到 3 轮（文档 08、09）淘汰了大多数“重新给各轮加权”的想法，找到了一条有用的规则（按天数而不是按轮数遗忘）。这一轮深入问 30 个问题，按使用者要做的决定分组：怎样合并各轮、问谁、隐私的代价、需要多少人和多少轮、服务器能对自己测量什么、怎么设半衰期，以及结论是否依赖评价指标。每个问题在运行前都登记了要超过的数字。

## 二、先做的调研（五组文献扫描，由同一提供方的调研代理完成，每个说法都在来源处核对）

- **合并各轮。** 阻尼趋势平滑是预测里难以超越的默认方法（Gardner 2006），但漂移缓慢时趋势项只会增加方差。倒数排名融合在检索里胜过分数融合（Cormack 等 2009）；在元搜索里分数相加胜过 Borda（Renda 与 Straccia 2003）。曝光相同时对计数做收缩不会改变排序。变化检测（ADWIN）在突变时有用，而我们测得的漂移是渐变的。
- **问谁。** 轮换小组只在同一个人的回答跨轮重复时才有帮助（Cantwell 2008，Kish 1999）；评论日志上的物品热度几乎没有重复。按活跃度成比例抽人再直接累加计数会有“按规模偏倚”（Hansen-Hurwitz）。各轮人数不同时，标准做法是按方差倒数加权。
- **隐私。** epsilon 不小于 1 时应使用解析高斯机制（Balle 与 Wang 2018），而不是经典公式；衰减累加是对各轮加噪直方图的后处理，所以不需要树机制；本地差分隐私每个计数的误差是 4 n e^eps/(e^eps-1)^2（Wang 等 2017）；实际部署的最小计数约为 10 到 50 个用户。
- **自我监测。** 没有来源表明相邻两期榜单之间的变动能度量榜单有多旧。下一轮自己的样本可以给出 DCG 的无偏代理。半样本一致度只是噪声下限，只有 n 大时才有信息。
- **额外数据。** 又下载了六份公开日志（三个 Goodreads 类别、Steam 评论、BeerAdvocate、RateBeer）来扩充非 Amazon 一侧；全部日志现在共 {n_disc + n_hold} 份（{n_disc} 份发现用，{n_hold} 份留出）。

## 三、每个问题怎么检验

- **引擎。** `scripts/ssp/deep_core.py`：约 20 种建榜规则和 7 种抽汇报者的方式，在完全相同的人上运行；指标有 NDCG@10、30、100、300、recall@100、榜单变动、发布榜单长度、半样本一致度，以及服务器自己的评分。
- **配对问题（D01 到 D14）。** 对每份日志：新规则与参照规则在 100 人、30 轮时的差，对各次抽样取平均。发现用日志：单侧 t 检验，对“声称有提升”的各项做 10% 的 Benjamini-Hochberg 筛选，效应至少 0.01。只有留出日志也显示至少 0.01 且 p 小于 0.05 除以被选中的项数，才算“确认”。每个问题还要根据全部日志的 95% 区间给出通俗标签，并统计在多少个其他（人数、轮长）组合中方向相同。
- **计算类问题（D15 到 D30）。** 事先写好数值预测；答案按所述数字判为“成立”或“不成立”，并附完整表格。
- **深度。** 剂量响应的各个版本（例如 3 种上限、5 种隐私预算、8 种以天计的半衰期）、30 轮和 60 轮、30 到 3000 人、16 次随机抽样、分层（Amazon、其他、新类别）。

| 阶段 | 每轮人数 | 轮数 | 抽样次数 |
| --- | --- | --- | --- |
{stage_en}

## 四、30 个假设和事先登记的预测（预测原文为英文）

| 编号 | 类别 | 想法 | 问题 | 登记的预测 |
| --- | --- | --- | --- | --- |
{rows_zh}

## 五、披露

- 搭建引擎时，看过一次在一份小日志（乐器类，2 次抽样，100 人）上的试跑结果。其中倒数排名聚合 +0.043、按活跃度抽人 +0.048、可变轮长加权 +0.033，而过期的全量统计榜单远远领先样本。D03、D12、D13、D15 的预测是在看到这些之后写的，注册表里已标明。
- 复用了早先的日志。有九份留出日志在早先的研究里决定了基线半衰期；只有五个新的 Amazon 类别和六份新的非 Amazon 日志是完全全新的。
- 有几个想法在早先几轮里已经以别的形式被淘汰（上升物品、混合、滑动窗口）；这里只在机制不同时才重新测试。
- 审查者是同一提供方的子代理，不是外部审计。

## 六、文件

`configs/deep_registry.json`（已冻结）、`scripts/ssp/deep_core.py`、`run_deep.py`、`an_deep.py`、`an_deep_paired.py`、`an_deep_calc1.py`、`an_deep_calc2.py`、`an_deep_lib.py`、`run_deep_night.sh`、`prep_extra.py`、`datasets/fetch_extra.sh`。结果：`results/small_sample_popularity/03_tables/deep/`，卡片见文档 11。
"""

open("docs/small_sample/10_deep_plan_en.md", "w").write(EN)
open("docs/small_sample/10_deep_plan_zh.md", "w").write(ZH)
print("ok", n_disc, n_hold)
