# 从这里开始：小样本热门估计

英文版：[`00_start_here_en.md`](00_start_here_en.md)。写给想要阅读并评判这项工作的同事。

## 1. 这是什么，三句话

从一个大群体里只抽一小批人，每人只留几条过去的选择，只用这些数据排出一份推荐清单。
**这份清单有多准？需要多少人？什么时候，用同一批样本训练的个性化推荐，才比直接推热门榜更值？**
我们把热门榜当作统计量（像均值、方差），只不过被估计的东西是一份排序。

## 结论摘要

- 只用几百个人建出来的热门榜，在软指标上就已经很接近全群体的榜。**更大的差距来自时间，而不是样本量：** 在我们能判定的 10 个数据集里，
  群体"更早的榜"都与"之后的榜"相差很远；11 个数据集里有 10 个，近期窗口样本比均匀样本更能预测之后的榜；按近期加权的计数在 80% 的格子里胜过朴素计数。
  但修复的幅度很小（榜单层面约占差距的 2%），所以漂移是还没解决的问题。
- 同一个样本训练的个性化模型，在最大的被测样本量处仍在提升，而热门榜早得多就饱和了；在事后加做的小 N 阶段里，7 个数据集中的 3 个，只要 20 个人个性化就能胜过热门榜。
- 我们预登记的三个预测，按原文写的**都没有**成立（C1 无定论，C2 被证伪，C3 混合）；原因，以及我们对自己解释的检验，见 [`02_results_zh.md`](02_results_zh.md) 第 2 节。

## 2. 阅读顺序

| 序号 | 文件 | 你会得到什么 | 用时 |
| ---: | --- | --- | --- |
| 1 | 本页 | 问题、结论摘要、文件地图 | 5 分钟 |
| 2 | [`01_preregistration_zh.md`](01_preregistration_zh.md) | 我们预测了什么、每个假设如何判定，**写在实验运行之前**（外加带日期的修订） | 15 分钟 |
| 3 | [`02_results_zh.md`](02_results_zh.md) | 全部结果：假设、结论、白话解读、图 | 20 分钟 |
| 4 | `results/small_sample_popularity/03_tables/cells_*.csv` | 每张图背后的数字：每个格子一行，含均值和标准误 | 按需 |

## 3. 文件放在哪里（按产生顺序编号）

```text
docs/small_sample/                      上面三份文档，中英文各一份
datasets/fetch.sh                       下载 11 个数据集的公开来源（原始文件不放进 Git）
scripts/ssp/prep_data.py                原始文件 -> 每个数据集一份紧凑格式
scripts/ssp/frame.py, est.py, ...       引擎：抽样、估计方法、指标
scripts/ssp/run_all.py                  任务队列（可续跑）；run_overnight.sh 以安静模式运行它
scripts/ssp/an_*.py                     每一族假设一个分析脚本
results/small_sample_popularity/
  00_safety/    风扇测试（calibration.csv）、调度器日志、队列日志
  01_datasets/  datasets.json：每个数据集的规模、集中程度和理论数值
  02_runs/      原始行，每次随机抽样一行（很大，不放进 Git）
  03_tables/    格子均值（cells_*.csv）、假设表格、*_verdicts.json
  04_figures/   结果文档里用到的全部图
```

## 4. 如何重新运行

```sh
make venv                               # 锁定的 Python 和软件包（一次）
datasets/fetch.sh                       # 下载公开数据
.venv/bin/python3 scripts/ssp/prep_data.py --all
.venv/bin/python3 scripts/ssp/describe_datasets.py
scripts/ssp/run_overnight.sh            # 所有阶段；可以随时停止并重启
.venv/bin/python3 scripts/ssp/aggregate.py
# 分析，每族一个脚本（各自产出表格和图）：
for s in an_confirm2 an_window an_explore_a an_explore_b an_extra an_stoprule an_curves an_hunt; do .venv/bin/python3 scripts/ssp/$s.py; done
.venv/bin/python3 scripts/ssp/make_report.py
```

## 5. 运行过程被设计得很安静

通宵在一台笔记本上运行，风扇保持空闲转速。`scripts/ssp/quiet.py` 每 3 秒读一次芯片温度、风扇转速、内存和磁盘
（`macmon`，不需要 sudo），在风扇升速之前放慢或冻结工作进程。细节和风扇实测数据在
`results/small_sample_popularity/00_safety/`。

## 6. 这项工作不能说明什么

- 样本是从群体里随机抽的，不是真正的小群体（子群体阶段除外）。
- 数据是公开的自愿评分或平台日志；结论是"在这些数据上看到的趋势"。
- 联邦是模拟的；没有设备、掉线或真实的安全聚合。
- 为了让机器保持安静，个性化模型最多只在 1,500 个候选物品上训练；见修订 A15。

## 7. 后续工作：每周一轮的记忆长度计划（文档 03–19 和 90）

主研究（文档 01–02）问的是需要多少人。之后的一切问的是服务器的实际问题：**它的记忆应该多长，它能不能只靠自己的数据来选这个记忆？** 后面每份文档都按写作顺序编号；“预登记”指计划在运行之前就已提交，“事后”指看过结果以后才分析。

| 文档 | 中文 | 你会得到什么 |
| ---: | --- | --- |
| 03 | [`03_conclusions_zh.md`](03_conclusions_zh.md) | 用白话讲主研究：八个术语、三个预测以及结果 |
| 04 | [`04_next_hypotheses_zh.md`](04_next_hypotheses_zh.md) | 主研究结束时候选的下一批想法 |
| 05 | [`05_refresh_and_two_round_zh.md`](05_refresh_and_two_round_zh.md) | 季节性和“两轮提问”：两个想法都没有帮助 |
| 06–07 | [`06_tracking_design_zh.md`](06_tracking_design_zh.md)、[`07_tracking_results_zh.md`](07_tracking_results_zh.md) | 在 22 份带时间戳的日志上逐轮跟踪热门榜：每轮 100 人给出的榜，和 1.3 到 8 轮之前的全量计数差不多好 |
| 08–09 | [`08_tournament_design_zh.md`](08_tournament_design_zh.md)、[`09_tournament_results_zh.md`](09_tournament_results_zh.md) | 用固定的误差预算筛查 12 个候选想法 |
| 10–12 | [`10_deep_plan_zh.md`](10_deep_plan_zh.md)、[`11_deep_cards_zh.md`](11_deep_cards_zh.md)、[`12_deep_results_zh.md`](12_deep_results_zh.md) | 30 份日志上的 30 个预登记问题：没有任何一个替代方案胜过简单的衰减求和；服务器可以不靠真实答案给自己的榜打分；隐私代价很大 |
| 13 | [`13_weekly_results_zh.md`](13_weekly_results_zh.md) | 每周一轮：最佳记忆长得多，并取决于每轮人数（30、100、300、1,000、3,000 人时分别约为 32、16、8、4、2 轮） |
| 14 | [`14_selfsel_exploration_zh.md`](14_selfsel_exploration_zh.md) | 事后分析：服务器每轮发布近期自评分最好的那个记忆，相对最佳固定记忆损失很小 |
| 15 | [`15_selfsel_confirmation_zh.md`](15_selfsel_confirmation_zh.md) | 在第二个窗口上对文档 14 的预登记确认：成立，主要从每轮约 100 人起 |
| 16 | [`16_selfsel_stress_zh.md`](16_selfsel_stress_zh.md) | 预登记的压力测试：自愿者偏差、隐私噪声、已发布榜单的抖动 |
| 17 | [`17_selfsel_guard_zh.md`](17_selfsel_guard_zh.md) | 在第三个窗口上的预登记检验：用先验自评分预置的选择器，能补上薄弱区域（30 人；每月一轮） |
| 18 | [`18_privacy_price_zh.md`](18_privacy_price_zh.md) | 预登记：限制每个人的贡献（报告缩放到单位长度）在 1,000 人时大幅降低隐私的代价（ε = 4 时 0.273 → 0.082），100–300 人时降幅较小；中心模型，名义记账 |
| 19 | [`19_volunteer_weights_zh.md`](19_volunteer_weights_zh.md) | 预登记：在小规模随机审计样本上选出的权重给重度自愿者带来增益（100–300 人时 +0.012 NDCG），但没有把轻度自愿者提升到随机人群的水平；8 条正向说法中 7 条成立；事后：同样人头数的随机汇报者胜过自愿者加审计 |
| 90 | [`90_materials_ledger_zh.md`](90_materials_ledger_zh.md) | 用到的每个数据集（含哈希）、工具和文献来源，以及安全规则 |

**服务器能从文档 13–19 里带走什么（通俗读法；一切都是在公开日志上的模拟）。**

1. **记忆长度。** 每周一轮时，每轮人数越多，最佳记忆越短（30、100、300、1,000、3,000 人时约为 32、16、8、4、2 轮；文档 13）。服务器可以用自己的数据来选（文档 14）；文档 15 主要从每轮约 100 人起确认了这一点。
2. **薄弱区域。** 30 人或每月一轮时，用先验自评分预置选择器有帮助（文档 17）；增益在 Amazon 日志上，文档 17 不称它可以直接部署。
3. **自愿者。** 谁来汇报比选哪个记忆更重要：自愿者偏差的代价最高 0.025 NDCG（轻度自愿者；服务器没有先知时唯一能用的重度自愿者规则，在 100 人及以上时只损失 0.0015–0.0035），而选择器的遗憾至多 0.0066（文档 16）。在小规模随机审计样本上选出的权重给重度自愿者带来约 0.012 的增益，没有把轻度自愿者提升到随机人群的水平；人头数相同时，随机汇报者胜过自愿者加审计（文档 19）。
4. **隐私。** 代价取决于一个人能贡献多少：把每份报告缩成单位长度后，ε = 4 每轮、1,000 人时选择器的代价从 0.273（每人 10 行上限）降到 0.082；100 人时榜单仍损失 61% 的质量（文档 18）。事后：每周的私有前 10 名从 300 人起（ε = 4 到 8）能保住约 80% 的无噪声质量，前 100 名在 ε = 4 时需要约 1,000 人。中心模型，名义记账，不是经过认证的保证。
5. **证据。** 每条结论都是在公开日志上的模拟，大多数是预登记的，事后项目都有标注，子代理的所有评审都是同一提供方的检查。

文档 13–19 的数字和判定规则在 `results/small_sample_popularity/03_tables/{weekly,confirm,stress,guard,price,volunteer}/` 和登记文件 `configs/*_registry.json` 里。子代理对这些文档的所有评审都是同一提供方的检查，不是外部审计。
