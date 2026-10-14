# 从这里开始：小样本热门估计

英文版：[`00_start_here_en.md`](00_start_here_en.md)。写给想要阅读并评判这项工作的同事。

## 1. 这是什么，三句话

从一个大群体里只抽一小批人，每人只留几条过去的选择，只用这些数据排出一份推荐清单。
**这份清单有多准？需要多少人？什么时候，用同一批样本训练的个性化推荐，才比直接推热门榜更值？**
我们把热门榜当作统计量（像均值、方差），只不过被估计的东西是一份排序。

<!-- HEADLINE -->

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
