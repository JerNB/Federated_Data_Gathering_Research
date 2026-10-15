"""Writes configs/deep_registry.json: 30 pre-registered hypotheses, arms, stages, log split, rules.

Run once and commit the JSON BEFORE the programme runs. Literature anchors come from five research scans (see docs/small_sample/10_deep_plan_en.md).
Disclosure: an exploratory smoke test on ONE small log (amazon-musical-instruments, 2 draws, n=100) was run while building the engine and showed
rrf:k=60 +0.043, pps +0.048, ivw (variable rounds) +0.033 and census lists far ahead of samples; D03, D12, D13 and D15 predictions were written after seeing it.
A design review then found that the rrf gain was a tie-breaking artifact (ties were broken by item id, which follows item age); with random ties the smoke shows rrf -0.026,
and the D03 prediction was reset to no gain before the real run; pps collects 1.8x more rows (oracle); the feasible ppsprev shows +0.0015.
"""
import json
import zlib

import tourney_registry as R   # reuse the generation 1 split lists

SALT = "deep-split-v1"
NEW = ["goodreads-poetry", "goodreads-comics", "goodreads-children", "steam", "beeradvocate", "ratebeer"]


def split_new():
    s = sorted(NEW, key=lambda x: zlib.crc32(f"{x}|{SALT}".encode()))
    return sorted(s[:3]), sorted(s[3:])


nd, nh = split_new()
BASE = "decay:h=2"
HG = ["decay:h=0.25", "decay:h=0.5", "decay:h=1", "decay:h=4", "decay:h=8", "decay:h=inf"]
DAYS = [3, 10, 30, 100, 200, 300, 400, 534, 700, 1000, 1500]


def P(i, fam, title_en, title_zh, st_en, st_zh, primary, secondary, pred, anchor, direction="sup", ref=None, extra=None):
    d = dict(id=i, family=fam, kind="paired", direction=direction, title_en=title_en, title_zh=title_zh, statement_en=st_en, statement_zh=st_zh,
             stage="agg", primary=primary, secondary=secondary, ref=ref or BASE, prediction=pred, anchor=anchor)
    if extra:
        d.update(extra)
    return d


def C(i, fam, title_en, title_zh, st_en, st_zh, stage, arms, pred, anchor, spec):
    return dict(id=i, family=fam, kind="calc", title_en=title_en, title_zh=title_zh, statement_en=st_en, statement_zh=st_zh, stage=stage,
                arms=arms, prediction=pred, anchor=anchor, analysis=spec)


H = [
    # ---------------- A. how to combine rounds
    P("D01", "A combine rounds", "Damped trend", "带阻尼的趋势项",
      "Holt damped-trend smoothing (level plus damped trend of the shares) gives a better list than the decayed sum.",
      "Holt 阻尼趋势平滑（份额的水平加上带阻尼的趋势）给出的榜单比衰减累加更好。",
      "holt:a=0.3,b=0.1,p=0.9", ["holt:a=0.3,b=0.3,p=0.8", "holt:a=0.3,b=0.1,p=1.0"],
      dict(text="No gain: |mean diff| <= 0.005 for the damped primary at n=100; undamped (p=1.0) loses at least 0.02.", gain=False, undamped_loss=0.02),
      "Gardner 2006: damped trend hard to beat on average, but a trend term is a smoothed difference of noisy levels (variance); local drift slope median 0.72 and H2 (rising items) lost 0.039"),
    P("D02", "A combine rounds", "Clip one-round spikes", "截断单轮尖峰",
      "Clipping each item's count in a round at 3 rows before sharing gives a better list.",
      "把每轮中每个物品的计数截断在 3 条再计算份额，榜单更好。",
      "clip:c=3", ["clip:c=1", "clip:c=10"],
      dict(text="No difference: |mean diff| < 0.003.", gain=False, absdiff=0.003),
      "Gardner 2006 s4.1 (trimmed smoothing helped on daily supermarket series); our caps of 30 rows lost 0.0006, so little heavy-tail damage is left"),
    P("D03", "A combine rounds", "Reciprocal-rank aggregation", "倒数排名聚合",
      "Adding per-round reciprocal-rank scores 1/(60+rank), decayed with half-life 2, gives a better list than adding shares.",
      "把每轮的倒数排名分数 1/(60+排名) 按半衰期 2 累加，比累加份额得到更好的榜单。",
      "rrf:k=60", ["rrf:k=10", "rrf:k=1"],
      dict(text="No gain: the 95% interval of the mean diff excludes +0.01. (A one-log smoke test first showed +0.043; the design review found that it came from breaking ties by item id, which follows item age; with random tie-breaks the same smoke shows -0.026, so the prediction was reset to no gain before the real run.)", gain=False),
      "Cormack et al. 2009 RRF k=60 beat score fusion by 4-5% MAP in retrieval; Renda and Straccia 2003 found score sums beat Borda in metasearch",
      extra=dict(exclude_discovery=["amazon-musical-instruments"])),
    P("D04", "A combine rounds", "Fast forgetting for young items", "对新物品更快遗忘",
      "Items first seen less than 4 rounds ago use half-life 1, older items half-life 4; this beats one half-life for everything.",
      "首次出现不到 4 轮的物品用半衰期 1，更老的物品用半衰期 4，比所有物品用同一半衰期更好。",
      "twospeed:hf=1,hs=4,age=4", ["twospeed:hf=1,hs=4,age=2", "twospeed:hf=1,hs=8,age=4"],
      dict(text="No gain (likely a loss): mean diff <= +0.005.", gain=False),
      "[INFERENCE] earlier tournament killed every re-weighting of the baseline; item-age mixing splices two score scales"),
    P("D05", "A combine rounds", "Finer half-life grid", "更细的半衰期网格",
      "A half-life of 3 rounds beats the baseline half-life of 2 rounds (the earlier grid steps by a factor of 2, so the true optimum may lie between grid points).",
      "半衰期 3 轮比基线半衰期 2 轮更好（早先的网格每档差 2 倍，真正的最优值可能落在两档之间）。",
      "decay:h=3", ["decay:h=1.5", "decay:h=2.5"],
      dict(text="No gain: the 95% interval of the mean diff excludes +0.01 (the optimum is flat around 2 rounds). Replaces an empirical-Bayes arm that the design review showed is algebraically a longer half-life.", gain=False),
      "Exponential smoothing practice: a flat optimum; earlier tracking study found best half-life 2 in 13 of 14 Amazon logs at the factor-2 grid"),
    P("D06", "A combine rounds", "Forget faster when the list jumps", "榜单跳变时加快遗忘",
      "When a round's shares differ from the tracker by more than 1.5 times the usual change, halving the tracker's memory gives a better list.",
      "当某一轮份额与追踪器的差异超过平常变化的 1.5 倍时，把追踪器的记忆减半，榜单更好。",
      "drift:z=1.5,f=0.5", ["drift:z=1.0,f=0.25"],
      dict(text="No gain: mean diff <= +0.005 (the detector rarely fires under slow drift).", gain=False),
      "ADWIN (Bifet and Gavalda 2007); Gama et al. 2014 survey: detectors pay off under abrupt change, our measured drift slope is 0.2-1.26 (gradual)"),
    P("D07", "A combine rounds", "Ensemble of half-lives", "多个半衰期的集成",
      "Adding the normalised decayed lists with half-lives 1, 2 and 4 gives a better list than half-life 2 alone.",
      "把半衰期 1、2、4 的归一化衰减榜单相加，比只用半衰期 2 更好。",
      "ens", [],
      dict(text="No gain on Amazon logs (mean diff <= +0.005); a gain of at least +0.005 on the non-Amazon logs where the best half-life is not 2.", gain=False),
      "Brown 2008 / forecast-combination literature: combining smoothers helps mainly when the best parameter is unknown"),
    P("D08", "A combine rounds", "Keep the list stable", "让榜单保持稳定",
      "Giving items on the previous list a 10% score bonus cuts list churn by at least 30% at a cost of less than 0.005 NDCG.",
      "给上一期榜单上的物品加 10% 的分数加成，使榜单变动至少减少 30%，而 NDCG 损失小于 0.005。",
      "hyst:d=0.1", ["hyst:d=0.25"],
      dict(text="Churn falls by >= 30% (relative) and NDCG diff >= -0.005.", churn_cut=0.30, margin=0.005),
      "Hysteresis / stability regularisation in top-k maintenance; exact null on NDCG alone is expected", direction="noninf",
      extra=dict(secondary_metric="churn")),
    P("D09", "A combine rounds", "Reserve slots for rising items", "为上升物品预留名额",
      "Replacing the last 10 list slots by the items whose recent share (half-life 1) most exceeds their older share (half-life 4) gives a better list.",
      "把榜单最后 10 个名额换成近期份额（半衰期 1）最超出旧份额（半衰期 4）的物品，榜单更好。",
      "reserve:r=10", ["reserve:r=20"],
      dict(text="No gain: mean diff <= +0.005 (reserved slots cost ranks 91-100 more than rising items earn).", gain=False),
      "Bandit-style exploration slots; tournament round 1 (H2) found extra weight for rising items cost 0.039"),
    dict(id="D10", family="A combine rounds", kind="paired", direction="sup", title_en="Same season last year", title_zh="去年同一季节",
         statement_en="At 90 rounds (rounds of about 3 months) adding the shares of the same season one year earlier (weight 0.5) gives a better list. Cells where the seasonal term is active in fewer than half of the steady steps are not run.",
         statement_zh="在 90 轮（每轮约 3 个月）下，加入一年前同一季节的份额（权重 0.5），榜单更好。季节项在不到一半稳态步中起作用的格子不运行。",
         stage="seas", primary="seas:g=0.5", secondary=["seas:g=0.25", "seas:g=1.0"], ref=BASE,
         prediction=dict(text="Gain at 90 rounds: mean diff >= +0.01 on Amazon logs; no gain on non-Amazon logs. (Earlier year-ago-list test: year-ago list alone was worse in 16 of 16 categories; here it is only an added term.)", gain=True),
         anchor="Seasonal naive / Holt-Winters; our S1 test (year-ago list alone, 0 of 16 better); Amazon December peaks"),
    # ---------------- B. who to ask
    P("D11", "B whom to ask", "Rotating panel", "轮换小组",
      "Keeping 75% of last round's reporters who are active again (rotation rate 0.25; gaps refilled with fresh people) gives a better list than a panel that is refreshed completely (rotation rate 1.0, the same sampler).",
      "保留上一轮仍然活跃的 75% 的汇报者（轮换率 0.25，空缺用新人补足），比完全更新的小组（轮换率 1.0，同一个抽样器）榜单更好。",
      "decay:h=2@panel:rho=0.25", ["decay:h=2@panel:rho=0.1", "decay:h=2@panel:rho=0.5"],
      dict(text="No gain, and a loss that grows with overlap: the 95% interval of the diff excludes +0.01; rho=0.1 worse than rho=0.25 in at least 60% of logs.", gain=False),
      "Cantwell 2008, Kish 1999: gain depends on within-unit correlation rho (about 0 for item-level popularity on review logs); plain decayed sum: fresh wins when drift is slow", ref="decay:h=2@panel:rho=1.0"),
    P("D12", "B whom to ask", "Ask people in proportion to activity", "按活跃度比例抽人",
      "Drawing the n reporters with probability proportional to their rows in the round (an oracle: the server cannot know this before asking) gives a better list than drawing them uniformly; a feasible variant uses last round's rows.",
      "按本轮行数成比例地抽 n 个汇报者（这是不可行的“先知”做法：服务器在提问前并不知道），比均匀抽样榜单更好；可行的变体使用上一轮的行数。",
      "decay:h=2@pps", ["decay:h=2@ppsprev"],
      dict(text="Oracle arm: gain of at least +0.01 at n=100, but with about 1.8 times more rows collected per round (a one-log smoke showed +0.067 with 1.8x rows); the feasible arm (last round's rows) is secondary and expected to show no gain. Rows per round are reported next to every difference.", gain=True),
      "Hansen-Hurwitz / Horvitz-Thompson: with p proportional to size and an uncorrected pooled count the estimate is size-biased, but the target (row counts) is itself dominated by heavy users; the oracle also collects more rows per person asked (reported as rows per round)",
      extra=dict(exclude_discovery=["amazon-musical-instruments"])),
    P("D13", "B whom to ask", "Weight rounds by their size", "按轮的样本量加权",
      "When the number of reporters varies from round to round, weighting each round by its number of reporters beats weighting rounds equally.",
      "当各轮汇报者人数不同时，按人数给各轮加权，比等权更好。",
      "ivw:g=1@varn", ["ivw:g=0.5@varn", "ivw:g=2@varn"],
      dict(text="Gain: mean diff >= +0.01 against equal weighting under the same variable round sizes (set after a one-log smoke showed +0.033).", gain=True),
      "Inverse-variance weighting (standard); variance of a round's share vector is proportional to 1/n_w", ref="decay:h=2@varn",
      extra=dict(exclude_discovery=["amazon-musical-instruments"])),
    # ---------------- G. census
    P("D14", "G census plus samples", "Do samples add to a periodic census", "样本能给定期普查再加分吗",
      "If a full count is available every 4 rounds, adding the continuous n=100 samples (the census counts 8 times one round's sample) to the census beats using the census alone.",
      "如果每 4 轮有一次全量统计，再加上每轮 n=100 的样本（普查的权重是一轮样本的 8 倍），比只用普查更好。",
      "cens:m=4,w=8", ["cens:m=4,w=2", "cens:m=8,w=2", "cens:m=8,w=8"],
      dict(text="No gain at n=100 (mean diff <= +0.005 against the stale census list); a gain at n=1000 would be reported as exploratory.", gain=False),
      "Composite / benchmark estimators need the sample to carry information the benchmark lacks; at n=100 a 4-round-old full count already beats the sample (eq. age 4.4 rounds)",
      ref="censold:m=4"),
    # ---------------- F. calendar and self-tuning
]

CALC = [
    C("D15", "G census plus samples", "Break-even n against a census", "样本追平定期普查所需的人数",
      "The number of people per round at which the sampled tracker matches a full count taken every 4 or 8 rounds.",
      "样本追踪器追平每 4 轮或 8 轮一次全量统计所需的每轮人数。",
      "plan", [BASE, "censold:m=4", "censold:m=8", "fulldecay:h=2"],
      dict(text="Break-even n (first n on the grid with NDCG(sampled) >= NDCG(censold m=4)) is above 300 in at least 70% of eligible logs.", op=">", value=300, share=0.7),
      "[INFERENCE] equivalent age 4.4 rounds at n=100 (tracking study)",
      "For each log and m in {4,8}: NDCG@100 (steady steps >= 8, mean over draws) of baseline at each n vs NDCG of censold:m; break-even = smallest n on the grid {10,30,100,300,1000,3000} with sampled >= census; censored if never. Report share of logs with break-even > 300 (censored counts as > 300) and the median."),
    C("D16", "C privacy", "Central DP price list", "中心化差分隐私的代价表",
      "How many people per round are needed for central differential privacy (each person limited to 10 rows) to cost less than 0.05 NDCG.",
      "中心化差分隐私（每人最多 10 条）要使 NDCG 损失小于 0.05，每轮需要多少人。",
      "dp", [BASE, "decay:h=2@gauss:eps=1000000,c=10", "decay:h=2@gauss:eps=1000000,c=3", "decay:h=2@cap:c=1"] + [f"decay:h=2@gauss:eps={e},c=10" for e in (1, 2, 4, 8)] + ["decay:h=2@gauss:eps=2,c=3", "decay:h=2@gauss:eps=8,c=3"],
      dict(text="At n=10000 (90 rounds), eps=2 per round loses < 0.05 against the non-private baseline in at least 50% of eligible logs; at n=1000, eps=8 does NOT reach < 0.05 in at least 50% of logs (any round length). Also reported: noise-only loss against the capped, noise-free reference (eps=1e6) on the same people.", cond="n10000_eps2_ok_and_n1000_eps8_fail"),
      "Balle-Wang analytic Gaussian (sigma 3.7/2.0/1.08 at eps 1/2/4 for L2 sensitivity 1); Desfontaines et al. / Korolova: thresholds need about 12 users at eps=1; amplification by subsampling not applied here (conservative)",
      "Per log, n in {1000,3000,10000} and round length (30 and 90 rounds): NDCG@100 steady mean of each gauss arm minus the non-private baseline (same people), and minus the noise-free capped reference (gauss eps=1e6 with the same cap). Table of both losses by (eps,c,n,n_win) with share of logs with loss < 0.05; smallest n with loss < 0.05 per eps. Noise is added to every item already seen in the catalogue (delta=1e-5, analytic Gaussian); epsilon is per round (the same person may be asked again, no composition across rounds is applied, amplification by subsampling is not applied); basic composition over 30 rounds multiplies epsilon by 30."),
    C("D17", "C privacy", "Local DP price list", "本地差分隐私的代价表",
      "Local differential privacy (optimised unary encoding, one row per person) at n=10000 loses more than 0.1 NDCG even at eps=4.",
      "本地差分隐私（优化的一元编码，每人一条）在 n=10000 时即使 eps=4 也损失超过 0.1 NDCG。",
      "dp", [BASE, "decay:h=2@cap:c=1"] + [f"decay:h=2@oue:eps={e}" for e in (1, 2, 4, 8)],
      dict(text="At n=10000 (90 rounds), eps=4 loses > 0.1 against the non-private baseline in at least 50% of eligible logs. Also reported: loss against the one-row-per-person noise-free reference.", op=">", value=0.1, share=0.5),
      "Wang et al. 2017: Var = 4 n e^eps/(e^eps-1)^2 per count (std 0.19 of share at eps=1, n=100 per LitPrivacy); local DP needs about 50-100x more people in our main study (X9)",
      "Same layout as D16 for the oue arms (reference decay:h=2@cap:c=1). Report loss by (eps,n,n_win) and the share of logs with loss > 0.1 at n=10000, eps=4 (90 rounds)."),
    C("D18", "C privacy", "How long a list can be published", "在计数阈值下能发布多长的榜单",
      "With a minimum running count of 5 reports per published item, a server with 100 people per round can publish fewer than 50 items.",
      "每个发布的物品要求累计至少 5 次汇报时，每轮 100 人的服务器只能发布不足 50 个物品。",
      "plan", [BASE, "athr:t=2", "athr:t=3", "athr:t=5", "athr:t=10"],
      dict(text="At n=100, t=5: median list length (len100 metric) < 50 and precision of the published items (prec100) >= 0.5.", op="<", value=50),
      "LitPrivacy: deployed thresholds 10-50 users (Chrome k=10-50, TrieHH theta=10-17); publishable items <= n*c/tau",
      "From the len100 and prec100 metrics of athr arms vs n in {10,30,100,300,1000,3000}: table of median published length and precision by (t,n); the smallest n for which t=5 publishes a full 100 items. In these logs each (person, item) pair appears once, so an item's count is its number of distinct reporters (decayed over rounds)."),
    C("D19", "D planning", "Equivalent age (replication)", "等效年龄（重复检验）",
      "A list from 100 people per round is worth a full count that is 3 to 6 rounds old, on the new categories and the new non-Amazon logs too.",
      "每轮 100 人的榜单，相当于一份 3 到 6 轮前的全量统计，在新类别和新的非 Amazon 日志上同样如此。",
      "plan", [BASE, "fulldecay:h=2"] + [f"full:lag={l}" for l in (0, 1, 2, 3, 4, 6, 8)],
      dict(text="Median equivalent age at n=100 over fresh and new logs lies in [3, 6] rounds.", lo=3, hi=6),
      "Tracking study (equivalent age 1.3-8, median 4.4 on Amazon)",
      "For each log and n: equivalent age = the lag L (linear interpolation between grid lags 0,1,2,3,4,6,8) at which NDCG of full:lag=L equals NDCG of the baseline tracker; censored at 8 if baseline is below lag 8 and 0 if above lag 0. Report by n in {30,100,300,1000} and by stratum (Amazon discovery / Amazon holdout and fresh / non-Amazon)."),
    C("D20", "D planning", "Best half-life by list size", "最佳半衰期随榜单长度的变化",
      "The best half-life is shorter for a top-10 list than for a top-300 list.",
      "前 10 名榜单的最佳半衰期短于前 300 名榜单。",
      "plan", [BASE] + HG,
      dict(text="In at least 60% of logs the best half-life for k=10 is <= the best half-life for k=300 (ties count as met), and strictly shorter in at least 25%.", share=0.6),
      "[INFERENCE] head items turn over faster than the tail",
      "Using nd10, nd30, nd100(from nd100 array), nd300 steady means of the seven decay arms at n=100: best half-life per k and log; share of logs with best(k=10) <= best(k=300), share strictly shorter; table of mean NDCG by (k,h)."),
    C("D21", "D planning", "People needed for 95% of the ceiling", "达到上限 95% 所需的人数",
      "Even 3000 people per round do not reach 95% of what a full count of all rounds with the same half-life achieves, in most logs.",
      "即使每轮 3000 人，也达不到用同样半衰期的全量统计所能达到的 95%，在多数日志里如此。",
      "plan", [BASE, "fulldecay:h=2"],
      dict(text="For k=100, NDCG(baseline, n=3000) < 0.95 x NDCG(fulldecay h=2) in at least 70% of logs eligible at n=3000.", share=0.7, level=0.95),
      "Dyer-Owen style bounds / main study: drift not sampling dominates; equivalent age 4.4 at n=100",
      "Ratio NDCG(baseline,n)/NDCG(fulldecay h=2) by n; for each log the smallest n with ratio >= 0.95 (censored if never) for k=100 and, from nd10/nd30/nd300 where available, k in {10,30,300}. Report share of logs that reach 95% by n, and the fitted slope of (1-ratio) against log n."),
    C("D22", "D planning", "Burn-in", "冷启动需要几轮",
      "With half-life 2 the list reaches 90% of its steady-state NDCG within 3 rounds at n=100 (a new tracker is useful after 3 rounds).",
      "半衰期为 2 时，榜单在 3 轮内达到稳态 NDCG 的 90%（新追踪器 3 轮后就有用）。",
      "plan", [BASE] + HG,
      dict(text="Median round at which mean NDCG (over draws) first reaches 90% of the steady mean is <= 3 for h=2 at n=100 (logs whose first scored round is already above 90% are left-censored and excluded). The h=inf row is reported but decides nothing (its list is not a steady state).", h2_max=3),
      "Exponential smoothing: effective memory 1/(1-2^(-1/h)) = 3.4 rounds at h=2",
      "From the per-step nd100 arrays (steps start at round 2): steady mean = mean over steps >= 8; first step index (round number = step) at which the draw-mean reaches 0.9 x steady; by half-life and n in {30,100,300}."),
    C("D23", "D planning", "Bad rounds", "糟糕的轮次有多糟",
      "At n=300 the 10th-percentile round is within 0.10 NDCG of the mean round; at n=30 it is more than 0.15 below.",
      "n=300 时，第 10 百分位的那一轮与平均轮相差不超过 0.10；n=30 时相差超过 0.15。",
      "plan", [BASE],
      dict(text="Median over logs of (mean - 10th percentile of per-round NDCG over steps >= 8 and draws): <= 0.10 at n=300 and >= 0.15 at n=30.", le300=0.10, ge30=0.15),
      "[INFERENCE] sampling noise shrinks as 1/sqrt(n); round-to-round shifts are part of drift",
      "From per-step nd100 arrays pooled over draws and steps >= 8: mean, p10, p5 and the gap by n; median over logs by n and stratum."),
    C("D24", "E self-monitoring", "Does the server's own score track true quality", "服务器自己的评分能反映真实质量吗",
      "NDCG of the published list against the next round's own sample (no ground truth needed) is rank-correlated with the true NDCG at 0.7 or more across logs, round lengths and n.",
      "已发布榜单对下一轮自身样本的 NDCG（不需要真值）与真实 NDCG 的秩相关在各日志、轮长、n 之间不低于 0.7。",
      "cal", [BASE],
      dict(text="MEDIAN over n values (those with at least 12 cells) of the within-n Spearman correlation across (log, n_win) cells between steady mean selfnd and steady mean nd100 of the baseline is >= 0.7; the pooled correlation across n is also reported (it is inflated because both rise with n).", rho=0.7),
      "LitMonitoring: the next-round sample gives an unbiased unnormalised DCG (prequential scheme, Gama et al. 2013); the sample-ideal normaliser is biased upward",
      "Cells from stages cal and plan (baseline arm only). x = mean of selfnd over steps >= 8 and draws, y = mean nd100. Spearman pooled; within each n; within each n_win; also Spearman between the cell-level ratio selfnd/nd100 and n (bias check)."),
    C("D25", "E self-monitoring", "Does list churn say how stale the list is", "榜单变动幅度能说明榜单有多旧吗",
      "The churn of the published list between rounds is rank-correlated at 0.5 or more with how far the list is behind a full count (NDCG(fulldecay h=2) minus NDCG(baseline)).",
      "榜单在相邻两轮之间的变动幅度，与它落后于全量统计的程度（全量衰减榜单 NDCG 减基线 NDCG）的秩相关不低于 0.5。",
      "cal", [BASE, "fulldecay:h=2"],
      dict(text="Not met expected: |Spearman| < 0.5 across (log, n_win) cells at n=100 (literature: churn measures noise and inertia, not staleness).", rho=0.5, expect_met=False),
      "Webber et al. 2010 (RBO between successive lists); Garg et al. 2022 (accuracy without labels is as hard as learning); decayed sums mechanically lower churn at longer half-lives",
      "Cells from stage cal at n=100 (all n_win): x = mean churn of baseline (steady), y = mean nd100 of fulldecay:h=2 minus baseline. Spearman pooled and by n_win; also against best-days stability."),
    C("D26", "E self-monitoring", "Split-half agreement as a quality meter", "半样本一致度作为质量指标",
      "The overlap between top-100 lists built from two random halves of the reporters is rank-correlated at 0.7 or more with the true NDCG, for n of 300 or more.",
      "由随机两半汇报者各自建出的前 100 名榜单的重合度，与真实 NDCG 的秩相关不低于 0.7（n 不小于 300 时）。",
      "plan", [BASE, "half:h=2"],
      dict(text="MEDIAN over n in {300,1000,3000} (those with at least 12 cells) of the within-n Spearman(agree, nd100) across logs is >= 0.7; at n=30 the within-n correlation is below 0.5.", rho=0.7, lo_rho=0.5),
      "LitMonitoring: split-half is the noise floor; Mogstad et al. 2022 rank confidence sets; informative only when n >= about 400",
      "From stage plan, arm half:h=2: agree metric (steady mean) and nd100 of the same arm; Spearman pooled over (log, n) for n >= 300, and for n = 30, 100 separately."),
    C("D27", "F calendar", "How flat is the best half-life in days", "最佳半衰期（按天）有多平坦",
      "On Amazon logs any half-life from 300 to 700 days is within 0.01 NDCG of the best of the days grid, for rounds of 3 to 9 months.",
      "在 Amazon 日志上，对于每轮 3 到 9 个月的情形，300 到 700 天之间的任意半衰期都与天数网格中的最佳值相差不超过 0.01 NDCG。",
      "cal", [BASE] + [f"decay:days={d}" for d in DAYS],
      dict(text="For at least 80% of Amazon (log, n_win in {30,45,60,90}) cells the NDCG regret of each of 300, 400, 534, 700 days is <= 0.01 relative to the best of the days values. Reported separately for the clean set (holdout_fresh and new logs) because the anchor numbers come from older logs.", share=0.8, margin=0.01),
      "Our tournament rounds 3 (400 days +0.057, 534 days +0.054, 700 days +0.031 at 90 rounds on holdout)",
      "Per Amazon log and n_win: NDCG (steady mean) by days; regret to best; share of cells where regret <= 0.01 for each days value; best days distribution by n_win; figure of mean regret against days."),
    C("D28", "F calendar", "Is the calendar half-life stable on non-Amazon logs", "非 Amazon 日志上日历半衰期稳定吗",
      "On each non-Amazon log the best half-life in days stays within a factor of 2 across 45, 60 and 90 rounds.",
      "在每个非 Amazon 日志上，最佳半衰期（按天）在 45、60、90 轮之间变化不超过 2 倍。",
      "cal", [BASE] + [f"decay:days={d}" for d in DAYS] + HG,
      dict(text="At least 60% of eligible non-Amazon logs have max/min best days <= 2 across n_win 45, 60, 90 (decay:days arms only). A log is censored (left out, counted) when its best days is at the edge of the grid (3 or 1500 days) at any of the three widths or when all days arms are within 0.001 NDCG of each other.", share=0.6, factor=2.0),
      "Earlier calendar check (11 of 18 logs same in days; Amazon 494-587; Gowalla 83); [INFERENCE] growing platforms prefer shorter spans",
      "Per non-Amazon log: best days among the eight decay:days arms at each n_win in {45,60,90} (n=100); ratio max/min; also list the best days per log and width; stratify new logs vs old."),
    C("D29", "F calendar", "Self-tuning half-life on other kinds of logs", "其他类型日志上的自调半衰期",
      "On non-Amazon logs a server that picks its half-life from its own samples recovers at least half of the gain of the best fixed half-life over half-life 2 in 60% of cases and is never worse than 0.01; on Amazon logs it is worse than 0.01 in at least 30% of cases.",
      "在非 Amazon 日志上，用自己样本选半衰期的服务器，在 60% 的情形里追回最佳固定半衰期相对半衰期 2 的至少一半收益，且从不差于 0.01；在 Amazon 日志上则有至少 30% 的情形差于 0.01。",
      "cal", [BASE, "selfsel"] + HG,
      dict(text="Non-Amazon (log, n_win) cells at n=100: share with (selfsel - base) >= 0.5 x (best fixed - base) is >= 0.6 among cells where best fixed - base >= 0.01, and no cell has selfsel - base < -0.01; Amazon cells: share with selfsel - base < -0.01 is >= 0.3.", share=0.6),
      "Tracking study PD (Other stratum 9 of 12 cells recovered half, 0 harmful; Amazon harmful in 51% of 59 cells)",
      "From stage cal arms: per cell selfsel vs baseline and vs the best of the seven half-life arms. Report the stratified shares above, by n_win, and the chosen half-life index distribution (chosen_h_idx)."),
    C("D30", "H metric choice", "Do conclusions depend on the metric", "结论是否依赖评价指标",
      "The ordering of the 30-odd aggregation arms by mean difference is the same under recall@100, NDCG@10, NDCG@30 and NDCG@300 as under NDCG@100 (rank correlation 0.9 or more), and no gain or loss verdict flips sign.",
      "三十多个聚合方式按平均差排序，在 recall@100、NDCG@10、NDCG@30、NDCG@300 下与 NDCG@100 下一致（秩相关不低于 0.9），且没有任何增益或损失的结论反号。",
      "agg", [BASE],
      dict(text="Among agg arms other than the oracle and census arms (fulldecay, cens, censold) and with |mean diff in NDCG@100| <= 0.02, the Spearman between mean diffs (arm minus baseline, n=100, 30 rounds, all logs) under NDCG@100 and under each other metric is >= 0.9; number of arms whose sign flips with |diff| > 0.005 under NDCG@100 is 0.", rho=0.9),
      "LitMonitoring: NDCG@k vs recall@k vs RBO agree on large effects, can disagree on small ones",
      "From stage agg metrics (nd10, nd30, nd300, rec100) and nd100: per arm mean diff vs baseline at n=100 over logs; Spearman across arms; sign flips."),
]
H = H + CALC

STAGES = dict(
    agg=dict(logs=["discovery", "holdout_old", "holdout_fresh", "new_discovery", "new_holdout"], n=[30, 100, 300, 1000], draws=16, n_win=[30, 60], seed_tag="agg"),
    seas=dict(logs=["discovery", "holdout_old", "holdout_fresh", "new_discovery", "new_holdout"], n=[100, 300], draws=16, n_win=[45, 60, 90], seed_tag="seas"),
    plan=dict(logs=["discovery", "holdout_old", "holdout_fresh", "new_discovery", "new_holdout"], n=[10, 30, 100, 300, 1000, 3000], draws=16, n_win=[30], seed_tag="plan"),
    cal=dict(logs=["discovery", "holdout_old", "holdout_fresh", "new_discovery", "new_holdout"], n=[100], draws=16, n_win=[20, 30, 45, 60, 90], seed_tag="cal"),
    dp=dict(logs=["discovery", "holdout_old", "holdout_fresh", "new_discovery", "new_holdout"], n=[1000, 3000, 10000], draws=12, n_win=[30, 90], seed_tag="dp"),
)

REG = dict(
    version=5, generation="deep", baseline=BASE, margin=0.01, alpha=0.05, bh_q=0.10, min_logs=6, clean_set=["holdout_fresh", "new_holdout"],
    split=dict(salt=SALT, discovery=R.REG["split"]["discovery"], holdout_old=R.REG["split"]["holdout_old"], holdout_fresh=R.REG["split"]["holdout_fresh"],
               new_discovery=nd, new_holdout=nh),
    stages=STAGES, hypotheses=H,
    protocol=dict(
        paired=("Primary combination: n=100, 30 rounds, nd100 steady-state (steps >= 8) mean over draws, one diff per log (arm minus ref). "
                "Discovery = logs of groups discovery + new_discovery; holdout = holdout_old + holdout_fresh + new_holdout. Per hypothesis one-sided t p-value on the discovery logs; "
                "Benjamini-Hochberg at q=0.10 across all paired hypotheses whose prediction claims a gain, a loss bound or a difference; "
                "a hypothesis is CONFIRMED only if it passes BH on discovery with mean diff >= margin (superiority) AND on the holdout logs the diff has the same sign, mean >= margin and one-sided p < 0.05 / (number of hypotheses selected). "
                "Non-inferiority (D08): lower one-sided t bound of the diff > -0.005 and churn cut >= 30% on discovery and holdout. "
                "Plain labels come from the 95% t interval of the mean diff over all logs: GAIN (lower bound > 0 and mean >= 0.01), SMALL GAIN, NO DIFFERENCE (interval inside +/- 0.01), SMALL LOSS, LOSS (upper bound < 0 and mean <= -0.01), INCONCLUSIVE. "
                "A prediction of 'no gain' counts as MET only if the upper 95% bound of the mean diff is below +0.01 (a gain of 0.01 is ruled out); a prediction of a gain counts as MET only if the holdout logs alone show label GAIN. "
                "CONFIRMED additionally requires a positive mean on the clean replication set (holdout_fresh plus new_holdout). Rows of secondary arms and of non-primary (people, rounds) combinations are EXPLORATORY (no multiplicity control). "
                "D03, D12 and D13 exclude the smoke-test log amazon-musical-instruments from the discovery set."),
        calc="Each calculator hypothesis states a numeric prediction; the verdict is MET or NOT MET on the stated numbers plus the full table. No multiplicity adjustment: these are measurements with pre-set thresholds.",
        eligibility="A (log, n, n_win) cell is run only if at least 10 steps have all of rounds w-7..w+1 with >= 5n active people (same rule as before).",
        seasons="seas arms are only run at round lengths of at most 182 days (lag of at least 2 rounds)."),
)

if __name__ == "__main__":
    json.dump(REG, open("configs/deep_registry.json", "w"), indent=1, ensure_ascii=False)
    print(len(H), "hypotheses;", {k: len(v) for k, v in R.REG["split"].items() if isinstance(v, (list,))} if False else "ok")
    print(REG["split"]["new_discovery"], REG["split"]["new_holdout"])
