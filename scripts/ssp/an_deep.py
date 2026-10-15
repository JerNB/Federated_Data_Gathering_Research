"""Deep programme analysis: runs every hypothesis analysis, writes tables, verdict JSON and bilingual hypothesis cards.

    .venv/bin/python3 scripts/ssp/an_deep.py [--rows PATH] [--out DIR] [--docs DIR]

Reads configs/deep_registry.json (frozen before the run) and results/small_sample_popularity/02_runs/deep.jsonl.
Writes <out>/deep_*.csv, <out>/deep_verdicts.json and <docs>/11_deep_cards_{en,zh}.md.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import an_deep_lib as L  # noqa: E402
import an_deep_paired as PA  # noqa: E402

OUT = "results/small_sample_popularity/03_tables"
ROWS = "results/small_sample_popularity/02_runs/deep.jsonl"
DOCS = "docs/small_sample"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{d}f}"


def paired_text(h, r):
    if not r.get("n_logs") or r.get("frac_pos") is None:
        return "No logs at the primary combination.", "主要组合下没有日志。"
    lab = r["label"]
    en = (f"Primary rule `{h['primary']}` against `{h['ref']}` (NDCG@100, higher is better): average change {fmt(r['mean'], 4)} over {r['n_logs']} logs "
          f"(95% interval {fmt(r['lo'], 4)} to {fmt(r['hi'], 4)}); better in {r['frac_pos']:.0%} of logs. Amazon logs {fmt(r['amazon'], 4)}, other logs {fmt(r['other'], 4)}. "
          f"Plain label: {lab}. ")
    zh = (f"主规则 `{h['primary']}` 对比 `{h['ref']}`（NDCG@100，越高越好）：{r['n_logs']} 份日志的平均变化 {fmt(r['mean'], 4)}"
          f"（95% 区间 {fmt(r['lo'], 4)} 到 {fmt(r['hi'], 4)}）；{r['frac_pos']:.0%} 的日志更好。Amazon 日志 {fmt(r['amazon'], 4)}，其他日志 {fmt(r['other'], 4)}。"
          f"通俗结论：{ZH_LABEL.get(lab, lab)}。")
    if r.get("selected_discovery") is not None and h["direction"] == "sup":
        if r.get("confirmed"):
            en += f"Pre-registered test: CONFIRMED (discovery {fmt(r['disc_mean'], 4)}, holdout {fmt(r['hold_mean'], 4)}, clean replication set {fmt(r['clean_mean'], 4)} over {r['n_clean']} logs)."
            zh += f"事先注册的检验：确认（发现用日志 {fmt(r['disc_mean'], 4)}，留出日志 {fmt(r['hold_mean'], 4)}，干净的重复检验集 {fmt(r['clean_mean'], 4)}，{r['n_clean']} 份日志）。"
        else:
            en += (f"Pre-registered test: not confirmed (discovery {fmt(r['disc_mean'], 4)}"
                   f"{', passed the false-discovery filter' if r['selected_discovery'] else ', did not pass the false-discovery filter'}; holdout {fmt(r['hold_mean'], 4)}).")
            zh += (f"事先注册的检验：未确认（发现用日志 {fmt(r['disc_mean'], 4)}"
                   f"{'，通过了错误发现率筛选' if r['selected_discovery'] else '，没有通过错误发现率筛选'}；留出日志 {fmt(r['hold_mean'], 4)}）。")
    return en, zh


ZH_LABEL = {"GAIN": "有提升", "SMALL GAIN": "有小幅提升", "NO DIFFERENCE": "没有差别", "SMALL LOSS": "有小幅下降", "LOSS": "变差", "INCONCLUSIVE": "无法判断", "NOT TESTABLE": "无法检验"}


def main():
    reg = json.load(open("configs/deep_registry.json"))
    rows = arg("--rows", ROWS)
    out = arg("--out", OUT + "/deep")
    docs = arg("--docs", DOCS)
    os.makedirs(out, exist_ok=True)
    deep = L.Deep(rows, reg)
    print(f"{len(deep.rows)} result rows from {rows}")
    have = {r["key"]: set(r["arms"]) for r in deep.rows}
    missing = {}
    for stage in reg["stages"]:
        p = f"results/small_sample_popularity/02_runs/deep_expected_{stage}.json"
        if os.path.exists(p):
            exp = json.load(open(p))
            missing[stage] = [k for k, arms in exp.items() if k not in have or not set(arms) <= have[k]]
    inc = {k: v for k, v in missing.items() if v}
    print("incomplete cells per stage:", {k: len(v) for k, v in inc.items()} or "none")
    verdicts = {}
    for r in PA.run(deep, reg, out):
        verdicts[r["id"]] = r
    for modname in ("an_deep_calc1", "an_deep_calc2"):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            print("missing module", modname, e)
            continue
        for i, fn in mod.RUNNERS.items():
            try:
                verdicts[i] = dict(kind="calc", **fn(deep, reg, out))
            except Exception as e:
                verdicts[i] = dict(kind="calc", id=i, verdict="NOT TESTABLE", numbers={"error": repr(e)[:200]},
                                   summary_en=f"Analysis failed: {e!r}", summary_zh=f"分析失败：{e!r}", tables=[])
                traceback.print_exc()
    json.dump(verdicts, open(f"{out}/deep_verdicts.json", "w"), indent=1, default=float)
    # scoreboard + cards
    sb_en, sb_zh, cards_en, cards_zh = [], [], [], []
    for h in reg["hypotheses"]:
        v = verdicts.get(h["id"])
        if not v:
            res_en = res_zh = "no result"
            body_en = body_zh = "No rows were available for this hypothesis."
            body_zh = "没有可用于该假设的结果。"
        elif v["kind"] == "paired":
            lab = v["label"]
            met = v["prediction_met"]
            res_en = f"{lab}; prediction {'met' if met else 'not met' if met is not None else 'n/a'}" + ("; CONFIRMED" if v.get("confirmed") else "")
            res_zh = f"{ZH_LABEL.get(lab, lab)}；预测{'成立' if met else '不成立' if met is not None else '无法判断'}" + ("；已确认" if v.get("confirmed") else "")
            body_en, body_zh = paired_text(h, v)
            body_en += f" Prediction check: {v['why']}. Same direction at {v['robust_same']} of {v['robust_total']} other (people, round length) combinations."
            body_zh += f" 预测检查：{v['why']}。在其他 {v['robust_total']} 个（人数、轮长）组合中，有 {v['robust_same']} 个方向相同。"
        else:
            res_en = f"{v['verdict']}"
            res_zh = {"MET": "预测成立", "NOT MET": "预测不成立", "NOT TESTABLE": "无法检验"}.get(v["verdict"], v["verdict"])
            body_en, body_zh = v.get("summary_en", ""), v.get("summary_zh", "")
        sb_en.append(f"| {h['id']} | {h['title_en']} | {res_en} |")
        sb_zh.append(f"| {h['id']} | {h['title_zh']} | {res_zh} |")
        ptxt = h["prediction"]["text"]
        cards_en.append(f"### {h['id']}. {h['title_en']} ({h['family']})\n\n- **Question.** {h['statement_en']}\n- **Registered prediction.** {ptxt}\n- **Literature anchor.** {h['anchor']}\n- **Result.** {res_en}\n\n{body_en}\n")
        cards_zh.append(f"### {h['id']}. {h['title_zh']}（{h['family']}）\n\n- **问题。** {h['statement_zh']}\n- **事先注册的预测（英文原文）。** {ptxt}\n- **文献依据（英文原文）。** {h['anchor']}\n- **结果。** {res_zh}\n\n{body_zh}\n")
    head_en = "# 11. Deep programme: hypothesis cards (generated)\n\nOne card per hypothesis, produced by `scripts/ssp/an_deep.py` from the frozen registry and the run. Tables: `results/small_sample_popularity/03_tables/deep/`.\n"
    head_zh = "# 11. 深度研究计划：假设卡片（自动生成）\n\n每个假设一张卡片，由 `scripts/ssp/an_deep.py` 根据冻结的注册表和运行结果生成。表格见 `results/small_sample_popularity/03_tables/deep/`。\n"
    note_en = ("\n**Completeness:** " + (", ".join(f"{k}: {len(v)} planned cells missing" for k, v in inc.items()) if inc else "every planned cell of every stage is present.") + "\n")
    note_zh = ("\n**完整性：** " + ("，".join(f"{k}：缺少 {len(v)} 个计划格子" for k, v in inc.items()) if inc else "所有阶段的所有计划格子都已完成。") + "\n")
    head_en += note_en + "\n## Scoreboard\n\n| # | Hypothesis | Result |\n| --- | --- | --- |\n"
    head_zh += note_zh + "\n## 成绩单\n\n| 编号 | 假设 | 结果 |\n| --- | --- | --- |\n"
    open(f"{docs}/11_deep_cards_en.md", "w").write(head_en + "\n".join(sb_en) + "\n\n## Cards\n\n" + "\n".join(cards_en))
    open(f"{docs}/11_deep_cards_zh.md", "w").write(head_zh + "\n".join(sb_zh) + "\n\n## 卡片\n\n" + "\n".join(cards_zh))
    print("wrote cards;", sum(1 for v in verdicts.values() if v.get("kind") == "paired"), "paired,", sum(1 for v in verdicts.values() if v.get("kind") == "calc"), "calc verdicts")


if __name__ == "__main__":
    main()
