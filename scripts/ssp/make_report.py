"""Assemble the two results documents (English, Chinese) from the verdict files, tables and figures.

    .venv/bin/python3 scripts/ssp/make_report.py

Reads   results/small_sample_popularity/03_tables/*_verdicts.json, 04_figures/*.png
        docs/small_sample/_summary_en.md, _summary_zh.md  (hand-written headline + interpretation, optional)
Writes  docs/small_sample/02_results_en.md, 02_results_zh.md

Nothing here invents a number: every sentence under a heading is the `reading_en` / `reading_zh` the analysis
script wrote next to the numbers it computed.
"""
import glob
import json
import os

BASE = "results/small_sample_popularity"
DOCS = "docs/small_sample"
FIG = "../../results/small_sample_popularity/04_figures"

FAMILIES = [
    # (verdict file, id prefix label en, zh, figure prefixes)
    ("an_confirm2", "Confirmatory claims", "确认性假设", ("F2-C",)),
    ("an_c3_checks", "Companions to C2/C3: in-universe score and smaller sample sizes", "C2/C3 的配套检验：候选集内得分与更小的样本量", ("an_c3_checks_",)),
    ("an_window", "Exploratory: stale list, shrinkage and recency weighting (window truth)", "探索性：旧榜、收缩和近期加权（窗口真相）", ("an_window_",)),
    ("an_explore_a", "Exploratory: scaling, theory check, drift", "探索性：缩放规律、理论检验、漂移", ("an_explore_a_",)),
    ("an_explore_b", "Exploratory: rank sets, biased samplers, recency, subgroups, privacy", "探索性：排名置信集、有偏抽样、近期偏好、子群体、隐私", ("an_explore_b_",)),
    ("an_hunt", "Exploratory: surprise hunt with fresh-seed replication", "探索性：惊喜搜索与新种子复现", ("hunt_",)),
    ("an_extra", "Exploratory: profile length vs training depth, self-assessment, unbiased vs plain count", "探索性：历史长度与训练深度、自我评估、无偏与朴素计数", ("an_extra_",)),
    ("an_curves", "Exploratory: two speeds of learning", "探索性：两种学习速度", ("an_curves_",)),
    ("an_stoprule", "Exploratory: a stop rule from inside the sample", "探索性：来自样本内部的切换规则", ("an_stoprule_",)),
]

BADGE_EN = {"supported": "SUPPORTED", "not supported": "NOT SUPPORTED", "mixed": "MIXED", "not testable": "NOT TESTABLE"}
BADGE_ZH = {"supported": "支持", "not supported": "不支持", "mixed": "混合", "not testable": "无法检验"}


def load(name):
    p = f"{BASE}/03_tables/{name}_verdicts.json"
    return json.load(open(p)) if os.path.exists(p) else []


def figs_for(prefixes, vid):
    out = []
    for pre in prefixes:
        for f in sorted(glob.glob(f"{BASE}/04_figures/{pre}*.png")):
            out.append(os.path.basename(f))
    return out


def fig_match(figs, vid):
    v = vid.lower().replace("-", "").replace("_", "")
    hit = [f for f in figs if v in f.lower().replace("-", "").replace("_", "")]
    return hit


def numbers_table(nums, lang):
    flat = []
    for k, v in (nums or {}).items():
        if isinstance(v, (dict, list)):
            continue
        flat.append((k, v))
    if not flat:
        return ""
    head = "| Number | Value |\n| --- | --- |\n" if lang == "en" else "| 数值 | 取值 |\n| --- | --- |\n"
    return head + "\n".join(f"| `{k}` | {v if not isinstance(v, float) else round(v, 4)} |" for k, v in flat[:14]) + "\n"


def render(lang):
    en = lang == "en"
    parts = []
    head = f"{DOCS}/_summary_{lang}.md"
    if os.path.exists(head):
        parts.append(open(head).read().rstrip() + "\n")
    else:
        parts.append("# Results\n" if en else "# 结果\n")
    parts.append(("\n## Every result, in order\n\nEach block is: the claim, its verdict, a plain reading, the rule that was fixed in advance, and the figures. Numbers come straight from `results/small_sample_popularity/03_tables/`.\n"
                  if en else
                  "\n## 全部结果（按顺序）\n\n每一块包含：假设、结论、白话解读、事先定好的判定规则，以及图。数值直接来自 `results/small_sample_popularity/03_tables/`。\n"))
    for name, t_en, t_zh, prefixes in FAMILIES:
        items = load(name)
        figs = figs_for(prefixes, "")
        if not items and not figs:
            continue
        parts.append(f"\n### {t_en if en else t_zh}\n")
        used = set()
        for it in items:
            vid = it.get("id", "?")
            title = it.get("title_en" if en else "title_zh", "")
            badge = (BADGE_EN if en else BADGE_ZH).get(it.get("verdict"), it.get("verdict"))
            parts.append(f"\n#### {vid}. {title} — **{badge}**\n")
            parts.append(it.get("reading_en" if en else "reading_zh", "") + "\n")
            if it.get("rule"):
                parts.append(("\n*Rule fixed in advance:* " if en else "\n*事先定好的规则：* ") + str(it["rule"]) + "\n")
            tbl = numbers_table(it.get("numbers"), lang)
            if tbl:
                parts.append("\n" + tbl)
            for f in fig_match(figs, vid):
                used.add(f)
                parts.append(f"\n![{f}]({FIG}/{f})\n")
        rest = [f for f in figs if f not in used]
        for f in rest:
            parts.append(f"\n![{f}]({FIG}/{f})\n")
    return "\n".join(parts)


if __name__ == "__main__":
    for lang in ("en", "zh"):
        text = render(lang)
        out = f"{DOCS}/02_results_{lang}.md"
        open(out, "w").write(text)
        print("wrote", out, len(text), "chars")
