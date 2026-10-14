"""Share of the panel's real later choices that fall inside the top-J_CAP_E2 items by full-group count.

A personalised model here can only recommend a candidate universe of J_CAP_E2 items; where this share is low,
a 'popularity wins' verdict is partly the cap talking. Output: results/small_sample_popularity/03_tables/cand_share.csv
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import exp_core as X

os.makedirs("results/small_sample_popularity/03_tables", exist_ok=True)
rows = []
for ds in ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software",
           "amazon-movies-tv", "amazon-books", "gowalla", "brightkite", "kuairand-pure", "lastfm-1k"]:
    fr = X.frame(ds)
    prow = X.panel_rows(fr)
    top = np.argsort(-fr.count_A, kind="stable")[:X.J_CAP_E2]
    inside = np.zeros(fr.n_items, bool)
    inside[top] = True
    ef = fr.e_f[prow]
    hit = inside[ef.indices].sum()
    rows.append((ds, round(float(hit) / max(ef.nnz, 1), 4), len(prow), int(ef.nnz)))
    print(ds, rows[-1][1], flush=True)
with open("results/small_sample_popularity/03_tables/cand_share.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "share", "panel_users", "panel_future_choices"]); w.writerows(rows)
