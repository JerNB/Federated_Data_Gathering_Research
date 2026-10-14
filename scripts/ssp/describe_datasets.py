"""Write one row per dataset: size, concentration, Zipf slope, and the theory-predicted sample sizes.

Output: results/small_sample_popularity/01_datasets/datasets.json (+ printed table)
"""
import json, sys, time
sys.path.insert(0, "scripts/ssp")
import numpy as np
from frame import load_frame
import scaling as SC

NAMES = ["ml-latest", "ml-1m", "amazon-video-games", "amazon-musical-instruments", "amazon-software",
         "amazon-movies-tv", "amazon-books", "gowalla", "brightkite", "kuairand-pure", "lastfm-1k"]
rows = []
for n in NAMES:
    fr = load_frame(n)
    q = np.sort(fr.count_A)[::-1] / fr.n_pop            # inclusion rate per user in the frame
    z = SC.zipf_slope(fr.count_A, 10, min(1000, int((fr.count_A > 0).sum() * 0.5)))
    zl = SC.zipf_slope_local(fr.count_A, (10, 30, 100))
    q = q[q > 0]
    lam = lambda N: N * q
    row = dict(
        name=n, n_pop=fr.n_pop, n_eval=fr.meta["n_eval"], n_items=fr.n_items,
        items_seen=int((fr.count_A > 0).sum()), min_rows=fr.min_rows,
        rows_collection=fr.meta["n_rows_coll"], rows_future=fr.meta["rows_future"],
        mean_rows_per_user=float(fr.n_c.mean()), median_rows_per_user=float(np.median(fr.n_c)),
        top1_share=float(q[0]), top10_share=float(q[:10].sum()), zipf_s=-z["s"], zipf_r2=z["r2"],
        zipf_local={str(k): -v for k, v in zl.items()},
        n_resolve_top10=float(SC.n_resolve_topm(q, 10, 0.05)),
        n_resolve_top100=float(SC.n_resolve_topm(q, 100, 0.05)),
        depth_at_N={str(N): int(SC.dyer_owen_depth(lam(N))) for N in (50, 200, 1000, 5000)},
    )
    rows.append(row)
    print(f"{n:27s} pop={fr.n_pop:7d} items={row['items_seen']:7d} top1={row['top1_share']:.3f} "
          f"s={row['zipf_s']:.2f} Nres10={row['n_resolve_top10']:.3g} Nres100={row['n_resolve_top100']:.3g} "
          f"depth@200={row['depth_at_N']['200']} depth@5000={row['depth_at_N']['5000']}", flush=True)
json.dump(rows, open("results/small_sample_popularity/01_datasets/datasets.json", "w"), indent=1)
