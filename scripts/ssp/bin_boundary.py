"""Bin the probe output by the sample's rank-100 count; write results/small_sample_popularity/03_tables/probe_boundary_bins.csv"""
import csv, pickle, sys
import numpy as np
out = []
for k in sys.argv[1:]:
    o = pickle.load(open(f"/tmp/h_check_{k}.pkl", "rb"))
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 40), (40, 80), (80, 1e9)]:
        ds = {}
        for d, N, h, x in o:
            if lo <= h < hi:
                ds.setdefault(d, []).append(x)
        if ds:
            m = [np.mean(v) for v in ds.values()]
            allx = [x for v in ds.values() for x in v]
            out.append(dict(k=k, rank100_count_lo=lo, rank100_count_hi=hi, draws=len(allx), datasets=len(ds),
                            mean_overlap100=round(float(np.mean(allx)), 3), ds_min=round(min(m), 3), ds_max=round(max(m), 3)))
w = csv.DictWriter(open("results/small_sample_popularity/03_tables/probe_boundary_bins.csv", "w", newline=""), fieldnames=list(out[0]))
w.writeheader(); w.writerows(out)
