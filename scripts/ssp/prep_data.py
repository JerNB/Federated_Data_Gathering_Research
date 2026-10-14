"""Convert each public dataset into one compact format: datasets/npz/<name>.npz.

Every file holds one row per (user, item) pair, the FIRST time the user
interacted with the item, sorted by (user, time):

    user  int32   dense 0..U-1
    item  int32   dense 0..I-1
    ts    int64   unix seconds (0 when the source has no timestamps)
    r     float32 rating as given (1.0 for implicit sources)
    positive_threshold  float, rows with r >= it count as a "choice"
    has_time            bool

Run:  .venv/bin/python3 scripts/ssp/prep_data.py ml-1m amazon_video_games ...
      .venv/bin/python3 scripts/ssp/prep_data.py --all
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
import tarfile
import zipfile
from array import array

import numpy as np

RAW = "datasets/raw"
OUT = "datasets/npz"


def _finish(name, user, item, ts, r, positive_threshold, has_time, extra=None):
    """Densify ids, keep the first (user,item) event, sort by (user, ts), save."""
    user = np.asarray(user)
    item = np.asarray(item)
    ts = np.asarray(ts, dtype=np.int64)
    r = np.asarray(r, dtype=np.float32)
    _, user = np.unique(user, return_inverse=True)
    uniq_items, item = np.unique(item, return_inverse=True)
    user = user.astype(np.int64)
    item = item.astype(np.int64)
    n_items = int(item.max()) + 1
    # first event per (user, item)
    o = np.lexsort((ts, item, user))
    user, item, ts, r = user[o], item[o], ts[o], r[o]
    key = user * n_items + item
    keep = np.ones(len(key), bool)
    keep[1:] = key[1:] != key[:-1]
    user, item, ts, r = user[keep], item[keep], ts[keep], r[keep]
    o = np.lexsort((item, ts, user))
    user, item, ts, r = user[o], item[o], ts[o], r[o]
    os.makedirs(OUT, exist_ok=True)
    arrays = dict(user=user.astype(np.int32), item=item.astype(np.int32), ts=ts, r=r,
                  positive_threshold=np.float32(positive_threshold), has_time=np.bool_(has_time),
                  raw_item_ids=uniq_items.astype(str))
    if extra:
        arrays.update(extra)
    np.savez(f"{OUT}/{name}.npz", **arrays)
    print(f"{name}: {len(user):,} rows, {int(user.max()) + 1:,} users, {n_items:,} items, "
          f"time={'yes' if has_time else 'no'}", flush=True)


class _Ids:
    """First-seen string -> dense int id."""

    def __init__(self):
        self.d: dict[str, int] = {}

    def __call__(self, s: str) -> int:
        v = self.d.get(s)
        if v is None:
            v = self.d[s] = len(self.d)
        return v


def _iso_to_epoch(strs: list[str]) -> np.ndarray:
    return np.array([s[:19] for s in strs], dtype="datetime64[s]").astype(np.int64)


def prep_ml_latest():
    z = np.load("artifacts/overnight_cache/ratings_v1.npz")
    user, item, r2, ts = z["user"], z["item"], z["r2"], z["ts"]
    genres = {}
    with open("ml-latest/movies.csv", newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            genres[int(row[0])] = row[2]
    _finish("ml-latest", user, item, ts, r2.astype(np.float32) / 2.0, 4.0, True,
            extra=None)
    # genres aligned to the dense item ids
    d = np.load(f"{OUT}/ml-latest.npz")
    raw = d["raw_item_ids"].astype(np.int64)
    g = np.array([genres.get(int(x), "") for x in raw])
    arrays = {k: d[k] for k in d.files}
    arrays["item_genres"] = g
    np.savez(f"{OUT}/ml-latest.npz", **arrays)


def prep_ml_1m():
    with zipfile.ZipFile(f"{RAW}/ml-1m.zip") as z:
        txt = z.read("ml-1m/ratings.dat").decode("latin-1").replace("::", ",")
        mv = z.read("ml-1m/movies.dat").decode("latin-1").splitlines()
    a = np.loadtxt(io.StringIO(txt), delimiter=",", dtype=np.int64)
    genres = {int(l.split("::")[0]): l.split("::")[2] for l in mv}
    _finish("ml-1m", a[:, 0], a[:, 1], a[:, 3], a[:, 2], 4.0, True)
    d = np.load(f"{OUT}/ml-1m.npz")
    arrays = {k: d[k] for k in d.files}
    arrays["item_genres"] = np.array([genres.get(int(x), "") for x in d["raw_item_ids"]])
    np.savez(f"{OUT}/ml-1m.npz", **arrays)


def _amazon(name, fname):
    uid, iid = _Ids(), _Ids()
    u, i, t, r = array("i"), array("i"), array("q"), array("f")
    with gzip.open(f"{RAW}/{fname}", "rt", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)  # user_id,parent_asin,rating,timestamp (ms)
        for row in rd:
            u.append(uid(row[0]))
            i.append(iid(row[1]))
            r.append(float(row[2]))
            t.append(int(row[3]) // 1000)
    _finish(name, u, i, t, r, 4.0, True)


def _checkins(name, fname):
    uid, iid = _Ids(), _Ids()
    u, i = array("i"), array("i")
    ts_str: list[str] = []
    with gzip.open(f"{RAW}/{fname}", "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or not p[4]:
                continue
            u.append(uid(p[0]))
            i.append(iid(p[4]))
            ts_str.append(p[1])
    ts = _iso_to_epoch(ts_str)
    _finish(name, u, i, ts, np.ones(len(u), np.float32), 1.0, True)


def prep_lastfm():
    uid, iid = _Ids(), _Ids()
    u, i = array("i"), array("i")
    ts_chunks, buf = [], []
    with tarfile.open(f"{RAW}/lastfm_1k.tar.gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("userid-timestamp-artid-artname-traid-traname.tsv"))
        f = io.TextIOWrapper(tf.extractfile(member), encoding="utf-8", errors="replace")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6 or not p[3]:
                continue
            u.append(uid(p[0]))
            i.append(iid(p[3]))  # artist name as the item
            buf.append(p[1])
            if len(buf) >= 1_000_000:
                ts_chunks.append(_iso_to_epoch(buf))
                buf = []
    if buf:
        ts_chunks.append(_iso_to_epoch(buf))
    ts = np.concatenate(ts_chunks)
    _finish("lastfm-1k", u, i, ts, np.ones(len(u), np.float32), 1.0, True)


def prep_kuairand():
    uid, iid = _Ids(), _Ids()
    u, i, t = array("i"), array("i"), array("q")
    with tarfile.open(f"{RAW}/kuairand_pure.tar.gz") as tf:
        for m in tf.getmembers():
            if "log_standard" in m.name and m.name.endswith(".csv"):
                f = io.TextIOWrapper(tf.extractfile(m), encoding="utf-8")
                rd = csv.DictReader(f)
                for row in rd:
                    if row["is_click"] != "1":
                        continue
                    u.append(uid(row["user_id"]))
                    i.append(iid(row["video_id"]))
                    t.append(int(row["time_ms"]) // 1000)
    _finish("kuairand-pure", u, i, t, np.ones(len(u), np.float32), 1.0, True)


JOBS = {
    "ml-latest": prep_ml_latest,
    "ml-1m": prep_ml_1m,
    "amazon-video-games": lambda: _amazon("amazon-video-games", "amazon_video_games.csv.gz"),
    "amazon-musical-instruments": lambda: _amazon("amazon-musical-instruments", "amazon_musical_instruments.csv.gz"),
    "amazon-movies-tv": lambda: _amazon("amazon-movies-tv", "amazon_movies_tv.csv.gz"),
    "amazon-books": lambda: _amazon("amazon-books", "amazon_books.csv.gz"),
    "amazon-software": lambda: _amazon("amazon-software", "amazon_software.csv.gz"),
    **{f"amazon-{s.lower().replace('_', '-')}": (lambda s=s: _amazon(f"amazon-{s.lower().replace('_', '-')}", f"amazon_{s.lower()}.csv.gz"))
       for s in ("Toys_and_Games", "Grocery_and_Gourmet_Food", "Pet_Supplies", "Sports_and_Outdoors",
                 "Cell_Phones_and_Accessories", "Arts_Crafts_and_Sewing", "Office_Products",
                 "Patio_Lawn_and_Garden", "Health_and_Household", "Electronics", "Automotive")},
    "gowalla": lambda: _checkins("gowalla", "gowalla.txt.gz"),
    "brightkite": lambda: _checkins("brightkite", "brightkite.txt.gz"),
    "lastfm-1k": prep_lastfm,
    "kuairand-pure": prep_kuairand,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for n in (JOBS if a.all else a.names):
        if os.path.exists(f"{OUT}/{n}.npz"):
            print("have", n)
            continue
        if n not in JOBS:
            sys.exit(f"unknown dataset {n}; choices: {sorted(JOBS)}")
        JOBS[n]()
