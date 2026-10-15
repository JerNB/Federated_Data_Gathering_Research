"""Prepare the extra non-Amazon logs for the deep programme (datasets/npz/<name>.npz, same format as prep_data.py).

    sh datasets/fetch_extra.sh && .venv/bin/python3 scripts/ssp/prep_extra.py goodreads-poetry goodreads-comics goodreads-children steam beeradvocate ratebeer

Every row = first time a user interacted with an item (shelf add / review / rating).
"""
from __future__ import annotations

import gzip
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import prep_data as P  # noqa: E402

RAW = "datasets/raw"
MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
_GR = re.compile(r'"user_id": "(\w+)", "book_id": "(\d+)".*?"date_added": "(\w{3}) (\w{3}) (\d\d) (\d\d:\d\d:\d\d) [+-]\d{4} (\d{4})"')


def goodreads(name: str, fname: str):
    ids = P._Ids()
    user, item, stamp = [], [], []
    with gzip.open(f"{RAW}/{fname}", "rt", errors="ignore") as f:
        for ln in f:
            m = _GR.search(ln)
            if not m:
                continue
            u, b, _, mon, day, hms, yr = m.groups()
            user.append(ids(u))
            item.append(int(b))
            stamp.append(f"{yr}-{MON[mon]:02d}-{day}T{hms}")
    ts = np.array(stamp, dtype="datetime64[s]").astype(np.int64)
    P._finish(name, user, item, ts, np.ones(len(user), np.float32), 1.0, True)


_ST_U = re.compile(r"u'username': u'(.*?)', u'product_id': u'(\d+)'")
_ST_D = re.compile(r"u'date': u'(\d{4}-\d{2}-\d{2})'")


def steam():
    ids = P._Ids()
    user, item, stamp = [], [], []
    with gzip.open(f"{RAW}/steam_reviews.json.gz", "rt", errors="ignore") as f:
        for ln in f:
            a = _ST_U.search(ln)
            d = _ST_D.search(ln)
            if not a or not d:
                continue
            user.append(ids(a.group(1)))
            item.append(int(a.group(2)))
            stamp.append(d.group(1))
    ts = np.array(stamp, dtype="datetime64[D]").astype("datetime64[s]").astype(np.int64)
    P._finish("steam", user, item, ts, np.ones(len(user), np.float32), 1.0, True)


def beer(name: str, fname: str):
    ids = P._Ids()
    user, item, ts = [], [], []
    cur = {}
    with gzip.open(f"{RAW}/{fname}", "rt", errors="ignore") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                if "beer/beerId" in cur and "review/profileName" in cur and "review/time" in cur:
                    try:
                        item.append(int(cur["beer/beerId"]))
                        ts.append(int(cur["review/time"]))
                        user.append(ids(cur["review/profileName"]))
                    except ValueError:
                        pass
                cur = {}
                continue
            k, _, v = ln.partition(": ")
            if k in ("beer/beerId", "review/profileName", "review/time"):
                cur[k] = v
    P._finish(name, user, item, np.array(ts, np.int64), np.ones(len(user), np.float32), 1.0, True)


JOBS = {
    "goodreads-poetry": lambda: goodreads("goodreads-poetry", "goodreads_interactions_poetry.json.gz"),
    "goodreads-comics": lambda: goodreads("goodreads-comics", "goodreads_interactions_comics_graphic.json.gz"),
    "goodreads-children": lambda: goodreads("goodreads-children", "goodreads_interactions_children.json.gz"),
    "steam": steam,
    "beeradvocate": lambda: beer("beeradvocate", "Beeradvocate.txt.gz"),
    "ratebeer": lambda: beer("ratebeer", "Ratebeer.txt.gz"),
}

if __name__ == "__main__":
    for n in sys.argv[1:]:
        if os.path.exists(f"datasets/npz/{n}.npz"):
            print("have", n)
            continue
        JOBS[n]()
