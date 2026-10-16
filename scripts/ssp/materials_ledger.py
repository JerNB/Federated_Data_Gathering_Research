"""Materials ledger: which external materials the programme uses, where they came from, and a hash check.

    .venv/bin/python3 scripts/ssp/materials_ledger.py write     # measure sizes + sha256 of datasets/npz, write the CSV
    .venv/bin/python3 scripts/ssp/materials_ledger.py check     # recompute and compare with the CSV (exit 1 on any difference)

Writes / reads results/small_sample_popularity/01_datasets/materials_ledger.csv (datasets only; software, tools, literature and
the safety rules are in docs/small_sample/90_materials_ledger_{en,zh}.md). The source URLs are the ones written in
datasets/fetch.sh and datasets/fetch_extra.sh (nothing is downloaded here; this script has no network code).
A processed .npz that changes (different bytes) is reported by `check`: do not run any analysis on it until the change is explained.
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys

NPZ = "datasets/npz"
OUT = "results/small_sample_popularity/01_datasets/materials_ledger.csv"
AM = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/"
GR = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
UC = "https://mcauleylab.ucsd.edu/public_datasets/data/"
SOURCES = {
    "amazon-arts-crafts-and-sewing": AM + "Arts_Crafts_and_Sewing.csv.gz", "amazon-automotive": AM + "Automotive.csv.gz",
    "amazon-baby-products": AM + "Baby_Products.csv.gz", "amazon-beauty-and-personal-care": AM + "Beauty_and_Personal_Care.csv.gz",
    "amazon-books": AM + "Books.csv.gz", "amazon-cds-and-vinyl": AM + "CDs_and_Vinyl.csv.gz",
    "amazon-cell-phones-and-accessories": AM + "Cell_Phones_and_Accessories.csv.gz", "amazon-electronics": AM + "Electronics.csv.gz",
    "amazon-grocery-and-gourmet-food": AM + "Grocery_and_Gourmet_Food.csv.gz", "amazon-health-and-household": AM + "Health_and_Household.csv.gz",
    "amazon-industrial-and-scientific": AM + "Industrial_and_Scientific.csv.gz", "amazon-musical-instruments": AM + "Musical_Instruments.csv.gz",
    "amazon-movies-tv": AM + "Movies_and_TV.csv.gz", "amazon-office-products": AM + "Office_Products.csv.gz",
    "amazon-patio-lawn-and-garden": AM + "Patio_Lawn_and_Garden.csv.gz", "amazon-pet-supplies": AM + "Pet_Supplies.csv.gz",
    "amazon-software": AM + "Software.csv.gz", "amazon-sports-and-outdoors": AM + "Sports_and_Outdoors.csv.gz",
    "amazon-tools-and-home-improvement": AM + "Tools_and_Home_Improvement.csv.gz", "amazon-toys-and-games": AM + "Toys_and_Games.csv.gz",
    "amazon-video-games": AM + "Video_Games.csv.gz",
    "beeradvocate": UC + "beer/Beeradvocate.txt.gz", "ratebeer": UC + "beer/Ratebeer.txt.gz", "steam": UC + "steam/steam_reviews.json.gz",
    "goodreads-children": GR + "goodreads_interactions_children.json.gz", "goodreads-comics": GR + "goodreads_interactions_comics_graphic.json.gz",
    "goodreads-poetry": GR + "goodreads_interactions_poetry.json.gz",
    "brightkite": "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz", "gowalla": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
    "kuairand-pure": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
    "lastfm-1k": "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
}
LICENSE = {"ml-latest": "MovieLens Usage License (data/dataset_manifest.json); commercial use needs permission from GroupLens",
           "ml-1m": "GroupLens MovieLens terms (not stated in the repo; check before redistribution)"}
# logs of the weekly study (27 usable), the confirmation (adds gowalla, brightkite), and the exclusion for length
WEEKLY_EXCLUDED_SHORT = {"gowalla", "brightkite", "kuairand-pure"}
WEEKLY_USED = {
    "amazon-arts-crafts-and-sewing", "amazon-books", "amazon-electronics", "amazon-musical-instruments", "amazon-office-products",
    "amazon-patio-lawn-and-garden", "amazon-pet-supplies", "amazon-video-games", "ml-latest", "amazon-automotive",
    "amazon-cell-phones-and-accessories", "amazon-grocery-and-gourmet-food", "amazon-health-and-household", "amazon-movies-tv",
    "amazon-sports-and-outdoors", "amazon-toys-and-games", "amazon-baby-products", "amazon-cds-and-vinyl", "amazon-industrial-and-scientific",
    "amazon-beauty-and-personal-care", "amazon-tools-and-home-improvement", "beeradvocate", "goodreads-children", "ratebeer",
    "goodreads-comics", "goodreads-poetry", "steam"}
HEAD = ["dataset", "bytes", "sha256", "source_url_from_repo_fetch_scripts", "license_note", "used_in"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def used_in(name: str) -> str:
    if name in WEEKLY_USED:
        return "weekly study (doc 13), selector (docs 14, 15), stress tests (doc 16)"
    if name in ("gowalla", "brightkite"):
        return "confirmation K11 only (docs 15); too short for the weekly study"
    return "earlier studies (docs 00-12) or unused; not re-checked in this ledger"


def measure() -> list[list]:
    rows = []
    for fn in sorted(os.listdir(NPZ)):
        if not fn.endswith(".npz"):
            continue
        name = fn[:-4]
        p = f"{NPZ}/{fn}"
        rows.append([name, os.path.getsize(p), sha256(p), SOURCES.get(name, "NOT RECORDED"),
                     LICENSE.get(name, "not stated in the repo"), used_in(name)])
    return rows


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "write":
        rows = measure()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", newline="") as f:
            cw = csv.writer(f, lineterminator="\n")
            cw.writerow(HEAD)
            cw.writerows(rows)
        print(f"wrote {OUT}: {len(rows)} datasets, {sum(r[3] == 'NOT RECORDED' for r in rows)} without a recorded source")
    elif cmd == "check":
        want = {r["dataset"]: r for r in csv.DictReader(open(OUT))}
        got = {r[0]: r for r in measure()}
        bad = [n for n in sorted(set(want) | set(got)) if n not in want or n not in got
               or want[n]["sha256"] != got[n][2] or int(want[n]["bytes"]) != got[n][1]]
        for n in bad:
            print("DIFFERENT:", n, "(missing from ledger)" if n not in want else "(missing on disk)" if n not in got else "(bytes or sha256 changed)")
        print("LEDGER CHECK", "FAIL" if bad else f"PASS ({len(got)} datasets)")
        sys.exit(1 if bad else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
