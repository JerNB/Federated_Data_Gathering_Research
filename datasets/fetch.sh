#!/bin/sh
# Public datasets for the small-sample popularity study. Raw files are untracked.
# Throttled so the download does not load the machine. Usage: datasets/fetch.sh
cd "$(dirname "$0")/raw" || exit 1
get() { [ -s "$2" ] && { echo "have $2"; return; }; curl -fL -A "Mozilla/5.0" --retry 3 --limit-rate 6M -C - -o "$2" "$1" && echo "ok $2"; }
AM=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only
get https://files.grouplens.org/datasets/movielens/ml-1m.zip ml-1m.zip
get $AM/Video_Games.csv.gz amazon_video_games.csv.gz
get $AM/Musical_Instruments.csv.gz amazon_musical_instruments.csv.gz
get $AM/Movies_and_TV.csv.gz amazon_movies_tv.csv.gz
get $AM/Books.csv.gz amazon_books.csv.gz
get $AM/Software.csv.gz amazon_software.csv.gz
get https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz gowalla.txt.gz
get https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz brightkite.txt.gz
get https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz kuairand_pure.tar.gz
get http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz lastfm_1k.tar.gz
# seasonality / more product categories (added for the refresh and two-round studies)
for c in Toys_and_Games Grocery_and_Gourmet_Food Pet_Supplies Sports_and_Outdoors Cell_Phones_and_Accessories Arts_Crafts_and_Sewing Office_Products Patio_Lawn_and_Garden Health_and_Household Electronics Automotive; do
  get $AM/$c.csv.gz amazon_$(echo $c | tr 'A-Z' 'a-z').csv.gz
done
