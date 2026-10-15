#!/bin/sh
# Extra non-Amazon logs for the deep programme. Raw files are untracked. Throttled. Usage: sh datasets/fetch_extra.sh
cd "$(dirname "$0")/raw" || exit 1
get() { [ -s "$2" ] && { echo "have $2"; return; }; curl -fL -A "Mozilla/5.0" --retry 3 --limit-rate 8M -C - -o "$2" "$1" && echo "ok $2"; }
GR=https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre
U=https://mcauleylab.ucsd.edu/public_datasets/data
get $GR/goodreads_interactions_poetry.json.gz goodreads_interactions_poetry.json.gz
get $GR/goodreads_interactions_comics_graphic.json.gz goodreads_interactions_comics_graphic.json.gz
get $GR/goodreads_interactions_children.json.gz goodreads_interactions_children.json.gz
get $U/beer/Beeradvocate.txt.gz Beeradvocate.txt.gz
get $U/beer/Ratebeer.txt.gz Ratebeer.txt.gz
get $U/steam/steam_reviews.json.gz steam_reviews.json.gz
