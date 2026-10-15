#!/bin/sh
# Overnight hypothesis tournament: screen -> full -> holdout -> calendar -> H7 -> ledger. Resumable, governor-limited, quiet.
#   nice -n 10 caffeinate -i sh scripts/ssp/run_tourney_night.sh
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python3 -W ignore"
LOG=results/small_sample_popularity/00_safety/tourney_night.log
mkdir -p "$(dirname "$LOG")"
run() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; }
stage() {
  run $PY scripts/ssp/run_tourney.py --stage "$1"
  run $PY scripts/ssp/an_tourney.py "$1" || { run $PY scripts/ssp/run_tourney.py --stage "$1"; run $PY scripts/ssp/an_tourney.py "$1"; }
}
stage screen
stage full
stage holdout
run $PY scripts/ssp/run_tourney.py --stage cal
run $PY scripts/ssp/an_tourney.py cal
run $PY scripts/ssp/an_tourney.py offline
run $PY scripts/ssp/an_tourney.py final
echo "[$(date +%H:%M:%S)] DONE" >> "$LOG"
