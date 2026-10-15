#!/bin/sh
# Overnight deep programme: stages agg, plan, cal, seas, dp (each resumable), then the analysis and cards.
#   nice -n 10 caffeinate -i sh scripts/ssp/run_deep_night.sh
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python3 -W ignore"
LOG=results/small_sample_popularity/00_safety/deep_night.log
mkdir -p "$(dirname "$LOG")"
run() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; }
for s in agg plan cal seas dp; do
  run $PY scripts/ssp/run_deep.py --stage "$s" || run $PY scripts/ssp/run_deep.py --stage "$s"
done
run $PY scripts/ssp/an_deep.py
echo "[$(date +%H:%M:%S)] DONE" >> "$LOG"
