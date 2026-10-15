#!/bin/sh
# Weekly-cadence programme: stages a through e (each resumable). Run an_weekly.py separately for analysis.
#   nice -n 10 caffeinate -i sh scripts/ssp/run_weekly_night.sh
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python3 -W ignore"
LOG=results/small_sample_popularity/00_safety/weekly_night.log
run() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; }
for s in a b c d e; do
  run $PY scripts/ssp/run_weekly.py --stage "$s" --workers 3 || run $PY scripts/ssp/run_weekly.py --stage "$s" --workers 3
done
echo "[$(date +%H:%M:%S)] DONE" >> "$LOG"
