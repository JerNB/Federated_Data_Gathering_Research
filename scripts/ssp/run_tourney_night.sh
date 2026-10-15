#!/bin/sh
# Overnight hypothesis tournament, one generation: screen -> full -> holdout -> robust (+ calendar and H7 for generation 1) -> ledger.
# Resumable, governor-limited, quiet.   Generation 1:  nice -n 10 caffeinate -i sh scripts/ssp/run_tourney_night.sh
#                                       Generation 2:  TOURNEY_GEN=2 nice -n 10 caffeinate -i sh scripts/ssp/run_tourney_night.sh
cd "$(dirname "$0")/../.." || exit 1
PY=".venv/bin/python3 -W ignore"
G="${TOURNEY_GEN:-}"
LOG=results/small_sample_popularity/00_safety/tourney${G}_night.log
mkdir -p "$(dirname "$LOG")"
run() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; }
stage() {
  run $PY scripts/ssp/run_tourney.py --stage "$1"
  run $PY scripts/ssp/an_tourney.py "$1" || { run $PY scripts/ssp/run_tourney.py --stage "$1"; run $PY scripts/ssp/an_tourney.py "$1"; }
}
stage_or_abort() { stage "$1" || { echo "[$(date +%H:%M:%S)] ABORT: stage $1 incomplete or failed" >> "$LOG"; exit 3; }; }
stage_or_abort screen
stage_or_abort full
stage_or_abort holdout
if grep -q '"robust"' "configs/tourney_registry${G}.json"; then stage_or_abort robust; fi
if [ -z "$G" ]; then
  run $PY scripts/ssp/run_tourney.py --stage cal
  run $PY scripts/ssp/an_tourney.py cal
  run $PY scripts/ssp/an_tourney.py offline
fi
run $PY scripts/ssp/an_tourney.py final
echo "[$(date +%H:%M:%S)] DONE" >> "$LOG"
