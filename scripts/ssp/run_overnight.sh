#!/bin/sh
# Quiet overnight run: low priority, no sleep, governor-limited. Safe to re-run: it resumes.
cd "$(dirname "$0")/../.." || exit 1
exec nice -n 10 caffeinate -i .venv/bin/python3 scripts/ssp/run_all.py "$@"
