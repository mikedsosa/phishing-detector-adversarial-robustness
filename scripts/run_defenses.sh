#!/bin/bash
# Resumable round-2 defense chain for one config: scripts/run_defenses.sh phresh|full [seeds...]
# A step is skipped if logs/done-def-<config>-<seed> exists.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src PHISHADV_THREADS=${PHISHADV_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
P=${PY:-.venv/bin/python}
cfg=$1; shift
for s in "${@:-42 43 44}"; do
  [ -f logs/done-def-$cfg-$s ] && { echo "skip $cfg-$s"; continue; }
  echo "$(date +%T) start $cfg-$s"
  "$P" -m phishadv.defenses --config "$cfg" --seed "$s" > logs/def-$cfg-$s.log 2>&1
  touch logs/done-def-$cfg-$s; echo "$(date +%T) done $cfg-$s"
done
