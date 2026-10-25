#!/bin/bash
# Resumable round-2 chain: domain-grouped PhiUSIIL baseline experiment (seeds 42-44),
# then the round-2 defenses on it. A step is skipped if logs/done-<step> exists.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src PHISHADV_THREADS=${PHISHADV_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
P=${PY:-.venv/bin/python}
step() { local name=$1; shift
  [ -f logs/done-$name ] && { echo "skip $name"; return; }
  echo "$(date +%T) start $name"; "$P" -m phishadv."$@" > logs/$name.log 2>&1; touch logs/done-$name
  echo "$(date +%T) done $name"; }
step grouped-42 experiment --config full-grouped --save-models models/phiusiil-grouped
step grouped-43 experiment --config full-grouped --seed 43
step grouped-44 experiment --config full-grouped --seed 44
for s in 42 43 44; do step def-full-grouped-$s defenses --config full-grouped --seed $s; done
