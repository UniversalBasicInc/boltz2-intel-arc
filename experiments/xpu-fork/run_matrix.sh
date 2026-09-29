#!/bin/bash
# Run every target x precision x seed on one device and keep going on failure.
#   run_matrix.sh <tag> <accelerator> <python> <targets_dir> <out_root> "<precisions>" "<seeds>"
# e.g. run_matrix.sh b70 xpu /work/venv/bin/python /work/exp/targets_pinned /work/runs "default fp32" "0 1 2 3 4"
# Layout: <out_root>/<tag>-<precision>/<target>/seed<N>/{bench.json,run.log,boltz_results_*}
set -u
TAG=$1; ACC=$2; PY=$3; TGT=$4; OUT=$5; PRECS=$6; SEEDS=$7
HERE="$(cd "$(dirname "$0")" && pwd)"
for prec in $PRECS; do
  for y in "$TGT"/*.yaml; do
    t=$(basename "$y" .yaml)
    for s in $SEEDS; do
      d="$OUT/$TAG-$prec/$t/seed$s"
      if [ -f "$d/bench.json" ] && grep -q '"status": "ok"' "$d/bench.json"; then continue; fi
      mkdir -p "$d"
      "$PY" "$HERE/bench_predict.py" "$y" "$d" "$ACC" "$prec" "$s" > "$d/run.log" 2>&1
      echo "$(date +%T) $TAG-$prec $t seed$s exit=$? $(tail -1 "$d/run.log" | cut -c1-160)"
    done
  done
done
echo "MATRIX DONE $TAG"
