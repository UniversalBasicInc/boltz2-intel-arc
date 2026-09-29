#!/bin/bash
# Same-seed reruns on the B70: default mode twice, deterministic mode twice.
for mode in default det; do
  for i in 1 2; do
    d=<container-work>/determinism2/$mode-$i; rm -rf $d; mkdir -p $d
    if [ $mode = det ]; then export BENCH_DETERMINISTIC=1; else unset BENCH_DETERMINISTIC; fi
    python <container-work>/exp/bench_predict.py <container-work>/exp/targets_pinned/sod1.yaml $d xpu default 0 > $d/run.log 2>&1
    echo "$mode-$i exit=$? $(tail -1 $d/run.log | grep -o "\"status\": \"[^\"]*\"\|\"wall_s\": [0-9.]*" | tr "\n" " ")"
  done
done
for mode in default det; do
  a=$(find <container-work>/determinism2/$mode-1 -name sod1_model_0.cif); b=$(find <container-work>/determinism2/$mode-2 -name sod1_model_0.cif)
  cmp -s "$a" "$b" && echo "$mode: byte-identical" || echo "$mode: DIFFERS ($(diff <(grep ^ATOM $a) <(grep ^ATOM $b) | grep -c "^<") atom lines)"
done
echo "== ops flagged as nondeterministic (det mode):"
grep -hoE "[A-Za-z_:.]+ does not have a deterministic implementation[^.]*" <container-work>/determinism2/det-*/run.log | sort | uniq -c | sort -rn | head -20
