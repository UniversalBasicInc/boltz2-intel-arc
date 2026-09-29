#!/bin/bash
export BENCH_DETERMINISTIC=1
for i in 1 2; do d=<container-work>/determinism2/abl-det-$i; rm -rf $d; mkdir -p $d
  python <container-work>/exp/bench_predict.py <container-work>/exp/targets_pinned/abl_imatinib.yaml $d xpu default 0 > $d/run.log 2>&1
  echo "abl det-$i $(tail -1 $d/run.log | grep -o "\"status\": \"[^\"]*\"\|\"wall_s\": [0-9.]*" | tr "\n" " ")"; done
a=$(find <container-work>/determinism2/abl-det-1 -name abl_imatinib_model_0.cif); b=$(find <container-work>/determinism2/abl-det-2 -name abl_imatinib_model_0.cif)
cmp -s $a $b && echo "abl structure: byte-identical" || echo "abl structure: DIFFERS"
a=$(find <container-work>/determinism2/abl-det-1 -name affinity_abl_imatinib.json); b=$(find <container-work>/determinism2/abl-det-2 -name affinity_abl_imatinib.json)
cmp -s $a $b && echo "abl affinity: byte-identical" || echo "abl affinity: DIFFERS"
