#!/bin/bash
cd <container-work>/boltz-community
echo "== pytest -m xpu"; python -m pytest tests/test_xpu.py -m xpu -v -p no:cacheprovider 2>&1 | grep -E "PASSED|FAILED|SKIPPED|ERROR|passed|failed" | tail -8
echo "== determinism (rerun seed 0, compare with matrix output)"
for t in sod1 abl_imatinib; do
  d=<container-work>/determinism/$t; rm -rf $d; mkdir -p $d
  python <container-work>/exp/bench_predict.py <container-work>/exp/targets_pinned/$t.yaml $d xpu default 0 > $d/run.log 2>&1
  a=$(find <container-work>/runs/b70-default/$t/seed0 -name "${t}_model_0.cif"); b=$(find $d -name "${t}_model_0.cif")
  if cmp -s "$a" "$b"; then echo "$t: CIF byte-identical"; else echo "$t: CIF DIFFERS ($(diff <(grep ^ATOM $a) <(grep ^ATOM $b) | grep -c "^<") atom lines differ)"; fi
  ja=$(find <container-work>/runs/b70-default/$t/seed0 -name "confidence_${t}_model_0.json"); jb=$(find $d -name "confidence_${t}_model_0.json")
  cmp -s "$ja" "$jb" && echo "$t: confidence JSON identical" || echo "$t: confidence JSON differs"
done
