#!/bin/bash
# One bf16 fold per target on the B70 with the public MSA server; keeps the MSAs for pinned reruns.
cd <container-work>/msa_harvest
for t in ${@:-sod1 vim1_blind mpro_paxlovid abl_imatinib}; do
  start=$(date +%s)
  boltz predict <container-work>/exp/targets/$t.yaml --out_dir <container-work>/msa_harvest --accelerator xpu \
    --use_msa_server --seed 0 --num_workers 0 --override > <container-work>/msa_harvest/$t.log 2>&1
  echo "$t exit=$? $(( $(date +%s)-start ))s $(grep -E "Running on Intel|Error|Traceback" <container-work>/msa_harvest/$t.log | tail -2 | tr "\n" " ")"
done
