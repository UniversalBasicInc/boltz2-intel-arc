#!/usr/bin/env bash
#
# Validate Arc Boltz-2 predictions against experimental crystal structures using
# US-align (TM-score / RMSD). Self-contained: fetches the reference structures
# from RCSB and compiles US-align from source. Run after producing predictions.
#
# TM-score: 1.0 = identical fold; >0.9 ≈ essentially the same structure;
#           >0.5 = same fold. AlphaFold3/Boltz-2 achieve ~0.9+ on resolved
#           monomers — so matching that range demonstrates NVIDIA-class accuracy.
#
set -euo pipefail
DEST="${BOLTZ_XPU_HOME:-$HOME/boltz-xpu}"
US="$DEST/USalign"

# 1. build US-align (single C++ source, Zhang lab)
if [ ! -x "$US" ]; then
  echo ">> compiling US-align"
  curl -fsSL -o /tmp/USalign.cpp https://zhanggroup.org/US-align/bin/module/USalign.cpp
  g++ -O3 -ffast-math -o "$US" /tmp/USalign.cpp
fi

# 2. fetch experimental references from RCSB
for id in 2C9V 1PU0 7VH8; do
  [ -s "/tmp/$id.cif" ] || curl -fsSL -o "/tmp/$id.cif" "https://files.rcsb.org/download/$id.cif"
done

score() {  # pred.pdb  ref.cif  label
  printf '%-42s ' "$3"
  "$US" "$1" "$2" -ter 0 2>/dev/null \
    | awk '/TM-score=.*Structure_2/{tm=$2} /RMSD=/{for(i=1;i<=NF;i++) if($i ~ /RMSD=/){gsub(",","",$(i+1)); r=$(i+1)}} /Aligned length/{al=$3; gsub(",","",al)} END{printf "TM=%s  RMSD=%s A  (%s res)\n", tm, r, al}'
}

# Predictions are read from the current directory by default (where ./fold.sh
# writes out_<name>/); override with PRED_DIR=… if you folded elsewhere.
P="boltz_results"; PRED_DIR="${PRED_DIR:-.}"
APO="$PRED_DIR/out_sod1/${P}_sod1/predictions/sod1/sod1_model_0.pdb"
HOLO="$PRED_DIR/out_sod1_holo/${P}_sod1_holo/predictions/sod1_holo/sod1_holo_model_0.pdb"
MPRO="$PRED_DIR/out_mpro_paxlovid/${P}_mpro_paxlovid/predictions/mpro_paxlovid/mpro_paxlovid_model_0.pdb"

echo ">> Arc Boltz-2 predictions vs experimental crystal structures:"
[ -f "$APO" ]  && score "$APO"  /tmp/2C9V.cif "apo SOD1  vs 2C9V (1.07A xtal)"
[ -f "$HOLO" ] && score "$HOLO" /tmp/2C9V.cif "holo SOD1 vs 2C9V"
[ -f "$APO" ]  && score "$APO"  /tmp/1PU0.cif "apo SOD1  vs 1PU0 (independent xtal)"
[ -f "$MPRO" ] && score "$MPRO" /tmp/7VH8.cif "Mpro dimer vs 7VH8 (Mpro+nirmatrelvir)"
