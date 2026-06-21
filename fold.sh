#!/usr/bin/env bash
#
# Fold a protein on Intel Arc — one command.
#
#   ./fold.sh <input.yaml|.fasta> [out_dir]
#
# Examples:
#   ./fold.sh examples/sod1.yaml
#   ./fold.sh my_protein.fasta my_results
#
# Wraps the GPU environment + the XPU launcher so you don't have to source env
# files or remember Boltz's flags. Run ./setup.sh once first.
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export BOLTZ_XPU_HOME="${BOLTZ_XPU_HOME:-$HOME/boltz-xpu}"

IN="${1:-}"
if [ -z "$IN" ]; then
  echo "usage: ./fold.sh <input.yaml|.fasta> [out_dir]"; exit 2
fi
if [ ! -f "$IN" ]; then
  echo "!! input not found: $IN"; exit 2
fi
OUT="${2:-out_$(basename "${IN%.*}")}"

if [ ! -f "$BOLTZ_XPU_HOME/xpu-env.sh" ] || [ ! -x "$BOLTZ_XPU_HOME/.venv/bin/python" ]; then
  echo "!! not set up yet (no $BOLTZ_XPU_HOME/.venv). Run ./setup.sh first."; exit 1
fi

# shellcheck disable=SC1090
source "$BOLTZ_XPU_HOME/xpu-env.sh"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
PY="$BOLTZ_XPU_HOME/.venv/bin/python"

DEV="$("$PY" -c 'import torch; torch.xpu.init(); print(torch.xpu.get_device_name(0))' 2>/dev/null || echo 'Intel XPU')"
echo ">> folding '$IN' on $DEV  →  $OUT/"

exec "$PY" "$HERE/scripts/run_boltz.py" predict "$IN" \
  --accelerator gpu --devices 1 --use_msa_server \
  --out_dir "$OUT" --output_format pdb --no_kernels
