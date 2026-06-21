#!/usr/bin/env bash
#
# One-command setup for Boltz-2 on Intel Arc.
#
#   git clone … && cd boltz2-intel-arc && ./setup.sh
#
# Builds the version-matched Intel GPU userspace stack (no sudo, system driver
# untouched), installs torch-xpu + Boltz-2, and verifies the GPU is visible.
# Everything lands in $BOLTZ_XPU_HOME (default ~/boltz-xpu). When it finishes:
#
#   ./fold.sh examples/sod1.yaml
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export BOLTZ_XPU_HOME="${BOLTZ_XPU_HOME:-$HOME/boltz-xpu}"

echo "== Boltz-2 on Intel Arc — setup =="
echo "   install dir: $BOLTZ_XPU_HOME"
echo

# --- preflight: catch the obvious "wrong machine" cases early -----------------
echo ">> preflight checks"
if [ "$(uname -s)" != "Linux" ]; then
  echo "!! Linux is required (Intel 'xe' GPU driver). Aborting."; exit 1
fi
if [ ! -e /dev/dri/renderD128 ]; then
  echo "!! /dev/dri/renderD128 not found — no GPU render node."
  echo "   Is an Intel Arc GPU installed and the 'xe' kernel driver loaded?"
  echo "   (Brand-new Arc Pro B70 may need xe.force_probe + a GuC firmware bump —"
  echo "    see docs/FINDINGS.md §6.)"
  exit 1
fi
if command -v lspci >/dev/null 2>&1; then
  if lspci -nn 2>/dev/null | grep -iqE "8086:e2(0b|23)"; then
    echo "   ✓ detected Intel Arc Battlemage (B580 e20b / B70 e223)"
  else
    echo "   note: did not see a B580(e20b)/B70(e223) via lspci — other Arc"
    echo "   Battlemage SKUs may still work; continuing."
  fi
fi
echo

# --- the two build steps ------------------------------------------------------
echo ">> [1/2] building the Intel GPU userspace stack"
"$HERE/scripts/setup_arc_stack.sh"
echo
echo ">> [2/2] installing torch-xpu + Boltz-2"
"$HERE/scripts/install_boltz_xpu.sh"

echo
echo "✅ Setup complete. Fold your first protein:"
echo "     ./fold.sh examples/sod1.yaml"
echo "   …or fold your own — point it at any sequence (.yaml or .fasta)."
