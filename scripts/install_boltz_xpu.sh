#!/usr/bin/env bash
#
# Create a Python env with torch-xpu and install Boltz-2 (without the CUDA extra)
# for the Arc B580 pipeline. Run scripts/setup_arc_stack.sh first.
#
set -euo pipefail

DEST="${BOLTZ_XPU_HOME:-$HOME/boltz-xpu}"
mkdir -p "$DEST"

# uv: userspace Python/venv manager (manages its own CPython, so no system
# python3.11/3.12-venv packages are required).
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv (userspace)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Boltz-2 requires Python >=3.10,<3.13 -> use 3.12.
echo ">> creating venv ($DEST/.venv, CPython 3.12)"
uv venv --python 3.12 "$DEST/.venv"
PY="$DEST/.venv/bin/python"

# PyTorch XPU. Nightly was used for development; the stable XPU channel
# (https://download.pytorch.org/whl/xpu) also works once the GPU stack is fixed.
echo ">> installing torch (XPU wheel channel)"
uv pip install --python "$PY" --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu
uv pip install --python "$PY" 'numpy>=1.26,<2.0'

# Boltz-2 WITHOUT [cuda] -> NVIDIA cuEquivariance is never installed.
echo ">> installing boltz (no CUDA extra)"
uv pip install --python "$PY" boltz

# sanity check (init() MUST precede any availability probe — see FINDINGS §4)
echo ">> sanity check"
if [ -f "$DEST/xpu-env.sh" ]; then
  # shellcheck disable=SC1090
  source "$DEST/xpu-env.sh"
  "$PY" - <<'PYEOF'
import torch
torch.xpu.init()
print("torch:", torch.__version__)
print("B580 :", torch.xpu.get_device_name(0))
import boltz; print("boltz:", getattr(boltz, "__version__", "?"))
PYEOF
else
  echo "   (run scripts/setup_arc_stack.sh, then re-source xpu-env.sh before folding)"
fi

echo
echo "Done. Fold with:  ./fold.sh examples/sod1.yaml"
