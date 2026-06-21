#!/usr/bin/env bash
# Coherent Intel GPU userspace stack for torch.xpu on the Arc B580.
# loader 1.29.0 + NEO 26.05.37020.3 + IGC 2.28.4 + gmmlib 22.9.0.
# Shadows the system 25.44 driver per-process only — system is untouched.
export LD_LIBRARY_PATH="$HOME/boltz-xpu/gpu-stack/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
