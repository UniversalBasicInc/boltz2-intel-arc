# Reproducing the Arc B580 Boltz-2 pipeline

## Prerequisites

- An **Intel Arc Battlemage** GPU (developed on B580) on **Linux** with the
  in-tree **`xe`** kernel driver loaded (`lsmod | grep '^xe'`). Developed on
  Debian 13 / kernel 6.19; recent Ubuntu should work too.
- Your user in the `render` and `video` groups (`id` should list both).
- Network egress (downloads driver `.deb`s, PyTorch wheels, Boltz weights, and
  queries the ColabFold MSA server).
- `curl`, `dpkg-deb` (both standard). **No sudo required** — the GPU stack is a
  userspace `LD_LIBRARY_PATH` shadow; your system driver is never touched.

Confirm the GPU is visible at all (this works even on the broken stock stack):

```bash
clinfo | grep -i "Device Name" ; sycl-ls 2>/dev/null
```

## Step 1 — build the coherent GPU stack

```bash
./scripts/setup_arc_stack.sh
```

Downloads the **version-matched** Level-Zero loader 1.29.0 + NEO 26.05.37020.3 +
IGC 2.28.4 + gmmlib 22.9.0, extracts the `.so` files into
`~/boltz-xpu/gpu-stack/lib/`, and writes `~/boltz-xpu/xpu-env.sh`. See
`docs/FINDINGS.md` for *why* these exact versions (they are version-locked).

Verify (note: `init()` must be called **before** any availability probe):

```bash
source ~/boltz-xpu/xpu-env.sh
~/boltz-xpu/.venv/bin/python - <<'PY'
import torch
torch.xpu.init()
print("device:", torch.xpu.get_device_name(0))
x = torch.randn(4096, 4096, device="xpu"); print("matmul ok:", float((x@x).sum()))
PY
```

(If the venv doesn't exist yet, run Step 2 first.)

## Step 2 — Python env + torch-xpu + Boltz-2

```bash
./scripts/install_boltz_xpu.sh
```

Creates a Python 3.12 venv with [`uv`](https://github.com/astral-sh/uv),
installs `torch` from the PyTorch XPU wheel channel, and installs `boltz`
**without** the `[cuda]` extra (so NVIDIA cuEquivariance is never pulled in).

## Step 3 — fold

```bash
./fold.sh examples/sod1.yaml          # or any input in examples/, or your own .fasta
```

`fold.sh` sources the GPU env and runs the launcher for you. Output lands in
`out_sod1/boltz_results_sod1/predictions/sod1/`:

- `sod1_model_0.pdb` — the structure
- `confidence_sod1_model_0.json` — confidence / pTM / pLDDT

**Manual invocation** (full control over Boltz's flags) — note `--accelerator gpu`
(Boltz's CLI only accepts `gpu/cpu/tpu`); the launcher rewrites it to `xpu`:

```bash
source ~/boltz-xpu/xpu-env.sh
~/boltz-xpu/.venv/bin/python scripts/run_boltz.py predict examples/sod1.yaml \
    --accelerator gpu --devices 1 --use_msa_server \
    --out_dir out_sod1 --output_format pdb --no_kernels
```

Example outputs are in [`../examples/`](../examples/).

## Troubleshooting

- **Segfault on first device touch** — your `LD_LIBRARY_PATH` isn't pointing at
  `gpu-stack/lib` (re-`source xpu-env.sh`), or you queried
  `torch.xpu.is_available()` before `torch.xpu.init()` (call `init()` first).
- **`gmm_helper/resource_info.cpp` abort** — mismatched component versions; the
  loader/NEO/IGC/gmmlib must be the matched set from Step 1.
- **Repeated segfaults after one crash** — a crashed XPU process wedges the `xe`
  device for ~30–60 s. Wait, health-check `get_device_name` in a throwaway
  process, then retry.
- **MSA step hangs/fails** — the ColabFold server needs network egress; or
  supply your own MSA and drop `--use_msa_server`.

## Optional — make the stack system-wide (requires sudo)

The userspace shadow is recommended (it can't disturb existing workloads). If you
*do* want it permanent, install the same matched `.deb`s
(`~/boltz-xpu/.driver-debs/`):

```bash
sudo dpkg -i ~/boltz-xpu/.driver-debs/libze1_*.deb \
             ~/boltz-xpu/.driver-debs/libze-intel-gpu1_*.deb \
             ~/boltz-xpu/.driver-debs/libigdgmm12_*.deb \
             ~/boltz-xpu/.driver-debs/intel-igc-core-2_*.deb \
             ~/boltz-xpu/.driver-debs/intel-igc-opencl-2_*.deb
```

This replaces the stock driver globally — keep a record of the prior versions to
roll back.
