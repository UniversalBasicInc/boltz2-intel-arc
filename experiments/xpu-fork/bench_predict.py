"""Run one `boltz predict` in-process and record wall time + peak device memory.

    python bench_predict.py <input.yaml> <out_dir> <accelerator> <precision> <seed>

accelerator: xpu | gpu | cpu     precision: default | fp32
"default" leaves boltz's own precision choice alone (bf16-mixed on xpu/gpu).
"fp32" forces 32-true for the comparison arm, without adding a CLI flag to boltz.
Writes <out_dir>/bench.json.
"""
import json
import sys
import time
from pathlib import Path

import torch

inp, out_dir, accel, prec, seed = sys.argv[1:6]
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)

if accel == "xpu":
    torch.xpu.init()
    dev = torch.xpu
elif accel == "gpu":
    dev = torch.cuda
else:
    dev = None

import os
import warnings

if os.environ.get("BENCH_DETERMINISTIC") == "1":
    # warn_only: ops without a deterministic implementation are named, not fatal
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.simplefilter("always", UserWarning)

import pytorch_lightning as pl

_used = {"deterministic": os.environ.get("BENCH_DETERMINISTIC") == "1"}
_base_init = pl.Trainer.__init__


def _record(self, *a, **kw):
    """Record the precision plugin Lightning actually builds."""
    _base_init(self, *a, **kw)
    plug = self.strategy.precision_plugin
    _used["precision_plugin"] = type(plug).__name__
    _used["autocast_device"] = getattr(plug, "device", None)
    _used["precision_actual"] = str(getattr(plug, "precision", None))


pl.Trainer.__init__ = _record

if prec == "fp32":

    _orig = pl.Trainer.__init__

    def _init(self, *a, **kw):
        kw.pop("plugins", None)  # drop boltz's explicit bf16 plugin (xpu path)
        kw["precision"] = "32-true"
        return _orig(self, *a, **kw)

    pl.Trainer.__init__ = _init

from boltz.main import predict  # noqa: E402

args = [
    inp,
    "--out_dir", str(out),
    "--accelerator", accel,
    "--seed", seed,
    "--override",
    "--num_workers", "0",
]
if dev is not None:
    dev.reset_peak_memory_stats()
t0 = time.perf_counter()
status = "ok"
try:
    predict.main(args=args, standalone_mode=False)
except SystemExit as e:
    status = f"exit {e.code}"
except Exception as e:  # record, don't hide
    status = f"error: {type(e).__name__}: {e}"
wall = time.perf_counter() - t0

# boltz catches out-of-memory, prints a warning and exits 0 without writing a
# structure; don't record that as a success
stem = Path(inp).stem
if status == "ok" and not list(out.rglob(f"{stem}_model_0.*")):
    status = "no structure written (boltz skipped the input, e.g. out of memory)"

meta = {
    "input": inp,
    "accelerator": accel,
    "precision": prec,
    "seed": int(seed),
    "status": status,
    "wall_s": round(wall, 2),
    "peak_mem_gib": round(dev.max_memory_allocated() / 2**30, 3) if dev else None,
    "device_name": dev.get_device_name(0) if dev else "cpu",
    "torch": torch.__version__,
    **_used,
}
(out / "bench.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta))
sys.exit(0 if status == "ok" else 1)
