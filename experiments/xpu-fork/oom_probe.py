"""Force a real out-of-memory in Boltz2.predict_step on an Intel GPU and record which cache is cleared.

    python oom_probe.py <plain|mixed> <input.yaml> <out_dir>

plain: the official torch+xpu wheel as is (it has no CUDA, so torch.cuda.is_available() is False).
mixed: emulates a torch build with both CUDA and XPU by making torch.cuda.is_available() return
       True for calls from boltz's model files outside setup() -- i.e. in the OOM handlers only.
       setup() keeps the real answer because it goes on to query CUDA device properties.
The OOM is real: just before predict_step, the per-process XPU memory cap is set to what is already
allocated plus 256 MiB. Prints one RESULT line: which empty_cache() the OOM handler ran, and XPU memory afterwards.
"""
import inspect
import json
import sys

import torch

mode, inp, out = sys.argv[1:4]
torch.xpu.init()

calls = {"cuda": 0, "mps": 0, "xpu": 0}
real_xpu_empty = torch.xpu.empty_cache


def counted(name, real):
    def empty_cache():
        # count only the OOM handlers in boltz's model files, not Lightning/accelerator teardown
        if inspect.stack()[1].filename.endswith(("boltz1.py", "boltz2.py")):
            calls[name] += 1
        if real is not None:
            real()

    return empty_cache


# cuda/mps are counted only: this build has neither, and on a real mixed build the call would
# clear the CUDA cache, leaving the XPU cache untouched either way.
torch.cuda.empty_cache = counted("cuda", None)
torch.mps.empty_cache = counted("mps", None)
torch.xpu.empty_cache = counted("xpu", real_xpu_empty)

if mode == "mixed":
    real_is_available = torch.cuda.is_available

    def is_available():
        caller = inspect.stack()[1]
        if caller.filename.endswith(("boltz1.py", "boltz2.py")) and caller.function != "setup":
            return True
        return real_is_available()

    torch.cuda.is_available = is_available

from boltz.model.models.boltz2 import Boltz2  # noqa: E402

real_predict_step = Boltz2.predict_step
mem = {}


def predict_step(self, *a, **kw):
    total = torch.xpu.get_device_properties(0).total_memory
    cap = torch.xpu.memory_allocated() + 256 * 2**20
    torch.xpu.set_per_process_memory_fraction(cap / total)
    result = real_predict_step(self, *a, **kw)
    mem["allocated_mib"] = torch.xpu.memory_allocated() // 2**20
    mem["reserved_mib"] = torch.xpu.memory_reserved() // 2**20
    mem["peak_reserved_mib"] = torch.xpu.max_memory_reserved() // 2**20
    mem["returned"] = "exception" if isinstance(result, dict) and result.get("exception") else "other"
    torch.xpu.set_per_process_memory_fraction(1.0)
    return result


Boltz2.predict_step = predict_step

import boltz  # noqa: E402
from boltz.main import predict  # noqa: E402

predict.main(
    args=[inp, "--out_dir", out, "--accelerator", "xpu", "--seed", "0", "--override",
          "--num_workers", "0"],
    standalone_mode=False,
)
print("RESULT", json.dumps({"mode": mode, "boltz": boltz.__file__, "empty_cache_calls": calls, **mem}))
