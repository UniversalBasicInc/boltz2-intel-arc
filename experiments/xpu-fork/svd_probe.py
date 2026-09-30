"""How much does the XPU->CPU fallback of torch.linalg.svd in weighted_rigid_align cost a prediction?

    python svd_probe.py <asis|timed|stub> <input.yaml> <out_dir>

asis:  unmodified; counts SVD calls and times predict_step.
timed: also times each SVD call with an XPU sync before and after, i.e. the fallback's own cost
       (copy to CPU, CPU SVD, copy back). The fallback syncs at that point anyway.
stub:  replaces the SVD on XPU inputs with identity U, V and unit S built on the device, with no
       sync and no CPU work. Structures are wrong, but the work around it is the same shape, so the
       predict_step time is a floor: no on-device SVD could make the step faster than this.
Every predict_step is timed with a sync on both sides, and the times are summed. Prints one RESULT line.
"""
import json
import sys
import time

import torch

mode, inp, out = sys.argv[1:4]
torch.xpu.init()

real_svd = torch.linalg.svd
stats = {"svd_calls": 0, "svd_seconds": 0.0, "svd_shapes": set()}


def svd(a, *args, **kw):
    if a.device.type != "xpu":
        return real_svd(a, *args, **kw)
    stats["svd_calls"] += 1
    stats["svd_shapes"].add(tuple(a.shape))
    if mode == "stub":
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand_as(a).clone()
        return eye, torch.ones(a.shape[:-1], dtype=a.dtype, device=a.device), eye.clone()
    if mode == "timed":
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        r = real_svd(a, *args, **kw)
        torch.xpu.synchronize()
        stats["svd_seconds"] += time.perf_counter() - t0
        return r
    return real_svd(a, *args, **kw)


torch.linalg.svd = svd

from boltz.model.models.boltz2 import Boltz2  # noqa: E402

real_predict_step = Boltz2.predict_step


def predict_step(self, *a, **kw):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    r = real_predict_step(self, *a, **kw)
    torch.xpu.synchronize()
    # summed: a target with affinity runs two predict passes (structure, then affinity)
    stats["predict_step_seconds"] = stats.get("predict_step_seconds", 0.0) + time.perf_counter() - t0
    stats["predict_steps"] = stats.get("predict_steps", 0) + 1
    return r


Boltz2.predict_step = predict_step

from boltz.main import predict  # noqa: E402

predict.main(
    args=[inp, "--out_dir", out, "--accelerator", "xpu", "--seed", "0", "--override",
          "--num_workers", "0"],
    standalone_mode=False,
)
stats["svd_shapes"] = sorted(stats["svd_shapes"])
print("RESULT", json.dumps({"mode": mode, "input": inp, **stats}))
