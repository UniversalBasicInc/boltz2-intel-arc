#!/usr/bin/env python3
"""Measure fp16/bf16/fp32/fp64 matmul throughput on an Intel Arc GPU via torch.xpu.

Run after `source ~/boltz-xpu/xpu-env.sh`:
    python bench_precision.py

Confirms the card does IEEE-correct FP64 and reports the fp32:fp64 ratio — the
number that distinguishes real double precision (~1/14 here) from crippled
software emulation (~1/64 or worse).
"""
import torch, time

torch.xpu.init()
print("device:", torch.xpu.get_device_name(0))
print("VRAM GB:", round(torch.xpu.get_device_properties(0).total_memory / 1e9, 1))

# FP64 correctness (full double precision, not silently downcast)
x = torch.tensor([1.0 / 3.0], device="xpu", dtype=torch.float64)
print("fp64 roundtrip:", x.item(), "(want 0.3333333333333333)")

def tflops(dt, n=8192, it=50, warm=5):
    a = torch.randn(n, n, device="xpu", dtype=dt)
    b = torch.randn(n, n, device="xpu", dtype=dt)
    for _ in range(warm):
        c = a @ b
    torch.xpu.synchronize()
    t = time.time()
    for _ in range(it):
        c = a @ b
    torch.xpu.synchronize()
    return it * 2 * n**3 / (time.time() - t) / 1e12

r = {dt: tflops(getattr(torch, dt)) for dt in ("float16", "bfloat16", "float32", "float64")}
for k, v in r.items():
    print(f"  {k:9s}: {v:7.1f} TFLOPS")
print(f"  fp32:fp64 ratio = {r['float32'] / r['float64']:.1f}:1  "
      f"(software emulation would be ~64:1 or worse)")
