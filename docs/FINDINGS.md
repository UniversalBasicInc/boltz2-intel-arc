# Findings — getting Boltz-2 / PyTorch-XPU working on Intel Arc B580

This is the engineering record behind `boltz-arc`: the bugs hit, how they were
diagnosed, and the fixes. Several items are **directly actionable for Intel /
PyTorch** and are flagged as such — this doubles as a bug report.

## Environment

- **GPU:** Intel Arc B580 (Battlemage G21, PCI `8086:e20b`), 12 GB, driven by
  the in-tree `xe` kernel driver.
- **Host:** Debian 13 (trixie), kernel `6.19.14`, AMD Ryzen 9 3900XT, 31 GB RAM.
  The GPU is shared with a live KDE Plasma desktop.
- **Stock (as-shipped) Intel userspace** — *this is the broken baseline:*
  - `libze-intel-gpu1` (NEO) `25.44.36015.5`
  - `intel-opencl-icd` `25.44.36015.5`
  - `libigdgmm12` (gmmlib) `22.8.2`
  - `intel-igc-core-2` / `intel-igc-opencl-2` (IGC) `2.22.2`
  - `libze1` (Level-Zero loader) `1.20.6`
- **PyTorch:** `torch` XPU build (tested 2.12.1+xpu and 2.14.0.dev+xpu), Python
  3.12, `triton-xpu` 3.7.1, oneAPI 2026.0 also present on the box.

A useful diagnostic property: `clinfo` and `sycl-ls` both enumerate the B580
perfectly on the stock stack — because they never query floating-point configs.
Any *real* workload that does (PyTorch's `initDeviceProperties`, or any SYCL
program reading `half_fp_config`) crashes. So "the GPU shows up fine" is
misleading.

---

## Finding 1 — NULL `zeDeviceGetVectorWidthPropertiesExt` segfault on `half_fp_config`

**Symptom.** Any SYCL query of `device::half_fp_config` (and
`native_vector_width_half`) on the B580 segfaults. In PyTorch this is the *first*
device touch: `torch.xpu.get_device_name(0)` →
`c10::xpu::initDeviceProperties` → crash.

**gdb backtrace (the load-bearing frames):**

```
#0  0x0000000000000000 in ?? ()                          <- call through a NULL pointer
#1  std::_Function_handler<void (ZeStruct<_ze_device_vector_width_properties_ext_t>&), ...>
        ::_M_invoke(...)                                  in libur_adapter_level_zero_v2.so
#4  ur::level_zero::urDeviceGetInfo(...)                  in libur_adapter_level_zero_v2.so
#6  urDeviceGetInfo()                                     in libur_loader.so
#7  sycl::..::device_impl::get_info<info::device::half_fp_config>()   in libsycl.so
#9  c10::xpu::initDeviceProperties(...)                   in libc10_xpu.so
```

**Root cause.** The Unified Runtime Level-Zero **v2** adapter, while building
device properties, invokes `zeDeviceGetVectorWidthPropertiesExt` (the
`ZE_extension_device_vector_sizes` / `ze_device_vector_width_properties_ext_t`
extension). On this driver that function pointer is **NULL**, and the adapter
**calls it without a null-check**. The extension was added around Level-Zero spec
v1.13 (loader ≈ v1.21); the **stock loader is `1.20.6`**, which predates it — so
the loader's dispatch yields NULL **regardless of which compute-runtime is
installed**.

**It is not a PyTorch bug.** Reproduced with a 10-line pure-SYCL program built
with the system oneAPI 2026.0 compiler (no torch in the process):

```cpp
auto c = dev.get_info<sycl::info::device::half_fp_config>();   // <- SIGSEGV
```

**Actionable for Intel.**
1. The UR Level-Zero v2 adapter should **null-check** the
   `zeDeviceGetVectorWidthPropertiesExt` pointer (or check extension
   availability) before calling it, and degrade gracefully when the driver/loader
   doesn't provide it.
2. The loader-version dependency for that extension should be documented.

**Fix used here.** Move to a version-matched stack whose loader **does** dispatch
the extension and whose driver exports it: **loader 1.29.0 + NEO 26.05.37020.3 +
IGC 2.28.4 + gmmlib 22.9.0**, shadowed via `LD_LIBRARY_PATH` (the system driver
is left in place — see `scripts/setup_arc_stack.sh`).

---

## Finding 2 — the documented `SYCL_UR_USE_LEVEL_ZERO_V2=0` workaround does **not** work

Intel's SYCL docs say `SYCL_UR_USE_LEVEL_ZERO_V2=0` forces the legacy Level-Zero
**v1** adapter, which "does not make that extension call." On this stack that is
**false**: with the variable set to `0`, *and* with the v1 adapter force-loaded
via `UR_ADAPTERS_FORCE_LOAD`, the `half_fp_config` query **still segfaults**.
Tested on torch 2.12 (runtime 2025.3.2), torch 2.14-nightly (runtime 2026.0.0),
and the system SYCL 2026.0 — all crash.

**Actionable for Intel.** Either the v1 adapter on these builds also calls the
ext unconditionally, or the env var isn't honored as documented for Battlemage.
The documented mitigation gives users a false sense that there's an escape hatch
when, on this hardware/driver, there isn't.

---

## Finding 3 — `gmm_helper/resource_info.cpp:15` abort when component versions are mixed

**Symptom.** Swapping in *only* a newer compute-runtime (e.g. NEO 25.48 / 26.05 /
26.22) while leaving the stock IGC 2.22.2 / gmmlib 22.8.2 aborts during device
creation:

```
Abort was called at 15 line in file: ../../neo/shared/source/gmm_helper/resource_info.cpp
```

**Root cause.** Line 15 is `UNRECOVERABLE_IF(resourceInfo->peekHandle() == 0)`,
which `abort()`s (SIGABRT) when gmmlib returns a NULL `GmmResourceInfo` handle —
i.e. the NEO GMM client and the gmmlib it's talking to disagree at the API level.
**NEO, IGC, and gmmlib are version-locked**, stated per release in each
compute-runtime "Additional components revisions used in build" section. All
shared objects resolving at the linker level does **not** imply runtime
compatibility.

**Actionable for Intel.** A version-mismatch should produce a clear diagnostic,
not a bare `abort()`. The known-good per-release matrix (below) should be easier
to discover.

| compute-runtime | IGC | gmmlib |
| --- | --- | --- |
| 25.44.36015.5 (stock) | 2.22.2 | 22.8.2 |
| 25.48.36300.8 | 2.24.8 | 22.8.2 |
| **26.05.37020.3 (used here)** | **2.28.4** | **22.9.0** |
| 26.22.38646.4 | 2.36.3 | 22.10.0 |

> Avoid **26.14.37833.4** — it has a separate multi-process / device-enumeration
> regression on Battlemage (intel/compute-runtime#922, #921). 26.05 is the
> confirmed-working baseline.

---

## Finding 4 — PyTorch-XPU init-order segfault (availability probe before `init()`)

**Symptom.** With the *fixed* GPU stack, `torch.xpu.init()` and
`torch.xpu.get_device_name()` work 12/12 in isolation — but Boltz still
segfaulted in `torch.xpu.set_device()` → `_lazy_init`. Minimal repro:

```python
import torch
torch.xpu.is_available()   # or torch.xpu.device_count()
torch.xpu.init()           # <- SIGSEGV
```

vs. the working order:

```python
import torch
torch.xpu.init()           # init FIRST
torch.xpu.is_available()   # now safe
```

**Root cause.** Calling `torch.xpu.is_available()` / `device_count()` *before*
`torch.xpu.init()` leaves the driver in a state where the subsequent full init
crashes. PyTorch **Lightning** triggers exactly this: its accelerator connector
calls `Accelerator.is_available()` / `auto_device_count()` before
`setup_device()`.

**Actionable for Intel / PyTorch.** An availability/count probe must not poison a
later `init()`. This breaks any framework that checks availability before
initializing the device (Lightning, and likely others).

**Fix used here.** The launcher calls `torch.xpu.init()` as its very first XPU
operation, before any other import or probe (`scripts/run_boltz.py`).

---

## Finding 5 — PyTorch Lightning 2.5 has no XPU accelerator

`pytorch_lightning` 2.5 registers only `cpu / cuda / mps / tpu`. The launcher
registers a minimal `XPUAccelerator` and pins `SingleDeviceStrategy(xpu:0)`. A
custom accelerator + `strategy="auto"` resolves `root_device` to `cpu`; pinning
the strategy is required.

---

## Finding 6 — Arc Pro B70 (`e223`) bring-up: `force_probe` + GuC firmware

Upgrading from the B580 (12 GB, `e20b`) to an **Arc Pro B70 (32 GB,
`8086:e223`)** — same Battlemage family — took two kernel-side fixes before the
GPU was usable. Symptom: `/dev/dri` empty, screen black after install.

1. **`e223` is not in the `xe` driver's device table** (kernel 6.19). `xe` loaded
   but bound to zero devices. Fix: boot with **`xe.force_probe=e223`**.
2. **GuC firmware too old.** With `force_probe` set, `xe` got further but failed:
   `GuC firmware (70.49.4) is recommended, but only (70.40.2) was found` →
   `Failed to initialize uC (-ENXIO)`. The distro `firmware-intel-graphics`
   (2025-04) predates this SKU. Fix: drop the upstream `linux-firmware`
   `xe/bmg_guc_70.bin` (loaded as **70.65.0**) into `/lib/firmware/xe/`,
   `update-initramfs`, reload `xe`. GuC then inits clean, `/dev/dri/renderD128`
   appears, 32 GB enumerated.

**The whole compute stack transferred unchanged** — the same `gpu-stack`
(loader 1.29 / NEO 26.05 / IGC 2.28.4 / gmm 22.9.0) + launcher ran torch.xpu on
`e223` immediately (fp32 + bf16, 32.5 GB). The Battlemage pipeline is portable
B580 → B70 Pro.

**Actionable for Intel:** ship `e223` enablement (drop the `force_probe` gate)
and the matching GuC firmware (≥70.49.4) in distro `linux-firmware` sooner — out
of the box, a brand-new Arc Pro is a black screen on a current stable distro
until you hand-patch firmware and add a boot flag.

---

## Finding 7 — the B70 does real FP64 (double precision) — usable for science beyond AI

A common assumption is that double-precision (FP64) compute is effectively
NVIDIA-only, or that consumer/workstation GPUs cripple it via slow software
emulation. **Measured on the Arc Pro B70, that is not the case.**

- The device advertises the SYCL **`fp64` aspect** (in `sycl-ls --verbose`).
- `torch.float64` runs and is **IEEE-correct**: `1/3 → 0.3333333333333333`.
- Saturating matmul throughput (`scripts/bench_precision.py`):

| Precision | B70 throughput |
| --- | --- |
| fp16 / bf16 (XMX matrix engines) | ~145–170 TFLOPS |
| fp32 | ~21 TFLOPS |
| **fp64 (double)** | **~1.4 TFLOPS** |

The **fp32:fp64 ratio is ~14:1**. That is the load-bearing number: pure software
FP64 emulation would be ~64:1 or worse (~0.1 TFLOPS). ~1.4 TFLOPS is a *usable*
double-precision rate — comparable to a consumer RTX 4090's FP64 (~1.3 TFLOPS),
though well below datacenter parts (A100/H100 ≈ 10–34 TFLOPS; Intel Data Center
GPU Max is the high-FP64 Intel part).

**Why it matters.** Boltz-2 itself uses fp16/fp32 only — FP64 is irrelevant to
folding (and the ~170 TFLOPS fp16 path is what makes folds fast). But this shows
the B70 is not just an AI accelerator: it can run genuine double-precision
scientific workloads (molecular dynamics, quantum chemistry, CFD) on Arc,
NVIDIA-free. The real barrier for those fields is the **CUDA-only software
ecosystem**, not Arc silicon — the same porting gap this project closes for
Boltz-2.

---

## Operational note — a crashed XPU process wedges the device

When an XPU process segfaults mid-execution, the `xe` device is left in a fault
state for ~30–60 s; subsequent `_lazy_init` calls segfault until it self-recovers
(no GPU reset / no root needed). When iterating, health-check with
`torch.xpu.get_device_name(0)` in a throwaway process before re-running. This
made several bugs *look* intermittent when they were deterministic.

---

## What did **not** need changing

- **Boltz-2 source:** unchanged. cuEquivariance is an optional `[cuda]` extra and
  is auto-disabled on non-CUDA hardware (`--no_kernels` belt-and-suspenders).
- **Precision:** ran fp32 (`precision="32-true"`), which also sidesteps the
  hardcoded `torch.autocast("cuda")` blocks in the OpenFold-derived attention
  code (they only matter under mixed precision). bf16 is future work.
- **The system GPU driver:** never modified — the fix is a userspace
  `LD_LIBRARY_PATH` shadow, so the box's existing inference workloads keep
  running on the stock 25.44 stack.
