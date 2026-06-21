import sys, torch
# Force a clean XPU init BEFORE Boltz's heavy imports (numba/scipy/etc.) can perturb
# the GPU stack — otherwise the first device touch deep in Lightning segfaults.
# CRITICAL: torch.xpu.init() MUST run before any is_available()/device_count() probe —
# on this Arc stack an availability query before init() poisons init and segfaults.
try:
    torch.xpu.init(); _t=torch.zeros(1, device="xpu"); torch.xpu.synchronize(); del _t
    print("[run_boltz] XPU pre-initialized cleanly")
except Exception as e:
    print("[run_boltz] early xpu init note:", e)
import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry

class XPUAccelerator(Accelerator):
    def setup_device(self, device): torch.xpu.set_device(device)
    def teardown(self): torch.xpu.empty_cache()
    def get_device_stats(self, device): return {}
    @staticmethod
    def parse_devices(devices):
        if isinstance(devices, int): return list(range(max(1, devices)))
        return devices
    @staticmethod
    def get_parallel_devices(devices):
        if isinstance(devices, int): devices = list(range(max(1, devices)))
        return [torch.device("xpu", i) for i in devices]
    @staticmethod
    def auto_device_count(): return torch.xpu.device_count()
    @staticmethod
    def is_available(): return torch.xpu.is_available()

try:
    AcceleratorRegistry.register("xpu", XPUAccelerator, description="Intel XPU")
except Exception as e:
    print("xpu register note:", e)

_orig = pl.Trainer.__init__
def _init(self, *a, **kw):
    # Boltz passes accelerator="gpu"; route to the Arc GPU (xpu) and force fp32 first.
    if kw.get("accelerator") in ("gpu", "xpu") and torch.xpu.is_available():
        from pytorch_lightning.strategies import SingleDeviceStrategy
        kw["accelerator"] = "xpu"
        kw["precision"] = "32-true"
        kw["strategy"] = SingleDeviceStrategy(device=torch.device("xpu", 0))
    return _orig(self, *a, **kw)
pl.Trainer.__init__ = _init

if __name__ == "__main__":
    from boltz.main import cli
    sys.exit(cli())
