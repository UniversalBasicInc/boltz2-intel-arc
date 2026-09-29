"""Minimal repro: a registered custom accelerator + precision="bf16-mixed" runs in fp32."""
import torch, pytorch_lightning as pl
from pytorch_lightning.plugins.precision import MixedPrecision
from boltz.xpu import register_xpu_accelerator
register_xpu_accelerator()

class M(pl.LightningModule):
    def __init__(self): super().__init__(); self.l = torch.nn.Linear(64, 64)
    def predict_step(self, batch, idx):
        y = self.l(batch)
        return {"out_dtype": str(y.dtype), "autocast_xpu": torch.is_autocast_enabled("xpu"),
                "autocast_cuda": torch.is_autocast_enabled("cuda")}

data = torch.utils.data.DataLoader(torch.randn(4, 64), batch_size=4)
for label, kw in [("precision='bf16-mixed'", {"precision": "bf16-mixed"}),
                  ("plugins=[MixedPrecision(device='xpu')]", {"plugins": [MixedPrecision("bf16-mixed", device="xpu")]})]:
    t = pl.Trainer(accelerator="xpu", strategy=pl.strategies.SingleDeviceStrategy(torch.device("xpu", 0)), logger=False, enable_progress_bar=False, **kw)
    print(f"{label:42s} plugin.device={t.precision_plugin.device!r:7s} ->", t.predict(M(), data)[0])
