# Boltz-2 on Intel GPUs: cross-vendor validation for `--accelerator xpu`

Evidence behind the Intel GPU support PR to
[boltz-community](https://github.com/Novel-Therapeutics/boltz-community): the same
Boltz-2 predictions run on an Intel Arc Pro B70, an NVIDIA RTX A5000 and an NVIDIA
RTX 3060, scored against experimental crystal structures and against each other.

## Setup

- **Code:** boltz-community at `401f181` (v2.10.12) plus [`patch/xpu-support.diff`](patch/xpu-support.diff), identical on every machine.
- **Inputs:** four targets with MSAs fetched once and pinned ([`targets_pinned/`](targets_pinned/), checksums in `msa/SHA256SUMS`):
  SOD1, VIM-1 (PDB 9RFM, released 2026-05-27, after Boltz-2's training data), SARS-CoV-2 Mpro dimer + nirmatrelvir (affinity), ABL kinase + imatinib (affinity).
- **Matrix:** 3 GPUs × 4 targets × {bf16 (Boltz-2 default), fp32} × seeds 0–4 = 120 runs.
- **Software:** PyTorch 2.14.0 on every device (`+xpu`, `+cu126`, `+cu130`), Lightning 2.6.6.
- **Hardware:** B70 and A5000 in the same machine (AMD EPYC 7742); the 3060 in a different machine.
- **Scoring:** US-align, first chain vs first chain, TM-score normalised by the reference. References in [`refs/`](refs/) (RCSB, checksummed).

## Results

| Target (reference) | TM vs crystal | Cα RMSD | B70 vs A5000, bf16, same seed (max) |
|---|---|---|---|
| SOD1 (2C9V) | 0.991 | 0.68 Å | 0.01 Å |
| VIM-1 (9RFM) | 0.993 | 0.64 Å | 0.02 Å |
| Mpro + nirmatrelvir (7VH8) | 0.999 | 0.19 Å | 0.02 Å |
| ABL + imatinib (1IEP) | 0.910 | 0.67 Å | 0.01 Å |

- Accuracy against the crystals is identical on all three GPUs in bf16 and fp32.
- Same seed, bf16: B70 vs A5000 ≤ 0.02 Å, and 3060 vs A5000 is also ≤ 0.02 Å. The A5000's own bf16 vs fp32 differ by up to 0.03 Å.
- Same seed, fp32: B70 vs A5000 < 0.001 Å on every target.
- Same-seed affinity predictions agree within 0.025 log10 units (Mpro −2.47 on every card).

Wall time, bf16, median of 5 (seconds):

| Target | Arc Pro B70 | RTX A5000 | RTX 3060 |
|---|---|---|---|
| SOD1 | 52 | 43 | 63 |
| VIM-1 | 56 | 46 | 69 |
| ABL + imatinib | 113 | 105 | 177 |
| Mpro + nirmatrelvir | 202 | 164 | 292 |

Peak memory on the Mpro dimer: B70 5.3 GiB (bf16) / 7.1 GiB (fp32); A5000 8.5 / 13.4 GiB; 3060 8.5 GiB (bf16) and out of memory in fp32 (0 of 5 structures written).

**Determinism:** with a fixed seed, CUDA reruns are byte-identical. XPU reruns differ in the last digits unless `torch.use_deterministic_algorithms(True)` is on; with it they are byte-identical (~3 % slower). The patch turns it on for `--seed` on XPU. Evidence: [`results/determinism.txt`](results/determinism.txt).

## Files

| Path | What |
|---|---|
| `raw/<config>/<target>/seed<N>/` | per-run `bench.json` (time, memory, device, precision actually used), `run.log`, structure (`.cif`), confidence and affinity JSON. Configs: `b70-default` = B70 bf16, `a5000-default`, `3060-default` (bf16), and the `-fp32` variants |
| `results/runs.csv`, `results/summary.md` | per-run scores and per-config summary from `score.py` |
| `results/test_xpu_on_B70.log`, `results/test_cpu_suite_on_HQ.log` | test runs (10/10 XPU tests on the B70; CPU suite unchanged by the patch) |
| `results/independent-audit/` | a second, independent recomputation of every number from the raw files (own parser + Kabsch RMSD, not `score.py`) |
| `bench_predict.py`, `run_matrix.sh`, `score.py` | harness |
| `scripts/remote/` | scripts run inside the Intel GPU container |
| `lightning_bf16_repro.py` | minimal repro: Lightning's `precision="bf16-mixed"` runs fp32 on a custom `xpu` accelerator |

Absolute paths in logs are replaced with placeholders such as `<jarvis-work>`. Only the
predicted structures and JSON outputs are included; Boltz's internal `.npz` arrays are
omitted for size. The 3060 fp32 Mpro runs are recorded with `status: ok` in `bench.json`
because Boltz exits 0 after skipping an input that runs out of memory; `score.py` and
the current `bench_predict.py` mark these as failures.

## Reproduce

```bash
# Intel GPU: PyTorch XPU build, then the patched fork
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/xpu
pip install -e <boltz-community checkout with patch/xpu-support.diff applied>
./run_matrix.sh b70 xpu "$(which python)" targets_pinned runs "default fp32" "0 1 2 3 4"

# NVIDIA: same, with the matching CUDA build of torch and accelerator "gpu"
# Score (US-align compiled from the source whose checksum is in refs/SHA256SUMS):
python score.py runs /path/to/USalign refs a5000-default
```
