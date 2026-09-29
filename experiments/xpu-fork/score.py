"""Score every run against experimental structures and summarise per device config.

    python score.py <runs_root> <usalign_binary> <refs_dir> [reference_config]

For each run: TM-score / Ca-RMSD vs the experimental structure (US-align),
Boltz confidence (pTM, pLDDT), affinity outputs, wall time, peak memory.
If reference_config is given (e.g. 3060-default), also scores each run's
structure against the reference config's structure for the same target+seed.
Writes runs.csv and summary.md into <runs_root>.
"""
import csv
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

REFS = {
    "sod1": "2C9V",
    "vim1_blind": "9RFM",
    "mpro_paxlovid": "7VH8",
    "abl_imatinib": "1IEP",
}

root, usalign, refs_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ref_cfg = sys.argv[4] if len(sys.argv) > 4 else None


def usalign_pair(a: Path, b: Path) -> tuple[float, float]:
    """First chain vs first chain: TM-score normalised by the second structure, Ca RMSD."""
    out = subprocess.run(
        [usalign, str(a), str(b), "-outfmt", "2"],
        capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    rows = [l for l in out if l and not l.startswith("#")]
    if not rows:
        return float("nan"), float("nan")
    f = rows[0].split("\t")
    return float(f[3]), float(f[4])  # TM2 (by reference), RMSD


def find_one(d: Path, pattern: str):
    hits = sorted(d.rglob(pattern))
    return hits[0] if hits else None


rows = []
for bench in sorted(root.glob("*/*/seed*/bench.json")):
    run = bench.parent
    cfg, target, seed = run.parent.parent.name, run.parent.name, run.name[4:]
    meta = json.loads(bench.read_text())
    r = {
        "config": cfg, "target": target, "seed": int(seed),
        "status": meta["status"], "wall_s": meta["wall_s"],
        "peak_mem_gib": meta["peak_mem_gib"], "device": meta["device_name"],
        "precision_used": f"{meta.get('precision_actual', '?')}@{meta.get('autocast_device', '-')}",
    }
    model = find_one(run, f"{target}_model_0.cif") or find_one(run, f"{target}_model_0.pdb")
    if meta["status"] == "ok" and not model:
        r["status"] = "no structure written (boltz skipped the input, e.g. out of memory)"
    if meta["status"] == "ok" and model:
        r["model"] = str(model)
        ref = refs_dir / f"{REFS[target]}.cif"
        r["tm_vs_exp"], r["rmsd_vs_exp"] = usalign_pair(model, ref)
        conf = find_one(run, f"confidence_{target}_model_0.json")
        if conf:
            c = json.loads(conf.read_text())
            r["ptm"], r["plddt"] = c.get("ptm"), c.get("complex_plddt")
            r["iptm"] = c.get("iptm")
        aff = find_one(run, f"affinity_{target}.json")
        if aff:
            a = json.loads(aff.read_text())
            r["affinity_value"] = a.get("affinity_pred_value")
            r["affinity_prob"] = a.get("affinity_probability_binary")
    rows.append(r)

if ref_cfg:
    ref_models = {
        (r["target"], r["seed"]): r["model"]
        for r in rows if r["config"] == ref_cfg and "model" in r
    }
    for r in rows:
        m = ref_models.get((r["target"], r["seed"]))
        if m and "model" in r and r["config"] != ref_cfg:
            r["tm_vs_ref"], r["rmsd_vs_ref"] = usalign_pair(Path(r["model"]), Path(m))

cols = ["config", "target", "seed", "status", "tm_vs_exp", "rmsd_vs_exp",
        "tm_vs_ref", "rmsd_vs_ref", "ptm", "iptm", "plddt", "affinity_value",
        "affinity_prob", "wall_s", "peak_mem_gib", "device", "precision_used"]
with open(root / "runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


def ms(vals, nd=3):
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{st.mean(v):.{nd}f} ± {st.stdev(v):.{nd}f}"


lines = ["| target | config | n ok | TM vs exp | RMSD vs exp (Å) | TM vs ref | pTM | pLDDT | affinity | wall (s) | peak mem (GiB) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for target in REFS:
    for cfg in sorted({r["config"] for r in rows}):
        g = [r for r in rows if r["target"] == target and r["config"] == cfg]
        if not g:
            continue
        ok = [r for r in g if r["status"] == "ok"]
        lines.append(
            f"| {target} | {cfg} | {len(ok)}/{len(g)} | "
            f"{ms([r.get('tm_vs_exp') for r in ok])} | {ms([r.get('rmsd_vs_exp') for r in ok], 2)} | "
            f"{ms([r.get('tm_vs_ref') for r in ok])} | {ms([r.get('ptm') for r in ok])} | "
            f"{ms([r.get('plddt') for r in ok])} | {ms([r.get('affinity_value') for r in ok], 2)} | "
            f"{ms([r.get('wall_s') for r in ok], 1)} | {ms([r.get('peak_mem_gib') for r in ok], 2)} |"
        )
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
failed = [r for r in rows if r["status"] != "ok"]
if failed:
    print(f"\n{len(failed)} failed runs:")
    for r in failed:
        print(f"  {r['config']} {r['target']} seed{r['seed']}: {r['status'][:150]}")
