#!/usr/bin/env python3
"""Render the Boltz-2 SOD1 prediction as poster-quality images (matplotlib only)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

PDB = sys.argv[1] if len(sys.argv) > 1 else "../examples/sod1_model_0.pdb"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."

# --- parse C-alpha trace + pLDDT (B-factor column) ---
xs, ys, zs, bb = [], [], [], []
for ln in open(PDB):
    if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
        xs.append(float(ln[30:38])); ys.append(float(ln[38:46]))
        zs.append(float(ln[46:54])); bb.append(float(ln[60:66]))
P = np.array([xs, ys, zs]).T
B = np.array(bb)
n_res = len(P)

# --- Catmull-Rom spline for a smooth ribbon (pure numpy) ---
def catmull_rom(P, m=18):
    Pp = np.vstack([P[0], P, P[-1], P[-1]])
    out = []
    for i in range(1, len(Pp) - 2):
        p0, p1, p2, p3 = Pp[i-1], Pp[i], Pp[i+1], Pp[i+2]
        for t in np.linspace(0, 1, m, endpoint=False):
            t2, t3 = t*t, t*t*t
            out.append(0.5*((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t2 + (-p0+3*p1-3*p2+p3)*t3))
    return np.array(out)

S = catmull_rom(P)
# value arrays along the spline
idx = np.linspace(0, 1, len(S))                          # rainbow N->C
Bs  = np.interp(np.linspace(0, len(B)-1, len(S)), np.arange(len(B)), B)  # pLDDT

# AlphaFold pLDDT colormap (orange<50 .. yellow .. cyan .. blue>90)
plddt_cmap = LinearSegmentedColormap.from_list(
    "plddt", ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"])

def ribbon(ax, vals, cmap):
    segs = np.stack([S[:-1], S[1:]], axis=1)
    c = (vals[:-1] + vals[1:]) / 2
    norm = plt.Normalize(c.min(), c.max())
    # glow: thick faint passes under a bright thin core
    for lw, a in [(16, 0.06), (11, 0.10), (7, 0.20)]:
        g = Line3DCollection(segs, linewidths=lw, alpha=a, capstyle="round")
        g.set_array(c); g.set_cmap(cmap); g.set_norm(norm); ax.add_collection3d(g)
    core = Line3DCollection(segs, linewidths=3.2, capstyle="round")
    core.set_array(c); core.set_cmap(cmap); core.set_norm(norm); ax.add_collection3d(core)
    return norm

def style(ax):
    c = P.mean(0); r = (P.max(0) - P.min(0)).max() / 2 * 1.05
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=-60)
    ax.set_facecolor("black")

# --- 1) hero rainbow ---
fig = plt.figure(figsize=(10, 10), facecolor="black")
ax = fig.add_subplot(111, projection="3d"); ribbon(ax, idx, plt.cm.turbo); style(ax)
fig.text(.5, .085, "Human Superoxide Dismutase 1  (SOD1)", color="white",
         ha="center", fontsize=17, weight="bold")
fig.text(.5, .052, "folded by Boltz-2 on an Intel Arc B580  ·  NVIDIA-free  ·  pTM 0.967",
         color="#9fb4c8", ha="center", fontsize=10.5)
fig.savefig(f"{OUT}/sod1_rainbow.png", dpi=300, facecolor="black", bbox_inches="tight")

# --- 2) pLDDT confidence ---
fig = plt.figure(figsize=(10, 10), facecolor="black")
ax = fig.add_subplot(111, projection="3d"); norm = ribbon(ax, Bs, plddt_cmap); style(ax)
sm = plt.cm.ScalarMappable(cmap=plddt_cmap, norm=norm)
cb = fig.colorbar(sm, ax=ax, shrink=.45, pad=-.02); cb.set_label("pLDDT (confidence)", color="white")
cb.ax.yaxis.set_tick_params(color="white"); plt.setp(plt.getp(cb.ax,"yticklabels"), color="white")
fig.text(.5, .085, f"SOD1 — per-residue confidence (mean pLDDT {B.mean():.0f})",
         color="white", ha="center", fontsize=15, weight="bold")
fig.savefig(f"{OUT}/sod1_plddt.png", dpi=300, facecolor="black", bbox_inches="tight")

# --- 3) two-panel poster ---
fig = plt.figure(figsize=(17, 9), facecolor="black")
for j, (vals, cmap, lab) in enumerate(
        [(idx, plt.cm.turbo, "chain N → C"), (Bs, plddt_cmap, "pLDDT confidence")]):
    ax = fig.add_subplot(1, 2, j+1, projection="3d"); ribbon(ax, vals, cmap); style(ax)
    ax.text2D(.5, .02, lab, transform=ax.transAxes, color="#9fb4c8", ha="center", fontsize=11)
fig.suptitle("Human SOD1 · Boltz-2 on Intel Arc B580 · pTM 0.967 · 154 aa · ~17 s",
             color="white", fontsize=16, weight="bold", y=.95)
fig.savefig(f"{OUT}/sod1_poster.png", dpi=220, facecolor="black", bbox_inches="tight")

print(f"rendered {n_res} residues -> sod1_rainbow.png, sod1_plddt.png, sod1_poster.png")
