#!/usr/bin/env python3
"""Render the Abl-kinase + imatinib complex (Boltz-2 on Arc Pro B70). matplotlib only."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PDB = "../examples/abl_imatinib_model_0.pdb"
ELEM = {'C':'#cfd8dc','N':'#3a5bff','O':'#ff3320','S':'#ffd000'}

ca, lig = [], []   # protein CA ; ligand heavy atoms (name,elem,xyz)
for ln in open(PDB):
    if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
        ca.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    elif ln.startswith("HETATM") and ln[21] == "B":
        el = (ln[76:78].strip() or ln[12:16].strip()[0])
        lig.append((el, np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])))
ca = np.array(ca)
L = np.array([a[1] for a in lig]); Lel = [a[0] for a in lig]

def catmull_rom(P, m=18):
    Pp = np.vstack([P[0], P, P[-1], P[-1]]); out = []
    for i in range(1, len(Pp)-2):
        p0,p1,p2,p3 = Pp[i-1],Pp[i],Pp[i+1],Pp[i+2]
        for t in np.linspace(0,1,m,endpoint=False):
            t2,t3=t*t,t*t*t
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3))
    return np.array(out)

def draw(ax, ligand_zoom=False):
    S = catmull_rom(ca); idx = np.linspace(0,1,len(S)); segs = np.stack([S[:-1],S[1:]],axis=1)
    c = (idx[:-1]+idx[1:])/2
    a_scale = .5 if not ligand_zoom else .25
    for lw,al in [(9,.04*a_scale*10),(5,.10),(2.4,.7 if not ligand_zoom else .35)]:
        lc=Line3DCollection(segs, linewidths=lw, alpha=min(al,.7), capstyle="round")
        lc.set_array(c); lc.set_cmap(plt.cm.turbo); ax.add_collection3d(lc)
    # ligand bonds
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            if np.linalg.norm(L[i]-L[j]) < 1.85:
                p=np.stack([L[i],L[j]]); ax.plot(*p.T, color="#f5f5f5", lw=5 if ligand_zoom else 3, zorder=5, solid_capstyle="round")
    # ligand atoms (glow)
    for el,xyz in zip(Lel,L):
        base = 260 if ligand_zoom else 90
        for s,al in [(base*2.4,.18),(base*1.5,.32),(base,1)]:
            ax.scatter(*xyz, s=s, c=ELEM.get(el,'#cfd8dc'), alpha=al, edgecolors="none", depthshade=False, zorder=6)
    if ligand_zoom:
        ctr=L.mean(0); r=(L.max(0)-L.min(0)).max()/2*2.0
    else:
        ctr=ca.mean(0); r=(ca.max(0)-ca.min(0)).max()/2*1.05
    ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
    ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(16,-72); ax.set_facecolor("black")

# hero: full complex
fig = plt.figure(figsize=(10,10), facecolor="black"); ax=fig.add_subplot(111,projection="3d")
draw(ax)
fig.text(.5,.10,"Abl kinase + imatinib (Gleevec)", color="white", ha="center", fontsize=17, weight="bold")
fig.text(.5,.068,"Boltz-2 on an Intel Arc Pro B70 (32 GB) — NVIDIA-free", color="#9fb4c8", ha="center", fontsize=11)
fig.text(.5,.040,"predicted binder  P=0.61  (~0.35 µM)   vs   aspirin decoy  P=0.16  (~240 µM)",
         color="#7fd6a0", ha="center", fontsize=10.5)
fig.savefig("abl_imatinib.png", dpi=300, facecolor="black", bbox_inches="tight")

# pocket zoom
fig = plt.figure(figsize=(10,10), facecolor="black"); ax=fig.add_subplot(111,projection="3d")
draw(ax, ligand_zoom=True)
fig.text(.5,.075,"Imatinib in the ATP pocket (37 heavy atoms, C₂₉N₇O)", color="white", ha="center", fontsize=15, weight="bold")
fig.savefig("abl_imatinib_pocket.png", dpi=300, facecolor="black", bbox_inches="tight")
print("wrote abl_imatinib.png, abl_imatinib_pocket.png")
