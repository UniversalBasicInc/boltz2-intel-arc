#!/usr/bin/env python3
"""Render holo SOD1 (Boltz-2 on Arc B580): full structure with Cu/Zn, and an
active-site close-up of the metal coordination shell. matplotlib only."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PDB = "../examples/sod1_holo_model_0.pdb"
ELEM = {'C':'#c2c2c2','N':'#3a5bff','O':'#ff3320','S':'#ffd000',
        'CU':'#e08a3c','ZN':'#9aa6c0'}

atoms, ca = [], []
for ln in open(PDB):
    if ln[:6] in ("ATOM  ", "HETATM"):
        a = dict(nm=ln[12:16].strip(), res=ln[17:20].strip(), ch=ln[21],
                 ri=int(ln[22:26]), el=(ln[76:78].strip() or ln[12:16].strip()[0]),
                 xyz=np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]))
        atoms.append(a)
        if a['nm'] == "CA": ca.append(a['xyz'])
ca = np.array(ca)
CU = next(a['xyz'] for a in atoms if a['el'] == 'CU')
ZN = next(a['xyz'] for a in atoms if a['el'] == 'ZN')

def catmull_rom(P, m=18):
    Pp = np.vstack([P[0], P, P[-1], P[-1]]); out = []
    for i in range(1, len(Pp)-2):
        p0,p1,p2,p3 = Pp[i-1],Pp[i],Pp[i+1],Pp[i+2]
        for t in np.linspace(0,1,m,endpoint=False):
            t2,t3=t*t,t*t*t
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3))
    return np.array(out)

# ---------- 1) full structure + metals ----------
S = catmull_rom(ca); idx = np.linspace(0,1,len(S))
fig = plt.figure(figsize=(10,10), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
segs = np.stack([S[:-1], S[1:]], axis=1); c=(idx[:-1]+idx[1:])/2
for lw,al in [(13,.05),(8,.12),(4,.9)]:
    lc=Line3DCollection(segs, linewidths=lw, alpha=al, capstyle="round")
    lc.set_array(c); lc.set_cmap(plt.cm.turbo); ax.add_collection3d(lc)
for M,el,lab in [(CU,'CU','Cu²⁺'),(ZN,'ZN','Zn²⁺')]:
    for s,al in [(900,.18),(500,.30),(220,1)]:
        ax.scatter(*M, s=s, c=ELEM[el], alpha=al, edgecolors="none", depthshade=False)
    ax.text(*(M+np.array([1.5,1.5,1.5])), lab, color="white", fontsize=13, weight="bold")
ctr=ca.mean(0); r=(ca.max(0)-ca.min(0)).max()/2*1.05
ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(18,-60); ax.set_facecolor("black")
fig.text(.5,.085,"Holo SOD1 — Cu²⁺ (catalytic) + Zn²⁺ (structural)", color="white",
         ha="center", fontsize=16, weight="bold")
fig.text(.5,.052,"Boltz-2 on an Intel Arc B580  ·  pTM 0.980  ·  metals placed in the active site",
         color="#9fb4c8", ha="center", fontsize=10.5)
fig.savefig("sod1_holo.png", dpi=300, facecolor="black", bbox_inches="tight")

# ---------- 2) active-site close-up ----------
site_res=set()
for a in atoms:
    if a['el'] in ('CU','ZN'): continue
    if a['nm'] in ('ND1','NE2','OD1','OD2'):
        if min(np.linalg.norm(a['xyz']-CU), np.linalg.norm(a['xyz']-ZN)) < 2.8:
            site_res.add((a['ch'], a['ri']))
site=[a for a in atoms if (a['ch'],a['ri']) in site_res and a['el']!='H' and a['nm'] not in ('N','C','O')]
fig=plt.figure(figsize=(11,10), facecolor="black"); ax=fig.add_subplot(111,projection="3d")
# intra-residue bonds
for r in site_res:
    ra=[a for a in site if (a['ch'],a['ri'])==r]
    for i in range(len(ra)):
        for j in range(i+1,len(ra)):
            if np.linalg.norm(ra[i]['xyz']-ra[j]['xyz'])<1.8:
                p=np.stack([ra[i]['xyz'],ra[j]['xyz']])
                ax.plot(*p.T, color="#888", lw=2, zorder=1)
for a in site:
    ax.scatter(*a['xyz'], s=70, c=ELEM.get(a['el'],'#c2c2c2'), edgecolors="none", depthshade=False, zorder=3)
# metals + dashed coordination bonds
for M,el,lab in [(CU,'CU','Cu²⁺'),(ZN,'ZN','Zn²⁺')]:
    for s,al in [(1300,.18),(700,.32),(300,1)]:
        ax.scatter(*M, s=s, c=ELEM[el], alpha=al, edgecolors="none", depthshade=False, zorder=4)
    ax.text(*(M+np.array([.6,.6,.8])), lab, color="white", fontsize=14, weight="bold", zorder=6)
    for a in atoms:
        if a['nm'] in ('ND1','NE2','OD1','OD2') and np.linalg.norm(a['xyz']-M)<2.8:
            p=np.stack([M,a['xyz']]); ax.plot(*p.T, color="white", lw=1.4, ls=(0,(2,2)), zorder=5)
# residue labels at CA
canon={47:'His46',49:'His48',64:'His63',72:'His71',81:'His80',84:'Asp83',121:'His120'}
for r in site_res:
    cax=[a for a in atoms if (a['ch'],a['ri'])==r and a['nm']=='CA']
    if cax: ax.text(*cax[0]['xyz'], canon.get(r[1], f"{r[1]}"), color="#cfe", fontsize=10, zorder=6)
allp=np.array([a['xyz'] for a in site]+[CU,ZN]); ctr=allp.mean(0); r=(allp.max(0)-allp.min(0)).max()/2*1.15
ax.set_xlim(ctr[0]-r,ctr[0]+r); ax.set_ylim(ctr[1]-r,ctr[1]+r); ax.set_zlim(ctr[2]-r,ctr[2]+r)
ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(20,-70); ax.set_facecolor("black")
fig.text(.5,.075,"SOD1 active site — metal coordination (~2.0 Å bonds)", color="white",
         ha="center", fontsize=15, weight="bold")
fig.text(.5,.045,"His63 bridges Cu and Zn · predicted by Boltz-2 on Intel Arc, NVIDIA-free",
         color="#9fb4c8", ha="center", fontsize=10)
fig.savefig("sod1_active_site.png", dpi=300, facecolor="black", bbox_inches="tight")
print("wrote sod1_holo.png, sod1_active_site.png")
