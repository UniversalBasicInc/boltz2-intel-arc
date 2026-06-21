#!/usr/bin/env python3
"""SARS-CoV-2 Mpro dimer + nirmatrelvir (Paxlovid), Boltz-2 on Arc Pro B70. matplotlib only."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PDB = "../examples/mpro_paxlovid_model_0.pdb"
ELEM = {'C':'#d0d8dc','N':'#3a5bff','O':'#ff3320','F':'#7ee06a','S':'#ffd000'}

chA, chB, lig, site = [], [], [], []
for ln in open(PDB):
    if ln[:6] not in ("ATOM  ", "HETATM"): continue
    res, ch, nm, ri = ln[17:20].strip(), ln[21], ln[12:16].strip(), int(ln[22:26])
    xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    el = (ln[76:78].strip() or nm[0])
    if nm == "CA" and ch == "A": chA.append(xyz)
    elif nm == "CA" and ch == "B": chB.append(xyz)
    if ch == "C": lig.append((el, xyz))
    if ch == "A" and ri in (145, 41) and nm not in ("N","C","O") and el != "H":
        site.append((res, ri, nm, el, xyz))
chA, chB = np.array(chA), np.array(chB)
L = np.array([a[1] for a in lig]); Lel = [a[0] for a in lig]
SG = next(a[4] for a in site if a[1]==145 and a[2]=="SG")

def cr(P, m=16):
    Pp=np.vstack([P[0],P,P[-1],P[-1]]); o=[]
    for i in range(1,len(Pp)-2):
        p0,p1,p2,p3=Pp[i-1],Pp[i],Pp[i+1],Pp[i+2]
        for t in np.linspace(0,1,m,endpoint=False):
            t2,t3=t*t,t*t*t
            o.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3))
    return np.array(o)

def ribbon(ax, P, cmap, dim=1.0):
    S=cr(P); seg=np.stack([S[:-1],S[1:]],axis=1); c=np.linspace(0,1,len(seg))
    for lw,al in [(8,.05),(4.5,.12),(2.2,.75*dim)]:
        lc=Line3DCollection(seg,linewidths=lw,alpha=min(al,.75),capstyle="round")
        lc.set_array(c); lc.set_cmap(cmap); ax.add_collection3d(lc)

def ligand(ax, big=False):
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            if np.linalg.norm(L[i]-L[j])<1.85:
                p=np.stack([L[i],L[j]]); ax.plot(*p.T,color="#f5f5f5",lw=5 if big else 3,zorder=6,solid_capstyle="round")
    for el,xyz in zip(Lel,L):
        b=240 if big else 80
        for s,a in [(b*2.3,.18),(b*1.5,.32),(b,1)]:
            ax.scatter(*xyz,s=s,c=ELEM.get(el,'#d0d8dc'),alpha=a,edgecolors="none",depthshade=False,zorder=7)

def frame(ax, pts, pad=1.05):
    c=pts.mean(0); r=(pts.max(0)-pts.min(0)).max()/2*pad
    ax.set_xlim(c[0]-r,c[0]+r);ax.set_ylim(c[1]-r,c[1]+r);ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(14,-66);ax.set_facecolor("black")

# hero: dimer + drug
fig=plt.figure(figsize=(11,10),facecolor="black");ax=fig.add_subplot(111,projection="3d")
ribbon(ax,chA,plt.cm.winter); ribbon(ax,chB,plt.cm.autumn); ligand(ax)
frame(ax,np.vstack([chA,chB]))
fig.text(.5,.10,"SARS-CoV-2 Mpro dimer + nirmatrelvir (Paxlovid)",color="white",ha="center",fontsize=16,weight="bold")
fig.text(.5,.068,"Boltz-2 on an Intel Arc Pro B70 (32 GB) — NVIDIA-free   ·   612 residues",color="#9fb4c8",ha="center",fontsize=10.5)
fig.text(.5,.040,"pTM 0.989 · ipTM 0.987 · predicted ~5 nM (exp. ~3 nM) · warhead 2.2 Å from Cys145",color="#7fd6a0",ha="center",fontsize=10)
fig.savefig("mpro_paxlovid.png",dpi=300,facecolor="black",bbox_inches="tight")

# zoom: warhead on Cys145
fig=plt.figure(figsize=(10,10),facecolor="black");ax=fig.add_subplot(111,projection="3d")
ligand(ax,big=True)
for i in range(len(site)):
    for j in range(i+1,len(site)):
        if np.linalg.norm(site[i][4]-site[j][4])<1.85:
            p=np.stack([site[i][4],site[j][4]]);ax.plot(*p.T,color="#9aa",lw=3,zorder=4)
for res,ri,nm,el,xyz in site:
    ax.scatter(*xyz,s=120,c=ELEM.get(el,'#d0d8dc'),edgecolors="none",depthshade=False,zorder=5)
# warhead bond: nearest ligand atom -> Cys145 SG
nl=min(L,key=lambda x:np.linalg.norm(x-SG))
ax.plot(*np.stack([nl,SG]).T,color="#ffe14d",lw=2,ls=(0,(2,2)),zorder=8)
ax.text(*((nl+SG)/2+np.array([.3,.3,.4])),"2.2 Å",color="#ffe14d",fontsize=13,weight="bold",zorder=9)
ax.text(*(SG+np.array([.4,.4,.4])),"Cys145",color="#cfe",fontsize=11,zorder=9)
his=[a for a in site if a[1]==41]
if his: ax.text(*(his[0][4]+np.array([.4,.4,.4])),"His41",color="#cfe",fontsize=11,zorder=9)
frame(ax,np.vstack([L,[a[4] for a in site]]),pad=1.5)
fig.text(.5,.075,"Nirmatrelvir warhead on the catalytic Cys145–His41 dyad",color="white",ha="center",fontsize=14,weight="bold")
fig.text(.5,.045,"the covalent target of Paxlovid — predicted, NVIDIA-free, on Intel Arc",color="#9fb4c8",ha="center",fontsize=10)
fig.savefig("mpro_active_site.png",dpi=300,facecolor="black",bbox_inches="tight")
print("wrote mpro_paxlovid.png, mpro_active_site.png")
