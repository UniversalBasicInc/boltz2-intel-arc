#!/usr/bin/env python3
"""Generic protein+ligand render: hero ribbon+ligand and an active-site warhead/Cys zoom.
Usage: render_complex.py <pdb> <ligand_chain> <cys_resnum> <title> <subtitle> <outprefix>"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PDB, LCH, CYS, TITLE, SUB, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
LABEL = sys.argv[7] if len(sys.argv) > 7 else f"Cys{CYS}"
ELEM = {'C':'#d0d8dc','N':'#3a5bff','O':'#ff3320','F':'#7ee06a','S':'#ffd000','CL':'#1fd14b'}

ca, lig, cysat = [], [], []
for ln in open(PDB):
    if ln[:6] not in ("ATOM  ", "HETATM"): continue
    nm, ch, ri = ln[12:16].strip(), ln[21], int(ln[22:26])
    el = (ln[76:78].strip() or nm[0]); xyz = np.array([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])])
    if nm == "CA" and ch == "A": ca.append(xyz)
    if ch == LCH: lig.append((el, xyz))
    if ch == "A" and ri == CYS and nm not in ("N","C","O") and el != "H": cysat.append((nm, el, xyz))
ca = np.array(ca); L = np.array([a[1] for a in lig]); Lel = [a[0] for a in lig]
SG = next((a[2] for a in cysat if a[0]=="SG"), None)
warhead = min(L, key=lambda x: np.linalg.norm(x-SG)) if SG is not None else None
dist = np.linalg.norm(warhead-SG) if SG is not None else None

def cr(P, m=16):
    Pp=np.vstack([P[0],P,P[-1],P[-1]]); o=[]
    for i in range(1,len(Pp)-2):
        p0,p1,p2,p3=Pp[i-1],Pp[i],Pp[i+1],Pp[i+2]
        for t in np.linspace(0,1,m,endpoint=False):
            t2,t3=t*t,t*t*t; o.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3))
    return np.array(o)

def lig_draw(ax, big=False):
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
    ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(15,-68);ax.set_facecolor("black")

# hero
fig=plt.figure(figsize=(10,10),facecolor="black");ax=fig.add_subplot(111,projection="3d")
S=cr(ca);seg=np.stack([S[:-1],S[1:]],axis=1);c=np.linspace(0,1,len(seg))
for lw,al in [(8,.05),(4.5,.12),(2.2,.72)]:
    lc=Line3DCollection(seg,linewidths=lw,alpha=al,capstyle="round");lc.set_array(c);lc.set_cmap(plt.cm.turbo);ax.add_collection3d(lc)
lig_draw(ax); frame(ax,ca)
fig.text(.5,.095,TITLE,color="white",ha="center",fontsize=16,weight="bold")
fig.text(.5,.060,SUB,color="#9fb4c8",ha="center",fontsize=10.5)
fig.text(.5,.034,"Boltz-2 on an Intel Arc Pro B70 (32 GB) — NVIDIA-free",color="#7fd6a0",ha="center",fontsize=10)
fig.savefig(f"{OUT}.png",dpi=300,facecolor="black",bbox_inches="tight")

# active-site warhead zoom
if SG is not None:
    fig=plt.figure(figsize=(10,10),facecolor="black");ax=fig.add_subplot(111,projection="3d")
    lig_draw(ax,big=True)
    for i in range(len(cysat)):
        for j in range(i+1,len(cysat)):
            if np.linalg.norm(cysat[i][2]-cysat[j][2])<1.9:
                p=np.stack([cysat[i][2],cysat[j][2]]);ax.plot(*p.T,color="#9aa",lw=3,zorder=4)
    for nm,el,xyz in cysat:
        ax.scatter(*xyz,s=150,c=ELEM.get(el,'#d0d8dc'),edgecolors="none",depthshade=False,zorder=5)
    ax.plot(*np.stack([warhead,SG]).T,color="#ffe14d",lw=2,ls=(0,(2,2)),zorder=8)
    ax.text(*((warhead+SG)/2+np.array([.3,.3,.4])),f"{dist:.1f} Å",color="#ffe14d",fontsize=14,weight="bold",zorder=9)
    ax.text(*(SG+np.array([.4,.4,.5])),LABEL,color="#cfe",fontsize=12,zorder=9)
    frame(ax,np.vstack([L,[a[2] for a in cysat]]),pad=1.5)
    fig.text(.5,.075,f"Covalent warhead on {LABEL}",color="white",ha="center",fontsize=15,weight="bold")
    fig.text(.5,.045,SUB,color="#9fb4c8",ha="center",fontsize=10)
    fig.savefig(f"{OUT}_site.png",dpi=300,facecolor="black",bbox_inches="tight")
print(f"{OUT}: ligand {len(L)} atoms, warhead-Cys{CYS} {dist:.2f} A" if dist else f"{OUT}: rendered")
