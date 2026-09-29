import json,glob,os,subprocess,re,itertools
import numpy as np
S='<scratch>/sc'
US='<repo>/experiments/xpu-fork/tools/USalign'
REF={'sod1':'2C9V','vim1_blind':'9RFM','mpro_paxlovid':'7VH8','abl_imatinib':'1IEP'}
CF=['b70-default','b70-fp32','a5000-default','a5000-fp32','3060-default','3060-fp32']
T=list(REF)
def cif(c,t,s):
    g=glob.glob(f'{S}/{c}/{t}/seed{s}/boltz_results_*/predictions/{t}/{t}_model_0.cif');return g[0] if g else None
def us(a,b):
    o=subprocess.run([US,a,b,'-outfmt','2'],capture_output=True,text=True).stdout
    l=[x for x in o.splitlines() if x and not x.startswith('#')][0].split('\t')
    return float(l[3]),float(l[4])
def ca(p):
    # mmCIF atom_site parse
    lines=open(p).read().splitlines();cols=[];i=0;out={}
    while i<len(lines):
        if lines[i].startswith('_atom_site.'):
            while lines[i].startswith('_atom_site.'): cols.append(lines[i].split('.')[1].strip());i+=1
            while i<len(lines) and (lines[i].startswith('ATOM') or lines[i].startswith('HETATM')):
                f=lines[i].split();d=dict(zip(cols,f))
                ch=d.get('auth_asym_id',d.get('label_asym_id'))
                if d['label_atom_id']=='CA' and f[0]=='ATOM':
                    out.setdefault(ch,[]).append((d.get('label_seq_id'),float(d['Cartn_x']),float(d['Cartn_y']),float(d['Cartn_z'])))
                i+=1
            break
        i+=1
    first=list(out)[0];return [r[0] for r in out[first]],np.array([r[1:] for r in out[first]])
def kabsch(P,Q):
    P=P-P.mean(0);Q=Q-Q.mean(0);H=P.T@Q;U,Sv,Vt=np.linalg.svd(H);d=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,d]);R=Vt.T@D@U.T;return np.sqrt(((P@R.T-Q)**2).sum(1).mean())
R={}
for c in CF:
  for t in T:
    for s in range(5):
      d=f'{S}/{c}/{t}/seed{s}'
      if not os.path.isdir(d): R[(c,t,s)]=None;continue
      b=json.load(open(d+'/bench.json'));log=open(d+'/run.log',errors='ignore').read()
      p=cif(c,t,s);r=dict(b=b,cif=p,xpu='Running on Intel GPU (XPU)' in log,oom='out of memory' in log.lower())
      if p:
        r['tm'],r['rmsd']=us(p,f'<repo>/experiments/xpu-fork/refs/{REF[t]}.cif')
        a=glob.glob(os.path.dirname(p)+f'/affinity_{t}.json')
        r['aff']=json.load(open(a[0]))['affinity_pred_value'] if a else None
      R[(c,t,s)]=r
print('== counts/status/provenance')
for c in CF:
  for t in T:
    rs=[R[(c,t,s)] for s in range(5) if R[(c,t,s)]]
    print(c,t,len(rs),'cif',sum(1 for r in rs if r['cif']),'status',{r['b']['status'] for r in rs},'dev',{r['b'].get('device_name') for r in rs},
      'prec',{(r['b'].get('precision_actual'),r['b'].get('autocast_device')) for r in rs},'xpulog',sum(r['xpu'] for r in rs),'oom',sum(r['oom'] for r in rs))
print('== accuracy mean TM / RMSD (min-max)')
for t in T:
  for c in CF:
    rs=[R[(c,t,s)] for s in range(5) if R[(c,t,s)] and R[(c,t,s)]['cif']]
    if rs: tm=[r['tm'] for r in rs];rm=[r['rmsd'] for r in rs];print(t,c,f'{np.mean(tm):.4f} [{min(tm):.4f}-{max(tm):.4f}] {np.mean(rm):.3f} [{min(rm):.2f}-{max(rm):.2f}]')
print('== pairwise same seed max RMSD (USalign / kabsch)')
for a,b in [('b70-default','a5000-default'),('3060-default','a5000-default'),('a5000-fp32','a5000-default'),('b70-fp32','a5000-fp32'),('b70-fp32','b70-default'),('3060-fp32','a5000-fp32')]:
  for t in T:
    u=[];k=[]
    for s in range(5):
      x,y=R[(a,t,s)],R[(b,t,s)]
      if x and y and x['cif'] and y['cif']:
        u.append(us(x['cif'],y['cif'])[1]);s1,P=ca(x['cif']);s2,Q=ca(y['cif']);assert s1==s2;k.append(kabsch(P,Q))
    if u: print(a,'vs',b,t,'n',len(u),f'US max {max(u):.3f} kabsch max {max(k):.4f}')
print('== affinity')
for t in ['mpro_paxlovid','abl_imatinib']:
  for c in CF:
    v=[R[(c,t,s)]['aff'] for s in range(5) if R[(c,t,s)] and R[(c,t,s)]['cif']]
    if v: print(t,c,[round(x,3) for x in v],f'mean {np.mean(v):.3f}')
  for a,b in [('b70-default','a5000-default'),('3060-default','a5000-default'),('b70-fp32','a5000-fp32'),('a5000-fp32','a5000-default')]:
    d=[abs(R[(a,t,s)]['aff']-R[(b,t,s)]['aff']) for s in range(5) if R[(a,t,s)] and R[(b,t,s)] and R[(a,t,s)]['cif'] and R[(b,t,s)]['cif']]
    if d: print(' ',a,b,f'max|d| {max(d):.4f}')
print('== wall / mem')
W={}
for c in CF:
  for t in T:
    rs=[R[(c,t,s)] for s in range(5) if R[(c,t,s)] and R[(c,t,s)]['cif']]
    if rs:
      W[(c,t)]=np.mean([r['b']['wall_s'] for r in rs]);print(c,t,f"wall {W[(c,t)]:.2f} sd {np.std([r['b']['wall_s'] for r in rs]):.1f} mem max {max(r['b']['peak_mem_gib'] for r in rs):.3f} mean {np.mean([r['b']['peak_mem_gib'] for r in rs]):.3f}")
for t in T:
  b,a,g=W[('b70-default',t)],W[('a5000-default',t)],W[('3060-default',t)]
  print(t,f'B70/A5000 {b/a:.3f}  A5000 less time by {(1-a/b)*100:.1f}%  3060/B70 {g/b:.3f}')
for t in T:
  m=lambda c:np.mean([R[(c,t,s)]['b']['peak_mem_gib'] for s in range(5)])
  print(t,f'B70 bf16 vs fp32 mem reduction {(1-m("b70-default")/m("b70-fp32"))*100:.1f}%')
