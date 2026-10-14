"""Quick probe (exploratory, post hoc): top-100 overlap vs the sample's own count at its rank-100 item.
Usage: .venv/bin/python3 scripts/ssp/probe_boundary.py <k|all>   -> /tmp/h_check_<k>.pkl, then bin_boundary.py
"""
import sys,os,pickle,numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'scripts/ssp')
import frame as F, metrics as M, est as E
DS=["ml-latest","ml-1m","amazon-video-games","amazon-musical-instruments","amazon-software","amazon-movies-tv","amazon-books","gowalla","brightkite","kuairand-pure","lastfm-1k"]
rng=np.random.default_rng(7)
K=None if sys.argv[1]=="all" else int(sys.argv[1]); out=[]
for d in DS:
    fr=F.load_frame(d)
    for N in [100,400,1000,4000,10000]:
        if N>fr.n_pop*0.5: continue
        for r in range(12):
            u,pi=F.draw_users(fr,N,rng,"uniform")
            S,_,_,_=F.draw_rows(fr,u,K,rng)
            c=E.pop_count(S)
            h_obs=np.sort(c)[::-1][99]
            ov=M.overlap_at_m(c,fr.count_A,100,rng)
            out.append((d,N,h_obs,ov))
    del fr
import pickle; pickle.dump(out,open(f'/tmp/h_check_{sys.argv[1]}.pkl','wb'))
print(len(out))
