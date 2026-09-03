"""
Build the model frame: team-game EPA splits, personnel availability, Kalman
ratings, per-stadium home field, and the QB rating timeline.
"""
import numpy as np, pandas as pd, warnings
from collections import defaultdict
from sklearn.linear_model import Ridge
from .config import (CACHE, SEASON, HIST_START, KALMAN, EPA_RIDGE_ALPHA, EPA_DECAY,
                     QB_DECAY, QB_PRIOR_DB, QB_PRIOR_MU, HFA_SHRINK, STADIUM, TEAM_DIV, ALIAS)
warnings.filterwarnings('ignore')

PBP_COLS = ['game_id','season','week','posteam','defteam','play_type','epa','pass','rush',
            'penalty','penalty_team','penalty_type','penalty_yards','qb_epa',
            'passer_player_id','passer_player_name','interception','fumble_lost','season_type']


def team_games(years):
    """One row per team per game: opponent-split EPA, turnovers, penalties, starter."""
    frames=[]
    for y in years:
        f = CACHE/f"pbp_{y}.parquet"
        if not f.exists(): continue
        d = pd.read_parquet(f, columns=PBP_COLS)
        frames.append(d[d.season_type=='REG'])
    if not frames: raise RuntimeError("no play-by-play frames available")
    pbp = pd.concat(frames, ignore_index=True)

    sc = pbp[pbp.posteam.notna() & pbp.epa.notna() & pbp.play_type.isin(['pass','run'])].copy()
    sc['is_pass'] = (sc.play_type=='pass').astype(int)
    agg = sc.groupby(['game_id','season','week','posteam','defteam']).apply(
        lambda g: pd.Series({
            'off_pass_epa': g.loc[g.is_pass==1,'epa'].mean(),
            'off_rush_epa': g.loc[g.is_pass==0,'epa'].mean(),
            'off_pass_n':   (g.is_pass==1).sum(),
            'off_rush_n':   (g.is_pass==0).sum(),
            'off_epa':      g.epa.mean()})).reset_index().rename(
        columns={'posteam':'team','defteam':'opp'})

    to = pbp[pbp.posteam.notna()].groupby(['game_id','posteam']).agg(
        ints=('interception','sum'), fum=('fumble_lost','sum')).reset_index()
    to['giveaways']=to.ints.fillna(0)+to.fum.fillna(0)
    to=to.rename(columns={'posteam':'team'})[['game_id','team','giveaways']]

    pen = pbp[(pbp.penalty==1)&pbp.penalty_team.notna()].copy()
    pen['is_dpi']=(pen.penalty_type=='Defensive Pass Interference').astype(int)
    pn = pen.groupby(['game_id','penalty_team']).agg(
        pen_n=('penalty','sum'), pen_yds=('penalty_yards','sum'),
        dpi_n=('is_dpi','sum')).reset_index().rename(columns={'penalty_team':'team'})

    qb = sc[(sc.is_pass==1)&sc.passer_player_id.notna()]
    qbg = qb.groupby(['game_id','posteam','passer_player_id','passer_player_name']).agg(
        db=('qb_epa','size'), qb_epa=('qb_epa','mean')).reset_index()
    qbg = qbg.sort_values('db',ascending=False).groupby(['game_id','posteam']).head(1)
    qbg = qbg.rename(columns={'posteam':'team'})

    tg = (agg.merge(to,on=['game_id','team'],how='left')
             .merge(pn,on=['game_id','team'],how='left')
             .merge(qbg,on=['game_id','team'],how='left'))
    for c in ['giveaways','pen_n','pen_yds','dpi_n']: tg[c]=tg[c].fillna(0)
    tg['t_abs']=(tg.season-HIST_START)*18+tg.week
    tg.to_parquet(CACHE/"team_games.parquet")
    return tg


def personnel(years):
    """Snap-share weighted availability loss from weekly injury reports."""
    sn, ij = [], []
    for y in years:
        for lst, kind in ((sn,'snap_counts'), (ij,'injuries')):
            f = CACHE/f"{kind}_{y}.parquet"
            if f.exists(): lst.append(pd.read_parquet(f))
    if not sn or not ij:
        return pd.DataFrame(columns=['season','week','team','off_lost','def_lost'])
    snaps = pd.concat(sn); snaps = snaps[snaps.game_type=='REG']
    inj   = pd.concat(ij); inj   = inj[inj.game_type=='REG']
    xw = pd.read_parquet(CACHE/"players.parquet")[['gsis_id','pfr_id']].dropna()
    snaps = snaps.merge(xw, left_on='pfr_player_id', right_on='pfr_id', how='left')
    snaps['t_abs']=(snaps.season-HIST_START)*18+snaps.week
    inj['t_abs']  =(inj.season-HIST_START)*18+inj.week

    snaps = snaps.sort_values('t_abs')
    snaps['off_pct']=snaps.offense_pct.fillna(0); snaps['def_pct']=snaps.defense_pct.fillna(0)
    DEC=0.80; state={}; roll=[]
    for r in snaps.itertuples():
        k=(r.gsis_id,r.team); o,d,n = state.get(k,(0.0,0.0,0.0))
        roll.append((o/n if n else np.nan, d/n if n else np.nan))
        state[k]=(o*DEC+r.off_pct, d*DEC+r.def_pct, n*DEC+1)
    snaps['off_share_pre']=[x[0] for x in roll]; snaps['def_share_pre']=[x[1] for x in roll]

    share={}
    for r in snaps.itertuples():
        if pd.notna(r.gsis_id):
            share.setdefault((r.gsis_id,r.team),[]).append((r.t_abs,r.off_share_pre,r.def_share_pre))
    def get(pid,team,t):
        v=share.get((pid,team));  best=None
        if not v: return 0.0,0.0
        for tt,o,d in v:
            if tt<t: best=(o,d)
            else: break
        if best is None: return 0.0,0.0
        return (0.0 if pd.isna(best[0]) else best[0], 0.0 if pd.isna(best[1]) else best[1])

    W={'Out':1.0,'Doubtful':0.85,'Questionable':0.30}
    inj['w']=inj.report_status.map(W).fillna(0.0); inj=inj[inj.w>0]
    recs=[]
    for r in inj.itertuples():
        o,d=get(r.gsis_id,r.team,r.t_abs)
        recs.append((r.season,r.week,r.team,r.w*o,r.w*d))
    L=pd.DataFrame(recs,columns=['season','week','team','off_lost','def_lost'])
    L=L.groupby(['season','week','team'],as_index=False).sum()
    L.to_parquet(CACHE/"personnel.parquet")
    return L


def kalman(played, teams):
    """State-space team strength. Returns final ratings, per-stadium HFA, and per-game preds."""
    TIDX={t:i for i,t in enumerate(teams)}; N=len(teams)
    K=KALMAN; x=np.zeros(N); P=np.eye(N)*K['p0']; cur=None; preds=[]
    for r in played.itertuples():
        if r.season!=cur:
            if cur is not None: x*=K['regress']; P+=np.eye(N)*K['off_var']
            cur=r.season
        h,a=TIDX[r.home_team],TIDX[r.away_team]
        P+=np.eye(N)*K['q']; pred=x[h]-x[a]; preds.append(pred)
        S=P[h,h]+P[a,a]-2*P[h,a]+K['R']
        Hv=np.zeros(N); Hv[h]=1; Hv[a]=-1
        Kg=(P@Hv)/S; x=x+Kg*(r.result-K['hfa']-pred); P=P-np.outer(Kg,Hv@P); x-=x.mean()
    if cur is not None and cur < SEASON: x=x*K['regress']
    played=played.assign(kal_pred=preds)
    hfa={}
    for t in teams:
        d=played[played.home_team==t]; res=(d.result-d.kal_pred).values
        hfa[t]=(res.sum()+K['hfa']*HFA_SHRINK)/(len(res)+HFA_SHRINK)
    return pd.Series(x,index=teams), pd.Series(hfa), played


def epa_ratings(tg, teams, upto=None):
    """Recency-weighted ridge: epa ~ offense + defense."""
    h = tg if upto is None else tg[tg.t_abs<upto]
    if len(h) < 200: return None
    TIDX={t:i for i,t in enumerate(teams)}; N=len(teams)
    w = EPA_DECAY**(h.t_abs.max()-h.t_abs.values)
    ti=h.team.map(TIDX).values; oi=h.opp.map(TIDX).values
    ok = ~pd.isna(ti) & ~pd.isna(oi)
    h=h[ok]; w=w[ok]; ti=ti[ok].astype(int); oi=oi[ok].astype(int)
    X=np.zeros((len(h),2*N)); X[np.arange(len(h)),ti]=1; X[np.arange(len(h)),N+oi]=1
    out={}
    for lbl,yc,nc in [('pass','off_pass_epa','off_pass_n'),('rush','off_rush_epa','off_rush_n')]:
        y=h[yc].values; m=~np.isnan(y)
        r=Ridge(alpha=EPA_RIDGE_ALPHA).fit(X[m],y[m],sample_weight=w[m]*np.sqrt(h[nc].values[m]))
        out['off_'+lbl]=pd.Series(r.coef_[:N],index=teams)
        out['def_'+lbl]=pd.Series(r.coef_[N:],index=teams)
    return out


def qb_timeline(tg):
    """Rating keyed by player id so it survives a change of team."""
    tg=tg.sort_values('t_abs'); st={}
    for r in tg.itertuples():
        if pd.isna(r.passer_player_id): continue
        s,n=st.get(r.passer_player_id,(0.0,0.0))
        st[r.passer_player_id]=(s*QB_DECAY+r.qb_epa*r.db, n*QB_DECAY+r.db)
    return {pid:((s+QB_PRIOR_MU*QB_PRIOR_DB)/(n+QB_PRIOR_DB), n) for pid,(s,n) in st.items()}


def travel_miles(a,b):
    if a not in STADIUM or b not in STADIUM: return 0.0
    (la1,lo1),(la2,lo2)=STADIUM[a],STADIUM[b]; p=np.pi/180
    x=0.5-np.cos((la2-la1)*p)/2+np.cos(la1*p)*np.cos(la2*p)*(1-np.cos((lo2-lo1)*p))/2
    return 7912*np.arcsin(np.sqrt(x))
