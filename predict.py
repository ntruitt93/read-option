"""
Fit the model on history and project the current season.

The historical frame is built walk-forward (each game sees only prior data) and
cached, because it only changes when new results land.
"""
import numpy as np, pandas as pd, json, warnings
from scipy.stats import norm
from sklearn.linear_model import Ridge
from .config import (CACHE, DATA, SEASON, HIST_START, LIVE_FEATURES, MODEL_ALPHA,
                     STADIUM, TEAM_DIV, ALIAS, SCHEMA_VERSION, WATCH_WEIGHTS)
from . import features as F
warnings.filterwarnings('ignore')


def _teams(games):
    ts = sorted(set(games[games.season==SEASON].home_team)|set(games[games.season==SEASON].away_team))
    return [t for t in ts if t in STADIUM]


def build_history(games, tg, pers):
    """Walk-forward feature frame over completed games. Cached."""
    g = games[(games.game_type=='REG') & games.result.notna() &
              games.season.between(HIST_START, SEASON)].copy()
    g['home_team']=g.home_team.replace(ALIAS); g['away_team']=g.away_team.replace(ALIAS)
    g = g.sort_values(['season','week','game_id']).reset_index(drop=True)
    g['t_abs']=(g.season-HIST_START)*18+g.week
    teams = sorted(set(g.home_team)|set(g.away_team))
    teams = [t for t in teams if t in STADIUM]
    g = g[g.home_team.isin(teams)&g.away_team.isin(teams)].reset_index(drop=True)

    ratings, hfa, gp = F.kalman(g, teams)
    g['kal_margin']=gp.kal_pred.values
    g['team_hfa']=g.home_team.map(hfa)

    qbtl = F.qb_timeline(tg)
    tg_s = tg.sort_values('t_abs')
    qb_at={}
    for pid, sub in tg_s.groupby('passer_player_id'):
        qb_at[pid]=(sub.t_abs.values, sub.qb_epa.values, sub.db.values)

    rows=[]
    for t in sorted(g.t_abs.unique()):
        R = F.epa_ratings(tg, teams, upto=t)
        cur = g[g.t_abs==t]
        for r in cur.itertuples():
            h,a=r.home_team,r.away_team
            d={'game_id':r.game_id}
            if R is None:
                for k in ['off_pass_diff','off_rush_diff','def_pass_diff','def_rush_diff',
                          'mismatch','flag_pass_mismatch']: d[k]=0.0
            else:
                d['off_pass_diff']=R['off_pass'][h]-R['off_pass'][a]
                d['off_rush_diff']=R['off_rush'][h]-R['off_rush'][a]
                d['def_pass_diff']=R['def_pass'][a]-R['def_pass'][h]
                d['def_rush_diff']=R['def_rush'][a]-R['def_rush'][h]
                mm=(R['off_pass'][h]-R['def_pass'][a])-(R['off_pass'][a]-R['def_pass'][h])
                d['mismatch']=mm; d['flag_pass_mismatch']=mm
            rows.append(d)
    g = g.merge(pd.DataFrame(rows), on='game_id', how='left')

    qmap = tg.set_index(['game_id','team']).qb_epa.to_dict()
    dmap = tg.set_index(['game_id','team']).db.to_dict()
    def qb_pre(team, gid, t):
        pid = tg[(tg.game_id==gid)&(tg.team==team)].passer_player_id
        pid = pid.iloc[0] if len(pid) else None
        if pid is None or pid not in qb_at: return -0.005, 0.0
        ts,ep,db = qb_at[pid]; i=np.searchsorted(ts,t)
        if i==0: return -0.005, 0.0
        w=0.968**(t-ts[:i]); n=(db[:i]*w).sum()
        return float((ep[:i]*db[:i]*w).sum()/n) if n else -0.005, float(n)
    hq=[qb_pre(r.home_team,r.game_id,r.t_abs) for r in g.itertuples()]
    aq=[qb_pre(r.away_team,r.game_id,r.t_abs) for r in g.itertuples()]
    conf=np.minimum([x[1] for x in hq],[x[1] for x in aq]).clip(0,400)/400
    g['qb_diff_w']=(np.array([x[0] for x in hq])-np.array([x[0] for x in aq]))*conf

    if len(pers):
        for side in ('home','away'):
            m=pers.rename(columns={'team':f'{side}_team','off_lost':f'{side}_off_lost',
                                   'def_lost':f'{side}_def_lost'})
            g=g.merge(m,on=['season','week',f'{side}_team'],how='left')
        for c in ['home_off_lost','home_def_lost','away_off_lost','away_def_lost']:
            g[c]=g[c].fillna(0)
        g['off_avail_diff']=g.away_off_lost-g.home_off_lost
        g['def_avail_diff']=g.away_def_lost-g.home_def_lost
    else:
        g['off_avail_diff']=0.0; g['def_avail_diff']=0.0

    g['rest_diff']=(g.home_rest-g.away_rest).fillna(0)
    g['is_div']=g.div_game.fillna(0)
    g['away_travel']=[F.travel_miles(a,h)/1000 for a,h in zip(g.away_team,g.home_team)]
    neutral = g.location.eq('Neutral') if 'location' in g else pd.Series(False,index=g.index)
    g.loc[neutral,'team_hfa']=0.0; g.loc[neutral,'away_travel']=0.0
    for c in LIVE_FEATURES: g[c]=g[c].fillna(0)
    g.to_parquet(CACHE/"history.parquet")
    return g, ratings, hfa, qbtl, teams


def fit(hist):
    m = Ridge(alpha=MODEL_ALPHA).fit(hist[LIVE_FEATURES], hist.result)
    sigma = float((hist.result - m.predict(hist[LIVE_FEATURES])).std())
    return m, sigma


def project(games, tg, hist, model, sigma, ratings, hfa, qbtl, teams):
    """Project every remaining game of SEASON."""
    up = games[(games.season==SEASON)&(games.game_type=='REG')].copy()
    for c in ('weekday','stadium','location'):
        if c not in up: up[c]=''
    up['t_abs']=(up.season-HIST_START)*18+up.week
    R = F.epa_ratings(tg, teams)
    last_starter = (tg.sort_values('t_abs').dropna(subset=['passer_player_id'])
                      .groupby('team').tail(1).set_index('team'))
    rows=[]
    for r in up.itertuples():
        h,a=r.home_team,r.away_team
        if h not in teams or a not in teams: continue
        neutral = getattr(r,'location','')=='Neutral'
        def qb(team):
            if team not in last_starter.index: return -0.005,0.0,'?'
            pid=last_starter.loc[team,'passer_player_id']
            nm=last_starter.loc[team,'passer_player_name']
            v=qbtl.get(pid,(-0.005,0.0))
            return v[0],v[1],nm
        qh,eh,nh=qb(h); qa,ea,na=qb(a)
        conf=min(eh,ea)/400 if min(eh,ea)<400 else 1.0
        mm=(R['off_pass'][h]-R['def_pass'][a])-(R['off_pass'][a]-R['def_pass'][h])
        d=dict(game_id=r.game_id, week=int(r.week), home=h, away=a,
               day=str(r.gameday), time=str(r.gametime), neutral=bool(neutral),
               wd=str(getattr(r,'weekday','')), venue=str(getattr(r,'stadium','')),
               kal_margin=float(ratings.get(h,0)-ratings.get(a,0)),
               team_hfa=0.0 if neutral else float(hfa.get(h,1.6)),
               off_pass_diff=float(R['off_pass'][h]-R['off_pass'][a]),
               off_rush_diff=float(R['off_rush'][h]-R['off_rush'][a]),
               def_pass_diff=float(R['def_pass'][a]-R['def_pass'][h]),
               def_rush_diff=float(R['def_rush'][a]-R['def_rush'][h]),
               qb_diff_w=float((qh-qa)*conf),
               rest_diff=float(r.home_rest-r.away_rest) if pd.notna(r.home_rest) else 0.0,
               is_div=int(r.div_game) if pd.notna(r.div_game) else 0,
               mismatch=float(mm), flag_pass_mismatch=float(mm), qb_home=nh, qb_away=na,
               away_travel=0.0 if neutral else F.travel_miles(a,h)/1000,
               off_avail_diff=0.0, def_avail_diff=0.0,
               spread=float(r.spread_line) if pd.notna(r.spread_line) else None,
               result=float(r.result) if pd.notna(r.result) else None,
               home_score=int(r.home_score) if pd.notna(r.home_score) else None,
               away_score=int(r.away_score) if pd.notna(r.away_score) else None)
        rows.append(d)
    P=pd.DataFrame(rows)
    P['pred_margin']=model.predict(P[LIVE_FEATURES])
    P['wp']=norm.cdf(P.pred_margin/sigma)
    return P


def watchability(P):
    a=P.pred_margin.abs().values
    close=np.clip(1-a/14.0,0,1)
    q=P.kal_margin.abs().values*0 + (P.team_hfa.values*0)   # placeholder, replaced below
    return close


def _clean(v):
    """NaN/NaT are not valid JSON. Convert to None so json.dump can stay strict."""
    if v is None: return None
    if isinstance(v,(float,np.floating)) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return float(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    return v


def write_json(P, sigma, ratings, meta):
    def nz(s):
        s=np.asarray(s,float); rng=(s.max()-s.min()) or 1
        return (s-s.min())/rng
    close = np.clip(1-P.pred_margin.abs().values/14.0,0,1)
    qual  = nz([ratings.get(h,0)+ratings.get(a,0) for h,a in zip(P.home,P.away)])
    stake = P.is_div.values
    W=WATCH_WEIGHTS
    score = W['closeness']*close + W['quality']*qual + W['scoring']*0.5 + W['stakes']*stake
    P=P.assign(watch=np.round(100*nz(score),1))
    P['tier']=P.watch.map(lambda v:'must' if v>=70 else ('good' if v>=45 else 'redzone'))

    games=[{k:_clean(v) for k,v in dict(id=r.game_id, wk=int(r.week), home=r.home, away=r.away, day=r.day, time=r.time,
                neutral=bool(r.neutral), margin=round(float(r.pred_margin),1),
                wp=round(float(r.wp),4), spread=r.spread, watch=float(r.watch), tier=r.tier,
                div=int(r.is_div), kal=round(float(r.kal_margin),2), hfa=round(float(r.team_hfa),2),
                qbd=round(float(r.qb_diff_w),3), travel=round(float(r.away_travel)*1000),
                rest=round(float(r.rest_diff),1), wd=r.wd, venue=r.venue,
                qbH=r.qb_home, qbA=r.qb_away,
                result=r.result, hs=r.home_score, as_=r.away_score).items()}
           for r in P.itertuples()]
    from .config import DIVISIONS
    payload=dict(schema=SCHEMA_VERSION, season=SEASON, sigma=round(sigma,2),
                 generated=meta['generated'], played=meta['played'],
                 weights=W, divisions=DIVISIONS, games=games)
    (DATA/"season.json").write_text(json.dumps(payload,separators=(',',':'),allow_nan=False))

    results={g['id']:dict(hs=g['hs'],as_=g['as_'],result=g['result'])
             for g in games if g['result'] is not None}
    (DATA/"results.json").write_text(json.dumps(
        dict(schema=SCHEMA_VERSION, generated=meta['generated'], results=results),
        separators=(',',':'), allow_nan=False))
    return len(games), len(results)
