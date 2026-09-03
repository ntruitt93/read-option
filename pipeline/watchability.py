import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json
from sklearn.linear_model import Ridge
from live import build_slate, fit_and_predict, LIVE_FEATURES, KAL, load_games
import live

# ---- rebuild the 2026 slate with totals as well as margins ----
g = load_games()
played = g[g.result.notna()]
neutral = set(g[(g.season==2026) & (g.location=='Neutral')].game_id)

hist = pd.read_parquet('data/model_frame_v3.parquet')
hist = hist[hist.result.notna()]
TOT_FE = ['pace_sum','offepa_sum','is_dome','wind_out','cold','inj_sum','no_crowd']
sc = pd.read_parquet('data/eval_scores.parquet')
tot_hist = sc[sc.total_actual.notna()]
mt = Ridge(alpha=3.0).fit(tot_hist[TOT_FE], tot_hist.total_actual)
BASE_TOTAL = float(tot_hist.total_actual.mean())

rows=[]
for w in range(1,19):
    s = build_slate(w)
    s.loc[s.game_id.isin(neutral),'team_hfa']=0.0
    s.loc[s.game_id.isin(neutral),'away_travel']=0.0
    p,sig,_ = fit_and_predict(s)
    rows.append(p)
P = pd.concat(rows)

# team strength from the Kalman, for the "two good teams" component
TEAMS=sorted(set(g[g.season==2026].home_team)|set(g[g.season==2026].away_team))
TEAMS=[t for t in TEAMS if t in live.LOC]
ratings, hfa = live.run_kalman(played[played.home_team.isin(TEAMS)&played.away_team.isin(TEAMS)], TEAMS)

# projected total: we lack 2026 pace/EPA sums, so use the league base shifted by
# how much better than average the two offences were last year (via kal proxy)
P['proj_total'] = BASE_TOTAL
sched = pd.read_csv('data/games_live.csv', low_memory=False)
sched = sched[sched.season==2026][['game_id','div_game','gameday','gametime','weekday','roof']]
P = P.merge(sched, on='game_id', how='left', suffixes=('','_s'))

def nz(s):
    s=np.asarray(s,dtype=float)
    return (s-s.min())/((s.max()-s.min()) or 1)

absm  = P.pred_margin.abs().values
qual  = np.array([ratings.get(h,0)+ratings.get(a,0) for h,a in zip(P.home_team,P.away_team)])
divg  = P.div_game.fillna(0).values

# ---- components, each 0-1, higher = more watchable ----
closeness = np.clip(1 - absm/14.0, 0, 1)          # a pick'em is the ceiling
quality   = nz(qual)                               # both teams good
scoring   = nz(P.proj_total.values) if P.proj_total.nunique()>1 else np.full(len(P),0.5)
stakes    = divg

W = dict(closeness=0.40, quality=0.30, scoring=0.20, stakes=0.10)
score = (W['closeness']*closeness + W['quality']*quality +
         W['scoring']*scoring     + W['stakes']*stakes)
P['watch'] = np.round(100*nz(score),1)
P['c_close']=np.round(closeness,3); P['c_qual']=np.round(quality,3); P['c_stake']=divg

def tier(v):
    if v>=70: return 'must'
    if v>=45: return 'good'
    return 'redzone'
P['tier']=P.watch.map(tier)

out=[]
for r in P.itertuples():
    out.append(dict(id=r.game_id, wk=int(r.week), home=r.home_team, away=r.away_team,
        day=str(r.gameday), time=str(r.gametime), wd=r.weekday,
        watch=float(r.watch), tier=r.tier,
        close=float(r.c_close), qual=float(r.c_qual), div=int(r.c_stake),
        margin=round(float(r.pred_margin),1), wp=round(float(r.wp_home),4)))
json.dump(dict(weights=W, games=out), open('watch.json','w'), separators=(',',':'))

print('scored %d games | weights %s' % (len(out), W))
print('\nWeek 1 by watchability:')
for r in sorted([o for o in out if o['wk']==1], key=lambda x:-x['watch'])[:6]:
    print('  %5.1f  %-3s %-9s %-3s @ %-3s  margin %+.1f%s' % (r['watch'], r['tier'],'',r['away'],r['home'],r['margin'],'  (div)' if r['div'] else ''))
print('  ...')
for r in sorted([o for o in out if o['wk']==1], key=lambda x:-x['watch'])[-3:]:
    print('  %5.1f  %-3s %-9s %-3s @ %-3s  margin %+.1f' % (r['watch'], r['tier'],'',r['away'],r['home'],r['margin']))
print('\ntier counts (season):', pd.Series([o['tier'] for o in out]).value_counts().to_dict())
