import pandas as pd, numpy as np, json, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

SEA=2025
pbp = pd.read_parquet('data/pbp_2025.parquet',
      columns=['game_id','season','week','season_type','posteam','defteam','home_team','away_team',
               'play_type','epa','wp','home_wp','qtr','game_seconds_remaining','score_differential'])
pbp = pbp[(pbp.season_type=='REG') & pbp.posteam.notna()]
sc  = pbp[(pbp.epa.notna()) & pbp.play_type.isin(['pass','run'])].copy()

# garbage-time filter: only plays where the game was still in doubt
sc['live'] = sc.wp.between(0.05, 0.95)

WP = {gid: d.home_wp.dropna() for gid, d in pbp[pbp.home_wp.notna()].groupby('game_id')}
rows=[]
for gid, d in sc.groupby('game_id'):
    h, a = d.home_team.iloc[0], d.away_team.iloc[0]
    L = d[d.live]
    if len(L) < 40: L = d                      # blowouts: fall back to all plays
    ho = L[L.posteam==h].epa.mean(); ao = L[L.posteam==a].epa.mean()
    if np.isnan(ho) or np.isnan(ao): continue
    # control = share of the game the HOME team spent as favourite (home_wp, not posteam wp)
    w = WP.get(gid)
    ctrl = float((w>0.5).mean()) if w is not None and len(w) else 0.5
    meanwp = float(w.mean()) if w is not None and len(w) else 0.5
    rows.append(dict(id=gid, wk=int(d.week.iloc[0]), home=h, away=a,
                     epa_marg=round(float(ho-ao),4), ctrl=round(ctrl,3), meanwp=round(meanwp,3),
                     live_n=int(len(L)), all_n=int(len(d))))
D=pd.DataFrame(rows)

g = pd.read_csv('data/games.csv', low_memory=False)
g = g[(g.season==SEA)&(g.game_type=='REG')&g.result.notna()][['game_id','result','home_score','away_score']]
D = D.merge(g, left_on='id', right_on='game_id')
D['score_marg']=D.result

# how often does the scoreboard disagree with the play-by-play?
D['sb_win']=np.sign(D.score_marg); D['pbp_win']=np.sign(D.epa_marg)
mismatch = (D.sb_win!=D.pbp_win).mean()
print('games: %d | scoreboard winner lost the EPA battle in %.0f%% of them' % (len(D), 100*mismatch))
print('corr(score margin, adj EPA margin): %.3f' % np.corrcoef(D.score_marg,D.epa_marg)[0,1])
print('corr(score margin, control): %.3f'      % np.corrcoef(D.score_marg,D.ctrl)[0,1])

# biggest disagreements
D['gap']=(D.score_marg-D.score_marg.mean())/D.score_marg.std() - (D.epa_marg-D.epa_marg.mean())/D.epa_marg.std()
ex=D.reindex(D.gap.abs().sort_values(ascending=False).index).head(3)
print('\nbiggest scoreboard/play disagreements:')
for r in ex.itertuples():
    print('  %-18s home %s %+d on the board, %+.3f EPA/play, home control %.0f%%'
          % (r.id, r.home, r.score_marg, r.epa_marg, 100*r.ctrl))

# ---------- quadrant ratings (opponent-adjusted, full 2025) ----------
tg=pd.read_parquet('data/team_games.parquet'); tg=tg[tg.season==SEA]
teams=sorted(tg.team.unique()); TIDX={t:i for i,t in enumerate(teams)}; N=len(teams)
ti=tg.team.map(TIDX).values; oi=tg.opp.map(TIDX).values
X=np.zeros((len(tg),2*N)); X[np.arange(len(tg)),ti]=1; X[np.arange(len(tg)),N+oi]=1
R={}
for lbl,yc,nc in [('pass','off_pass_epa','off_pass_n'),('rush','off_rush_epa','off_rush_n')]:
    y=tg[yc].values; m=~np.isnan(y)
    r=Ridge(alpha=55.0).fit(X[m],y[m],sample_weight=np.sqrt(tg[nc].values[m]))
    R['off_'+lbl]=dict(zip(teams,r.coef_[:N])); R['def_'+lbl]=dict(zip(teams,r.coef_[N:]))
R['off_all']={t:0.58*R['off_pass'][t]+0.42*R['off_rush'][t] for t in teams}
R['def_all']={t:0.58*R['def_pass'][t]+0.42*R['def_rush'][t] for t in teams}
quad={t:{k:round(float(R[k][t]),4) for k in R} for t in teams}

json.dump(dict(games=D[['id','wk','home','away','epa_marg','ctrl','meanwp','score_marg']]
                 .to_dict('records'), quad=quad), open('stats.json','w'), separators=(',',':'))
print('\nstats.json:', round(len(open('stats.json').read())/1024,1),'KB')
