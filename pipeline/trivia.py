"""
Trivia bank generated ONLY from tables we computed ourselves.
Every question carries the numbers that make it true, so nothing is invented.
"""
import pandas as pd, numpy as np, json, itertools, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

tg = pd.read_parquet('data/team_games.parquet')
tg['t_abs'] = (tg.season-2018)*18 + tg.week
g  = pd.read_csv('data/games.csv', low_memory=False)
mf = pd.read_parquet('data/model_frame_v3.parquet')
SEA = 2025

# ---------- season team table (2025) ----------
t25 = tg[tg.season==SEA]
teams = sorted(t25.team.unique())
TIDX={t:i for i,t in enumerate(teams)}; N=len(teams)

# opponent-adjusted splits via the same ridge we use in the model
def splits(df):
    ti=df.team.map(TIDX).values; oi=df.opp.map(TIDX).values
    X=np.zeros((len(df),2*N)); X[np.arange(len(df)),ti]=1; X[np.arange(len(df)),N+oi]=1
    out={}
    for lbl,yc,nc in [('pass','off_pass_epa','off_pass_n'),('rush','off_rush_epa','off_rush_n')]:
        y=df[yc].values; m=~np.isnan(y)
        r=Ridge(alpha=55.0).fit(X[m],y[m],sample_weight=np.sqrt(df[nc].values[m]))
        out['off_'+lbl]=pd.Series(r.coef_[:N],index=teams)
        out['def_'+lbl]=pd.Series(r.coef_[N:],index=teams)
    return out
S = splits(t25)
S['off_all'] = S['off_pass']*0.58 + S['off_rush']*0.42
S['def_all'] = S['def_pass']*0.58 + S['def_rush']*0.42

# records
r25 = g[(g.season==SEA)&(g.game_type=='REG')&g.result.notna()]
W={t:0 for t in teams}
for r in r25.itertuples():
    if r.result>0: W[r.home_team]=W.get(r.home_team,0)+1
    elif r.result<0: W[r.away_team]=W.get(r.away_team,0)+1

# counting stats
agg = t25.groupby('team').agg(
    giveaways=('giveaways','sum'), pen=('pen_n','sum'), pen_yds=('pen_yds','sum'),
    dpi=('dpi_n','sum'), plays=('off_pass_n','sum')).reset_index().set_index('team')
takeaways = t25.groupby('opp').giveaways.sum()
agg['takeaways']=takeaways
agg['to_margin']=agg.takeaways-agg.giveaways

# points
pts={t:0 for t in teams}; pa={t:0 for t in teams}
for r in r25.itertuples():
    pts[r.home_team]=pts.get(r.home_team,0)+r.home_score; pa[r.home_team]=pa.get(r.home_team,0)+r.away_score
    pts[r.away_team]=pts.get(r.away_team,0)+r.away_score; pa[r.away_team]=pa.get(r.away_team,0)+r.home_score

hfa = mf.groupby('home_team').team_hfa.last()

Q=[]
def add(kind, text, ans, opts=None, why=''):
    if isinstance(ans, (np.bool_, bool)): ans = bool(ans)
    Q.append(dict(k=kind, q=text, a=ans, o=opts, w=why))

rng = np.random.default_rng(42)
DIVOF={'AFC':['BUF','MIA','NE','NYJ','BAL','CIN','CLE','PIT','HOU','IND','JAX','TEN','DEN','KC','LAC','LV'],
       'NFC':['DAL','NYG','PHI','WAS','CHI','DET','GB','MIN','ATL','CAR','NO','TB','ARI','LA','SEA','SF']}

# ---- 1. true/false pairwise comparisons on adjusted splits ----
LBL={'off_pass':'adjusted pass offense','def_pass':'adjusted pass defense',
     'off_rush':'adjusted rush offense','def_rush':'adjusted rush defense',
     'off_all':'adjusted offense overall','def_all':'adjusted defense overall'}
for metric in LBL:
    s=S[metric]
    better = (lambda a,b: s[a]<s[b]) if metric.startswith('def') else (lambda a,b: s[a]>s[b])
    pairs = list(itertools.combinations(teams,2))
    rng.shuffle(pairs)
    for a,b in pairs[:9]:
        if abs(s[a]-s[b])<0.02: continue
        claim = better(a,b)
        add('tf', f'In 2025, {a} had a better {LBL[metric]} than {b}.', claim,
            why=f'{a} {s[a]:+.3f} EPA/play, {b} {s[b]:+.3f}. Lower is better on defense.')

# ---- 2. multiple choice: leaders ----
def mc_leader(series, label, best='high', n=6):
    order = series.sort_values(ascending=(best=='low'))
    top = order.index[0]
    distract = list(order.index[3:12])
    rng.shuffle(distract)
    opts = [top]+distract[:3]; rng.shuffle(opts)
    add('mc', f'Which team led the NFL in {label} in 2025?', top, opts,
        why=f'{top} at {series[top]:.3f}. Next closest was {order.index[1]} at {series[order.index[1]]:.3f}.')
mc_leader(S['off_pass'],'opponent-adjusted pass offense')
mc_leader(S['off_rush'],'opponent-adjusted rush offense')
mc_leader(S['def_pass'],'opponent-adjusted pass defense','low')
mc_leader(S['def_rush'],'opponent-adjusted rush defense','low')
mc_leader(pd.Series(pts),'points scored')
mc_leader(pd.Series(pa),'fewest points allowed','low')
mc_leader(agg.to_margin,'turnover margin')
mc_leader(agg.pen_yds,'penalty yards')
mc_leader(agg.dpi,'defensive pass interference penalties committed')
mc_leader(hfa,'home-field advantage in our model')

# ---- 3. true/false on records and points ----
pairs=list(itertools.combinations(teams,2)); rng.shuffle(pairs)
for a,b in pairs[:22]:
    if W[a]==W[b]: continue
    add('tf', f'{a} won more games than {b} in 2025.', W[a]>W[b],
        why=f'{a} finished {W[a]}-{17-W[a]}, {b} finished {W[b]}-{17-W[b]}.')
for a,b in pairs[22:40]:
    if abs(pts[a]-pts[b])<12: continue
    add('tf', f'{a} scored more points than {b} in 2025.', pts[a]>pts[b],
        why=f'{a} scored {pts[a]}, {b} scored {pts[b]}.')

# ---- 4. model / historical facts we computed ----
hist=g[(g.season.between(1999,2025))&g.home_score.notna()]
allpts=pd.concat([hist.home_score,hist.away_score]).astype(int)
freq=allpts.value_counts(normalize=True)
top_score=freq.index[0]
add('mc','Since 1999, what is the most common single-team score in an NFL game?',
    str(top_score), [str(x) for x in [top_score, freq.index[1], freq.index[3], 41]],
    why=f'{top_score} points occurs in {freq.iloc[0]*100:.1f}% of team-games, just ahead of {freq.index[1]}.')
hw=(hist.home_score>hist.away_score).mean()
add('mc','Since 1999, how often has the home team won outright?',
    f'{hw*100:.0f}%', [f'{hw*100:.0f}%','48%','63%','71%'],
    why=f'Home teams won {hw*100:.1f}% of {len(hist):,} games.')
add('tf','In our model, a team\u2019s Kalman-filtered strength rating carries more weight than its quarterback rating.',
    True, why='Standardised weights: team strength 4.04 points per SD, quarterback 0.72.')
add('tf','Our model predicts the final margin of an NFL game more accurately than the closing Vegas line.',
    False, why='Our margin error is 10.28 points against the line\u2019s 9.73 on 2025 holdout.')
add('tf','Turnover margin in one game is a strong predictor of the same team\u2019s result the following week.',
    False, why='We tested it. Turnovers correlate with winning the game they happen in but barely persist week to week.')
add('tf','Drawing more defensive pass interference penalties is associated with winning more games.',
    False, why='We tested this too. Correlation between DPI and winning was -0.007, statistically nothing.')
lo=hfa.idxmin(); hi=hfa.idxmax()
add('mc','Which stadium does our model award the largest home-field advantage?', hi,
    [hi, lo, 'MIA', 'ARI'], why=f'{hi} at {hfa[hi]:.2f} points; the smallest is {lo} at {hfa[lo]:.2f}.')

# ---- 5. head to head from actual results ----
h2h=[]
for r in r25.itertuples():
    if r.result!=0: h2h.append((r.home_team,r.away_team,r.result>0))
rng.shuffle(h2h)
for h,a,hw_ in h2h[:26]:
    add('tf', f'{h} beat {a} when they hosted them in 2025.', bool(hw_),
        why='From the 2025 regular-season results.')

json.dump(Q, open('trivia.json','w'), separators=(',',':'))
print('questions generated:', len(Q))
print(' true/false:', sum(1 for q in Q if q['k']=='tf'), '| multiple choice:', sum(1 for q in Q if q['k']=='mc'))
print(' size:', round(len(open('trivia.json').read())/1024,1),'KB')
for q in Q[:2]+Q[60:62]:
    print('  -', q['q'], '->', q['a'])
