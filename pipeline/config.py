"""Shared configuration. Every tuned constant lives here, nowhere else."""
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
CACHE  = ROOT / "cache"          # downloaded nflverse data + intermediate frames
DATA   = ROOT / "data"           # JSON the app fetches
STATIC = DATA / "static"         # rarely regenerated (territory, trivia)
for p in (CACHE, DATA, STATIC): p.mkdir(parents=True, exist_ok=True)

SEASON      = 2026               # season being predicted
HIST_START  = 2018               # first season used for fitting
BACKTEST    = (2018, 2025)

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_CSV = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# ---- tuned model constants (grid-searched on 2021-2024, holdout 2025) ----
KALMAN = dict(q=0.20, R=850, regress=0.50, hfa=1.6, p0=42.0, off_var=30.0)
EPA_RIDGE_ALPHA = 55.0
EPA_DECAY       = 0.982
QB_DECAY, QB_PRIOR_DB, QB_PRIOR_MU = 0.968, 200.0, -0.005
HFA_SHRINK      = 60.0
MODEL_ALPHA     = 3.0

# Features used for live prediction. Weather is deliberately excluded: temp and
# wind in games.csv are recorded after the fact and are not knowable pre-kickoff.
LIVE_FEATURES = ['kal_margin','team_hfa','off_pass_diff','off_rush_diff','def_pass_diff',
                 'def_rush_diff','qb_diff_w','rest_diff','is_div','mismatch','away_travel',
                 'flag_pass_mismatch','off_avail_diff','def_avail_diff']

WATCH_WEIGHTS = dict(closeness=0.40, quality=0.30, scoring=0.20, stakes=0.10)

DIVISIONS = {
 'AFC East':['BUF','MIA','NE','NYJ'], 'AFC North':['BAL','CIN','CLE','PIT'],
 'AFC South':['HOU','IND','JAX','TEN'], 'AFC West':['DEN','KC','LAC','LV'],
 'NFC East':['DAL','NYG','PHI','WAS'], 'NFC North':['CHI','DET','GB','MIN'],
 'NFC South':['ATL','CAR','NO','TB'], 'NFC West':['ARI','LA','SEA','SF']}
TEAM_DIV  = {t:d for d,ts in DIVISIONS.items() for t in ts}
TEAM_CONF = {t:d[:3] for t,d in TEAM_DIV.items()}
ALIAS = {'OAK':'LV','SD':'LAC','STL':'LA','LAR':'LA'}
for _o,_n in ALIAS.items():
    TEAM_DIV[_o]=TEAM_DIV[_n]; TEAM_CONF[_o]=TEAM_CONF[_n]

# stadium coordinates; NYG/NYJ and LA/LAC are offset so shared markets split cleanly
STADIUM = {
'ARI':(33.53,-112.26),'ATL':(33.76,-84.40),'BAL':(39.28,-76.62),'BUF':(42.77,-78.79),
'CAR':(35.23,-80.85),'CHI':(41.86,-87.62),'CIN':(39.10,-84.52),'CLE':(41.51,-81.70),
'DAL':(32.75,-97.09),'DEN':(39.74,-105.02),'DET':(42.34,-83.05),'GB':(44.50,-88.06),
'HOU':(29.68,-95.41),'IND':(39.76,-86.16),'JAX':(30.32,-81.64),'KC':(39.05,-94.48),
'LV':(36.09,-115.18),'MIA':(25.96,-80.24),'MIN':(44.97,-93.26),'NE':(42.09,-71.26),
'NO':(29.95,-90.08),'PHI':(39.90,-75.17),'PIT':(40.45,-80.02),'SEA':(47.60,-122.33),
'SF':(37.40,-121.97),'TB':(27.98,-82.50),'TEN':(36.17,-86.77),'WAS':(38.91,-76.86),
'NYG':(40.95,-74.30),'NYJ':(40.68,-73.85),'LA':(34.15,-118.45),'LAC':(33.75,-118.20)}

SCHEMA_VERSION = 1               # bump when JSON shape changes; app migrates on mismatch
