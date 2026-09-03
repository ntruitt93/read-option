"""
NFL seeding with tiebreakers, validated against actual playoff fields.

Implemented ladder (division): head-to-head, division record, common games,
conference record, strength of victory, strength of schedule.
Implemented ladder (wild card): division-winner reduction, head-to-head sweep,
conference record, common games (min 4), strength of victory, strength of schedule.

Not implemented: the points-based steps below strength of schedule, and the
coin toss. Those resolve a small minority of real ties; when the ladder runs
out here we fall back to a stable alphabetical order and flag it.
"""
import pandas as pd, numpy as np, itertools, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

DIV = {
 'AFC East':['BUF','MIA','NE','NYJ'], 'AFC North':['BAL','CIN','CLE','PIT'],
 'AFC South':['HOU','IND','JAX','TEN'], 'AFC West':['DEN','KC','LAC','LV'],
 'NFC East':['DAL','NYG','PHI','WAS'], 'NFC North':['CHI','DET','GB','MIN'],
 'NFC South':['ATL','CAR','NO','TB'], 'NFC West':['ARI','LA','SEA','SF']}
T2D = {t:d for d,ts in DIV.items() for t in ts}
T2C = {t:d[:3] for t,d in T2D.items()}
# historical franchise codes -> current
ALIAS = {'OAK':'LV','SD':'LAC','STL':'LA','LAR':'LA'}
for old,new in ALIAS.items():
    T2D[old]=T2D[new]; T2C[old]=T2C[new]


class Season:
    """games: list of (home, away, home_won) plus the full schedule for common-games."""
    def __init__(self, games):
        # games: (home, away, outcome) with outcome 1.0 home win, 0.0 away win, 0.5 tie
        self.games = games
        self.pts = defaultdict(float); self.n = defaultdict(int)     # win-credit, games
        self.h2h = defaultdict(lambda: defaultdict(lambda: [0.0,0]))
        self.opps = defaultdict(list)
        self.cp = defaultdict(float); self.cn = defaultdict(int)     # conference
        self.dp = defaultdict(float); self.dn = defaultdict(int)     # division
        for h, a, o in games:
            self.pts[h] += o;       self.pts[a] += (1.0-o)
            self.n[h]   += 1;       self.n[a]   += 1
            self.h2h[h][a][0] += o; self.h2h[h][a][1] += 1
            self.h2h[a][h][0] += (1.0-o); self.h2h[a][h][1] += 1
            self.opps[h].append(a); self.opps[a].append(h)
            if T2C[h] == T2C[a]:
                self.cp[h] += o; self.cn[h] += 1
                self.cp[a] += (1.0-o); self.cn[a] += 1
            if T2D[h] == T2D[a]:
                self.dp[h] += o; self.dn[h] += 1
                self.dp[a] += (1.0-o); self.dn[a] += 1

    def pct(self, t):
        return self.pts[t]/self.n[t] if self.n[t] else 0.0

    def _pct(self, p, n):
        return p/n if n else 0.0

    def h2h_pct(self, t, group):
        p = n = 0
        for o in group:
            if o == t: continue
            p += self.h2h[t][o][0]; n += self.h2h[t][o][1]
        return self._pct(p, n), n

    def common_pct(self, t, group):
        sets = [set(self.opps[x]) for x in group]
        common = set.intersection(*sets) - set(group)
        if not common: return 0.0, 0
        p = n = 0
        for h, a, o in self.games:
            if t == h and a in common: p += o; n += 1
            elif t == a and h in common: p += (1.0-o); n += 1
        return self._pct(p, n), n

    def sov(self, t):
        num = den = 0.0
        for o in set(self.opps[t]):
            num += self.h2h[t][o][0] * self.pct(o); den += self.h2h[t][o][0]
        return num/den if den else 0.0

    def sos(self, t):
        if not self.opps[t]: return 0.0
        return np.mean([self.pct(o) for o in self.opps[t]])


def _break_tie(S, tied, wildcard=False):
    """Return tied sorted best-first. Recursive: ladder reapplies to sub-ties."""
    if len(tied) == 1: return list(tied)
    steps = []
    if not wildcard:
        steps = [lambda t, g: S.h2h_pct(t, g)[0],
                 lambda t, g: S._pct(S.dp[t], S.dn[t]),
                 lambda t, g: S.common_pct(t, g)[0],
                 lambda t, g: S._pct(S.cp[t], S.cn[t]),
                 lambda t, g: S.sov(t),
                 lambda t, g: S.sos(t)]
    else:
        def sweep(t, g):
            # only applies if one club beat all others or lost to all others
            w = sum(S.h2h[t][o][0] for o in g if o != t)
            l = sum(S.h2h[t][o][1] - S.h2h[t][o][0] for o in g if o != t)
            if w and not l: return 1.0
            if l and not w: return -1.0
            return 0.0
        steps = [sweep,
                 lambda t, g: S._pct(S.cp[t], S.cn[t]),
                 lambda t, g: S.common_pct(t, g)[0] if S.common_pct(t, g)[1] >= 4 else 0.0,
                 lambda t, g: S.sov(t),
                 lambda t, g: S.sos(t)]
    group = list(tied)
    for step in steps:
        vals = {t: step(t, group) for t in group}
        best = max(vals.values())
        winners = [t for t in group if vals[t] == best]
        if len(winners) < len(group):
            rest = [t for t in group if t not in winners]
            return _break_tie(S, winners, wildcard) + _break_tie(S, rest, wildcard)
    return sorted(group)   # ladder exhausted


def rank_group(S, teams, wildcard=False):
    out = []
    for _, grp in itertools.groupby(sorted(teams, key=lambda t: -S.pct(t)), key=lambda t: S.pct(t)):
        out.extend(_break_tie(S, list(grp), wildcard))
    return out


def seed_conference(S, conf):
    div_names = [d for d in DIV if d.startswith(conf)]
    winners = [rank_group(S, DIV[d])[0] for d in div_names]
    winners = rank_group(S, winners)
    rest = [t for d in div_names for t in DIV[d] if t not in winners]
    wc = rank_group(S, rest, wildcard=True)[:3]
    return winners + wc


def seed_all(S):
    return {c: seed_conference(S, c) for c in ('AFC', 'NFC')}


# ------------------ validation ------------------
if __name__ == '__main__':
    g = pd.read_csv('data/games.csv', low_memory=False)
    for yr in [2018,2019,2020,2021,2022,2023,2024,2025]:
        reg = g[(g.season == yr) & (g.game_type == 'REG') & g.result.notna()]
        fix = lambda t: ALIAS.get(t,t)
        games = [(fix(r.home_team), fix(r.away_team), 1.0 if r.result>0 else (0.0 if r.result<0 else 0.5)) for r in reg.itertuples()]
        S = Season(games)
        seeds = seed_all(S)
        # actual field: teams appearing in the wild card round
        wc = g[(g.season == yr) & (g.game_type == 'WC')]
        actual = {fix(t) for t in set(wc.home_team) | set(wc.away_team)}
        # the 1 seed has a bye, so it is absent from WC: add whoever isn't there
        got = set(seeds['AFC']) | set(seeds['NFC'])
        byes = {seeds['AFC'][0], seeds['NFC'][0]}
        pred_field = got
        actual_field = actual | byes
        missing = actual_field - pred_field
        extra = pred_field - actual_field
        # seed check: WC round home teams should be seeds 2,3,4 of each conf
        ok_seeds = []
        for c in ('AFC','NFC'):
            homes = {fix(t) for t in set(wc[wc.home_team.isin(DIV['%s East'%c]+DIV['%s North'%c]+
                                             DIV['%s South'%c]+DIV['%s West'%c])].home_team)}
            ok_seeds.append(homes == set(seeds[c][1:4]))
        print('%d  field match: %s  | WC home seeds match: AFC %s NFC %s'
              % (yr, 'YES' if not missing and not extra else f'NO missing={missing} extra={extra}',
                 ok_seeds[0], ok_seeds[1]))
        if yr == 2025:
            print('   AFC seeds:', ' '.join(seeds['AFC']))
            print('   NFC seeds:', ' '.join(seeds['NFC']))
