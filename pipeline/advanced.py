"""
Advanced stats for the Stats tab — the charts that aren't on every other site.

Six payloads, written to data/static/advanced.json:

  qb        CPOE against EPA per dropback for each team's main passer, with the
            model's own shrunk estimate alongside the raw one
  proe      pass rate over expected against offensive success — coaching identity
  leverage  EPA weighted by how much the game was in doubt, with the evidence
            that it does NOT persist year to year
  market    is the closing line biased in any slice? (measured 2018-2025)
  linefit   does beating the line in the first half of a season predict the second?
  kalman    the model's own team-strength state, week by week

Two of these are null results. That is deliberate: "we looked and there's nothing
there" is a finding, and it's a more honest thing to show than a chart implying an
edge that eight seasons of data say isn't there. The nulls are computed here, not
asserted in the app's copy, so they update if the data ever changes its mind.
"""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import (CACHE, STATIC, SEASON, ALIAS, SCHEMA_VERSION, QB_PRIOR_DB,
                     STATS_SEASONS)

warnings.filterwarnings('ignore')

# seasons used for the two market questions; these barely change, so a wide window
MARKET_FROM, MARKET_TO = 2018, SEASON - 1
# the Kalman trajectory chart starts here rather than at HIST_START — a nine-season
# line chart is unreadable and nobody is asking what a team's rating was in 2018
TRAJECTORY_FROM = SEASON - 1
MIN_TRAIL_DB = 5          # a week needs this many dropbacks to add a trail point


# ───────────────────────── quarterbacks ─────────────────────────

QB_COLS = ['season', 'week', 'season_type', 'posteam', 'passer_player_id',
           'passer_player_name', 'qb_dropback', 'epa', 'cpoe', 'complete_pass',
           'pass_attempt', 'passing_yards', 'pass_touchdown', 'interception',
           'sack', 'air_yards', 'wp']


def _qb_frame(year):
    f = CACHE / f"pbp_{year}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=QB_COLS)
    d = d[(d.season_type == 'REG') & (d.qb_dropback == 1)
          & d.passer_player_id.notna() & d.epa.notna()].copy()
    d['posteam'] = d.posteam.replace(ALIAS)
    return d if len(d) else None


def quarterbacks(year):
    """One passer per team: whoever took the most dropbacks.

    Deliberately one per team rather than a minimum-attempt cut. A backup who
    mopped up two series is not a data point anyone wants on this chart, and a
    per-team pick gives exactly 32 without an arbitrary threshold.

    Each passer gets a raw point and a shrunk one. The shrinkage is the model's
    own: blend toward the league mean with a prior worth QB_PRIOR_DB dropbacks.
    In September that pulls a passer most of the way back to average, which is
    the honest read and the reason the two dots are drawn joined.
    """
    d = _qb_frame(year)
    if d is None:
        return None

    lg_epa = float(d.epa.mean())
    lg_cpoe = float(d.cpoe.mean()) if d.cpoe.notna().any() else 0.0

    # the main passer for each team
    counts = d.groupby(['posteam', 'passer_player_id', 'passer_player_name']).size()
    counts = counts.reset_index(name='db').sort_values('db', ascending=False)
    main = counts.groupby('posteam').head(1)

    def shrink(val, n, prior):
        return (val * n + prior * QB_PRIOR_DB) / (n + QB_PRIOR_DB)

    out = []
    for r in main.itertuples():
        p = d[(d.passer_player_id == r.passer_player_id) & (d.posteam == r.posteam)]
        n = len(p)
        cp = p[p.pass_attempt == 1]
        epa = float(p.epa.mean())
        cpoe = float(p.cpoe.mean()) if p.cpoe.notna().any() else None

        # per-week running totals, for the trail
        trail = []
        for wk in sorted(p.week.unique()):
            upto = p[p.week <= wk]
            if len(upto) < MIN_TRAIL_DB:
                continue
            trail.append(dict(wk=int(wk), n=int(len(upto)),
                              epa=round(float(upto.epa.mean()), 4),
                              cpoe=round(float(upto.cpoe.mean()), 2)
                              if upto.cpoe.notna().any() else None))

        out.append(dict(
            id=r.passer_player_id, name=r.passer_player_name, team=r.posteam,
            db=int(n),
            att=int(len(cp)), cmp=int(cp.complete_pass.sum()),
            cmp_pct=round(float(cp.complete_pass.mean() * 100), 1) if len(cp) else None,
            yards=int(p.passing_yards.fillna(0).sum()),
            td=int(p.pass_touchdown.fillna(0).sum()),
            int=int(p.interception.fillna(0).sum()),
            sacks=int(p.sack.fillna(0).sum()),
            ay=round(float(cp.air_yards.mean()), 1) if len(cp) and cp.air_yards.notna().any() else None,
            epa=round(epa, 4),
            cpoe=round(cpoe, 2) if cpoe is not None else None,
            epa_shrunk=round(shrink(epa, n, lg_epa), 4),
            cpoe_shrunk=round(shrink(cpoe, n, lg_cpoe), 2) if cpoe is not None else None,
            trail=trail))

    return dict(passers=sorted(out, key=lambda x: x['team']),
                league=dict(epa=round(lg_epa, 4), cpoe=round(lg_cpoe, 2)),
                prior_db=QB_PRIOR_DB,
                weeks=sorted(int(w) for w in d.week.unique()))


# ───────────────────────── leverage ─────────────────────────

def leverage(year):
    """EPA weighted by how much the game was still in doubt.

    Weight is 4*wp*(1-wp): full credit at a coin flip, near zero at 95/5. This is
    the same instinct as the garbage-time filter on the dominance chart, applied
    continuously instead of as a cutoff.
    """
    d = _qb_frame(year)
    if d is None:
        return None
    d = d[d.wp.notna()].copy()
    d['lev'] = 4 * d.wp * (1 - d.wp)

    counts = d.groupby(['posteam', 'passer_player_id', 'passer_player_name']).size()
    main = counts.reset_index(name='db').sort_values('db', ascending=False).groupby('posteam').head(1)

    out = []
    for r in main.itertuples():
        p = d[(d.passer_player_id == r.passer_player_id) & (d.posteam == r.posteam)]
        if p.lev.sum() <= 0:
            continue
        raw = float(p.epa.mean())
        clutch = float(np.average(p.epa, weights=p.lev))
        out.append(dict(name=r.passer_player_name, team=r.posteam, db=int(len(p)),
                        raw=round(raw, 4), clutch=round(clutch, 4),
                        gap=round(clutch - raw, 4),
                        lev_share=round(float(p.lev.mean()), 3)))
    return sorted(out, key=lambda x: -x['gap'])


def leverage_persistence(years):
    """Does the clutch gap carry from one season to the next? (Spoiler: no.)

    Computed rather than asserted, so the caveat the app prints stays true if the
    answer ever changes.
    """
    per = {}
    for y in years:
        d = _qb_frame(y)
        if d is None:
            continue
        d = d[d.wp.notna()].copy()
        d['lev'] = 4 * d.wp * (1 - d.wp)
        g = d.groupby('passer_player_id').agg(n=('epa', 'size'), raw=('epa', 'mean'))
        g['clutch'] = d.groupby('passer_player_id').apply(
            lambda x: np.average(x.epa, weights=x.lev))
        g['gap'] = g.clutch - g.raw
        per[y] = g[g.n >= 200]

    pairs = []
    ys = sorted(per)
    for a, b in zip(ys[:-1], ys[1:]):
        j = per[a].join(per[b], lsuffix='_a', rsuffix='_b', how='inner')
        if len(j) < 10:
            continue
        pairs.append(dict(
            from_season=int(a), to_season=int(b), n=int(len(j)),
            gap_r=round(float(np.corrcoef(j.gap_a, j.gap_b)[0, 1]), 3),
            raw_r=round(float(np.corrcoef(j.raw_a, j.raw_b)[0, 1]), 3)))
    if not pairs:
        return None
    return dict(pairs=pairs,
                mean_gap_r=round(float(np.mean([p['gap_r'] for p in pairs])), 3),
                mean_raw_r=round(float(np.mean([p['raw_r'] for p in pairs])), 3))


# ───────────────────────── pass rate over expected ─────────────────────────

def proe(year):
    """Pass rate over expected against offensive success.

    `xpass` / `pass_oe` are computed upstream by nflverse, so this is a read
    rather than a model. PROE is the cleanest available measure of what a coach
    *chooses* once down, distance, score and clock are accounted for.
    """
    f = CACHE / f"pbp_{year}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=['season_type', 'posteam', 'pass_oe', 'success',
                                    'epa', 'play_type', 'qb_dropback'])
    d = d[(d.season_type == 'REG') & d.posteam.notna()].copy()
    d['posteam'] = d.posteam.replace(ALIAS)
    p = d[d.pass_oe.notna()]
    if not len(p):
        return None
    sc = d[d.play_type.isin(['pass', 'run']) & d.epa.notna()]

    out = []
    for t in sorted(p.posteam.unique()):
        tp, ts = p[p.posteam == t], sc[sc.posteam == t]
        if not len(ts):
            continue
        out.append(dict(team=t, proe=round(float(tp.pass_oe.mean()), 2),
                        plays=int(len(ts)),
                        success=round(float(ts.success.mean()), 4),
                        epa=round(float(ts.epa.mean()), 4),
                        dropbacks=int(ts.qb_dropback.fillna(0).sum())))
    return out


# ───────────────────────── the market ─────────────────────────

BINS = [
    ('Rest', 'restdiff', [-99, -3, -0.5, 0.5, 3, 99],
     ['home on short rest', 'home -1 to -3', 'equal rest', 'home +1 to +3', 'home +4 or more']),
    ('Spread', 'spread_line', [-99, -10, -3, 0, 3, 10, 99],
     ['home dog 10+', 'home dog 3-10', 'home dog 0-3', 'home fav 0-3', 'home fav 3-10', 'home fav 10+']),
    ('Week', 'week', [0, 4, 13, 17, 18],
     ['weeks 1-4', 'weeks 5-13', 'weeks 14-17', 'week 18']),
]


def market(games):
    """Is the closing line biased in any slice we can name?

    `resid` is how much the home team beat its closing line by. An efficient
    market means every bin averages zero. Standard errors are reported so the app
    can draw them — the point of the chart is that the bars all straddle zero.
    """
    g = games[(games.game_type == 'REG') & games.result.notna() & games.spread_line.notna()
              & games.season.between(MARKET_FROM, MARKET_TO)].copy()
    if not len(g):
        return None
    g['resid'] = g.result - g.spread_line
    g['restdiff'] = g.home_rest - g.away_rest

    groups = []
    for label, col, edges, names in BINS:
        b = pd.cut(g[col], edges, labels=names)
        rows = []
        for name in names:
            sub = g[b == name]
            if len(sub) < 20:
                continue
            m, sd, n = float(sub.resid.mean()), float(sub.resid.std()), len(sub)
            se = sd / np.sqrt(n)
            rows.append(dict(bin=name, n=int(n), mean=round(m, 3),
                             se=round(se, 3), t=round(m / se, 2)))
        if rows:
            groups.append(dict(label=label, bins=rows))

    # division games don't bin, they split
    div_rows = []
    for name, sel in (('division', g.div_game == 1), ('non-division', g.div_game == 0)):
        sub = g[sel]
        if len(sub) < 20:
            continue
        m, sd, n = float(sub.resid.mean()), float(sub.resid.std()), len(sub)
        se = sd / np.sqrt(n)
        div_rows.append(dict(bin=name, n=int(n), mean=round(m, 3),
                             se=round(se, 3), t=round(m / se, 2)))
    if div_rows:
        groups.append(dict(label='Division', bins=div_rows))

    biggest = max((b for grp in groups for b in grp['bins']), key=lambda b: abs(b['t']))
    return dict(groups=groups, n_games=int(len(g)),
                sd=round(float(g.resid.std()), 2),
                overall=round(float(g.resid.mean()), 3),
                seasons=[MARKET_FROM, MARKET_TO],
                biggest_t=biggest)


def linefit(games):
    """Does beating the closing line in the first half of a season predict the second?

    Each team-game contributes a residual from that team's own point of view. The
    chart is first-half average against second-half average, one dot per team-season.
    """
    g = games[(games.game_type == 'REG') & games.result.notna() & games.spread_line.notna()
              & games.season.between(MARKET_FROM, MARKET_TO)].copy()
    if not len(g):
        return None
    rows = []
    for r in g.itertuples():
        d = r.result - r.spread_line
        rows.append((r.season, r.week, r.home_team, d))
        rows.append((r.season, r.week, r.away_team, -d))
    t = pd.DataFrame(rows, columns=['season', 'week', 'team', 'resid'])
    t['team'] = t.team.replace(ALIAS)

    pts, by_season = [], []
    for s, d in t.groupby('season'):
        h1 = d[d.week <= 9].groupby('team').resid.mean()
        h2 = d[d.week > 9].groupby('team').resid.mean()
        j = pd.concat([h1, h2], axis=1, keys=['h1', 'h2']).dropna()
        if len(j) < 10:
            continue
        r = float(np.corrcoef(j.h1, j.h2)[0, 1])
        by_season.append(dict(season=int(s), r=round(r, 3), n=int(len(j))))
        for team, row in j.iterrows():
            pts.append(dict(season=int(s), team=team,
                            h1=round(float(row.h1), 2), h2=round(float(row.h2), 2)))

    seas = t.groupby(['season', 'team']).resid.mean().unstack(0)
    yoy = []
    cols = sorted(seas.columns)
    for a, b in zip(cols[:-1], cols[1:]):
        j = seas[[a, b]].dropna()
        if len(j) >= 10:
            yoy.append(float(np.corrcoef(j[a], j[b])[0, 1]))

    return dict(points=pts, by_season=by_season,
                mean_r=round(float(np.mean([b['r'] for b in by_season])), 3),
                yoy_r=round(float(np.mean(yoy)), 3) if yoy else None,
                seasons=[MARKET_FROM, MARKET_TO])


# ───────────────────────── model trajectory ─────────────────────────

def trajectory(snaps, teams):
    """The Kalman state, week by week — the model's own opinion as it changed.

    `snaps` comes from features.kalman(..., snap={}), which records the rating
    vector at the end of every week it processes.
    """
    keys = [k for k in sorted(snaps) if k[0] >= TRAJECTORY_FROM]
    if not keys:
        return None
    series = {t: [] for t in teams}
    labels = []
    for (season, week) in keys:
        labels.append(dict(season=int(season), week=int(week)))
        vec = snaps[(season, week)]
        for i, t in enumerate(teams):
            series[t].append(round(float(vec[i]), 3))
    return dict(points=labels, teams={t: v for t, v in series.items() if any(v)})


# ───────────────────────── assembly ─────────────────────────

def build(games, snaps, teams):
    payload = dict(
        schema=SCHEMA_VERSION,
        generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        current=SEASON)

    # both charts are season-pickable, so build every season the Stats tab offers
    qbs = {}
    for y in STATS_SEASONS:
        q = quarterbacks(y)
        if q:
            qbs[str(y)] = q
    if qbs:
        payload['qb'] = qbs
        print("    qb: " + ", ".join(f"{k} ({len(v['passers'])} passers)" for k, v in qbs.items()))

    levs = {}
    for y in STATS_SEASONS:
        l = leverage(y)
        if l:
            levs[str(y)] = l
    if levs:
        payload['leverage'] = levs
        pers = leverage_persistence(range(SEASON - 5, SEASON))
        if pers:
            payload['leverage_persistence'] = pers
            print("    leverage: " + ", ".join(f"{k} ({len(v)})" for k, v in levs.items())
                  + f" | gap carries over at r={pers['mean_gap_r']:+.2f} "
                    f"vs raw EPA r={pers['mean_raw_r']:+.2f}")

    pr = {}
    for y in STATS_SEASONS:
        p = proe(y)
        if p:
            pr[str(y)] = p
    if pr:
        payload['proe'] = pr
        print(f"    proe: {', '.join(f'{k} ({len(v)} teams)' for k, v in pr.items())}")

    mk = market(games)
    if mk:
        payload['market'] = mk
        print(f"    market: {mk['n_games']} games {mk['seasons'][0]}-{mk['seasons'][1]}, "
              f"largest bias t={mk['biggest_t']['t']:+.2f} ({mk['biggest_t']['bin']})")

    lf = linefit(games)
    if lf:
        payload['linefit'] = lf
        print(f"    linefit: split-half r={lf['mean_r']:+.2f}, year-over-year r={lf['yoy_r']:+.2f}")

    tr = trajectory(snaps, teams)
    if tr:
        payload['kalman'] = tr
        print(f"    kalman: {len(tr['points'])} weekly snapshots from {TRAJECTORY_FROM}")

    return payload


def write(payload):
    path = STATIC / "advanced.json"
    path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    return path, len(path.read_text())
