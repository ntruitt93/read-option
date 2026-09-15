"""
Season stats for the Stats tab: per-game dominance rows and opponent-adjusted
quadrant ratings, for any set of seasons.

Rewritten 2026-09-15. The previous version was a one-off script: it hardcoded
SEA=2025, read `data/pbp_2025.parquet` and `data/team_games.parquet` from an
earlier layout that no longer exists, and wrote stats.json into the working
directory. It had not been runnable for some time. This version is an importable
module that reads from config.CACHE and is called by refresh.py.

Output is keyed by season so the app can offer a year picker:

    {schema, generated, current, seasons: {"2025": {...}, "2026": {...}}}

Each season carries `games` (per-game dominance rows), `quad` (opponent-adjusted
offense/defense ratings) and `teams` (per-team offense/defense/special-teams
splits with league ranks — see teamsplits.py), which is what the team pages read.

A season with no played games is skipped rather than emitted empty. Thin seasons
are emitted with the counts that produced them (`n_games`, `weeks`) so the app can
say how much data is behind a chart instead of presenting one week as settled.
"""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import teamsplits as TS
from .config import CACHE, STATIC, SEASON, EPA_RIDGE_ALPHA, ALIAS, SCHEMA_VERSION

warnings.filterwarnings('ignore')

PBP_COLS = ['game_id', 'season', 'week', 'season_type', 'posteam', 'defteam',
            'home_team', 'away_team', 'play_type', 'epa', 'wp', 'home_wp',
            'qtr', 'game_seconds_remaining', 'score_differential']

# a game needs this many in-doubt plays before we trust the garbage-time filter;
# below it we fall back to every play rather than judge a blowout on a handful
MIN_LIVE_PLAYS = 40


def _load_pbp(year):
    """Play-by-play for one season, or None if it hasn't been published yet."""
    f = CACHE / f"pbp_{year}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=PBP_COLS)
    d = d[(d.season_type == 'REG') & d.posteam.notna()]
    for c in ('posteam', 'defteam', 'home_team', 'away_team'):
        d[c] = d[c].replace(ALIAS)
    return d if len(d) else None


def dominance(pbp, results):
    """One row per game: opponent-agnostic EPA margin with garbage time removed.

    `ctrl` is the share of the game the home team spent as favourite, taken from
    home_wp rather than possession wp so it reads the same way for both sides.
    """
    sc = pbp[pbp.epa.notna() & pbp.play_type.isin(['pass', 'run'])].copy()
    if not len(sc):
        return []
    sc['live'] = sc.wp.between(0.05, 0.95)

    wp_by_game = {gid: d.home_wp.dropna()
                  for gid, d in pbp[pbp.home_wp.notna()].groupby('game_id')}

    rows = []
    for gid, d in sc.groupby('game_id'):
        h, a = d.home_team.iloc[0], d.away_team.iloc[0]
        live = d[d.live]
        if len(live) < MIN_LIVE_PLAYS:
            live = d
        ho = live[live.posteam == h].epa.mean()
        ao = live[live.posteam == a].epa.mean()
        if np.isnan(ho) or np.isnan(ao):
            continue
        w = wp_by_game.get(gid)
        has_wp = w is not None and len(w)
        rows.append(dict(
            id=gid, wk=int(d.week.iloc[0]), home=h, away=a,
            epa_marg=round(float(ho - ao), 4),
            ctrl=round(float((w > 0.5).mean()), 3) if has_wp else 0.5,
            meanwp=round(float(w.mean()), 3) if has_wp else 0.5,
            live_n=int(len(live)), all_n=int(len(d))))

    # attach the actual scoreboard margin; a game without a final score is dropped
    out = []
    for r in rows:
        res = results.get(r['id'])
        if res is None:
            continue
        r['score_marg'] = int(res)
        out.append(r)
    return sorted(out, key=lambda r: (r['wk'], r['id']))


def quadrant(tg, year):
    """Opponent-adjusted offense/defense EPA per play, one season, ridge-shrunk.

    With a full season this separates teams cleanly. With one week of games the
    ridge correctly collapses nearly everything toward zero — that is the honest
    answer at that sample size, not a bug, but the app should say so.
    """
    t = tg[tg.season == year]
    if not len(t):
        return {}
    teams = sorted(t.team.unique())
    tidx = {x: i for i, x in enumerate(teams)}
    n = len(teams)
    ti = t.team.map(tidx).values
    oi = t.opp.map(tidx).values
    ok = ~pd.isna(ti) & ~pd.isna(oi)
    t, ti, oi = t[ok], ti[ok].astype(int), oi[ok].astype(int)
    if not len(t):
        return {}

    X = np.zeros((len(t), 2 * n))
    X[np.arange(len(t)), ti] = 1
    X[np.arange(len(t)), n + oi] = 1

    R = {}
    for lbl, ycol, ncol in [('pass', 'off_pass_epa', 'off_pass_n'),
                            ('rush', 'off_rush_epa', 'off_rush_n')]:
        y = t[ycol].values
        m = ~np.isnan(y)
        if m.sum() < 2:
            return {}
        fit = Ridge(alpha=EPA_RIDGE_ALPHA).fit(
            X[m], y[m], sample_weight=np.sqrt(t[ncol].values[m]))
        R['off_' + lbl] = dict(zip(teams, fit.coef_[:n]))
        R['def_' + lbl] = dict(zip(teams, fit.coef_[n:]))

    # league pass/rush split, so "overall" isn't a straight average of unequal volumes
    R['off_all'] = {x: 0.58 * R['off_pass'][x] + 0.42 * R['off_rush'][x] for x in teams}
    R['def_all'] = {x: 0.58 * R['def_pass'][x] + 0.42 * R['def_rush'][x] for x in teams}
    return {x: {k: round(float(R[k][x]), 4) for k in R} for x in teams}


def build(seasons, games, tg):
    """Payload for every requested season that has played games."""
    out = {}
    for year in seasons:
        pbp = _load_pbp(year)
        if pbp is None:
            print(f"    {year}: no play-by-play yet — skipped")
            continue

        g = games[(games.season == year) & (games.game_type == 'REG') & games.result.notna()]
        results = dict(zip(g.game_id, g.result))
        dom = dominance(pbp, results)
        quad = quadrant(tg, year)
        if not dom:
            print(f"    {year}: no completed games — skipped")
            continue

        weeks = sorted({r['wk'] for r in dom})
        teams = TS.build_for(year)
        out[str(year)] = dict(games=dom, quad=quad, teams=teams,
                              n_games=len(dom), weeks=weeks)
        print(f"    {year}: {len(dom)} games, weeks {weeks[0]}-{weeks[-1]}, "
              f"{len(quad)} teams rated, {len(teams)} team splits")

    if not out:
        raise RuntimeError("stats: no season produced any data")
    return dict(schema=SCHEMA_VERSION,
                generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                current=max(int(k) for k in out),
                seasons=out)


def write(payload):
    path = STATIC / "stats.json"
    path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    return path, len(path.read_text())
