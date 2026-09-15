"""
Per-team season splits for the team pages: offense, defense, and a deliberately
modest special-teams block, each with a league rank.

Design notes worth keeping:

* **Ranks are the point.** "+0.031 EPA per play" means nothing to most people;
  "4th in the league" means something immediately. Every headline number carries
  its rank, and the rank is out of however many teams actually have data.

* **Special teams is reported in two pieces, not one.** A single special-teams
  EPA number is dominated by made field goals, which makes a good kicker look
  like a good unit. Kicking (FG/XP) and the return/coverage game are separated so
  they can't launder each other.

* **Defensive numbers are what the team ALLOWED**, so for every defensive metric
  lower is better. The app has to know that to rank them correctly; `LOWER_BETTER`
  below is the authority and the app should not re-derive it.
"""
import numpy as np
import pandas as pd

from .config import CACHE, ALIAS

# This module needs a much wider slice of the play-by-play than the dominance
# chart does, so it loads its own frame rather than widening stats.py's read.
# Parquet is columnar, so the extra pass only decodes these columns.
COLS = ['game_id', 'season', 'week', 'season_type', 'posteam', 'defteam',
        'play_type', 'epa', 'success', 'yards_gained', 'sack', 'interception',
        'fumble_lost', 'third_down_converted', 'third_down_failed',
        'yardline_100', 'fixed_drive', 'fixed_drive_result', 'special',
        'field_goal_result', 'kick_distance', 'extra_point_result']


def load(year):
    """Regular-season plays for one season, or None if not published yet."""
    f = CACHE / f"pbp_{year}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=COLS)
    d = d[d.season_type == 'REG']
    for c in ('posteam', 'defteam'):
        d[c] = d[c].replace(ALIAS)
    return d if len(d) else None


def build_for(year):
    """Splits for one season, loading the play-by-play itself."""
    d = load(year)
    return build(d) if d is not None else {}

# metrics where a smaller number is a better team, for ranking
LOWER_BETTER = {'def_epa', 'def_pass_epa', 'def_rush_epa', 'def_success',
                'def_explosive', 'def_third', 'def_rz_td', 'def_pts_per_drive',
                'off_giveaways', 'off_sack_rate'}

SCRIMMAGE = ['pass', 'run']


def _rate(num, den):
    return round(float(num) / float(den), 4) if den else None


def _side(sc, drives, team, side):
    """Offensive or defensive splits for one team.

    `side` is 'posteam' for offense (this team has the ball) or 'defteam' for
    defense (this team is defending), so the same code produces both and the
    defensive numbers are literally what the offense did against them.
    """
    p = sc[sc[side] == team]
    if not len(p):
        return None
    passes = p[p.play_type == 'pass']
    rushes = p[p.play_type == 'run']

    explosive = ((passes.yards_gained >= 20).sum() + (rushes.yards_gained >= 10).sum())

    third = p[p.third_down_converted.notna() | p.third_down_failed.notna()]
    third_att = float(third.third_down_converted.sum() + third.third_down_failed.sum())

    d = drives[drives[side] == team]
    rz = d[d.reached_rz]
    n_drives = len(d)

    return dict(
        plays=int(len(p)),
        epa=round(float(p.epa.mean()), 4),
        pass_epa=round(float(passes.epa.mean()), 4) if len(passes) else None,
        rush_epa=round(float(rushes.epa.mean()), 4) if len(rushes) else None,
        success=_rate(p.success.sum(), len(p)),
        explosive=_rate(explosive, len(p)),
        third=_rate(third.third_down_converted.sum(), third_att),
        third_att=int(third_att),
        rz_td=_rate((rz.result == 'Touchdown').sum(), len(rz)),
        rz_trips=int(len(rz)),
        giveaways=int(p.interception.fillna(0).sum() + p.fumble_lost.fillna(0).sum()),
        sack_rate=_rate(p.sack.fillna(0).sum(), len(passes) + p.sack.fillna(0).sum()),
        drives=int(n_drives),
        pts_per_drive=round(float(d.points.mean()), 3) if n_drives else None)


DRIVE_POINTS = {'Touchdown': 7.0, 'Field goal': 3.0, 'Opp touchdown': -7.0, 'Safety': -2.0}


def _drives(pbp):
    """One row per drive: who had it, who defended it, did it reach the red zone,
    and what it was worth. Points are the scoreboard value of the drive result,
    with an extra point assumed on a touchdown."""
    d = pbp[pbp.fixed_drive.notna() & pbp.posteam.notna()]
    if not len(d):
        return pd.DataFrame(columns=['posteam', 'defteam', 'result', 'points', 'reached_rz'])
    g = d.groupby(['game_id', 'fixed_drive'], sort=False).agg(
        posteam=('posteam', 'first'),
        defteam=('defteam', 'first'),
        result=('fixed_drive_result', 'first'),
        min_yl=('yardline_100', 'min')).reset_index(drop=True)
    g['points'] = g.result.map(DRIVE_POINTS).fillna(0.0)
    g['reached_rz'] = g.min_yl.le(20).fillna(False)
    return g


def _special(pbp, team):
    """Kicking and the return/coverage game, kept apart on purpose."""
    own = pbp[pbp.posteam == team]

    fg = own[own.field_goal_result.notna()]
    made = fg.field_goal_result == 'made'
    long_fg = fg[made].kick_distance.max() if made.any() else None

    def bucket(lo, hi):
        b = fg[(fg.kick_distance >= lo) & (fg.kick_distance < hi)]
        return dict(m=int((b.field_goal_result == 'made').sum()), a=int(len(b)))

    xp = own[own.extra_point_result.notna()]
    punts = own[own.play_type == 'punt']

    # return/coverage EPA, with kicks from scrimmage excluded so a good kicker
    # can't carry a bad coverage unit
    rc = pbp[(pbp.play_type.isin(['punt', 'kickoff']))]
    ours = rc[rc.posteam == team]

    return dict(
        fg_m=int(made.sum()), fg_a=int(len(fg)),
        fg_long=int(long_fg) if long_fg == long_fg and long_fg is not None else None,
        fg_short=bucket(0, 40), fg_mid=bucket(40, 50), fg_deep=bucket(50, 99),
        xp_m=int((xp.extra_point_result == 'good').sum()), xp_a=int(len(xp)),
        punts=int(len(punts)),
        punt_avg=round(float(punts.kick_distance.mean()), 1) if len(punts) and punts.kick_distance.notna().any() else None,
        rc_epa=round(float(ours.epa.mean()), 4) if len(ours) and ours.epa.notna().any() else None,
        rc_plays=int(len(ours)))


def build(pbp):
    """Splits plus league ranks for every team with plays in this frame."""
    sc = pbp[pbp.play_type.isin(SCRIMMAGE) & pbp.epa.notna()
             & pbp.posteam.notna() & pbp.defteam.notna()].copy()
    if not len(sc):
        return {}
    drives = _drives(pbp)
    teams = sorted(set(sc.posteam.dropna()) | set(sc.defteam.dropna()))

    out = {}
    for t in teams:
        off = _side(sc, drives, t, 'posteam')
        dfn = _side(sc, drives, t, 'defteam')
        if off is None or dfn is None:
            continue
        out[t] = dict(off=off, **{'def': dfn}, st=_special(pbp, t))

    # league ranks: 1 is best, and "best" means smallest for the LOWER_BETTER set
    metrics = [('off', k) for k in ('epa', 'pass_epa', 'rush_epa', 'success', 'explosive',
                                    'third', 'rz_td', 'pts_per_drive', 'giveaways', 'sack_rate')]
    metrics += [('def', k) for k in ('epa', 'pass_epa', 'rush_epa', 'success', 'explosive',
                                     'third', 'rz_td', 'pts_per_drive')]
    for side, key in metrics:
        name = f'{side}_{key}'
        vals = {t: out[t][side].get(key) for t in out if out[t][side].get(key) is not None}
        if not vals:
            continue
        asc = name in LOWER_BETTER
        order = sorted(vals, key=lambda t: vals[t], reverse=not asc)
        for i, t in enumerate(order, 1):
            out[t].setdefault('rank', {})[name] = i

    n = len(out)
    for t in out:
        out[t]['rank_of'] = n
    return out
