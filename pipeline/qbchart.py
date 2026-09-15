"""
Quarterback passing chart: completion rate and EPA by field zone, against the
league baseline for that same zone, for any set of seasons.

Rewritten 2026-09-15 alongside stats.py, for the same reason — the previous
version was a one-off script pinned to 2025 and to a path layout that no longer
exists.

One behaviour change worth knowing about. The old script qualified passers at a
flat 150 charted attempts, which is a reasonable full-season bar and an
impossible one in September: nobody has 150 attempts in Week 1, so a 2026 chart
would have been empty until roughly midseason. The threshold now scales with the
weeks actually played and never drops below MIN_FLOOR. The bar that was used is
reported in the payload (`min_att`) so the app can show it rather than implying
the list is complete.
"""
import json
import warnings
from datetime import datetime, timezone

import pandas as pd

from .config import CACHE, STATIC, ALIAS, SCHEMA_VERSION

warnings.filterwarnings('ignore')

PBP_COLS = ['season', 'week', 'season_type', 'play_type', 'passer_player_id',
            'passer_player_name', 'posteam', 'pass_location', 'pass_length',
            'air_yards', 'complete_pass', 'incomplete_pass', 'interception',
            'pass_touchdown', 'epa', 'cpoe', 'yards_gained', 'sack']

ZONES = [(loc, length) for length in ('deep', 'short')
         for loc in ('left', 'middle', 'right')]

FULL_SEASON_ATT = 150      # the bar a qualified passer clears over a full year
FULL_SEASON_WKS = 17       # games a starter plays in a 18-week season
MIN_FLOOR = 20             # below this a zone chart is six cells of noise


def _qualify_threshold(weeks_played):
    """Scale the attempt bar to the season so far, with a floor."""
    if weeks_played >= FULL_SEASON_WKS:
        return FULL_SEASON_ATT
    return max(MIN_FLOOR, int(round(FULL_SEASON_ATT * weeks_played / FULL_SEASON_WKS)))


def _load(year):
    f = CACHE / f"pbp_{year}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=PBP_COLS)
    d = d[(d.season_type == 'REG') & (d.play_type == 'pass') & d.passer_player_id.notna()]
    d = d[d.pass_location.notna() & d.pass_length.notna()].copy()
    d['posteam'] = d.posteam.replace(ALIAS)
    return d if len(d) else None


def _zone_cell(z):
    n = len(z)
    if not n:
        return dict(n=0, cmp=None, epa=None, td=0, int=0, ay=None, cpoe=None)
    return dict(
        n=int(n),
        cmp=round(float(z.complete_pass.mean() * 100), 1),
        epa=round(float(z.epa.mean()), 3),
        td=int(z.pass_touchdown.sum()),
        int=int(z.interception.sum()),
        ay=round(float(z.air_yards.mean()), 1),
        cpoe=round(float(z.cpoe.mean()), 1) if z.cpoe.notna().any() else None)


def build_season(p, year):
    weeks = sorted(p.week.unique())
    min_att = _qualify_threshold(len(weeks))

    qual = (p.groupby(['passer_player_id', 'passer_player_name', 'posteam'])
             .size().reset_index(name='att'))
    qual = qual[qual.att >= min_att].sort_values('att', ascending=False)

    qbs = {}
    for r in qual.itertuples():
        sub = p[p.passer_player_id == r.passer_player_id]
        qbs[r.passer_player_id] = dict(
            name=r.passer_player_name, team=r.posteam, att=int(r.att),
            cmp=round(float(sub.complete_pass.mean() * 100), 1),
            epa=round(float(sub.epa.mean()), 3),
            td=int(sub.pass_touchdown.sum()),
            int=int(sub.interception.sum()),
            cpoe=round(float(sub.cpoe.mean()), 1) if sub.cpoe.notna().any() else None,
            zones={f'{length}_{loc}': _zone_cell(
                       p[(p.passer_player_id == r.passer_player_id) &
                         (p.pass_location == loc) & (p.pass_length == length)])
                   for loc, length in ZONES})

    base = {}
    for loc, length in ZONES:
        z = p[(p.pass_location == loc) & (p.pass_length == length)]
        base[f'{length}_{loc}'] = dict(
            cmp=round(float(z.complete_pass.mean() * 100), 1) if len(z) else None,
            epa=round(float(z.epa.mean()), 3) if len(z) else None,
            n=int(len(z)))

    return dict(qbs=qbs, base=base, min_att=min_att,
                weeks=[int(w) for w in weeks], n_passers=len(qbs))


def build(seasons):
    out = {}
    for year in seasons:
        p = _load(year)
        if p is None:
            print(f"    {year}: no charted pass data yet — skipped")
            continue
        s = build_season(p, year)
        if not s['qbs']:
            print(f"    {year}: no passer cleared {s['min_att']} attempts — skipped")
            continue
        out[str(year)] = s
        print(f"    {year}: {s['n_passers']} passers at {s['min_att']}+ attempts "
              f"({len(s['weeks'])} week(s) played)")

    if not out:
        raise RuntimeError("qbchart: no season produced any data")
    return dict(schema=SCHEMA_VERSION,
                generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                current=max(int(k) for k in out),
                seasons=out)


def write(payload):
    path = STATIC / "qb.json"
    path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    return path, len(path.read_text())
