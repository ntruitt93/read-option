"""
Download nflverse data into cache/.

Fails loudly. A refresh that cannot get fresh data must not overwrite good JSON
with garbage, so every caller is expected to let these exceptions propagate.
"""
import sys, time, urllib.request, urllib.error
import pandas as pd
from .config import CACHE, NFLVERSE, GAMES_CSV, SEASON, HIST_START

TIMEOUT, RETRIES = 180, 3

def _get(url, dest, required=True):
    for attempt in range(1, RETRIES+1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                if r.status != 200: raise IOError(f"HTTP {r.status}")
                body = r.read()
            if len(body) < 1000 and required:
                raise IOError(f"suspiciously small response ({len(body)} bytes)")
            dest.write_bytes(body)
            print(f"    {dest.name}  {len(body)/1e6:.1f} MB")
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404 and not required:
                print(f"    {dest.name}  not published yet (404) — skipping")
                return False
            if attempt == RETRIES: raise
        except Exception:
            if attempt == RETRIES: raise
        time.sleep(2*attempt)
    return False


def schedules():
    """games.csv: schedule, results, betting lines, 1999-present."""
    dest = CACHE / "games.csv"
    _get(GAMES_CSV, dest)
    g = pd.read_csv(dest, low_memory=False)
    if SEASON not in set(g.season):
        raise RuntimeError(f"games.csv has no {SEASON} rows — upstream schema or timing change")
    return g


def play_by_play(years=None):
    """Historical seasons are required; the current season may not exist yet."""
    years = years or range(HIST_START, SEASON+1)
    got = []
    for y in years:
        dest = CACHE / f"pbp_{y}.parquet"
        if dest.exists() and y < SEASON:
            got.append(y); continue          # completed seasons never change
        if _get(f"{NFLVERSE}/pbp/play_by_play_{y}.parquet", dest, required=(y < SEASON)):
            got.append(y)
    if not got:
        raise RuntimeError("no play-by-play could be downloaded")
    return got


def supporting(years=None):
    """Snap counts, injuries, rosters. Current season 404s until games are played."""
    years = years or range(HIST_START, SEASON+1)
    out = {}
    for kind in ("snap_counts", "injuries"):
        out[kind] = []
        for y in years:
            dest = CACHE / f"{kind}_{y}.parquet"
            if dest.exists() and y < SEASON:
                out[kind].append(y); continue
            if _get(f"{NFLVERSE}/{kind}/{kind}_{y}.parquet", dest, required=(y < SEASON)):
                out[kind].append(y)
    _get(f"{NFLVERSE}/players/players.parquet", CACHE / "players.parquet")
    return out


def all_data():
    print("  schedules...");      g = schedules()
    print("  play-by-play...");   yrs = play_by_play()
    print("  snaps/injuries..."); sup = supporting()
    played = g[(g.season==SEASON) & g.result.notna()]
    print(f"  {SEASON}: {len(played)} of {len(g[g.season==SEASON])} games played")
    return dict(games=g, pbp_years=yrs, support=sup, played=len(played))
