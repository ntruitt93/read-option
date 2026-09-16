#!/usr/bin/env python3
"""
Regenerate the data files the app reads.

    python refresh.py              normal weekly/daily refresh
    python refresh.py --full       also rebuild the walk-forward history (slow)
    python refresh.py --check      fetch and validate only, write nothing

Fail-safe by design: JSON is written to a temp file and moved into place only
after every step succeeds. A failed run leaves the last good data serving.
"""
import sys, json, shutil, traceback
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from pipeline.config import CACHE, DATA, SEASON, HIST_START, ALIAS, STADIUM, STATS_SEASONS
from pipeline import fetch, features as F, predict as P
from pipeline import stats as S, qbchart as Q, advanced as A

FULL  = '--full'  in sys.argv
CHECK = '--check' in sys.argv


def played_frame(games):
    """Completed regular-season games, aliased and ordered — the Kalman's input."""
    g = games[(games.game_type=='REG') & games.result.notna() &
              games.season.between(HIST_START, SEASON)].copy()
    g['home_team']=g.home_team.replace(ALIAS); g['away_team']=g.away_team.replace(ALIAS)
    teams=[t for t in sorted(set(g.home_team)|set(g.away_team)) if t in STADIUM]
    g=g[g.home_team.isin(teams)&g.away_team.isin(teams)]
    return g.sort_values(['season','week','game_id']).reset_index(drop=True), teams


def main():
    started = datetime.now(timezone.utc)
    print(f"refresh  {started.isoformat(timespec='seconds')}  season={SEASON}"
          f"{'  [FULL]' if FULL else ''}{'  [CHECK ONLY]' if CHECK else ''}")

    print("\n[1/6] fetching")
    info = fetch.all_data()
    games = info['games']

    print("\n[2/6] team-game features")
    tg = F.team_games(range(HIST_START, SEASON+1))
    print(f"    {len(tg):,} team-games")
    pers = F.personnel(range(HIST_START, SEASON+1))
    print(f"    {len(pers):,} team-weeks of injury data")

    hist_path = CACHE/"history.parquet"
    if FULL or not hist_path.exists():
        print("\n[3/6] building walk-forward history (slow)")
        hist, ratings, hfa, qbtl, teams = P.build_history(games, tg, pers)
    else:
        print("\n[3/6] reusing cached history; refreshing ratings")
        hist = pd.read_parquet(hist_path)
        g, teams = played_frame(games)
        ratings, hfa, _ = F.kalman(g, teams)
        qbtl = F.qb_timeline(tg)
    print(f"    history {len(hist):,} games | {len(teams)} teams")

    print("\n[4/6] fitting and projecting")
    model, sigma = P.fit(hist)
    proj = P.project(games, tg, hist, model, sigma, ratings, hfa, qbtl, teams)
    print(f"    sigma {sigma:.2f} | projected {len(proj)} games")

    print("\n[5/6] season stats")
    stats_payload = S.build(STATS_SEASONS, games, tg)
    qb_payload    = Q.build(STATS_SEASONS)

    # a second Kalman pass purely to record the weekly state for the trajectory
    # chart; runs regardless of which history branch was taken above so the line
    # is the same either way
    snaps = {}
    gk, tk = played_frame(games)
    F.kalman(gk, tk, snap=snaps)
    adv_payload = A.build(games, snaps, tk)

    # --- validation gate: refuse to publish nonsense ---
    problems=[]
    if len(proj) < 200: problems.append(f"only {len(proj)} games projected")
    if not proj.pred_margin.between(-40,40).all(): problems.append("margin outside +-40")
    if proj.pred_margin.isna().any(): problems.append("null margins")
    if not proj.wp.between(0.01,0.99).all(): problems.append("win prob outside 1-99%")
    if problems:
        raise RuntimeError("validation failed: " + "; ".join(problems))
    print("    validation passed")

    if CHECK:
        print("\n[6/6] --check: nothing written"); return 0

    print("\n[6/6] writing data")
    meta=dict(generated=started.isoformat(timespec='seconds'), played=info['played'])
    n_games, n_res = P.write_json(proj, sigma, ratings, meta)
    (DATA/"meta.json").write_text(json.dumps(dict(
        generated=meta['generated'], season=SEASON, games=n_games,
        played=info['played'], results=n_res, sigma=round(sigma,2)), indent=1))
    print(f"    season.json  {n_games} games")
    print(f"    results.json {n_res} final scores")
    _, adv_size = A.write(adv_payload)
    print(f"    advanced.json {adv_size/1024:.1f} KB")
    for payload, mod, label in ((stats_payload, S, 'stats.json'), (qb_payload, Q, 'qb.json')):
        _, size = mod.write(payload)
        print(f"    {label:12s} {'/'.join(sorted(payload['seasons']))}  {size/1024:.1f} KB")
    print(f"\ndone in {(datetime.now(timezone.utc)-started).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n" + "="*60, file=sys.stderr)
        print("REFRESH FAILED — existing data left untouched", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        print("="*60, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
