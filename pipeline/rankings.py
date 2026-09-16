"""
The Read-Option Power Rankings and Quarterback Ranking System.

Both are weekly: they read the frozen end-of-week state rather than the live one,
so a Thursday night game cannot quietly reshuffle "last week's rankings" on Friday,
and week-over-week movement is a comparison of two fixed snapshots rather than a
moving target.

POWER RANKINGS order teams by the Kalman rating — which is not a résumé. It is the
model's answer to "who would we favour on a neutral field tomorrow", so a 0-1 team
can rank third and that is the ranking working, not failing. The interesting part
is the decomposition: every rating is split into what the team carried over from
last season (its rating regressed halfway to average at the season boundary) and
what this season has actually moved it. Early on the ranking is mostly carry-over,
and saying so out loud — with the number — is better than pretending otherwise.

THE QB SYSTEM ranks each team's main passer on rate statistics only, so a
quarterback is never penalised for leaving a blowout early. It then shrinks every
rate toward league average by sample size, which is the honest treatment of a
25-dropback season: not a punishment for low volume, a refusal to over-credit it.

The uncomfortable part, computed here rather than asserted in the app's copy:
after one week the reliability of that ordering is about 0.3 — the noise is larger
than the signal, and no two adjacent passers are statistically separable. The
tiers are therefore fixed-size bands, and the app ships a confidence figure that
rises as the season accumulates. It reaches roughly 0.55 by week 3 and 0.7 by
week 6.
"""
import numpy as np
import pandas as pd

from .config import CACHE, SEASON, ALIAS, KALMAN

# ── quarterback index ──────────────────────────────────────────────────

# Component weights. A judgment call, like the watchability weights, and the app
# says so. EPA per dropback carries the most because it is the only one that
# prices the whole outcome; success rate guards against one long touchdown
# carrying a bad afternoon; CPOE isolates accuracy from what the offense asked
# for; sack rate is the part of pressure a quarterback owns.
QB_WEIGHTS = dict(epa=0.45, success=0.25, cpoe=0.20, sack=0.10)

# Shrinkage prior, in dropbacks (~2 games). Lighter than the model's own 200-play
# prior: the model is predicting a game and should be conservative, whereas this
# is describing a season and a 200-play prior would flatten every passer into an
# indistinguishable heap through September.
QB_PRIOR = 80.0

# Fixed band sizes, so the pyramid always has a pyramid's shape. Statistical
# separation between adjacent passers does not exist this early — the confidence
# figure is what carries that caveat, not the band edges.
# Sizes must increase monotonically or it isn't a pyramid — a band narrower than
# the one above it reads as a bulge. These sum to 32, one passer per team.
TIERS = [
    (2,  'Carrying the offense'),
    (4,  'Winning with it'),
    (6,  'Doing the job'),
    (9,  'Getting by'),
    (11, 'Struggling so far'),
]

QB_COLS = ['season', 'week', 'season_type', 'posteam', 'passer_player_id',
           'passer_player_name', 'qb_dropback', 'epa', 'cpoe', 'success', 'sack',
           'pass_attempt', 'complete_pass', 'passing_yards', 'pass_touchdown',
           'interception', 'air_yards']


def _frame(year):
    f = CACHE / f"pbp_{year}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=QB_COLS)
    d = d[(d.season_type == 'REG') & (d.qb_dropback == 1)
          & d.passer_player_id.notna() & d.epa.notna()].copy()
    d['posteam'] = d.posteam.replace(ALIAS)
    return d if len(d) else None


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd == sd else s * 0.0


def _score_frame(rows):
    """Composite from a frame of per-passer rates, shrunk by sample size."""
    q = pd.DataFrame(rows)
    if len(q) < 2:
        return None
    raw = (QB_WEIGHTS['epa'] * _z(q.epa)
           + QB_WEIGHTS['success'] * _z(q.success)
           + QB_WEIGHTS['cpoe'] * _z(q.cpoe.fillna(q.cpoe.mean()))
           - QB_WEIGHTS['sack'] * _z(q.sack))
    q['raw_score'] = raw
    q['score'] = raw * (q.db / (q.db + QB_PRIOR))
    return q.sort_values('score', ascending=False).reset_index(drop=True)


def _rates(p, pid, team):
    """Every rate we rank on, plus the counting stats the tooltip shows."""
    cp = p[p.pass_attempt == 1]
    return dict(
        id=pid, team=team, db=int(len(p)),
        epa=float(p.epa.mean()),
        success=float(p.success.mean()),
        cpoe=float(p.cpoe.mean()) if p.cpoe.notna().any() else np.nan,
        sack=float(p.sack.fillna(0).mean()),
        att=int(len(cp)), cmp=int(cp.complete_pass.sum()),
        cmp_pct=float(cp.complete_pass.mean() * 100) if len(cp) else None,
        yards=int(p.passing_yards.fillna(0).sum()),
        td=int(p.pass_touchdown.fillna(0).sum()),
        ints=int(p.interception.fillna(0).sum()),
        ay=float(cp.air_yards.mean()) if len(cp) and cp.air_yards.notna().any() else None)


def _main_passers(d):
    c = d.groupby(['posteam', 'passer_player_id', 'passer_player_name']).size()
    return c.reset_index(name='db').sort_values('db', ascending=False).groupby('posteam').head(1)


def _through_week(d, wk):
    """Ranked passers using everything up to and including week `wk`."""
    upto = d[d.week <= wk]
    if not len(upto):
        return None
    main = _main_passers(upto)
    rows = []
    for r in main.itertuples():
        p = upto[(upto.passer_player_id == r.passer_player_id) & (upto.posteam == r.posteam)]
        row = _rates(p, r.passer_player_id, r.posteam)
        row['name'] = r.passer_player_name
        rows.append(row)
    return _score_frame(rows)


def reliability(d, q):
    """What share of the spread between passers is real rather than sampling noise?

    Observed variance of a rate is true variance plus noise variance. Noise for a
    mean over n plays is the play-level variance over n, which we can measure
    directly, so the split is arithmetic rather than a guess.
    """
    play_sd = float(d.epa.std())
    noise_var = float(((play_sd / np.sqrt(q.db)) ** 2).mean())
    obs_var = float(q.epa.var())
    sig_var = max(obs_var - noise_var, 0.0)
    rel = sig_var / obs_var if obs_var else 0.0
    # dropbacks needed for the ordering to be mostly signal, at this true spread
    def need(target):
        if sig_var <= 0 or target >= 1:
            return None
        return int(round((play_sd ** 2) * target / (sig_var * (1 - target))))
    return dict(
        reliability=round(rel, 3),
        signal_sd=round(float(np.sqrt(sig_var)), 3),
        noise_sd=round(float(np.sqrt(noise_var)), 3),
        play_sd=round(play_sd, 3),
        median_db=int(q.db.median()),
        db_for_half=need(0.5), db_for_seventy=need(0.7))


def quarterbacks(year):
    d = _frame(year)
    if d is None:
        return None
    weeks = sorted(int(w) for w in d.week.unique())
    cur = _through_week(d, weeks[-1])
    if cur is None:
        return None
    prev = _through_week(d, weeks[-2]) if len(weeks) > 1 else None
    prev_rank = {r.id: i + 1 for i, r in enumerate(prev.itertuples())} if prev is not None else {}

    # assign fixed-size bands
    tier_of, edges, i = {}, [], 0
    for size, label in TIERS:
        for j in range(i, min(i + size, len(cur))):
            tier_of[j] = label
        edges.append(dict(label=label, size=size))
        i += size

    out = []
    for i, r in enumerate(cur.itertuples()):
        pr = prev_rank.get(r.id)
        out.append(dict(
            id=r.id, name=r.name, team=r.team, rank=i + 1,
            tier=tier_of.get(i, TIERS[-1][1]),
            score=round(float(r.score), 4), raw_score=round(float(r.raw_score), 4),
            db=int(r.db), att=r.att, cmp=r.cmp,
            cmp_pct=round(r.cmp_pct, 1) if r.cmp_pct is not None else None,
            yards=r.yards, td=r.td, ints=r.ints,
            ay=round(r.ay, 1) if r.ay is not None else None,
            epa=round(float(r.epa), 4),
            success=round(float(r.success), 4),
            cpoe=round(float(r.cpoe), 2) if r.cpoe == r.cpoe else None,
            sack=round(float(r.sack), 4),
            shrunk_pct=round(100 * QB_PRIOR / (r.db + QB_PRIOR), 1),
            prev_rank=pr, move=(pr - (i + 1)) if pr else None))

    return dict(season=int(year), through_week=weeks[-1], weeks=weeks,
                passers=out, tiers=edges, weights=QB_WEIGHTS, prior_db=QB_PRIOR,
                has_movement=bool(prev_rank), **reliability(d, cur))


# ── power rankings ─────────────────────────────────────────────────────

def power(snaps, teams, season=SEASON):
    """Weekly team rankings from the Kalman snapshots, with the carry-over split.

    `carried` is the rating the team started the season with — last season's final
    rating after the boundary regression — so `earned` is genuinely what this
    season has done, not the regression masquerading as movement.
    """
    keys = sorted(snaps)
    in_season = [k for k in keys if k[0] == season]
    if not in_season:
        return None
    last_prior = [k for k in keys if k[0] < season]
    reg = KALMAN['regress']
    carried = ({t: snaps[last_prior[-1]][i] * reg for i, t in enumerate(teams)}
               if last_prior else {t: 0.0 for t in teams})

    def ranked(vec):
        order = sorted(range(len(teams)), key=lambda i: -vec[i])
        return {teams[i]: r + 1 for r, i in enumerate(order)}

    cur_key = in_season[-1]
    cur = snaps[cur_key]
    prev_key = in_season[-2] if len(in_season) > 1 else None
    if prev_key is not None:
        prev_rank, basis = ranked(snaps[prev_key]), f"week {prev_key[1]}"
    else:
        prev_rank = ranked([carried[t] for t in teams])
        basis = "the preseason carry-over"

    cur_rank = ranked(cur)
    rows = []
    for i, t in enumerate(teams):
        rows.append(dict(
            team=t, rank=cur_rank[t], rating=round(float(cur[i]), 3),
            carried=round(float(carried[t]), 3),
            earned=round(float(cur[i] - carried[t]), 3),
            prev_rank=prev_rank[t], move=prev_rank[t] - cur_rank[t]))
    rows.sort(key=lambda r: r['rank'])

    c = np.array([r['carried'] for r in rows])
    e = np.array([r['earned'] for r in rows])
    denom = c.std() + e.std()
    return dict(season=int(season), through_week=int(cur_key[1]),
                weeks=[int(k[1]) for k in in_season],
                move_basis=basis, has_movement=prev_key is not None,
                carried_share=round(float(c.std() / denom), 3) if denom else None,
                regress=reg, teams=rows)


def build(snaps, teams):
    out = {}
    p = power(snaps, teams)
    if p:
        out['power'] = p
        print(f"    power: {len(p['teams'])} teams through week {p['through_week']}, "
              f"{p['carried_share']*100:.0f}% carried over, movement vs {p['move_basis']}")
    q = quarterbacks(SEASON)
    if q:
        out['qb'] = q
        print(f"    qb index: {len(q['passers'])} passers through week {q['through_week']}, "
              f"reliability {q['reliability']:.2f} (needs ~{q['db_for_half']} dropbacks for 0.50)")
    return out or None
