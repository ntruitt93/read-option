# The Read Option

An NFL prediction model for the 2026 season, plus the tools to argue with it.

The model rates teams from play-by-play data, projects every game, and keeps score
of how it does — against the closing line, against naive baselines, and against you.

## Honest performance

Walk-forward validated on 2018–2025, holdout year 2025, never used for fitting:

| | Model | Benchmark |
|---|---|---|
| Straight-up accuracy | **60.1%** | always-home: 53.9% |
| Margin error (MAE) | 10.28 pts | closing line: 9.73 |
| Brier score | .2230 | closing line: .2121 |

We beat naive baselines comfortably and lose to the betting market. An encompassing
regression says our prediction adds nothing conditional on the closing line
(t = 1.54, p = 0.12). The app displays this rather than hiding it.

## What's in the model

Ridge over 14 features. The ones that carry weight, standardised:

| Feature | Weight | Source |
|---|---|---|
| Kalman team strength | 4.04 | state-space filter over game margins |
| Per-stadium home field | 1.65 | shrunk residuals by venue |
| Offensive availability | 1.29 | injury reports × trailing snap share |
| Quarterback rating | 0.72 | EPA per dropback, confidence-weighted |

Tuned constants live in `pipeline/config.py` and nowhere else.

Weather is deliberately excluded. `temp` and `wind` in games.csv are *recorded*
conditions, not forecasts, so they aren't knowable before kickoff. Including them
would have leaked.

## Setup

```bash
git clone <your-repo>
cd read-option
pip install -r requirements.txt
python refresh.py --full        # first run: builds history, ~1-2 min
```

Then serve `app/` however you like. For GitHub Pages, Settings → Pages →
deploy from branch `main`, folder `/` or `/app`.

## Refreshing

```bash
python refresh.py               # normal run, ~20s once history is cached
python refresh.py --full        # rebuild walk-forward history (after schema changes)
python refresh.py --check       # fetch and validate, write nothing
```

`.github/workflows/refresh.yml` runs this on a schedule: Tuesday (results plus new
lines), Thursday and Saturday (injuries), Sunday morning (final pre-kickoff), Monday
(results). Adjust the cron lines to taste, and use **Actions → Refresh data → Run
workflow** to trigger it by hand.

**Enable this first:** Settings → Actions → General → Workflow permissions →
*Read and write permissions*. Without it the bot can't commit.

## Fail-safe behaviour

The refresh is built to fail loudly rather than publish garbage:

- Downloads retry three times, then raise.
- A validation gate rejects the run if fewer than 200 games project, or any margin
  falls outside ±40 points, or any win probability lands outside 1–99%.
- **Any failure leaves the previous data in place.** Stale but correct beats fresh
  but wrong.
- A failed scheduled run opens a GitHub issue automatically.

Current-season files (`pbp_2026`, `injuries_2026`, `snap_counts_2026`) return 404
until games are played. That's expected and handled — those inputs degrade to
documented defaults rather than silently becoming zero.

## Output

| File | Contents | Changes |
|---|---|---|
| `data/season.json` | all 272 games: predictions, watchability, lines | every refresh |
| `data/results.json` | final scores as they land | during the season |
| `data/meta.json` | generation timestamp, counts, sigma | every refresh |
| `data/static/*.json` | territory geography, trivia bank | rarely |

`schema` is stamped on every file. Bump `SCHEMA_VERSION` in `config.py` when the
shape changes so the app can migrate stored user data instead of breaking on it.

## Layout

```
refresh.py              single entry point
pipeline/
  config.py             all tuned constants and team metadata
  fetch.py              nflverse downloads, fail-loud
  features.py           EPA splits, personnel, Kalman, QB timeline
  predict.py            walk-forward history, fitting, projection, JSON
  seeding_src.py        NFL tiebreakers (validated 2020-2025)
  adjust.py             user adjustment layer, kickoff locking
  territory.py trivia.py stats.py qbchart.py    static generators
app/
  index.html            the front end
data/                   generated — committed so Pages can serve it
cache/                  downloads and intermediate frames — gitignored
```

## Status

**Working:** the full data pipeline. `refresh.py` regenerates predictions for all
272 games from scratch in about 20 seconds once history is cached.

**Not yet done:** `app/index.html` still has its data compiled in rather than
fetching `data/*.json`. Until that split happens, a data refresh updates the JSON
but not what the page displays. That's the next piece of work, and it's also what
makes user picks survive deploys — `localStorage` is keyed to the origin, so as long
as the domain is stable, picks persist across every data update automatically.

**Deliberately not built:** any backend. Everything is per-browser `localStorage`.
That means no cross-device sync, no shared leaderboard, and no tamper-proof pick
locking. For a real pool with standings, the cheap path is Supabase or Firebase.

## Data

All of it from [nflverse](https://github.com/nflverse). Play-by-play back to 1999,
schedules and betting lines, snap counts, injury reports, rosters.
