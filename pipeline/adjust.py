"""
User input layer + three-track scorekeeping.

Four input types, each with its own mechanics, each scored separately so we can
learn which of the user's intuitions are worth anything.

    moxie   eye test / vibes          +-4 pts   deadband 1.5   decays 3 wks
    news    late-breaking fact        +-7 pts   no deadband    single game
    stakes  motivation / seeding      +-3 pts   deadband 1.0   single game
    no_read low confidence flag       widens sigma, never shifts the mean

Picks lock at kickoff. Adjustments are immutable once locked.
"""
import pandas as pd, numpy as np, json, hashlib
from datetime import datetime, timezone
from scipy.stats import norm

SPEC = {
    'moxie':  dict(cap=4.0, deadband=1.5, decay_weeks=3, shifts_mean=True),
    'news':   dict(cap=7.0, deadband=0.0, decay_weeks=1, shifts_mean=True),
    'stakes': dict(cap=3.0, deadband=1.0, decay_weeks=1, shifts_mean=True),
    'no_read':dict(cap=1.0, deadband=0.0, decay_weeks=1, shifts_mean=False),
}
NO_READ_SIGMA_MULT = 1.35   # widen predictive sd when user has no read


class AdjustmentLog:
    """Append-only. Entries lock at kickoff and cannot be edited afterward."""
    def __init__(self, path='data/adjustments.jsonl'):
        self.path = path; self.rows = []
        try:
            with open(path) as f:
                self.rows = [json.loads(l) for l in f if l.strip()]
        except FileNotFoundError:
            pass

    def add(self, game_id, team, kind, value, kickoff_utc, note='', now=None):
        if kind not in SPEC: raise ValueError(f'unknown input type: {kind}')
        now = now or datetime.now(timezone.utc)
        if now >= kickoff_utc:
            raise PermissionError(f'{game_id} already kicked off — locked')
        s = SPEC[kind]
        v = float(np.clip(value, -s['cap'], s['cap']))
        row = dict(game_id=game_id, team=team, kind=kind, value=v, note=note,
                   entered_utc=now.isoformat(), kickoff_utc=kickoff_utc.isoformat())
        row['hash'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:12]
        self.rows.append(row)
        with open(self.path, 'a') as f: f.write(json.dumps(row) + '\n')
        return row

    def frame(self):
        return pd.DataFrame(self.rows) if self.rows else pd.DataFrame(
            columns=['game_id','team','kind','value','entered_utc','kickoff_utc','hash'])


def resolve(adj_frame, game_id, home_team, away_team):
    """Collapse this game's adjustments into a mean shift and a sigma multiplier."""
    shift, sigma_mult, by_kind = 0.0, 1.0, {}
    if len(adj_frame) == 0: return shift, sigma_mult, by_kind
    d = adj_frame[adj_frame.game_id == game_id]
    for kind, s in SPEC.items():
        sub = d[d.kind == kind]
        if len(sub) == 0: continue
        if not s['shifts_mean']:
            if sub.value.abs().max() > 0: sigma_mult *= NO_READ_SIGMA_MULT
            by_kind[kind] = 0.0
            continue
        h = sub[sub.team == home_team].value.sum()
        a = sub[sub.team == away_team].value.sum()
        net = np.clip(h, -s['cap'], s['cap']) - np.clip(a, -s['cap'], s['cap'])
        if abs(net) < s['deadband']: net = 0.0
        shift += net; by_kind[kind] = net
    return shift, sigma_mult, by_kind


def predict(base_margin, sigma, adj_frame, game_id, home_team, away_team):
    shift, smult, by_kind = resolve(adj_frame, game_id, home_team, away_team)
    adj_margin = base_margin + shift
    return dict(
        base_margin=base_margin, adj_margin=adj_margin, shift=shift,
        by_kind=by_kind, sigma=sigma*smult,
        wp_base=float(norm.cdf(base_margin/sigma)),
        wp_adj=float(norm.cdf(adj_margin/(sigma*smult))))


# ---------------- three-track scorekeeping ----------------
def scorecard(df):
    """df needs: result, base_margin, adj_margin, sigma, sigma_adj, spread_line."""
    y = (df.result > 0).astype(int).values
    out = {}
    tracks = {
        'model only':  norm.cdf(df.base_margin/df.sigma),
        'model+you':   norm.cdf(df.adj_margin/df.sigma_adj),
        'vegas':       norm.cdf(df.spread_line/df.sigma),
        'always home': np.full(len(df), (df.result > 0).mean()),
    }
    for k, p in tracks.items():
        p = np.clip(np.asarray(p), 1e-9, 1-1e-9)
        out[k] = dict(brier=float(np.mean((p-y)**2)),
                      logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                      su=float(np.mean((p > .5) == y)),
                      correct=int(np.sum((p > .5) == y)), n=len(y))
    out['your_edge'] = dict(
        brier_delta=out['model only']['brier'] - out['model+you']['brier'],
        games_delta=out['model+you']['correct'] - out['model only']['correct'])
    return out


def per_input_skill(df, log_frame):
    """Estimate each input type's contribution separately: corr(adjustment, true error)."""
    rows = []
    for kind in SPEC:
        if not SPEC[kind]['shifts_mean']: continue
        vals, errs = [], []
        for r in df.itertuples():
            _, _, bk = resolve(log_frame, r.game_id, r.home_team, r.away_team)
            v = bk.get(kind, 0.0)
            if v != 0.0:
                vals.append(v); errs.append(r.result - r.base_margin)
        n = len(vals)
        if n < 8:
            rows.append((kind, n, np.nan, np.nan, 'too few to judge')); continue
        rho = float(np.corrcoef(vals, errs)[0,1])
        gain = float(np.mean(np.abs(np.array(errs))) -
                     np.mean(np.abs(np.array(errs) - np.array(vals))))
        verdict = ('helping' if gain > 0.15 else 'hurting' if gain < -0.15 else 'no signal yet')
        rows.append((kind, n, rho, gain, verdict))
    return pd.DataFrame(rows, columns=['input','n_used','corr_with_error','mae_gain_pts','verdict'])
