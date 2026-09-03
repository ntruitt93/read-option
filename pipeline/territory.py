import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

STAD = {'ARI':(33.53,-112.26),'ATL':(33.76,-84.40),'BAL':(39.28,-76.62),'BUF':(42.77,-78.79),
'CAR':(35.23,-80.85),'CHI':(41.86,-87.62),'CIN':(39.10,-84.52),'CLE':(41.51,-81.70),
'DAL':(32.75,-97.09),'DEN':(39.74,-105.02),'DET':(42.34,-83.05),'GB':(44.50,-88.06),
'HOU':(29.68,-95.41),'IND':(39.76,-86.16),'JAX':(30.32,-81.64),'KC':(39.05,-94.48),
'LV':(36.09,-115.18),'MIA':(25.96,-80.24),'MIN':(44.97,-93.26),'NE':(42.09,-71.26),
'NO':(29.95,-90.08),'PHI':(39.90,-75.17),'PIT':(40.45,-80.02),'SEA':(47.60,-122.33),
'SF':(37.40,-121.97),'TB':(27.98,-82.50),'TEN':(36.17,-86.77),'WAS':(38.91,-76.86),
# co-located pairs get offset virtual centers so the shared market splits cleanly
'NYG':(40.95,-74.30),'NYJ':(40.68,-73.85),
'LA':(34.15,-118.45),'LAC':(33.75,-118.20)}

def decode(topo, key):
    """Yield (id, [rings]) with rings as lists of (x,y) in the file's own coordinate space."""
    tr = topo.get('transform')
    sx, sy = (tr['scale'] if tr else (1,1))
    tx, ty = (tr['translate'] if tr else (0,0))
    arcs = []
    for arc in topo['arcs']:
        x = y = 0; pts = []
        for dx, dy in arc:
            x += dx; y += dy
            pts.append((x*sx+tx, y*sy+ty) if tr else (x, y))
        arcs.append(pts)
    def ring(idx):
        out = []
        for i in idx:
            a = arcs[~i][::-1] if i < 0 else arcs[i]
            out.extend(a if not out else a[1:])
        return out
    for g in topo['objects'][key]['geometries']:
        t = g.get('type')
        if t == 'Polygon':   rings = [ring(r) for r in g['arcs']]
        elif t == 'MultiPolygon': rings = [ring(r) for poly in g['arcs'] for r in poly]
        else: continue
        yield g.get('id'), rings, g.get('properties', {})

# ---------- 1. assignment from lat/lon file ----------
ll = json.load(open('geo/counties-10m.json'))
cents = {}
for cid, rings, props in decode(ll, 'counties'):
    pts = np.array([p for r in rings for p in r])
    if not len(pts): continue
    cents[cid] = (pts[:,1].mean(), pts[:,0].mean())   # (lat, lon)
print('counties with centroids:', len(cents))

teams = list(STAD); TLL = np.array([STAD[t] for t in teams])
def nearest(lat, lon):
    dlat = (TLL[:,0]-lat)*69.0
    dlon = (TLL[:,1]-lon)*69.0*np.cos(np.radians(lat))
    return teams[int(np.argmin(dlat**2 + dlon**2))]

# continental only: drop AK(02), HI(15), PR(72) and territories
own0 = {}
for cid,(lat,lon) in cents.items():
    st = str(cid)[:2] if len(str(cid))>=4 else None
    if st in ('02','15','60','66','69','72','78'): continue
    own0[cid] = nearest(lat, lon)
print('continental counties assigned:', len(own0))
cnt = pd.Series(own0).value_counts()
print('largest territories:', ', '.join(f'{t}:{n}' for t,n in cnt.head(5).items()))
print('smallest:', ', '.join(f'{t}:{n}' for t,n in cnt.tail(4).items()))

# ---------- 2. SVG paths from the albers file ----------
alb = json.load(open('geo/counties-albers-10m.json'))
paths = {}
for cid, rings, props in decode(alb, 'counties'):
    if cid not in own0: continue
    d = []
    for r in rings:
        if len(r) < 3: continue
        d.append('M' + 'L'.join(f'{x:.0f},{y:.0f}' for x,y in r) + 'Z')
    if d: paths[cid] = ''.join(d)
print('paths built:', len(paths), '| approx KB:', round(sum(len(v) for v in paths.values())/1024))

# ---------- 3. conquest history ----------
def conquest(season):
    g = pd.read_csv('data/games.csv', low_memory=False)
    g = g[(g.season==season) & (g.game_type=='REG') & g.result.notna()].sort_values(['week','gameday'])
    ALIAS={'OAK':'LV','SD':'LAC','STL':'LA','LAR':'LA'}
    fix=lambda t: ALIAS.get(t,t)
    empire = {t:{t} for t in teams}     # team -> set of ORIGINAL owners it now controls
    hist = []
    for wk, sub in g.groupby('week'):
        moves = []
        for r in sub.itertuples():
            h,a = fix(r.home_team), fix(r.away_team)
            if h not in empire or a not in empire: continue
            if r.result == 0: continue
            win, lose = (h,a) if r.result>0 else (a,h)
            taken = empire[lose]
            if taken:
                moves.append(dict(win=win, lose=lose, n=len(taken)))
                empire[win] |= taken
                empire[lose] = set()
        # snapshot: original-owner -> current controller
        snap = {}
        for ctrl, orig in empire.items():
            for o in orig: snap[o] = ctrl
        counts = {t: sum(len([c for c,oo in own0.items() if oo==o]) for o in empire[t]) for t in teams}
        hist.append(dict(week=int(wk), map=snap, counts=counts, moves=moves))
    return hist

H25 = conquest(2025)
last = H25[-1]
alive = {t:c for t,c in last['counts'].items() if c>0}
print()
print('2025 end state: %d teams still hold territory' % len(alive))
for t,c in sorted(alive.items(), key=lambda x:-x[1])[:6]:
    print('   %-4s %5d counties (%.1f%%)' % (t, c, 100*c/len(own0)))

json.dump(dict(paths=paths, own0=own0,
               hist2025=[{'week':h['week'],'map':h['map'],'counts':h['counts'],'moves':h['moves']} for h in H25]),
          open('territory.json','w'), separators=(',',':'))
print()
print('territory.json:', round(len(open('territory.json').read())/1024,1), 'KB')
