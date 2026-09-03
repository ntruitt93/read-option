import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')

COLS=['season','season_type','play_type','passer_player_id','passer_player_name','posteam',
      'pass_location','pass_length','air_yards','complete_pass','incomplete_pass','interception',
      'pass_touchdown','epa','cpoe','yards_gained','sack']
d=pd.read_parquet('data/pbp_2025.parquet', columns=COLS)
p=d[(d.season_type=='REG')&(d.play_type=='pass')&d.passer_player_id.notna()].copy()
p=p[p.pass_location.notna()&p.pass_length.notna()]
print('charted pass attempts:', len(p))

ZONES=[(l,d_) for d_ in ['deep','short'] for l in ['left','middle','right']]
qual = p.groupby(['passer_player_id','passer_player_name','posteam']).size().reset_index(name='att')
qual = qual[qual.att>=150].sort_values('att',ascending=False)
print('qualified passers (150+ charted attempts):', len(qual))

out={}
for r in qual.itertuples():
    sub=p[p.passer_player_id==r.passer_player_id]
    zones={}
    for loc,ln in ZONES:
        z=sub[(sub.pass_location==loc)&(sub.pass_length==ln)]
        n=len(z)
        zones[f'{ln}_{loc}']=dict(
            n=int(n),
            cmp=round(float(z.complete_pass.mean()*100),1) if n else None,
            epa=round(float(z.epa.mean()),3) if n else None,
            td=int(z.pass_touchdown.sum()) if n else 0,
            int=int(z.interception.sum()) if n else 0,
            ay=round(float(z.air_yards.mean()),1) if n else None,
            cpoe=round(float(z.cpoe.mean()),1) if n and z.cpoe.notna().any() else None)
    out[r.passer_player_id]=dict(
        name=r.passer_player_name, team=r.posteam, att=int(r.att),
        cmp=round(float(sub.complete_pass.mean()*100),1),
        epa=round(float(sub.epa.mean()),3),
        td=int(sub.pass_touchdown.sum()), int=int(sub.interception.sum()),
        cpoe=round(float(sub.cpoe.mean()),1) if sub.cpoe.notna().any() else None,
        zones=zones)

# league baseline per zone, for colour scaling and context
base={}
for loc,ln in ZONES:
    z=p[(p.pass_location==loc)&(p.pass_length==ln)]
    base[f'{ln}_{loc}']=dict(cmp=round(float(z.complete_pass.mean()*100),1),
                             epa=round(float(z.epa.mean()),3), n=int(len(z)))
json.dump(dict(qbs=out, base=base), open('qb.json','w'), separators=(',',':'))
print('qb.json:', round(len(open('qb.json').read())/1024,1),'KB')
print()
print('league baseline by zone:')
for k,v in base.items(): print('  %-14s %5.1f%% cmp  %+.3f EPA  n=%d' % (k, v['cmp'], v['epa'], v['n']))
top=sorted(out.items(), key=lambda x:-x[1]['epa'])[:3]
print()
for pid,q in top:
    dl=q['zones']['deep_left']; dr=q['zones']['deep_right']
    print('%-16s %s  %.3f EPA/att overall | deep left %s%% (n=%d), deep right %s%% (n=%d)'
          % (q['name'], q['team'], q['epa'], dl['cmp'], dl['n'], dr['cmp'], dr['n']))
