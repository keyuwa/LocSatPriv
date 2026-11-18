"""Paired historical footprint replay for fixed areas and private service labels."""
import csv
import json
from pathlib import Path
import numpy as np
from record_orbit import historical_footprints
from schedule_privacy import COUNTS, evaluate, probabilities

OUT = Path(__file__).resolve().parent / 'historical-window-results'
SIDES = (1, 2, 4, 5, 8, 10, 20, 40)


def replay(hour):
    grid = np.indices((40, 40)).reshape(2, -1).T
    points = grid + .5 - 20
    local = (abs(points[:, 0]) < 10) & (abs(points[:, 1]) < 5)
    sites = points[local]
    region = (sites[:, 0] > 0).astype(int)
    groups = {s: np.unique(grid // s, axis=0, return_inverse=True)[1] for s in SIDES}
    # Append the two service areas to the list of fixed grid regions.
    offsets = np.cumsum([0] + [(40 // s)**2 for s in SIDES])
    count = offsets[-1] + 2
    n = max(COUNTS)
    tokens = np.zeros((count, n), int)
    sent = np.zeros_like(tokens, bool)
    waits = np.zeros_like(tokens)
    delivery = np.zeros((2, n, len(sites)), bool)
    footprints, hashes = historical_footprints(points, hour)
    for event in range(n):
        for wait in range(11):
            for cover, token in footprints(event * 16 + wait):
                field = cover.reshape(40, 40)
                eligible = [field.reshape(40//s,s,40//s,s).all(axis=(1,3)).ravel() for s in SIDES]
                eligible.append([cover[local][region == k].all() for k in (0,1)])
                chosen = np.concatenate(eligible) & ~sent[:,event]
                sent[chosen,event] = True
                tokens[chosen,event] = token
                waits[chosen,event] = wait
                delivery[chosen[-2:],event] = cover[local]
                if sent[:,event].all():
                    break
            if sent[:,event].all():
                break
    size_rows = []
    for offset,side in zip(offsets,SIDES):
        ids = groups[side] + offset
        for length in COUNTS:
            _,inv,sizes = np.unique(tokens[ids,:length], axis=0, return_counts=True, return_inverse=True)
            mass = sizes / len(points)
            centers = np.column_stack([np.bincount(inv,weights=c)/sizes for c in points.T])
            hits = sent[ids,:length]
            size_rows.append(dict(hour=hour,side_km=side,messages=length,
                mean_error_km=np.linalg.norm(centers[inv]-points,axis=1).mean(),
                leaked_bits=-mass@np.log2(mass),delivery=hits.mean(),
                p95_wait_delivered_s=float(np.quantile(waits[ids,:length][hits],.95)) if hits.any() else None))
    plan = tokens[-2:], sent[-2:].astype(int), waits[-2:], delivery
    dp = []
    for mode in ('fixed_region','random_reuse','random_refresh'):
        prob = np.eye(2)[region] if mode=='fixed_region' else probabilities(sites,[[-5,0],[5,0]],.3)
        for row in evaluate(sites,region,plan,prob,mode,.3 if mode!='fixed_region' else None,np.random.default_rng(20261117),1000):
            dp.append(dict(hour=hour,margin_km=0,epsilon_per_label_per_km=.3 if mode!='fixed_region' else None,**row))
    return size_rows, dp, hashes


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    sizes,records=[],[]
    for hour in (6,12,18):
        rows,dp,hashes=replay(hour)
        sizes.extend(rows)
        records.extend(dp)
        print('UTC',hour,'complete',flush=True)
    with (OUT/'region-size.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sizes[0])
        writer.writeheader()
        writer.writerows(sizes)
    (OUT/'summary.json').write_text(json.dumps(dict(date='2022-02-14',hours_utc=[6,12,18],input_sha256=hashes,results=records),indent=2,allow_nan=False))
