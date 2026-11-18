"""Paired historical replay of two-server private demand aggregation."""
import csv
import json
import secrets
from pathlib import Path

import numpy as np
from record_orbit import historical_footprints

OUT = Path(__file__).resolve().parent / "secret-share-results"
NREG = 16
PAYLOAD = 256  # modeled ciphertext bytes, excluding frame headers


def private_counts(regions):
    # Additive shares in Z/(2^32); distinct logical servers receive one vector each.
    requests = np.eye(NREG, dtype=np.uint32)[regions]
    first = np.frombuffer(secrets.token_bytes(requests.nbytes), dtype='<u4').reshape(requests.shape)
    second = requests - first
    return first.sum(axis=0, dtype=np.uint32) + second.sum(axis=0, dtype=np.uint32)


def schedule(counts, ready, deadline, footprints):
    pending = counts > 0
    actions = []
    for t in range(ready, deadline + 1):
        for beam, (covered, visible_token) in enumerate(footprints(t)):
            served = pending & covered
            if served.any():
                actions.append((t, beam, visible_token, tuple(np.flatnonzero(served))))
                pending = pending & ~served
        if not pending.any():
            break
    return actions


def main():
    grid = np.indices((40, 40)).reshape(2, -1).T
    rng = np.random.default_rng(20261117)
    regions = rng.integers(NREG, size=(64, 8))
    rows = []
    for hour in (6,12,18):
        geometry, hashes = historical_footprints(grid + .5 - 20, hour)
        for event, users in enumerate(regions):
            # One arrival per second. No scheme may deliver later than arrival + 10 s.
            base = event * 24
            cache = {}
            needed = np.unique(users)

            def footprints(t):
                if t not in cache:
                    beams = []
                    seen = np.zeros(NREG, bool)
                    for cover, token in geometry(base+t):
                        eligible = cover.reshape(4,10,4,10).all(axis=(1,3)).ravel()
                        beams.append((eligible, token))
                        seen |= eligible
                        if seen[needed].all():
                            break
                    cache[t] = beams
                return cache[t]
            for k in (1,2,4,8):
                targets = users[:k]
                plain = np.bincount(targets, minlength=NREG)
                private = private_counts(targets)
                assert np.array_equal(private, plain)
                baseline = []
                for t, label in enumerate(targets):
                    one = np.eye(NREG, dtype=int)[label]
                    baseline.extend((slot, (t,)) for slot, _, _, _ in schedule(one,t,t+10,footprints))
                plans = []
                for counts in (plain, private):
                    plans.append(schedule(counts,k-1,10,footprints))
                assert plans[0] == plans[1]
                for mode, plan in zip(('individual','plain_batch','shared_batch'),(None,*plans)):
                    deliveries = baseline if plan is None else [
                        (slot,tuple(np.flatnonzero(np.isin(targets, served))))
                        for slot, _, _, served in plan]
                    waits = [slot-user for slot, ids in deliveries for user in ids]
                    assert all(0 <= w <= 10 for w in waits)
                    rows.append(dict(hour=hour,batch=event,users=k,mode=mode,
                                     delivered=len(waits),wait_sum=sum(waits),frames=len(deliveries),
                                     ciphertext_bytes=len(deliveries)*PAYLOAD*(1 if plan is None else k),
                                     request_bytes=k*NREG*4*(2 if mode=='shared_batch' else 1)))
    OUT.mkdir(exist_ok=True)
    with (OUT/'batches.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for k in (1,2,4,8):
        for mode in ('individual','plain_batch','shared_batch'):
            subset=[r for r in rows if r['users']==k and r['mode']==mode]
            total=k*len(subset)
            delivered=sum(r['delivered'] for r in subset)
            summary.append(dict(users=k,mode=mode,delivery_rate=delivered/total,
                mean_wait_s=sum(r['wait_sum'] for r in subset)/delivered,
                downlink_bytes_per_request=sum(r['ciphertext_bytes'] for r in subset)/total,
                uplink_bytes_per_request=sum(r['request_bytes'] for r in subset)/total))
    (OUT/'summary.json').write_text(json.dumps(dict(input_hashes=hashes,results=summary),indent=2))
    print(summary[-3:])


if __name__=='__main__':
    main()
