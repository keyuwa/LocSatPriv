"""Evaluate location privacy of downlink schedules without position reports."""
import argparse
import json
from pathlib import Path

import numpy as np

COUNTS = (1, 4, 16, 32, 64)


def build_schedule(coverage, visible, areas):
    """Choose the earliest slot covering the designated area, then fewest beams."""
    events, slots, beams, points = coverage.shape
    masks = sorted(range(1, 1 << beams), key=int.bit_count)
    tokens = np.zeros((len(areas), events), dtype=int)
    copies = np.zeros_like(tokens)
    waits = np.zeros_like(tokens)
    delivery = np.zeros((len(areas), events, points), dtype=bool)
    for event in range(events):
        pending = np.ones(len(areas), dtype=bool)
        for slot in range(slots):
            for mask in masks:
                selected = [b for b in range(beams) if mask & (1 << b)]
                covered = coverage[event, slot, selected].any(axis=0)
                chosen = pending & (covered | ~areas).all(axis=1)
                if not chosen.any():
                    continue
                observed = sum(1 << b for b in selected if visible[event, slot, b])
                tokens[chosen, event] = 1 + slot * (1 << beams) + observed if observed else 0
                copies[chosen, event] = len(selected)
                waits[chosen, event] = slot
                delivery[chosen, event] = covered
                pending[chosen] = False
    return tokens, copies, waits, delivery


def synthetic_geometry(points=None):
    # Adjacent 10 km squares on either side of x=0.
    if points is None:
        points = np.array([(x + .5, y + .5) for x in range(-10, 10) for y in range(-5, 5)])
    times = np.arange(max(COUNTS))[:, None] * 16 + np.arange(11)
    centers = np.stack(((times[..., None] * 5 + [0, 35, 70]) % 105 - 52,
                        np.broadcast_to([-15, 4, 18], (*times.shape, 3))), axis=-1)

    def covers(targets):
        return ((centers[..., None, :] - targets) ** 2).sum(axis=-1) <= 31**2

    visible = covers(np.array([[-18, -8], [18, 10]])).any(axis=-1)
    return points, covers(points), visible


def probabilities(points, centers, epsilon):
    # The score and log normalizer are each epsilon/2-Lipschitz.
    scores = -.5 * epsilon * np.linalg.norm(points[:, None] - centers, axis=-1)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def observed_distribution(prob, tokens):
    _, inverse = np.unique(tokens, axis=0, return_inverse=True)
    return prob @ (inverse[:, None] == np.arange(inverse.max() + 1))


def pair_metrics(points, region, distribution, refresh_tokens=None, prob=None):
    distances = np.linalg.norm(points[:, None] - points, axis=-1)
    left, right = np.where(np.triu(np.isclose(distances, 1), k=1))
    rows = {}
    for name, select in (("within", region[left] == region[right]),
                         ("across", region[left] != region[right])):
        p, q = distribution[left[select]], distribution[right[select]]
        union = (p > 0) | (q > 0)
        mismatch = ((p == 0) != (q == 0)).any()
        ratio = np.abs(np.log(p[union]) - np.log(q[union])).max() if not mismatch else None
        tv = float(.5 * np.abs(p - q).sum(axis=1).max())
        if refresh_tokens is not None:
            # For independent events, each directional maximum adds across events.
            a, b = left[select], right[select]
            forward = np.zeros(len(a))
            reverse = np.zeros(len(a))
            for column in refresh_tokens.T:
                marginal = observed_distribution(prob, column[:, None])
                log_ratio = np.log(marginal[a]) - np.log(marginal[b])
                forward += log_ratio.max(axis=1)
                reverse -= log_ratio.min(axis=1)
            ratio = float(max(forward.max(), reverse.max()))
            tv = None
        rows[name] = dict(pairs=int(select.sum()), max_tv=tv, max_log_ratio=ratio)
    return rows


def evaluate(points, region, plan, prob, mode, epsilon, rng, trials):
    tokens, copies, waits, delivery = plan
    targets = rng.integers(len(points), size=trials)
    if mode == "random_refresh":
        draws = rng.random((trials, max(COUNTS)))
        labels = (draws > prob[targets, 0, None]).astype(int)
    else:
        draws = rng.random(trials)
        picked = (draws[:, None] > prob[targets].cumsum(axis=1)).sum(axis=1)
        labels = np.broadcast_to(picked[:, None], (trials, max(COUNTS)))
    events = np.arange(max(COUNTS))
    observations = tokens[labels, events]
    posterior = np.ones((trials, len(points)))
    rows = []
    for event in events:
        if mode == "random_refresh":
            matches = observations[:, event, None] == tokens[:, event]
            posterior *= matches @ prob.T
            posterior /= posterior.sum(axis=1, keepdims=True)
        count = int(event + 1)
        if count not in COUNTS:
            continue
        if mode != "random_refresh":
            matches = (observations[:, None, :count] == tokens[None, :, :count]).all(axis=-1)
            posterior = matches @ prob.T
            posterior /= posterior.sum(axis=1, keepdims=True)
        observed = observed_distribution(prob, tokens[:, :count])
        refresh = tokens[:, :count] if mode == "random_refresh" and count > 1 else None
        statistics = pair_metrics(points, region, observed, refresh, prob)
        bound = epsilon * (count if mode == "random_refresh" else 1) if mode.startswith("random") else None
        if statistics is not None and bound is not None:
            for stats in statistics.values():
                assert stats["max_log_ratio"] is not None and stats["max_log_ratio"] <= bound + 1e-10
        success = delivery[labels[:, :count], events[:count], targets[:, None]]
        selected_waits = waits[labels[:, :count], events[:count]][success]
        transmissions = copies[labels[:, :count], events[:count]]
        ordered = -np.sort(-posterior, axis=1)
        rows.append(dict(mode=mode, messages=count, epsilon_bound_per_km=bound,
                         neighbor_metrics=statistics,
                         centroid_within_1km=float((np.linalg.norm(posterior @ points - points[targets], axis=1) <= 1).mean()),
                         mean_error_km=float(np.linalg.norm(posterior @ points - points[targets], axis=1).mean()),
                         mean_credible95_km2=float(((ordered.cumsum(axis=1) < .95).sum(axis=1) + 1).mean()),
                         delivery_rate=float(success.mean()),
                         p95_delivered_wait_s=float(np.quantile(selected_waits, .95)) if success.any() else None,
                         transmissions_per_message=float(transmissions.mean())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("schedule-privacy-results"))
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    points, coverage, visible = synthetic_geometry()
    region = (points[:, 0] >= 0).astype(int)
    exact = build_schedule(coverage, visible, np.eye(len(points), dtype=bool))
    records = []
    for margin in (0, 5, 10):
        areas = np.stack([points[:, 0] < margin, points[:, 0] >= -margin])
        plan = build_schedule(coverage, visible, areas)
        settings = [("random_reuse", plan, probabilities(points, [[-5, 0], [5, 0]], e), e)
                    for e in (.1, .3, 1.)]
        settings += [("random_refresh", plan, p, e) for _, _, p, e in settings]
        if not margin:
            settings += [("point", exact, np.eye(len(points)), None),
                         ("fixed_region", plan, np.eye(2)[region], None)]
        for mode, schedule, prob, epsilon in settings:
            rng = np.random.default_rng(args.seed)
            for row in evaluate(points, region, schedule, prob, mode, epsilon, rng, args.trials):
                records.append(dict(margin_km=margin, epsilon_per_label_per_km=epsilon, **row))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(
        dict(seed=args.seed, trials=args.trials, grid_cells=len(points), results=records),
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
