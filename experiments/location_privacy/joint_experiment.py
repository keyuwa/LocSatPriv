"""Repeated synthetic joint-observation experiment; distances km, time seconds."""
import argparse
import csv
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def synthetic_schedule(counts):
    points = np.array([(x + .5, y + .5) for x in range(-20, 20)
                       for y in range(-20, 20)])
    region_ids = ((points + 20) // 10).astype(int)
    groups = np.unique(region_ids, axis=0)
    arrivals = np.arange(max(counts)) * 16
    time = np.arange(arrivals[-1] + 11)
    centers = np.stack(((time[:, None] * 5 + [0, 35, 70]) % 105 - 52,
                        np.broadcast_to([-15, 4, 18], (len(time), 3))), axis=-1)
    coverage = ((centers[:, :, None] - points) ** 2).sum(axis=-1) <= 31 ** 2
    receivers = np.array([[-18, -8], [18, 10]])
    visible = (((centers[:, :, None] - receivers) ** 2).sum(axis=-1) <= 31 ** 2).any(axis=-1)
    beam_sets = [c for size in (1, 2, 3) for c in combinations(range(3), size)]
    # Observable token encodes absolute send slot and visible beam subset; zero is silence.
    tokens = np.zeros((len(points), len(arrivals)), dtype=int)
    delivered = np.zeros_like(tokens, dtype=bool)
    waits = np.zeros_like(tokens, dtype=float)
    copies = np.zeros_like(tokens)
    for group in groups:
        members = np.flatnonzero((region_ids == group).all(axis=1))
        eligible = np.stack([coverage[:, beams][:, :, members].any(axis=1).all(axis=-1)
                             for beams in beam_sets], axis=1)
        for step, arrival in enumerate(arrivals):
            options = np.argwhere(eligible[arrival:arrival + 11])
            if not len(options):
                continue
            wait, option = map(int, options[0])
            slot = int(arrival) + wait
            beams = beam_sets[option]
            assert coverage[slot, list(beams)][:, members].any(axis=0).all()
            mask = sum(1 << beam for beam in beams if visible[slot, beam])
            tokens[members, step] = 1 + slot * 8 + mask if mask else 0
            delivered[members, step] = True
            waits[members, step] = wait
            copies[members, step] = len(beams)

    return points, tokens, delivered, waits, copies


def experiment(schedule, seed, users, counts, epsilon, total_budget):
    points, tokens, delivered, waits, copies = schedule
    rng = np.random.default_rng(seed)
    targets = rng.integers(len(points), size=users)
    truth = points[targets]
    per_report = epsilon / max(counts) if total_budget else epsilon
    radius = rng.gamma(2, 1 / per_report, size=(users, max(counts)))
    angles = rng.uniform(0, 2 * np.pi, size=radius.shape)
    reports = truth[:, None] + np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=-1)
    log_likelihood = np.zeros((users, len(points)))
    compatible = np.ones_like(log_likelihood, dtype=bool)
    distances = np.linalg.norm(points[None] - truth[:, None], axis=-1)
    rows = []
    for step in range(max(counts)):
        compatible &= tokens[:, step][None] == tokens[targets, step][:, None]
        assert compatible[np.arange(users), targets].all()
        log_likelihood -= per_report * np.linalg.norm(reports[:, step, None] - points, axis=-1)
        count = step + 1
        if count not in counts:
            continue
        delivered_count = delivered[targets, :count].sum()
        service = dict(delivery_rate=float(delivered_count / (users * count)),
                       mean_wait_s=float(waits[targets, :count].sum() / delivered_count) if delivered_count else None,
                       extra_transmissions_per_delivery=float(
                           copies[targets, :count].sum() / delivered_count - 1) if delivered_count else None)
        service["p95_wait_s"] = float(np.quantile(
            waits[targets, :count][delivered[targets, :count]], .95
        )) if delivered_count else None
        for mode in ("region_only", "region_exact", "region_dp"):
            if mode == "region_exact":
                weights = np.eye(len(points))[targets]
            else:
                scores = np.where(compatible, log_likelihood if mode == "region_dp" else 0, -np.inf)
                weights = np.exp(scores - scores.max(axis=1, keepdims=True))
                weights /= weights.sum(axis=1, keepdims=True)
            estimate = weights @ points
            errors = np.linalg.norm(estimate - truth, axis=1)
            ordered = np.sort(weights, axis=1)[:, ::-1]
            area95 = (np.cumsum(ordered, axis=1) < .95).sum(axis=1) + 1
            rows.append(dict(seed=seed, mode=mode, reports=count,
                             mean_error_km=float(errors.mean()),
                             success_1km=float((errors <= 1).mean()),
                             mean_credible95_km2=float(area95.mean()),
                             mean_support_km2=float((weights > 0).sum(axis=1).mean()),
                             **service))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--epsilon", type=float, default=.2)
    parser.add_argument("--fixed-total-budget", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "joint-results")
    parser.add_argument("--record-orbit", action="store_true")
    parser.add_argument("--paired", action="store_true",
                        help="compare point and region scheduling on historical coverage")
    args = parser.parse_args()
    if args.paired and not args.record_orbit:
        parser.error("--paired requires --record-orbit")
    if args.paired and args.output == Path(__file__).parent / "joint-results":
        args.output = Path(__file__).parent / "paired-orbit-results"
    if args.record_orbit and args.output == Path(__file__).parent / "joint-results":
        args.output = Path(__file__).parent / "record-orbit-results"
    if args.repeats < 2 or args.users < 1 or args.epsilon <= 0:
        parser.error("repeats >= 2, users >= 1 and epsilon > 0 required")
    counts = (1, 4, 16, 64)
    metadata = {}
    if args.record_orbit:
        from record_orbit import record_schedule
        schedule, metadata = record_schedule(counts)
    else:
        schedule = synthetic_schedule(counts)
    schedules = {"region": schedule}
    if args.paired:
        schedules["point"], point_metadata = record_schedule(counts, region_side=1)
        metadata["ordinary_schedule"] = point_metadata
    rows = []
    for name, current in schedules.items():
        for repeat in range(args.repeats):
            batch = experiment(current, args.seed + repeat, args.users, counts,
                               args.epsilon, args.fixed_total_budget)
            for row in batch:
                row["mode"] = row["mode"].replace("region_", name + "_", 1)
            rows.extend(batch)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["counts"] = counts
    config.update(metadata)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = ("mean_error_km", "success_1km", "mean_credible95_km2",
               "delivery_rate", "mean_wait_s", "extra_transmissions_per_delivery")
    labels = {
        "region_only": "仅区域调度",
        "region_exact": "区域调度＋精确报告",
        "region_dp": "区域调度＋差分报告",
    }
    if args.paired:
        labels.update(point_only="普通调度", point_exact="普通调度＋精确报告",
                      point_dp="普通调度＋差分报告")
    titles = {
        "mean_error_km": ("平均定位误差", "距离（千米）"),
        "success_1km": ("定位到 1 千米内的比例", "比例"),
        "mean_credible95_km2": ("平均 95% 可信区域面积", "面积（平方千米）"),
        "delivery_rate": ("消息送达率", "比例"),
        "mean_wait_s": ("已送达消息的平均等待", "时间（秒）"),
        "extra_transmissions_per_delivery": ("每条送达消息的额外发送", "额外发送次数"),
    }
    summaries = []
    for mode in labels:
        for count in counts:
            subset = [r for r in rows if r["mode"] == mode and r["reports"] == count]
            summary = dict(mode=mode, reports=count)
            for metric in (*metrics, "mean_support_km2", "p95_wait_s"):
                values = np.array([r[metric] for r in subset if r[metric] is not None])
                summary[metric] = dict(mean=float(values.mean()) if len(values) else None,
                                       sd=float(values.std(ddof=1)) if len(values) > 1 else None)
            summaries.append(summary)
        for axis, metric in zip(axes.flat, metrics):
            subset = [r for r in summaries if r["mode"] == mode]
            mean = np.array([r[metric]["mean"] for r in subset], dtype=float)
            sd = np.array([r[metric]["sd"] for r in subset], dtype=float)
            axis.plot(counts, mean, "o-", label=labels[mode])
            axis.fill_between(counts, mean - sd, mean + sd, alpha=.12)
            axis.set(xscale="log", xlabel="累计报告次数／下行事件数",
                     title=titles[metric][0], ylabel=titles[metric][1])
            axis.set_xticks(counts, labels=[str(count) for count in counts])
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    model = "RECORD 历史星历与波束" if args.record_orbit else "合成移动波束"
    fig.suptitle(f"联合观测实验（{model}，阴影表示重复实验标准差）")
    fig.tight_layout()
    fig.savefig(args.output / "joint-effects.png", dpi=180)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    if args.paired:
        comparisons = (("region_only", "point_only"), ("region_dp", "region_exact"),
                       ("region_dp", "region_only"))
        bootstrap = np.random.default_rng(args.seed)
        differences = []
        lookup = {(r["mode"], r["reports"], r["seed"]): r for r in rows}
        for first, second in comparisons:
            for count in counts:
                estimates = {}
                for metric in (*metrics, "mean_support_km2", "p95_wait_s"):
                    pairs = [(lookup[first, count, seed][metric],
                              lookup[second, count, seed][metric])
                             for seed in range(args.seed, args.seed + args.repeats)]
                    delta = np.array([a - b for a, b in pairs if a is not None and b is not None])
                    if len(delta) < 2:
                        continue
                    draws = bootstrap.choice(delta, size=(2000, len(delta))).mean(axis=1)
                    estimates[metric] = {"mean": float(delta.mean()),
                                         "ci95": np.quantile(draws, [.025, .975]).tolist()}
                differences.append(dict(first=first, second=second, reports=count,
                                        metrics=estimates))
        (args.output / "paired-differences.json").write_text(json.dumps(differences, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
