"""Compare synthetic coordinate disclosures conditioned on upstream RECORD output."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely import contains_xy

import polygon_utilities as pu
from beam_model.Generic_rec_processed_beam_model import Polygon  # Pickled upstream beam class.

import simulations_attackerTypes_fibo as upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    if args.trials < 1 or args.seconds < 1:
        parser.error("trials and seconds must be positive")
    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Noto Sans CJK SC", "SimHei"]
    epsilon = 0.2
    counts = (1, 4, 16, 64)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    outcomes = []

    def measure(polygon, origin, simulated_location, valid, area):
        result = {"record_area_km2": area, "contains_terminal": bool(valid)}
        outcomes.append(result)
        if not valid:
            return
        truth_itrs = pu.lonLatHeight_2_ITRS(*simulated_location)
        truth = np.asarray(pu.__itrs_2_stereographic(origin, [truth_itrs])[0])
        radius = rng.gamma(2, 1 / epsilon, (64, 64))
        angle = rng.uniform(0, 2 * np.pi, radius.shape)
        reports = truth + np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)
        # Integrate the planar Laplace likelihood over the original feasible region.
        # A report-centred importance proposal resolves the narrow likelihood even for large regions.
        min_x, min_y, max_x, max_y = polygon.bounds
        uniform = rng.uniform((min_x, min_y), (max_x, max_y), (24000, 2))
        uniform = uniform[contains_xy(polygon, *uniform.T)]
        if not len(uniform):
            return
        base_error = float(np.linalg.norm(uniform.mean(axis=0) - truth))
        errors = []
        for count in counts:
            means = reports[:, :count].mean(axis=1)
            proposal_sd = 20 / np.sqrt(count)
            distances = []
            for sequence, center in zip(reports[:, :count], means):
                samples = rng.normal(center, proposal_sd, (4096, 2))
                samples = samples[contains_xy(polygon, *samples.T)]
                if not len(samples):
                    samples = uniform
                    correction = 0
                else:
                    correction = ((samples - center) ** 2).sum(axis=1) / (2 * proposal_sd ** 2)
                log_weights = -epsilon * np.linalg.norm(
                    samples[:, None] - sequence, axis=-1).sum(axis=1) + correction
                weights = np.exp(log_weights - log_weights.max())
                distances.append(float(np.linalg.norm(np.average(samples, axis=0, weights=weights) - truth)))
            errors.append({"reports": count, "record_only_km": base_error,
                           "exact_km": 0.0, "dp_km": float(np.mean(distances)),
                           "dp_within_1km": float(np.mean(np.array(distances) <= 1))})
        result["comparison"] = errors

    upstream.__analyze_attackers_generic(
        distances=[100], sniffing_durations=[args.seconds], iterations=args.trials,
        eve_amounts=3, weak_events=True, noisy_prediction=True, on_result=measure)
    folder = Path(__file__).resolve().parent / "my_simulations" / "record-dp"
    folder.mkdir(parents=True, exist_ok=True)
    result = {**vars(args), "epsilon_per_report_per_km": epsilon,
              "noise_repeats_per_scenario": 64, "outcomes": outcomes}
    (folder / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    valid_runs = [r["comparison"] for r in outcomes if "comparison" in r]
    if not valid_runs:
        raise SystemExit("No valid synthetic RECORD region to compare")
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("record_only_km", "仅 RECORD 下行"),
                       ("exact_km", "下行＋精确报告"), ("dp_km", "下行＋差分报告")):
        ax.plot(counts, [np.mean([run[i][key] for run in valid_runs])
                         for i in range(len(counts))], "o-", label=label)
    ax.set(xscale="log", xlabel="坐标报告次数", ylabel="平均定位误差（千米）")
    ax.set_xticks(counts, labels=list(map(str, counts)))
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(folder / "record-dp.png", dpi=180)
    print(folder, flush=True)


if __name__ == "__main__":
    main()
