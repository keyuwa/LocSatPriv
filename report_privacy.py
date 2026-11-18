"""Planar Laplace report experiment. Coordinates and distances are in km."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
COUNTS = (1, 4, 16, 64)
TRIALS = 500
EPSILON = 0.2


def main():
    rng = np.random.default_rng(20261117)
    users = rng.uniform(-50, 50, size=(TRIALS, 2))
    rows = []
    summaries = []
    for mode in ("exact", "grid_20km", "fixed_per_report", "fixed_total"):
        for count in COUNTS:
            epsilon = EPSILON / count if mode == "fixed_total" else EPSILON
            if mode == "exact":
                reports = np.repeat(users[:, None, :], count, axis=1)
            elif mode == "grid_20km":
                centers = (np.floor(users / 20) + 0.5) * 20
                reports = np.repeat(centers[:, None, :], count, axis=1)
            else:
                # Planar Laplace: uniform angle, Gamma(shape=2, scale=1/epsilon).
                radius = rng.gamma(2, 1 / epsilon, size=(TRIALS, count))
                angle = rng.uniform(0, 2 * np.pi, size=(TRIALS, count))
                noise = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)
                reports = users[:, None, :] + noise
            estimate = reports.mean(axis=1)
            errors = np.linalg.norm(estimate - users, axis=1)
            utility = np.linalg.norm(reports - users[:, None, :], axis=-1).mean(axis=1)
            bootstrap = rng.choice(errors, size=(1000, TRIALS), replace=True).mean(axis=1)
            low, high = np.quantile(bootstrap, (0.025, 0.975))
            total_budget = epsilon * count if mode.startswith("fixed") else None
            summaries.append(dict(mode=mode, reports=count,
                                  epsilon_per_km=epsilon if mode.startswith("fixed") else None,
                                  composed_epsilon_per_km=total_budget,
                                  mean_error_km=float(errors.mean()),
                                  ci95_low=float(low), ci95_high=float(high),
                                  success_within_5km=float((errors <= 5).mean()),
                                  mean_report_error_km=float(utility.mean())))
            for trial in range(TRIALS):
                for step in range(count):
                    rows.append((mode, count, trial, step, *users[trial],
                                 *reports[trial, step]))
    with (ROOT / "reports.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(("mode", "count", "trial", "step", "true_x_km", "true_y_km",
                         "report_x_km", "report_y_km"))
        writer.writerows(rows)
    (ROOT / "report-summary.json").write_text(json.dumps(
        {"seed": 20261117, "trials": TRIALS, "attacker": "sample mean",
         "results": summaries}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in ("exact", "grid_20km", "fixed_per_report", "fixed_total"):
        data = [row for row in summaries if row["mode"] == mode]
        axes[0].plot(COUNTS, [row["mean_error_km"] for row in data], "o-", label=mode)
        axes[1].plot(COUNTS, [row["mean_report_error_km"] for row in data], "o-", label=mode)
    for axis, title in zip(axes, ("Attacker mean-estimate error", "Mean report distortion")):
        axis.set(xscale="log", yscale="symlog", xlabel="Number of reports",
                 ylabel="Distance (km)", title=title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "report-privacy.png", dpi=180)
    for row in summaries:
        print(row["mode"], row["reports"], round(row["mean_error_km"], 2),
              round(row["success_within_5km"], 3))


if __name__ == "__main__":
    main()
