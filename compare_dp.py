"""Compare location reports given a known fictional 10 km by 10 km region."""
import json
from pathlib import Path

import numpy as np

def main():
    rng = np.random.default_rng(20261117)
    grid = np.array([(x, y) for x in np.arange(0.25, 10, 0.5)
                     for y in np.arange(0.25, 10, 0.5)])
    users = grid[rng.integers(len(grid), size=500)]
    epsilon = 0.2
    radius = rng.gamma(2, 1 / epsilon, size=(len(users), 64))
    angle = rng.uniform(0, 2 * np.pi, size=radius.shape)
    reports = users[:, None] + np.stack((radius * np.cos(angle),
                                         radius * np.sin(angle)), axis=-1)
    results = []
    for count in (1, 4, 16, 64):
        log_weights = np.zeros((len(users), len(grid)))
        for report in reports[:, :count].transpose(1, 0, 2):
            log_weights -= epsilon * np.linalg.norm(report[:, None] - grid, axis=-1)
        weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
        estimate = (weights / weights.sum(axis=1, keepdims=True)) @ grid
        for mode, predicted in (("region_only", np.full_like(users, 5)),
                                ("exact_report", users),
                                ("dp_report", estimate)):
            error = np.linalg.norm(predicted - users, axis=1)
            results.append(dict(mode=mode, reports=count, mean_error_km=float(error.mean()),
                                within_1km=float((error <= 1).mean()),
                                within_5km=float((error <= 5).mean())))
    data = dict(seed=20261117, trials=len(users), region_side_km=10,
                grid_step_km=0.5, epsilon_per_report_per_km=epsilon, results=results)
    report = json.dumps(data, indent=2)
    Path(__file__).with_name("compare-dp.json").write_text(report + "\n", encoding="utf-8")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
