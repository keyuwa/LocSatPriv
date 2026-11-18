"""Audit region scheduling on fictional footprints and summarize upstream CSVs."""
import csv
import json
from itertools import combinations
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DURATIONS = (60, 180, 600, 1800, 3600, 7200, 14400, 28800, 57600)
PAPER = (307000, 122000, 25300, 3480, 700, 154, 92, 60.8, 48.5)


def published_baseline():
    rows = []
    for seconds, reference in zip(DURATIONS, PAPER):
        name = f"100kmFibo_cont_{seconds}sec_3eves_weakEvents_noisyPrediction.csv"
        values = {}
        for folder in ("duration_and_type", "public_model"):
            with (ROOT / "simulation_data" / folder / name).open() as source:
                combined = [float(row[1].rstrip("]")) for row in csv.reader(source)]
            values[folder] = {"runs": len(combined), "median_km2": float(np.median(combined))}
        rows.append({"seconds": seconds, "readme_paper_km2": reference, **values,
                     "public_over_paper_reference": values["public_model"]["median_km2"] / reference})
    return rows


def audit(region_side, deadline, messages=12):
    # Fictional kilometre grid. Each point represents a 1 km² cell.
    points = np.array([(x, y) for x in range(-30, 30) for y in range(-30, 30)])
    arrivals = np.arange(messages) * 16
    slots = int(arrivals[-1]) + deadline + 1
    # Three moving circular footprints, with the geometry repeated every 21 s.
    time = np.arange(slots)
    centers = np.stack(((time[:, None] * 5 + np.array([0, 35, 70])) % 105 - 52,
                        np.broadcast_to([-15, 4, 18], (slots, 3))), axis=-1)
    covered = np.sum((centers[:, :, None, :] - points) ** 2, axis=-1) <= 31 ** 2
    observed_beams = np.any(np.sum((centers[:, :, None, :] -
                                  np.array([[-18, -8], [18, 10]])) ** 2, axis=-1)
                            <= 31 ** 2, axis=-1)
    policies = {}
    for mode in ("point", "region", "region_multi"):
        beam_sets = [(beam,) for beam in range(3)]
        if mode == "region_multi":
            beam_sets += [group for size in (2, 3)
                          for group in combinations(range(3), size)]
        regions = defaultdict(list)
        for index, (x, y) in enumerate(points):
            key = (index,) if mode == "point" else ((x + 30) // region_side, (y + 30) // region_side)
            regions[key].append(index)
        received = defaultdict(list)
        waits = []
        deliveries = 0
        transmissions = 0
        for members in regions.values():
            eligible = np.stack([
                covered[:, group][:, :, members].any(axis=1).all(axis=-1)
                for group in beam_sets
            ], axis=1)
            transcript = []
            for arrival in arrivals:
                # First slot and lowest beam ID that cover the entire service set.
                options = np.argwhere(eligible[arrival:arrival + deadline + 1])
                if not options.size:
                    transcript.append(None)
                    continue
                wait, option = map(int, options[0])
                slot = int(arrival) + wait
                beams = beam_sets[option]
                deliveries += len(members)
                transmissions += len(members) * len(beams)
                waits.extend([wait] * len(members))
                visible = tuple(beam for beam in beams if observed_beams[slot, beam])
                transcript.append((slot, visible) if visible else None)
            received[tuple(transcript)].extend(members)
        # Observer knows the full deterministic policy, arrivals and receiver mask.
        # Group synthetic locations by complete observable transcript, including silence.
        areas = []
        errors = []
        for members in received.values():
            coords = points[members]
            areas.extend([len(members)] * len(members))
            errors.extend(np.linalg.norm(coords - coords.mean(axis=0), axis=1))
        assert len(areas) == len(points)
        policies[mode] = {
            "median_candidate_km2": float(np.median(areas)),
            "min_candidate_km2": min(areas),
            "mean_centroid_error_km": float(np.mean(errors)),
            "delivery_rate": deliveries / (len(points) * messages),
            "mean_wait_s_delivered": float(np.mean(waits)) if waits else None,
            "transmissions_per_user": transmissions / len(points),
            "extra_copies_per_delivered_message": (
                (transmissions - deliveries) / deliveries if deliveries else None
            ),
        }
    return {"region_side_km": region_side, "deadline_s": deadline,
            "messages_per_location": messages, "synthetic_locations": len(points), **policies}


def main():
    result = {
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "baseline_kind": "recomputed statistics from upstream CSVs",
        "baseline": published_baseline(),
        "defense_kind": "fictional footprints, offline synthetic transcript equivalence",
        "defense": [audit(side, deadline) for side in (10, 20, 30) for deadline in (3, 10)],
    }
    result["comparisons_at_98pct_delivery"] = [
        {"region_side_km": side, "deadline_s": deadline,
         "eligible_policies": [name for name in ("point", "region", "region_multi")
                               if row[name]["delivery_rate"] >= 0.98]}
        for row in result["defense"]
        for side, deadline in [(row["region_side_km"], row["deadline_s"])]
    ]
    path = ROOT / "privacy-evaluation.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
