"""Offline coverage for fictional terminals using the bundled RECORD artifact."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath
from pyproj import CRS, Transformer

RECORD = Path(__file__).resolve().parent / "RECORD"


def record_schedule(counts, region_side=10):
    if str(RECORD) not in sys.path:
        sys.path.insert(0, str(RECORD))
    from TLE_calculator import TLE_calculator
    from beam_model.Generic_rec_processed_beam_model import GenericRecordedProcessedModel
    from beam_model import beamModel_coordinate_transformations as transform

    tle = RECORD / "beam_model/iridium_TLE_2022_02_14.tle"
    beam_file = RECORD / "beam_model/beamModel_iridium_noisy.npy"
    propagator = TLE_calculator(str(tle))
    model = GenericRecordedProcessedModel()
    model.load_processed_model(str(beam_file))
    polygons = [tuple(PolygonPath(p.get_points_line()) for p in group)
                for group in model.beam_polygon_dict.values()]
    beam_names = list(model.beam_polygon_dict)
    points = np.array([(x + .5, y + .5) for x in range(-20, 20) for y in range(-20, 20)])
    receivers = np.array([[-18, -8], [18, 10]])
    local_crs = CRS.from_proj4("+proj=aeqd +lat_0=50.9 +lon_0=6.95 +datum=WGS84")
    to_ecef = Transformer.from_crs(local_crs, "EPSG:4978", always_xy=True)
    local = np.vstack((points, receivers)) * 1000
    ground = np.array(to_ecef.transform(*local.T, np.zeros(len(local)))).T / 1000
    groups = ((points + 20) // region_side).astype(int)
    members = [np.flatnonzero((groups == group).all(axis=1))
               for group in np.unique(groups, axis=0)]
    n = max(counts)
    tokens = np.zeros((len(points), n), dtype=int)
    delivered = np.zeros_like(tokens, dtype=bool)
    waits = np.zeros_like(tokens, dtype=float)
    copies = np.zeros_like(tokens)
    jd, fraction = propagator.utc_time_to_jDay(2022, 2, 14, 12, 0, 0)
    for event in range(n):
        pending = set(range(len(members)))
        for wait in range(11):
            elapsed = event * 16 + wait
            instant = fraction + elapsed / 86400
            errors, positions, velocities = propagator.satrec.sgp4(
                np.array([jd]), np.array([instant]))
            positions, velocities = propagator.TEME_2_ITRS(
                jd, instant, positions[:, 0], velocities[:, 0])
            for satellite, (pos, vel) in enumerate(zip(positions, velocities)):
                if errors[satellite, 0] or np.linalg.norm(pos - ground[0]) > 3200:
                    continue
                los = pos - ground
                horizon = np.sum(los * ground, axis=1) > 0
                lonlat = transform.ITRS_2_LonLatHeight([pos])[0]
                intermediate = transform.__ITRS_2_intermediate(-los, lonlat)
                x, y, z = transform.__intermediate_2_local(intermediate, pos, vel).T
                theta = np.rad2deg(np.arctan2(np.hypot(x, y), z))
                phi = np.arctan2(y, x)
                flat = np.column_stack((theta * np.sin(phi), theta * np.cos(phi)))
                for beam, shapes in enumerate(polygons):
                    covered = np.logical_or.reduce([shape.contains_points(flat) for shape in shapes]) & horizon
                    if region_side == 1:
                        accepted = np.flatnonzero(covered[:-2] & ~delivered[:, event])
                        if covered[-2:].any():
                            tokens[accepted, event] = 1 + (
                                (elapsed * len(positions) + satellite) * len(beam_names) + beam
                            )
                        delivered[accepted, event] = True
                        waits[accepted, event] = wait
                        copies[accepted, event] = 1
                        pending.difference_update(accepted)
                        continue
                    for group in sorted(pending):
                        locations = members[group]
                        if not covered[locations].all():
                            continue
                        if covered[-2:].any():
                            tokens[locations, event] = 1 + (
                                (elapsed * len(positions) + satellite) * len(beam_names) + beam
                            )
                        delivered[locations, event] = True
                        waits[locations, event] = wait
                        copies[locations, event] = 1
                        pending.remove(group)
                if not pending:
                    break
            if not pending:
                break
    metadata = {
        "epoch_utc": "2022-02-14T12:00:00Z",
        "grid_center_lat_lon": [50.9, 6.95],
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (tle, beam_file)},
        "scheduler": "first eligible single beam with up to 10 s waiting",
        "region_side_km": region_side,
        "satellites": len(propagator.satList),
        "beams_per_satellite": len(beam_names),
    }
    return (points, tokens, delivered, waits, copies), metadata
