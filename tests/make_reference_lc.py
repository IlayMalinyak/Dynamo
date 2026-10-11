"""Regression scenario for the Q=0 test (T-B0).

Runs a fixed multi-spot star with the Dynamo package found at REPO and saves
time + normalised Kepler flux. Generate the reference once on master with
only the mu-pairing fix (commit 4035b2e) cherry-picked:

    python tests/make_reference_lc.py <master_checkout> tests/data/ref_lc_noQ.npz

The same function is called by tests/test_faculae.py on the current code.
"""
import os
import sys
from pathlib import Path

import numpy as np

MODELS_ROOT = Path(os.environ.get("DYNAMO_MODELS_ROOT", Path.home() / ".dynamo"))

SPOTS = np.array([  # t_emerge, duration, colat, lon, radius (deg)
    [0.0, 60.0, 70.0, 10.0, 6.0],
    [5.0, 60.0, 100.0, 150.0, 4.0],
    [12.0, 60.0, 50.0, 250.0, 8.0],
    [20.0, 60.0, 120.0, 300.0, 3.0],
    [25.0, 60.0, 85.0, 40.0, 5.0],
])


def run_scenario():
    from Dynamo.star import Star, FLUX_SCALE
    s = Star(conf_file_path="star.conf")
    s.verbose = False
    p = {"Activity Rate": 1.0, "Shear": 0.1, "Period": 10.0, "convective_shift": 0.0,
         "R": 1.0, "mass": 1.0, "L": 1.0, "Teff": 5800.0, "logg": 4.5, "FeH": 0.0,
         "age": 4.6, "CDPP": 0.0, "Outlier Rate": 0.0, "Flicker Time Scale": 0.0,
         "Inclination": np.deg2rad(60.0), "Cycle Length": 11.0, "Cycle Overlap": 1.0,
         "Spot Max": 30.0, "Spot Min": 5.0, "Decay Time": 3.0, "Butterfly": True,
         "Distance": 10.0, "simulate_planet": 0}
    s.set_stellar_parameters(p)
    s.set_planet_parameters({"simulate_planet": 0})
    s.models_root = MODELS_ROOT
    s.photometry_names, s.photometry_filters, s.photometry_cadences = ["Kepler"], ["Kepler.dat"], [1 / 48]
    s.spectra_names = []
    s.spot_T_contrast = 1500.0  # pinned: independent of the contrast prior
    s.facular_area_ratio = 0.0
    s.spots_evo_law = "gaussian"
    s.spot_map = np.zeros((len(SPOTS), 14))
    s.spot_map[:, :5] = SPOTS
    s.reference_time = 0.0
    s.n_grid_rings = 20
    t = np.linspace(0.0, 40.0, 800)
    s.compute_forward(t=t)
    scale = (s.radius * 6.957e10 / (s.distance * 3.086e18)) ** 2 * FLUX_SCALE
    return t, np.asarray(s.results["lc"], dtype=float) / scale


if __name__ == "__main__":
    repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    t, f = run_scenario()
    np.savez(out, t=t, lc=f)
    print("saved", out, "flux range", f.min(), f.max())
