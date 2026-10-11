"""Shared helpers for the Dynamo tests (no pytest fixtures needed)."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_ROOT = Path(os.environ.get("DYNAMO_MODELS_ROOT", Path.home() / ".dynamo"))


def have_phoenix():
    d = MODELS_ROOT / "models" / "Phoenix_mu"
    return d.is_dir() and any(d.glob("lte*SPECINT*.fits"))


def solar_params(**over):
    """Parameter dict accepted by Star.set_stellar_parameters (solar-like star)."""
    p = {
        "Activity Rate": 1.0, "Shear": 0.0, "Period": 25.0, "convective_shift": 0.0,
        "R": 1.0, "mass": 1.0, "L": 1.0, "Teff": 5800.0, "logg": 4.5, "FeH": 0.0,
        "age": 4.6, "CDPP": 0.0, "Outlier Rate": 0.0, "Flicker Time Scale": 0.0,
        "Inclination": np.pi / 2, "Cycle Length": 11.0, "Cycle Overlap": 1.0,
        "Spot Max": 30.0, "Spot Min": 5.0, "Decay Time": 6.0, "Butterfly": True,
        "Distance": 10.0, "simulate_planet": 0,
    }
    p.update(over)
    return p


def make_star(conf="star.conf", **over):
    from Dynamo.star import Star
    s = Star(conf_file_path=conf)
    s.verbose = False
    s.set_stellar_parameters(solar_params(**over))
    s.set_planet_parameters({"simulate_planet": 0})
    s.models_root = MODELS_ROOT
    s.photometry_names, s.photometry_filters, s.photometry_cadences = ["Kepler"], ["Kepler.dat"], [1 / 48]
    s.spectra_names = []
    return s


def single_feature_lc(star, colat_deg=90.0, lon_deg=0.0, rad_deg=5.0, ntime=1000, n_rot=1.0):
    """Light curve of one constant-size spot over `n_rot` rotations (normalised flux)."""
    star.spots_evo_law = "constant"
    star.spot_map = np.zeros((1, 14))
    star.spot_map[0, :5] = [-1.0, 1e6, colat_deg, lon_deg, rad_deg]
    star.reference_time = 0.0
    star.n_grid_rings = 30
    t = np.linspace(0, n_rot * star.rotation_period, ntime)
    star.compute_forward(t=t)
    return t, np.asarray(star.results["lc"], dtype=float) / flux_scale(star)


def flux_scale(star):
    """Factor compute_forward multiplies the normalised flux by ((R/d)^2 * FLUX_SCALE)."""
    from Dynamo.star import FLUX_SCALE
    return (star.radius * 6.957e10 / (getattr(star, "distance", 10.0) * 3.086e18)) ** 2 * FLUX_SCALE
