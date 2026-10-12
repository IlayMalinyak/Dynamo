#!/usr/bin/env python3
"""Simulate the Sun as a Kepler star and measure its rotational variability.

Each run is one 1000-day, 30-min-cadence Kepler-band light curve of a solar-twin
(Teff 5772 K, R = 1, P = 25.4 d, seen equator-on) with a given time-mean spot
coverage, spot size, spot contrast, facular ratio and spot lifetime. Output per run:
R_per (McQuillan: median over rotation-period windows of the 5-95% range) and the
realised filling factors. Compare against the measured solar values in
docs/sun_calibration.md.

  python scripts/sun_calibration.py --coverage 0.001 0.003 --max-area 3 10 \
      --contrast-scale 1.0 --q 3 --decay 6 --seeds 3 --out sun_grid.csv
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
from helpers import make_star, flux_scale  # noqa: E402
from faculae_smoke_report import r_per  # noqa: E402

# [paper] Teff 5772 K (IAU 2015 nominal, Prsa et al. 2016); P = 25.4 d, sidereal
# equatorial rotation (the value behind ROSSBY_SUN in create_data).
# [own assumption] the observer is in the equatorial plane (i = 90 deg; the ecliptic view
# is i ~ 83 deg), no differential rotation (shear 0), spots at 5-30 deg latitude (the
# main sunspot belt, simplified to a fixed band) and one 1000-d window rather than a
# full cycle at varying coverage.
P_SUN = 25.4


def q_shapiro(spot_disc_fraction):
    """Facular-to-spot area ratio of the Sun at a given spot disc coverage, from the
    SATIRE-S fits of Shapiro et al. 2014 (arXiv:1406.2383, Sect. 3, Eqs. 1-2):
    A_S = 0.105 - 1.315 S + 4.102 S^2,  A_F = -0.233 + 1.400 S  (S = Ca II S-index).
    Q ~ 11 at maximum (A_S = 0.28%), ~100 at minimum."""
    a, b, c = 4.102, -1.315, 0.105 - spot_disc_fraction
    S = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    return (-0.233 + 1.400 * S) / spot_disc_fraction


def run(coverage, max_area, contrast_scale, q, decay, seed, ndays=1000, cadence=1 / 48,
        incl_deg=90.0, shear=0.0, lat=(5.0, 30.0)):
    s = make_star(**{"Teff": 5772.0, "Period": P_SUN, "Spot Coverage pct": 100 * coverage,
                     "Spot Max Area": max_area, "Spot Contrast Scale": contrast_scale,
                     "Facula Q": q, "Decay Time": decay, "Seed": seed, "Shear": shear,
                     "Inclination": np.deg2rad(incl_deg), "Spot Min": lat[0], "Spot Max": lat[1]})
    s.generate_spot_map(ndays)
    t = np.arange(0, ndays, cadence)
    t0 = time.time()
    s.compute_forward(t=t)
    y = np.asarray(s.results["lc"], dtype=float) / flux_scale(s)
    return dict(coverage=coverage, max_area=max_area, contrast_scale=contrast_scale, q=q,
                decay=decay, seed=seed, n_spots=len(s.spot_map), dT_spot=s.spot_T_contrast,
                rper_ppt=1e3 * r_per(y, t, P_SUN),
                range_ppt=1e3 * (np.quantile(y, 0.95) - np.quantile(y, 0.05)) / np.median(y),
                std_ppt=1e3 * np.std(y) / np.median(y),
                mean_ff_sp=float(np.mean(s.results["ff_sp"])), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", type=float, nargs="+", default=[0.001, 0.003])
    ap.add_argument("--max-area", type=float, nargs="+", default=[10.0])
    ap.add_argument("--contrast-scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--q", type=float, nargs="+", default=[3.0], help="facular ratio; -1 = Shapiro 2014 Q(coverage)")
    ap.add_argument("--decay", type=float, nargs="+", default=[6.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ndays", type=int, default=1000)
    ap.add_argument("--out", default="sun_grid.csv")
    a = ap.parse_args()
    rows = []
    grid = itertools.product(a.coverage, a.max_area, a.contrast_scale, a.q, a.decay, range(a.seeds))
    for cov, area, cs, q, dec, k in grid:
        rows.append(run(cov, area, cs, q_shapiro(cov) if q < 0 else q, dec, seed=1000 + k, ndays=a.ndays))
        print({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
