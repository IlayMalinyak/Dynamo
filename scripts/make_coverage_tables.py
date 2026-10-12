#!/usr/bin/env python3
"""Spot-coverage calibration tables (docs/spot_coverage.md).

One table drawn once, written once per arm. The arms differ only in the
'Spot Coverage pct' column; the per-star scatter draw, every other column and the
per-star Seed are shared, so the arms are paired. Arms:
  legacy     old spot-count rule (column NaN)
  fXXXX      single law f_sun * max(Ro/Ro_sun, ro_sat)^-0.881, f_sun = XXXX/1e4 (model "rossby")
  broken     the default broken law (SPOT_COVERAGE_DEFAULT)

  python scripts/make_coverage_tables.py --n 200 --out dataset_cov
  -> dataset_cov_legacy/ dataset_cov_f0030/ dataset_cov_f0100/ dataset_cov_f0290/ dataset_cov_broken/
  python scripts/make_coverage_tables.py --arms broken     # only the default-law arm
"""
import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dynamo.create_data import (generate_simdata, SPOT_COVERAGE_DEFAULT, ROSSBY_SUN,  # noqa: E402
                                spot_coverage_law)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="dataset_cov")
    ap.add_argument("--f-sun", type=float, nargs="+", default=[0.003, 0.01, 0.029])
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--table-seed", type=int, default=2026, help="seed for drawing the table itself")
    ap.add_argument("--arms", nargs="+", default=["legacy", "f_sun", "broken"],
                    help="any of: legacy, f_sun (one arm per --f-sun value), broken")
    a = ap.parse_args()
    arms = {}
    if "legacy" in a.arms:
        arms[f"{a.out}_legacy"] = None
    if "f_sun" in a.arms:
        arms.update({f"{a.out}_f{round(f * 1e4):04d}": f for f in a.f_sun})
    if "broken" in a.arms:
        arms[f"{a.out}_broken"] = "broken"
    for d in arms:
        if (Path(d) / "simulation_properties.csv").exists():
            sys.exit(f"{d}/simulation_properties.csv exists; refusing to overwrite")
    with tempfile.TemporaryDirectory() as tmp:
        np.random.seed(a.table_seed)
        generate_simdata(tmp, a.n, logging.getLogger("tables"), base_seed=a.base_seed, sim_name="dataset")
        t = pd.read_csv(Path(tmp) / "simulation_properties.csv")
    # per-star scatter factor of the default law (0 for stars without a dynamo)
    law = spot_coverage_law(t["Rossby"].to_numpy() / ROSSBY_SUN, SPOT_COVERAGE_DEFAULT)
    eps = t["Spot Coverage pct"].to_numpy() / 100 / law
    x = t["Rossby"].to_numpy() / ROSSBY_SUN
    for d, f in arms.items():
        Path(d).mkdir(parents=True, exist_ok=True)
        u = t.copy()
        if f is None:
            u["Spot Coverage pct"] = np.nan
        elif f != "broken":
            single = spot_coverage_law(x, dict(SPOT_COVERAGE_DEFAULT, model="rossby", f_sun=f))
            u["Spot Coverage pct"] = 100 * np.minimum(single * eps, SPOT_COVERAGE_DEFAULT["f_max"])
        u.to_csv(Path(d) / "simulation_properties.csv", index=False, float_format="%.6g")
        cov = u.loc[u["Has Dynamo"].astype(bool), "Spot Coverage pct"]
        print(f"{d}: {len(u)} stars, {int(u['Has Dynamo'].sum())} with a dynamo, "
              f"coverage % median {cov.median():.3g}" if f is not None else f"{d}: legacy spot-count rule")


if __name__ == "__main__":
    main()
