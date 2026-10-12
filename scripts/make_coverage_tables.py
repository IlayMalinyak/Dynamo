#!/usr/bin/env python3
"""Spot-coverage calibration tables (docs/spot_coverage.md).

One table drawn once, written once per arm. The arms differ only in the
'Spot Coverage pct' column: f_sun scales every star's coverage (same scatter
draw), and the 'legacy' arm uses the old spot-count rule (column NaN). Every
other column and the per-star Seed are shared, so the arms are paired.

  python scripts/make_coverage_tables.py --n 200 --out dataset_cov
  -> dataset_cov_legacy/ dataset_cov_f0030/ dataset_cov_f0100/ dataset_cov_f0290/
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dynamo.create_data import generate_simdata, SPOT_COVERAGE_DEFAULT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="dataset_cov")
    ap.add_argument("--f-sun", type=float, nargs="+", default=[0.003, 0.01, 0.029])
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--table-seed", type=int, default=2026, help="seed for drawing the table itself")
    a = ap.parse_args()
    arms = {f"{a.out}_legacy": None}
    arms.update({f"{a.out}_f{round(f * 1e4):04d}": f for f in a.f_sun})
    for d in arms:
        if (Path(d) / "simulation_properties.csv").exists():
            sys.exit(f"{d}/simulation_properties.csv exists; refusing to overwrite")
    first = Path(next(iter(arms)))
    first.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.table_seed)
    generate_simdata(str(first), a.n, logging.getLogger("tables"), base_seed=a.base_seed, sim_name="dataset")
    t = pd.read_csv(first / "simulation_properties.csv")
    base = t["Spot Coverage pct"].to_numpy() / SPOT_COVERAGE_DEFAULT["f_sun"]
    for d, f in arms.items():
        Path(d).mkdir(parents=True, exist_ok=True)
        u = t.copy()
        u["Spot Coverage pct"] = np.nan if f is None else base * f
        u.to_csv(Path(d) / "simulation_properties.csv", index=False, float_format="%.6g")
        cov = u.loc[u["Has Dynamo"].astype(bool), "Spot Coverage pct"]
        print(f"{d}: {len(u)} stars, {int(u['Has Dynamo'].sum())} with a dynamo, "
              f"coverage % median {cov.median():.3g}" if f is not None else f"{d}: legacy spot-count rule")


if __name__ == "__main__":
    main()
