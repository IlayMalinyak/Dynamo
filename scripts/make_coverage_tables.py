#!/usr/bin/env python3
"""Spot-coverage calibration tables (docs/spot_coverage.md).

Each arm is one coverage_config (overriding SPOT_COVERAGE_DEFAULT) applied to the same
table: same --table-seed and --base-seed, so every star has the same parameters, scatter
draw and per-star Seed in every arm; arms differ only in the coverage column and, with
spot properties scaled by activity, in Spot Contrast Scale / Spot Max Area / Facula Q.

Named arms:
  legacy           old spot-count rule and old spot properties
  fXXXX            single law, f_sun = XXXX/1e4, drawn spot properties (first calibration)
  broken           SPOT_COVERAGE_DEFAULT
  act_fbXXXX       default (Sun-anchored broken law, activity-scaled spots), f_break = XXXX/1e4

  python scripts/make_coverage_tables.py --arms act_fb0100 act_fb0200 act_fb0400
  -> dataset_cov_act_fb0100/ ...
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dynamo.create_data import generate_simdata  # noqa: E402


def arm_config(name):
    if name == "legacy":
        return {"model": "legacy"}
    if name == "broken":
        return {}
    m = re.fullmatch(r"f(\d{4})", name)
    if m:
        return {"model": "rossby", "f_sun": int(m.group(1)) / 1e4, "props": "fixed"}
    m = re.fullmatch(r"act_fb(\d{4})", name)
    if m:
        return {"f_break": int(m.group(1)) / 1e4}
    sys.exit(f"unknown arm {name!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="dataset_cov")
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--table-seed", type=int, default=2026, help="seed for drawing the table itself")
    a = ap.parse_args()
    arms = {f"{a.out}_{name}": arm_config(name) for name in a.arms}
    for d in arms:
        if (Path(d) / "simulation_properties.csv").exists():
            sys.exit(f"{d}/simulation_properties.csv exists; refusing to overwrite")
    for d, cfg in arms.items():
        Path(d).mkdir(parents=True, exist_ok=True)
        np.random.seed(a.table_seed)
        generate_simdata(d, a.n, logging.getLogger("tables"), base_seed=a.base_seed,
                         sim_name="dataset", coverage_config=cfg)
        u = pd.read_csv(Path(d) / "simulation_properties.csv")
        cov = u.loc[u["Has Dynamo"].astype(bool), "Spot Coverage pct"]
        print(f"{d}: {len(u)} stars, {int(u['Has Dynamo'].sum())} with a dynamo, config {cfg}, "
              f"coverage % median {cov.median():.3g}")


if __name__ == "__main__":
    main()
