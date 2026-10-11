#!/usr/bin/env python3
"""Write one simulation table twice: with faculae and with Q = 0.

Both copies share every parameter and per-star Seed, so the two datasets have
identical spot maps; flux(with) - flux(noQ) is then exactly the facular signal.
Run create_data on each directory afterwards (it reuses an existing table).

  python scripts/make_paired_tables.py --n 200 --out dataset_faculae_smoke --base-seed 100000
  -> dataset_faculae_smoke/ and dataset_faculae_smoke_noQ/
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dynamo.create_data import generate_simdata  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="dataset_faculae_smoke")
    ap.add_argument("--base-seed", type=int, default=100000)
    ap.add_argument("--table-seed", type=int, default=2026, help="seed for drawing the table itself")
    ap.add_argument("--q-activity-dep", action="store_true")
    a = ap.parse_args()
    out, out0 = Path(a.out), Path(f"{a.out}_noQ")
    for d in (out, out0):
        if (d / "simulation_properties.csv").exists():
            sys.exit(f"{d}/simulation_properties.csv exists; refusing to overwrite")
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.table_seed)
    generate_simdata(str(out), a.n, logging.getLogger("tables"), base_seed=a.base_seed,
                     sim_name="dataset", q_activity_dep=a.q_activity_dep)
    t = pd.read_csv(out / "simulation_properties.csv")
    t["Facula Q"] = 0.0
    t.to_csv(out0 / "simulation_properties.csv", index=False, float_format="%5.4f")
    print(f"wrote {out} and {out0}: {len(t)} stars, {int(t['Has Dynamo'].sum())} with a dynamo, "
          f"{int((t.n_planets > 0).sum())} planet hosts")


if __name__ == "__main__":
    main()
