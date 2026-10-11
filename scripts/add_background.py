"""Add stellar (granulation + oscillations) and instrumental background to the clean
light curves of a Dynamo dataset (see Dynamo/background.py).

Reads   DATASET/lc/<INSTR>/chunk_*.pqt  + DATASET/simulation_properties.csv
Writes  DATASET/lc/<INSTR>_bg[_r<k>]/chunk_*.pqt   (time, flux, sim_id; same chunks)
Magnitudes are survey-like by default (--mag_mode survey), one per star.
        DATASET/background_params_<INSTR>.csv       (numax, dnu, a_gran, A_max, mag, sigma)

The clean light curves are left untouched. Realisation k of star i uses the seed
(Seed_i, k, instrument), so the output is reproducible and independent per star.

    python scripts/add_background.py DATASET [--instrument Kepler] [--realizations 1]
"""
import argparse
import glob
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dynamo.background import add_background, star_background_params, survey_mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--instrument", default="Kepler", choices=["Kepler", "TESS"])
    ap.add_argument("--lc_name", default=None, help="LC folder to read (default: instrument)")
    ap.add_argument("--realizations", type=int, default=1)
    ap.add_argument("--components", default="granulation,oscillations,white")
    ap.add_argument("--mag_mode", default="survey", choices=["survey", "distance"],
                    help="survey: magnitude drawn per star from a survey-like distribution "
                         "(default); distance: from L, Teff and the table Distance")
    args = ap.parse_args()

    d = args.dataset_dir
    lc_name = args.lc_name or args.instrument
    sims = pd.read_csv(f"{d}/simulation_properties.csv").set_index("Simulation Number")
    comps = tuple(c.strip() for c in args.components.split(","))
    instr_key = zlib.crc32(args.instrument.encode())
    chunks = sorted(glob.glob(f"{d}/lc/{lc_name}/chunk_*.pqt"))
    if not chunks:
        raise SystemExit(f"no light curves in {d}/lc/{lc_name}")

    def base_seed(sid):
        return int(sims.loc[sid, "Seed"]) if "Seed" in sims.columns else int(sid)

    # one magnitude per star (shared by all realisations)
    mags = {sid: (survey_mag(np.random.default_rng([base_seed(sid), instr_key, 7]), args.instrument)
                  if args.mag_mode == "survey" else None) for sid in sims.index}

    rows = []
    for sid, r in sims.iterrows():
        p = star_background_params(r, args.instrument, mag=mags[sid])
        rows.append(dict(sim_id=sid, numax=p["numax"], dnu=p["dnu"], a_gran=p["gran"][0][0],
                         b1=p["gran"][0][1], b2=p["gran"][1][1], A_max=p["amax"],
                         linewidth=p["gamma"], mag=p["mag"], sigma_point_ppm=p["sigma_point"]))
    pd.DataFrame(rows).to_csv(f"{d}/background_params_{args.instrument}.csv", index=False)

    for k in range(args.realizations):
        out_dir = f"{d}/lc/{lc_name}_bg" + (f"_r{k}" if args.realizations > 1 else "")
        os.makedirs(out_dir, exist_ok=True)
        for ch in chunks:
            lc = pd.read_parquet(ch)
            parts = []
            for sid, g in lc.groupby("sim_id", sort=False):
                g = g.sort_values("time")
                seed = [base_seed(sid), k, instr_key]
                flux, _, _ = add_background(g["time"].values, g["flux"].values, sims.loc[sid],
                                            args.instrument, seed=seed, mag=mags[sid],
                                            components=comps)
                parts.append(pd.DataFrame({"time": g["time"].values, "flux": flux, "sim_id": sid}))
            pd.concat(parts).to_parquet(f"{out_dir}/{os.path.basename(ch)}")
        print(f"realization {k}: {len(chunks)} chunks -> {out_dir}")


if __name__ == "__main__":
    main()
