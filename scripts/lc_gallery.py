#!/usr/bin/env python3
"""Light-curve gallery for one coverage arm: dwarfs with a dynamo, spread in Ro/Ro_sun.

  python scripts/lc_gallery.py dataset_cov_act_fb0600 --out docs/lc_gallery_fb0600.png
"""
import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 14, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 1.5})
ROSSBY_SUN = 1.97  # same as create_data (Cranmer & Saar 2011 tau_c, as in Masuda 2022)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default="docs/spot_coverage_calibration_act.csv",
                    help="per-star table from coverage_calibration_report.py")
    a = ap.parse_args()
    u = pd.read_csv(f"{a.arm}/simulation_properties.csv")
    rel = pd.read_csv(a.csv)
    rel = rel[rel.arm == a.arm].set_index("sim_id")
    d = rel[(rel.evo == "dwarf") & (rel.rper_ppt > 0)].sort_values("ro_rel")
    pick = d.iloc[np.linspace(0, len(d) - 1, a.n).round().astype(int)]
    want = set(pick.index)
    lc = pd.concat(df[df.sim_id.isin(want)] for df in
                   (pd.read_parquet(f) for f in glob.glob(f"{a.arm}/lc/Kepler/*.pqt")))
    fig, axes = plt.subplots(a.n, 1, figsize=(14, 1.9 * a.n), sharex=True)
    for ax, (sid, r) in zip(axes, pick.iterrows()):
        g = lc[lc.sim_id == sid]
        ax.plot(g.time, 1e3 * (g.flux / g.flux.median() - 1), color="#264653", lw=0.8)
        ax.text(0.005, 0.95, f"Ro/Ro$_\\odot$ {r.ro_rel:.2f}   P {r.period:.1f} d   "
                f"{r.teff:.0f} K   R$_{{\\rm per}}$ {r.rper_ppt:.2g} ppt",
                transform=ax.transAxes, va="top", fontsize=12)
        ax.set_ylabel("ppt")
    axes[-1].set_xlabel("time [d]")
    fig.tight_layout(h_pad=0.3)
    fig.savefig(a.out, dpi=130)


if __name__ == "__main__":
    main()
