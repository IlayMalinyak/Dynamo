"""Clean vs background-added light curves (scripts/add_background.py output).

One row per star: full light curve and a zoom, clean (black) and with background.
    python scripts/plot_background_effect.py DATASET [--ids 0 1 2] [--zoom 2]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 15, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 1.6, "xtick.major.width": 1.4, "ytick.major.width": 1.4})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--instrument", default="Kepler")
    ap.add_argument("--ids", type=int, nargs="+", default=None)
    ap.add_argument("--zoom", type=float, default=2.0, help="zoom window [days]")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.dataset_dir)
    clean = pd.concat(pd.read_parquet(f) for f in sorted((d / "lc" / args.instrument).glob("chunk_*.pqt")))
    noisy = pd.concat(pd.read_parquet(f) for f in sorted((d / "lc" / f"{args.instrument}_bg").glob("chunk_*.pqt")))
    sims = pd.read_csv(d / "simulation_properties.csv").set_index("Simulation Number")
    bgp = pd.read_csv(d / f"background_params_{args.instrument}.csv").set_index("sim_id")
    ids = args.ids or sorted(clean.sim_id.unique())[:3]

    fig, axes = plt.subplots(len(ids), 2, figsize=(18, 3.6 * len(ids)),
                             gridspec_kw={"width_ratios": [3, 1]}, squeeze=False)
    for row, sid in zip(axes, ids):
        c = clean[clean.sim_id == sid].sort_values("time")
        n = noisy[noisy.sim_id == sid].sort_values("time")
        med = np.median(c.flux)
        yc, yn = 1e3 * (c.flux / med - 1), 1e3 * (n.flux.values / med - 1)
        s, b = sims.loc[sid], bgp.loc[sid]
        label = (f"Teff {s.Teff:.0f} K  logg {s.logg:.2f}  Kp {b.mag:.1f}"
                 + ("" if s["Has Dynamo"] else "  no spots"))
        for ax, sel in [(row[0], slice(None)), (row[1], (c.time.values - c.time.values[0]) < args.zoom)]:
            t = c.time.values[sel]
            ax.plot(t, yn[sel], ".", ms=3 if ax is row[0] else 6, color="tab:orange", alpha=0.7)
            ax.plot(t, yc.values[sel], "k-", lw=2)
            ax.set_ylabel("flux [ppt]")
        row[0].text(0.01, 0.9, label, transform=row[0].transAxes, fontsize=14)
    for ax in axes[-1]:
        ax.set_xlabel("time [d]")
    fig.tight_layout()
    out = args.out or str(d / "images" / "background_effect.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
