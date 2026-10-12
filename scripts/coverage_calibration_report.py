#!/usr/bin/env python3
"""Spot-coverage calibration readout (docs/spot_coverage.md).

For each arm (dataset_cov_legacy, dataset_cov_fXXXX):
  R_per percentiles of main-sequence stars below 6500 K vs McQuillan et al. 2014
  (950 / 5600 / 22700 ppm), Sarle's bimodality coefficient of log R_per, and
  R_per vs Ro/Ro_sun for 0.9-1.1 M_sun stars vs Masuda 2022 (~1% plateau for
  0.2 < Ro/Ro_sun < 0.4, steep decline above ~0.4).

  python scripts/coverage_calibration_report.py dataset_cov_legacy dataset_cov_f0030 \
      dataset_cov_f0100 dataset_cov_f0290 --out docs/spot_coverage_calibration
  -> .md (tables), .csv (per star), .png (figure)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faculae_smoke_report import load, norm, r_per, md_table  # noqa: E402
from Dynamo.create_data import ROSSBY_SUN  # noqa: E402

MCQ = (0.95, 5.6, 22.7)          # ppt, 5th / 50th / 95th percentile
RO_BINS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 3.0]

plt.rcParams.update({"font.size": 15, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 1.6, "xtick.major.width": 1.4, "ytick.major.width": 1.4})


def bimodality(x):
    """Sarle's bimodality coefficient; > 0.555 (uniform) suggests two modes."""
    n = len(x)
    g, k = stats.skew(x), stats.kurtosis(x)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def arm_table(d):
    lc, cfg = load(d)
    sims = pd.read_csv(f"{d}/simulation_properties.csv").set_index("Simulation Number")
    rows = []
    for sid, g in lc.groupby("sim_id"):
        g = g.sort_values("time")
        sp, ap = cfg[sid]["stellar_params"], cfg[sid]["activity_params"]
        r = sims.loc[sid]
        if not bool(r["Has Dynamo"]):
            continue
        y = norm(g.flux.to_numpy(), sp)
        rows.append(dict(sim_id=sid, rper_ppt=1e3 * r_per(y, g.time.to_numpy(), r.Period),
                         period=r.Period, teff=r.Teff, mass=r.mass, evo=r["Evolutionary Class"],
                         ro_rel=r.Rossby / ROSSBY_SUN, max_area=r["Spot Max Area"],
                         coverage_pct=100 * ap.get("spot_coverage", np.nan),
                         n_spots=ap.get("n_spots", np.nan), boost=ap.get("spot_area_boost", 1.0),
                         ff_sp_pct=ap["mean_ff_sp"]))
    t = pd.DataFrame(rows)
    t["arm"] = Path(d).name
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out", default="docs/spot_coverage_calibration")
    a = ap.parse_args()
    t = pd.concat([arm_table(d) for d in a.arms], ignore_index=True)
    t.to_csv(f"{a.out}.csv", index=False)

    summ, ro_rows = [], []
    for arm, g in t.groupby("arm", sort=False):
        ms = g[(g.evo == "dwarf") & (g.teff < 6500) & (g.rper_ppt > 0)]
        p = np.percentile(ms.rper_ppt, [5, 50, 95])
        summ.append({"arm": arm, "N (MS, <6500 K)": len(ms),
                     "R_per 5% [ppt]": f"{p[0]:.2f}", "50%": f"{p[1]:.2f}", "95%": f"{p[2]:.1f}",
                     "bimodality (log R_per)": f"{bimodality(np.log10(ms.rper_ppt)):.2f}",
                     "spots median": f"{g.n_spots.median():.0f}", "spots max": f"{g.n_spots.max():.0f}",
                     "boosted": f"{(g.boost > 1).mean():.0%}",
                     "rho(R_per, Ro)": f"{stats.spearmanr(g.rper_ppt, g.ro_rel)[0]:+.2f}",
                     "rho(R_per, area)": f"{stats.spearmanr(g.rper_ppt, g.max_area)[0]:+.2f}"})
        sm = g[g.mass.between(0.9, 1.1)]
        for lo, hi in zip(RO_BINS[:-1], RO_BINS[1:]):
            s = sm[(sm.ro_rel >= lo) & (sm.ro_rel < hi)]
            ro_rows.append({"arm": arm, "Ro/Ro_sun": f"{lo}-{hi}", "n": len(s),
                            "R_per median [ppt]": f"{s.rper_ppt.median():.2f}" if len(s) else "-"})
    summ.append({"arm": "McQuillan 2014", "R_per 5% [ppt]": MCQ[0], "50%": MCQ[1], "95%": MCQ[2]})
    ro = pd.DataFrame(ro_rows).pivot(index="Ro/Ro_sun", columns="arm", values="R_per median [ppt]")
    ro = ro.loc[[f"{lo}-{hi}" for lo, hi in zip(RO_BINS[:-1], RO_BINS[1:])]].reset_index()
    with open(f"{a.out}.md", "w") as f:
        f.write("# Spot-coverage calibration readout\n\n"
                "R_per of clean 1000-d light curves (faculae on, no noise). Main sequence, Teff < 6500 K.\n\n")
        f.write(md_table(pd.DataFrame(summ).fillna("")) + "\n\n")
        f.write("R_per median vs Ro/Ro_sun, 0.9-1.1 M_sun (Masuda 2022: ~10 ppt plateau at 0.2-0.4, "
                "steep decline above ~0.4)\n\n" + md_table(ro.fillna("-")) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["k", "tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]
    bins = np.logspace(-2, 2, 33)
    for c, (arm, g) in zip(colors, t.groupby("arm", sort=False)):
        ms = g[(g.evo == "dwarf") & (g.teff < 6500) & (g.rper_ppt > 0)]
        ax1.hist(ms.rper_ppt, bins=bins, histtype="step", lw=2.5, color=c, label=arm.split("_")[-1])
        sm = g[g.mass.between(0.9, 1.1) & (g.rper_ppt > 0)]
        ax2.scatter(sm.ro_rel, sm.rper_ppt, s=18, color=c, alpha=0.35)
        med = [sm[(sm.ro_rel >= lo) & (sm.ro_rel < hi)].rper_ppt.median() for lo, hi in zip(RO_BINS[:-1], RO_BINS[1:])]
        ax2.plot(np.sqrt(np.array(RO_BINS[:-1]) * RO_BINS[1:]), med, "-o", color=c, lw=2.5)
    for v in MCQ:
        ax1.axvline(v, color="0.5", ls="--", lw=1.6)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"R$_{\rm per}$ [ppt]")
    ax1.set_ylabel("stars")
    ax1.legend(frameon=False, fontsize=13)
    ax2.axhspan(7, 13, xmin=0, xmax=1, color="0.9", zorder=0)
    ax2.axvspan(0.2, 0.4, color="0.8", alpha=0.5, zorder=0)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"Ro / Ro$_\odot$")
    ax2.set_ylabel(r"R$_{\rm per}$ [ppt]  (0.9-1.1 M$_\odot$)")
    for ax, lab in zip([ax1, ax2], "ab"):
        ax.text(-0.12, 1.02, lab, transform=ax.transAxes, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{a.out}.png", dpi=150)
    print(open(f"{a.out}.md").read())


if __name__ == "__main__":
    main()
