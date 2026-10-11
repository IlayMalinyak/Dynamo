#!/usr/bin/env python3
"""Figure for the faculae smoke datasets (paired runs with and without faculae).

(a) example star: spots only, spots + faculae, and the facular signal (difference)
(b) var(facular) / var(spot) vs inclination, rotational component, coloured by period
(c) the same vs rotation period
(d) Kepler-style amplitude R_per with and without faculae

Needs the per-star table written by faculae_smoke_report.py (--out X.md -> X.csv).

  python scripts/faculae_smoke_figure.py dataset_faculae_smoke dataset_faculae_smoke_noQ \
      --table docs/faculae_smoke.csv --out docs/faculae_smoke.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faculae_smoke_report import load, norm

plt.rcParams.update({"font.size": 15, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 1.6, "xtick.major.width": 1.4, "ytick.major.width": 1.4})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("with_q")
    ap.add_argument("no_q")
    ap.add_argument("--table", required=True)
    ap.add_argument("--example", type=int, default=None, help="sim_id for panel a")
    ap.add_argument("--window", type=float, default=120.0, help="days shown in panel a")
    ap.add_argument("--out", default="docs/faculae_smoke.png")
    a = ap.parse_args()
    d = pd.read_csv(a.table)
    lq, cq = load(a.with_q)
    l0, _ = load(a.no_q)

    if a.example is None:   # a slow rotator seen at intermediate inclination, typical activity
        cand = d[(d.period > 15) & d.incl_deg.between(35, 75)]
        cand = cand if len(cand) else d
        a.example = int(cand.iloc[(cand.rper_spot_ppt - cand.rper_spot_ppt.median()).abs().argmin()].sim_id)
    sid = a.example
    sp = cq[sid]["stellar_params"]
    fq = lq[lq.sim_id == sid].sort_values("time")
    f0 = l0[l0.sim_id == sid].sort_values("time")
    t = fq.time.to_numpy()
    yq, y0 = norm(fq.flux.to_numpy(), sp), norm(f0.flux.to_numpy(), sp)
    t0 = t[0] + 0.5 * (t[-1] - t[0] - a.window)
    w = (t >= t0) & (t < t0 + a.window)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t[w] - t0, 1e3 * (y0[w] - 1), color="k", lw=2.2, label="spots")
    ax.plot(t[w] - t0, 1e3 * (yq[w] - 1), color="tab:orange", lw=2.2, label="spots + faculae")
    ax.plot(t[w] - t0, 1e3 * (yq[w] - y0[w]), color="tab:blue", lw=2.2, label="faculae")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("time [d]")
    ax.set_ylabel("flux [ppt]")
    r = d[d.sim_id == sid].iloc[0]
    ax.text(0.01, 0.06, f"P = {r.period:.1f} d   i = {r.incl_deg:.0f}°   Q = {r.Q:.1f}",
            transform=ax.transAxes, fontsize=14)
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0))

    ax_i = fig.add_subplot(gs[1, 0])
    sc = ax_i.scatter(d.incl_deg, d.rot_var_ratio, c=np.log10(d.period), cmap="viridis", s=40)
    ax_i.set_yscale("log")
    ax_i.axhline(1, color="0.4", ls="--", lw=1.4)
    ax_i.set_xlabel("inclination [deg]")
    ax_i.set_ylabel(r"var$_{\rm fac}$ / var$_{\rm spot}$")
    cb = plt.colorbar(sc, ax=ax_i)
    cb.set_label(r"log$_{10}$ P [d]")

    ax_p = fig.add_subplot(gs[1, 1], sharey=ax_i)
    sc = ax_p.scatter(d.period, d.rot_var_ratio, c=d.incl_deg, cmap="plasma", s=40)
    ax_p.set_xscale("log")
    ax_p.axhline(1, color="0.4", ls="--", lw=1.4)
    ax_p.set_xlabel("P$_{\\rm rot}$ [d]")
    cb = plt.colorbar(sc, ax=ax_p)
    cb.set_label("i [deg]")

    ax_a = fig.add_subplot(gs[1, 2])
    bins = np.logspace(np.log10(max(0.1, d[["rper_ppt", "rper_spot_ppt"]].min().min())),
                       np.log10(d[["rper_ppt", "rper_spot_ppt"]].max().max()), 20)
    ax_a.hist(d.rper_spot_ppt, bins=bins, histtype="step", lw=2.5, color="k", label="spots")
    ax_a.hist(d.rper_ppt, bins=bins, histtype="step", lw=2.5, color="tab:orange", label="spots + faculae")
    ax_a.axvspan(3, 8, color="0.85", zorder=0)
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"R$_{\rm per}$ [ppt]")
    ax_a.set_ylabel("stars")
    ax_a.legend(frameon=False, fontsize=12)

    for axx, lab in zip([ax, ax_i, ax_p, ax_a], "abcd"):
        axx.text(-0.06 if axx is ax else -0.2, 1.02, lab, transform=axx.transAxes, fontsize=20,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(a.out, "example sim_id", sid)


if __name__ == "__main__":
    main()
