#!/usr/bin/env python3
"""How large is the spot-on-grid discretisation artifact, and how does it scale?

Single constant spot, solar-like star, Kepler band, one rotation, no noise.

* Pole-on (i = 0): a spot circles the line of sight at fixed mu, so the TRUE
  light curve is flat. Any modulation is grid error. Reported as
  ptp(flux) / depth, where depth = 1 - mean(flux).
* i = 60 deg: error relative to a 120-ring exact-coverage reference,
  rms(f_N - f_ref) / ptp(f_ref).

Both coverage schemes are run: exact (default) and the original approximation.

  python scripts/grid_artifact.py --out docs/grid_artifact --n-cpu 8
"""
import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

RINGS = [10, 15, 20, 30, 40, 60, 80]
RADII = [2.0, 5.0, 10.0]
COLATS = [10.0, 30.0, 60.0]
REF_RINGS = 120
NTIME = 400


def light_curve(incl, colat, rad, rings, exact=True):
    from helpers import make_star, flux_scale
    s = make_star(Inclination=np.deg2rad(incl))
    s.conf_file.set("general", "wavelength_lower_limit", "4000")
    s.conf_file.set("general", "wavelength_upper_limit", "9500")
    s.spot_T_contrast = 1500.0
    s.facular_area_ratio = 0.0
    s.spots_evo_law = "constant"
    s.spot_map = np.zeros((1, 16))
    s.spot_map[0, :5] = [-1.0, 1e6, colat, 0.0, rad]
    s.reference_time = 0.0
    s.n_grid_rings = rings
    s.exact_coverage = exact
    t = np.linspace(0, s.rotation_period, NTIME, endpoint=False)
    t0 = time.time()
    s.compute_forward(t=t)
    return np.asarray(s.results["lc"], float) / flux_scale(s), time.time() - t0


def job(args):
    kind, incl, colat, rad, rings, exact = args
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        f, dt = light_curve(incl, colat, rad, rings, exact)
    return dict(kind=kind, coverage="exact" if exact else "approx", incl=incl, colat=colat, rad=rad,
                rings=rings, seconds=dt, flux=f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/grid_artifact")
    ap.add_argument("--n-cpu", type=int, default=8)
    a = ap.parse_args()
    jobs = [("pole_on", 0.0, c, r, n, e) for c, r, n, e in itertools.product(COLATS, RADII, RINGS, (True, False))]
    jobs += [("i60", 60.0, 60.0, r, n, e) for r, n, e in itertools.product(RADII, RINGS, (True, False))]
    jobs += [("i60", 60.0, 60.0, r, REF_RINGS, True) for r in RADII]
    with Pool(a.n_cpu) as pool:
        res = pool.map(job, jobs, chunksize=1)
    rows = []
    ref = {r["rad"]: r["flux"] for r in res if r["kind"] == "i60" and r["rings"] == REF_RINGS}
    res = [r for r in res if r["rings"] != REF_RINGS]
    for r in res:
        f = r.pop("flux")
        depth = 1.0 - f.mean()
        if r["kind"] == "pole_on":
            r.update(depth_ppt=1e3 * depth, spurious_ptp_ppt=1e3 * np.ptp(f),
                     artifact_frac=np.ptp(f) / depth if depth > 0 else np.nan)
        else:
            fr = ref[r["rad"]]
            r.update(depth_ppt=1e3 * depth, true_ptp_ppt=1e3 * np.ptp(fr),
                     artifact_frac=np.sqrt(np.mean((f - fr) ** 2)) / np.ptp(fr))
        rows.append(r)
    d = pd.DataFrame(rows).sort_values(["kind", "coverage", "colat", "rad", "rings"])
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out.with_suffix(".csv"), index=False)
    pd.set_option("display.width", 200)
    print(d.drop(columns=["kind"]).round(4).to_string(index=False))
    plot(d, out.with_suffix(".png"))


def plot(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "font.weight": "bold", "axes.labelweight": "bold",
                         "axes.linewidth": 1.6})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {2.0: "#1f6fb4", 5.0: "#d9822b", 10.0: "#2a9d55"}
    if "coverage" not in d:
        d = d.assign(coverage="approx")
    for cov, alpha, mk in (("approx", 0.3, "s"), ("exact", 1.0, "o")):
        po = d[(d.kind == "pole_on") & (d.coverage == cov)]
        for (c, r), g in po.groupby(["colat", "rad"]):
            ls = {10.0: "-", 30.0: "--", 60.0: ":"}[c]
            axes[0].plot(g.rings, g.artifact_frac, ls, color=colors[r], lw=2.5, marker=mk, alpha=alpha)
        i60 = d[(d.kind == "i60") & (d.rings < REF_RINGS) & (d.coverage == cov)]
        for r, g in i60.groupby("rad"):
            axes[1].plot(g.rings, g.artifact_frac, color=colors[r], lw=2.5, marker=mk, alpha=alpha)
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(RINGS))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("grid rings")
        ax.axvline(30, color="0.5", lw=1.5, ls="--")
    axes[0].set_ylabel("artifact / signal")
    axes[0].text(0.97, 0.94, "i = 0°", transform=axes[0].transAxes, ha="right", va="top", fontweight="bold")
    axes[1].text(0.97, 0.94, "i = 60°", transform=axes[1].transAxes, ha="right", va="top", fontweight="bold")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=2.5, label=f"r = {r:g}°") for r, c in colors.items()]
    h += [Line2D([], [], color="0.3", lw=2.5, ls=ls, label=f"colat {c:g}°")
          for c, ls in ((10.0, "-"), (30.0, "--"), (60.0, ":"))]
    axes[0].legend(handles=h, fontsize=10, frameon=False, loc="center left", ncol=2)
    h2 = [Line2D([], [], color="0.2", lw=2.5, marker="o", label="exact"),
          Line2D([], [], color="0.2", lw=2.5, marker="s", alpha=0.3, label="approx")]
    axes[1].legend(handles=h2, fontsize=11, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        out = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "docs/grid_artifact")
        plot(pd.read_csv(out.with_suffix(".csv")), out.with_suffix(".png"))
    else:
        main()
