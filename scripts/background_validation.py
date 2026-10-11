"""Validation figure for the stellar + instrumental background (Dynamo/background.py).

(a) observed PSD of three stars (dwarf, subgiant, red giant) at Kepler 30-min cadence:
    synthesized (smoothed) vs the input model
(b) 8-hr flicker: logg inferred with Bastien et al. 2016 vs true logg, for the stars of a
    simulation table inside the relation's range (4500 < Teff < 6650 K, 2.5 < logg < 4.6)
(c) 6.5-hr noise vs Kepler magnitude: instrument only (line) and instrument + star (dots)

    python scripts/background_validation.py TABLE.csv --out docs/background_validation.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Dynamo.background import (star_background_params, synthesize_background, point_noise_ppm,
                               survey_mag, granulation_psd, oscillation_psd, oscillation_modes)

plt.rcParams.update({"font.size": 15, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.linewidth": 1.6, "xtick.major.width": 1.4, "ytick.major.width": 1.4})
DT_D = 1765.5 / 86400.0


def bastien_logg(f8_ppt):
    x = np.log10(f8_ppt)
    return 1.3724221 - 3.5002686 * x - 1.6838185 * x ** 2 - 0.37909094 * x ** 3


def f8_ppt(x):
    return np.std(x - uniform_filter1d(x, 16, mode="nearest")) / 1e3


def panel_psd(ax):
    stars = [("dwarf", dict(Teff=5777., logg=4.44, mass=1., R=1., L=1.), "tab:blue"),
             ("subgiant", dict(Teff=5300., logg=3.6, mass=1.3, R=2.9, L=6.0), "tab:green"),
             ("red giant", dict(Teff=4700., logg=2.6, mass=1.3, R=9.5, L=40.), "tab:red")]
    t = np.arange(0, 1000, DT_D)
    for name, row, c in stars:
        p = star_background_params({**row, "Distance": 500.}, mag=12.0)
        bg = synthesize_background(t, p, np.random.default_rng(1))
        x = bg["total"]
        n = len(x)
        f = np.fft.rfftfreq(n, d=DT_D * 86400) * 1e6
        df = f[1]
        pw = np.abs(np.fft.rfft(x)) ** 2 * 2 / (n ** 2 * df)          # one-sided ppm^2/muHz
        sm = uniform_filter1d(pw[1:], 200)        # average power (not log power: that is biased by e^-0.577)
        ax.loglog(f[1:], sm, color=c, lw=2.5, label=name)
        fs = 1e6 / (DT_D * 86400)
        model = (granulation_psd(f, df, fs, DT_D * 86400, p["gran"])
                 + oscillation_psd(f, df, fs, DT_D * 86400,
                                   *oscillation_modes(p["numax"], p["dnu"], p["amax"], p["teff"]))
                 + p["sigma_point"] ** 2 / (fs / 2))
        ax.loglog(f[1:], uniform_filter1d(model[1:], 200), color="k", lw=1.2, ls="--")
    ax.set_xlabel(r"Frequency [$\mu$Hz]")
    ax.set_ylabel(r"PSD [ppm$^2/\mu$Hz]")
    ax.set_xlim(0.2, 283)
    ax.legend(frameon=False, fontsize=13)


def panel_flicker(ax, sims):
    sel = sims[(sims.Teff > 4500) & (sims.Teff < 6650) & (sims.logg > 2.5) & (sims.logg < 4.6)]
    t = np.arange(0, 90, DT_D)
    true, inferred = [], []
    for sid, r in sel.iterrows():
        p = star_background_params(r, mag=12.0)
        bg = synthesize_background(t, p, np.random.default_rng(int(sid)),
                                   components=("granulation", "oscillations"))
        true.append(r.logg)
        inferred.append(bastien_logg(f8_ppt(bg["granulation"] + bg["oscillations"])))
    true, inferred = np.array(true), np.array(inferred)
    ax.scatter(true, inferred, s=30, color="tab:purple", alpha=0.8)
    lim = [2.4, 4.8]
    ax.plot(lim, lim, "k--", lw=1.4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"true $\log g$")
    ax.set_ylabel(r"$\log g$ from $F_8$ (Bastien+16)")
    d = inferred - true
    ax.text(0.05, 0.92, f"bias {np.mean(d):+.2f}  rms {np.std(d):.2f}  N={len(d)}",
            transform=ax.transAxes, fontsize=13)
    return np.mean(d), np.std(d), len(d)


def panel_noise(ax, sims):
    kp = np.linspace(9, 16, 100)
    ax.semilogy(kp, point_noise_ppm(kp) / np.sqrt(13), "k-", lw=2.5, label="instrument")
    t = np.arange(0, 90, DT_D)
    mags, tot, logg = [], [], []
    for sid, r in sims.iterrows():
        m = survey_mag(np.random.default_rng([int(sid), 7]))
        p = star_background_params(r, mag=m)
        x = synthesize_background(t, p, np.random.default_rng(int(sid)))["total"]
        nb = len(x) // 13
        mags.append(m); logg.append(r.logg)
        tot.append(x[: nb * 13].reshape(nb, 13).mean(1).std())
    sc = ax.scatter(mags, tot, c=logg, cmap="viridis", s=30)
    plt.colorbar(sc, ax=ax, label=r"$\log g$")
    ax.set_xlabel(r"$K_p$")
    ax.set_ylabel("6.5-hr noise [ppm]")
    ax.legend(frameon=False, fontsize=13, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--out", default="docs/background_validation.png")
    args = ap.parse_args()
    sims = pd.read_csv(args.table).set_index("Simulation Number")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    panel_psd(axes[0])
    bias, rms, n = panel_flicker(axes[1], sims)
    panel_noise(axes[2], sims)
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.15, 1.02, lab, transform=ax.transAxes, fontsize=20, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"flicker check: bias {bias:+.3f} rms {rms:.3f} N={n}; saved {args.out}")


if __name__ == "__main__":
    main()
