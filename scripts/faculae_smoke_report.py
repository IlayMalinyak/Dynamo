#!/usr/bin/env python3
"""Smoke-dataset report for the faculae model (Part E of docs/FACULAE_PLAN.md).

Two runs with the same parameter table and per-star seeds, so their spot maps
are identical: one with faculae (Q from the table) and one with Q = 0. Because
the flux is linear in the coverage fractions, flux_Q - flux_noQ is exactly the
facular signal and flux_noQ - 1 the spot signal.

  python scripts/faculae_smoke_report.py dataset_faculae_smoke dataset_faculae_smoke_noQ \
      --out docs/faculae_smoke.md
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(d):
    lc = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/lc/Kepler/chunk_*.pqt")))
    cfg = {}
    for f in sorted(glob.glob(f"{d}/configs/chunk_*.json")):
        cfg.update({int(k): v for k, v in json.load(open(f)).items()})
    return lc, cfg


def norm(flux, sp):
    scale = (sp["radius"] * 6.957e10 / (sp["distance"] * 3.086e18)) ** 2 * 1e26
    return flux / scale


def md_table(df):
    """Markdown table without the tabulate dependency."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:g}" if isinstance(v, float) else str(v) for v in r]) + " |")
    return "\n".join(lines)


def rot_band(x, t, period):
    """Rotational-timescale component: subtract a running mean over 3 rotations."""
    dt = np.median(np.diff(t))
    w = max(3, int(round(3 * period / dt)))
    k = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return x - np.convolve(pad, k, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("with_q")
    ap.add_argument("no_q")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lq, cq = load(a.with_q)
    l0, c0 = load(a.no_q)
    props = pd.read_csv(Path(a.with_q) / "simulation_properties.csv")
    rows = []
    for sid in sorted(set(cq) & set(c0)):
        sp, act = cq[sid]["stellar_params"], cq[sid]["activity_params"]
        if not act.get("has_dynamo", True):
            continue
        fq = lq[lq.sim_id == sid].sort_values("time")
        f0 = l0[l0.sim_id == sid].sort_values("time")
        if len(fq) != len(f0) or len(fq) == 0:
            continue
        t = fq.time.to_numpy()
        yq, y0 = norm(fq.flux.to_numpy(), sp), norm(f0.flux.to_numpy(), sp)
        spot, fac = y0 - 1.0, yq - y0
        P = sp["rotation_period"]
        rs, rf = rot_band(spot, t, P), rot_band(fac, t, P)
        rows.append(dict(
            sim_id=sid, teff=sp["Teff"], period=P, incl_deg=np.rad2deg(sp["inclination"]),
            activity=act["activity_rate"], Q=act["facula_q"],
            amp_ppt=1e3 * (np.quantile(yq, .95) - np.quantile(yq, .05)) / np.median(yq),
            amp_spot_ppt=1e3 * (np.quantile(y0, .95) - np.quantile(y0, .05)) / np.median(y0),
            var_ratio=np.var(fac) / np.var(spot) if np.var(spot) > 0 else np.nan,
            rot_var_ratio=np.var(rf) / np.var(rs) if np.var(rs) > 0 else np.nan,
            mean_ff_sp=act["mean_ff_sp"], mean_ff_fc=act["mean_ff_fc"]))
    d = pd.DataFrame(rows)
    d["fac_dom"] = d.var_ratio > 1
    d["fac_dom_rot"] = d.rot_var_ratio > 1

    def frac(g):
        return pd.Series(dict(n=len(g), fac_dominated=g.fac_dom.mean(), fac_dominated_rot=g.fac_dom_rot.mean(),
                              median_var_ratio=g.var_ratio.median(), median_rot_var_ratio=g.rot_var_ratio.median(),
                              median_amp_ppt=g.amp_ppt.median()))
    out = []
    q = d.amp_ppt.quantile([.05, .25, .5, .75, .95]).round(2)
    q0 = d.amp_spot_ppt.quantile([.05, .25, .5, .75, .95]).round(2)
    out.append(f"# Faculae smoke dataset\n\nStars with a spot dynamo: {len(d)} (of {len(props)} simulated). "
               f"Kepler band, {int(np.ptp(t))} days, noise-free.\n")
    out.append("## Amplitude (q95 - q05, ppt)\n\n| percentile | with faculae | spots only |\n|---|---|---|")
    for p in q.index:
        out.append(f"| {int(p*100)} | {q[p]} | {q0[p]} |")
    out.append("\nKepler reference: median about 5 ppt with a long tail below 1 ppt (McQuillan et al. 2014).\n")
    by_p = d.assign(P_bin=pd.cut(d.period, [0, 15, 25, 1e3], labels=["P < 15 d", "15-25 d", "P > 25 d"])) \
        .groupby("P_bin", observed=True).apply(frac, include_groups=False)
    by_i = d.assign(i_bin=pd.cut(d.incl_deg, [0, 30, 60, 90.01], labels=["i < 30", "30-60", "60-90"])) \
        .groupby("i_bin", observed=True).apply(frac, include_groups=False)
    out.append("## Faculae-dominated fraction (var(facular signal) > var(spot signal))\n")
    out.append("`rot` columns use the rotational-timescale component only (running mean over 3 rotations removed).\n")
    out.append("### By rotation period\n\n" + md_table(by_p.round(3)) + "\n")
    out.append("### By inclination\n\n" + md_table(by_i.round(3)) + "\n")
    out.append(f"Overall faculae-dominated: {d.fac_dom.mean():.3f} (total), {d.fac_dom_rot.mean():.3f} (rotational).\n")
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        d.to_csv(Path(a.out).with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
