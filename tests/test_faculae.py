"""Part B tests: facular brightness with centre-to-limb variation.

Single hand-placed features on a solar-like star (5800 K, logg 4.5, [Fe/H] 0),
Kepler band, no noise. Requires the PHOENIX SpecInt models.
"""
from pathlib import Path

import numpy as np
import pytest

from helpers import have_phoenix, make_star, single_feature_lc, flux_scale

pytestmark = pytest.mark.skipif(not have_phoenix(), reason="PHOENIX SpecInt models not installed")

DATA = Path(__file__).parent / "data"
TINY = 0.01  # deg: negligible spot radius for facula-only features


def _lc(incl_deg=90.0, Q=0.0, rad=5.0, colat=90.0, **over):
    s = make_star(Inclination=np.deg2rad(incl_deg), **over)
    s.spot_T_contrast = 1500.0
    s.facular_area_ratio = Q
    return single_feature_lc(s, colat_deg=colat, rad_deg=rad)


def _facula_only(incl_deg=90.0, colat=90.0, fac_rad=10.0):
    # constant law: facular outer radius = rad * sqrt(1 + Q)
    Q = (fac_rad / TINY) ** 2 - 1
    return _lc(incl_deg, Q=Q, rad=TINY, colat=colat)


def test_B0_q0_reproduces_master():
    from make_reference_lc import run_scenario
    ref = np.load(DATA / "ref_lc_noQ.npz")
    t, f = run_scenario()
    assert np.allclose(t, ref["t"])
    assert np.allclose(f, ref["lc"], rtol=0, atol=1e-12), np.abs(f - ref["lc"]).max()


def test_B1_facula_is_limb_brightened():
    t, spot = _lc(90.0, Q=0.0, rad=5.0)
    _, fac = _facula_only(90.0)
    P = t[-1] - t[0]
    t_centre = t[np.argmin(spot)]                       # central-meridian crossing
    assert fac.min() >= 1.0 - 1e-9, "a facula alone must never darken the star"
    assert fac.max() > 1.0 + 1e-5, "facula produced no brightening"
    i_c = np.argmin(np.abs(t - t_centre))
    assert fac[i_c] < fac.max(), "brightening should not peak at disk centre"
    # Projected facular brightening ~ mu * dT(mu): with the Meunier law this
    # peaks at mu ~ 0.45 (~63 deg from centre, 0.175 rotations); limb darkening
    # of the underlying intensity pulls the peak inward (~0.15 rotations).
    dphase = np.abs(((t[np.argmax(fac)] - t_centre) / P + 0.5) % 1 - 0.5)
    print(f"facular maximum {dphase:.3f} rotations from central crossing")
    assert dphase > 0.10, f"facular maximum at {dphase:.3f} rotations from centre"
    # two maxima: one on each side of the central crossing
    vis = fac > 1.0 + 1e-7
    before = vis & (t < t_centre)
    after = vis & (t > t_centre)
    if before.any() and after.any():
        assert fac[before].max() > fac[i_c] and fac[after].max() > fac[i_c]


def test_B2_spot_with_faculae_dip_and_shoulders():
    _, f = _lc(90.0, Q=3.0, rad=5.0)
    assert f.min() < 1.0 < f.max()


def test_B3_facula_to_spot_ratio_rises_at_low_inclination():
    ratio = {}
    for inc in (90.0, 30.0):
        _, spot = _lc(inc, Q=0.0, rad=5.0)
        _, fac = _facula_only(inc)
        ratio[inc] = np.ptp(fac) / np.ptp(spot)
    print("ptp(facula)/ptp(spot):", ratio)
    assert ratio[30.0] > ratio[90.0]


def test_B4_faculae_outlive_spots():
    # pole-on star, near-polar feature: always visible, no rotational modulation
    s = make_star(Inclination=0.0, Period=10.0, **{"Decay Time": 6.0, "Facula Q": 3.0,
                                                   "Facula Lifetime Ratio": 3.0})
    s.spots_evo_law = "gaussian"
    s.spot_map = np.zeros((1, 16))
    s.spot_map[0, :5] = [0.0, 1e6, 10.0, 0.0, 5.0]
    s.reference_time = 0.0
    s.n_grid_rings = 30
    t = np.linspace(0, 200, 401)
    s.compute_forward(t=t)
    ff_sp, ff_fc = np.asarray(s.results["ff_sp"]), np.asarray(s.results["ff_fc"])
    k = np.argmin(np.abs(t - 150))       # tau_decay = max(6*10, 50) = 60 d; tau_fac = 180 d
    assert ff_sp[k] < 0.01 * ff_sp.max()
    assert ff_fc[k] > 0.1 * ff_fc.max()


def test_B5_area_fractions_conserved():
    s = make_star(Inclination=np.deg2rad(60.0), Period=10.0, Shear=0.1,
                  **{"Decay Time": 3.0, "Facula Q": 3.0})
    from make_reference_lc import SPOTS
    s.spot_T_contrast = 1500.0
    s.spots_evo_law = "gaussian"
    s.spot_map = np.zeros((len(SPOTS), 16))
    s.spot_map[:, :5] = SPOTS
    s.reference_time = 0.0
    s.n_grid_rings = 20
    s.compute_forward(t=np.linspace(0, 40, 400))
    r = s.results
    tot = np.asarray(r["ff_ph"]) + r["ff_sp"] + r["ff_fc"] + r["ff_pl"]
    assert np.allclose(tot, 100.0, atol=1e-6)
    assert (np.asarray(r["ff_ph"]) >= 0).all() and (np.asarray(r["ff_fc"]) > 0).any()


def test_B6_generated_map_has_facular_columns():
    s = make_star(Seed=3, **{"Facula Q": 2.0, "Facula Lifetime Ratio": 4.0})
    s.generate_spot_map(ndays=200)
    m = s.spot_map
    assert m.shape[1] == 16 and len(m) > 0
    assert (m[:, 14] > 0).all() and abs(np.median(m[:, 14]) / 2.0 - 1) < 0.3
    assert np.allclose(m[:, 15], max(s.rotation_period * s.spots_decay_time, 50) * 4.0)
