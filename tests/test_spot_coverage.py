"""Spot coverage vs Rossby number sets the number of spots (docs/spot_coverage.md)."""
import json
import logging
import tempfile

import numpy as np

from helpers import make_star
from Dynamo import spots
from Dynamo.create_data import SPOT_COVERAGE_DEFAULT, ROSSBY_SUN, spot_coverage_law

_CACHE = {}


def _table(config, seed=5, n=3000):
    key = (json.dumps(config, sort_keys=True), seed, n)
    if key not in _CACHE:
        from Dynamo.create_data import generate_simdata
        np.random.seed(2)
        with tempfile.TemporaryDirectory() as d:
            sims = generate_simdata(d, n, logging.getLogger("t"), base_seed=seed, coverage_config=config)
            pop = json.load(open(f"{d}/population.json"))
        _CACHE[key] = (sims, pop)
    return _CACHE[key]


def test_mean_area_matches_emerged_spots():
    np.random.seed(0)
    g = spots.SpotsGenerator(max_area=20.0)
    reg = g.emerge_regions(ndays=1000, nspots=20000, butterfly=False, min_lat=5, max_lat=30)
    r = np.deg2rad(np.sqrt(np.asarray(reg["bmax"])))          # radius used by Star.generate_spot_map
    mc = ((180 / np.pi) ** 2 * 2 * np.pi * (1 - np.cos(r))).mean()
    assert abs(mc / spots.mean_spot_area_deg2(20.0) - 1) < 0.02


def test_coverage_round_trip_gaussian_law():
    """Spots at the predicted rate give the requested time-mean coverage."""
    rng = np.random.default_rng(1)
    ndays, f, m, te, td = 3000, 0.01, 10.0, 2.0, 60.0
    n = rng.poisson(spots.nspots_for_coverage(f, ndays, m, te, td))
    t0 = rng.uniform(0, ndays, n)
    a = spots.mean_spot_area_deg2(m)
    t = np.linspace(300, ndays - 300, 2000)                    # away from the edges
    dt = t[:, None] - t0[None, :]
    fac = np.where(dt < 0, np.exp(-dt ** 2 / te ** 2), np.exp(-dt ** 2 / td ** 2))
    cov = (a * fac ** 2).sum(1) / spots.SURFACE_DEG2           # area ~ radius^2
    assert abs(cov.mean() / f - 1) < 0.05


def test_max_spots_keeps_coverage():
    m, boost = spots.area_for_max_spots(0.25, 1000, 3.0, 2.0, 50.0, 3000)
    assert boost > 1
    assert abs(spots.nspots_for_coverage(0.25, 1000, m, 2.0, 50.0) / 3000 - 1) < 1e-3
    m2, b2 = spots.area_for_max_spots(0.001, 1000, 3.0, 2.0, 50.0, 3000)
    assert b2 == 1.0 and m2 == 3.0


def test_population_follows_rossby_law():
    sims, pop = _table(None)
    c = pop["spot_coverage"]
    assert c["model"] == "broken" and c["f_break"] == SPOT_COVERAGE_DEFAULT["f_break"]
    f = sims["Spot Coverage pct"].values / 100
    dyn = sims["Has Dynamo"].values.astype(bool)
    assert (f[~dyn] == 0).all() and (f[dyn] > 0).all()
    x = sims.Rossby.values / ROSSBY_SUN
    plateau = dyn & (x > c["ro_flat"]) & (x < c["ro_break"])
    assert abs(np.median(f[plateau]) / c["f_break"] - 1) < 0.1
    sat = dyn & (x < c["ro_sat"])
    assert abs(np.median(f[sat]) / (c["f_break"] * (c["ro_sat"] / c["ro_flat"]) ** -c["slope"]) - 1) < 0.1
    resid = np.log10(f[dyn]) - np.log10(spot_coverage_law(x[dyn], c))
    assert abs(resid.std() - c["scatter_dex"]) < 0.01


def test_broken_law_shape():
    c = SPOT_COVERAGE_DEFAULT
    x = np.array([0.01, c["ro_sat"], c["ro_flat"], 0.35, c["ro_break"], c["ro_tail"], 2 * c["ro_tail"]])
    f = spot_coverage_law(x, c)
    assert f[0] == f[1] and np.isclose(f[2], c["f_break"]) and f[3] == f[4] == c["f_break"]
    assert np.isclose(f[5], c["f_break"] * (c["ro_tail"] / c["ro_break"]) ** -c["slope_unsat"])
    assert np.isclose(f[6], f[5] * 2 ** -c["slope_tail"])
    r = spot_coverage_law(x, dict(c, model="rossby"))
    assert np.isclose(spot_coverage_law(1.0, dict(c, model="rossby")), c["f_sun"]) and r[0] == r[1]


def test_legacy_model_and_other_draws_unchanged():
    new, _ = _table(None)
    old, _ = _table({"model": "legacy"})
    assert old["Spot Coverage pct"].isna().all()
    for col in ["Teff", "Period", "Inclination", "Activity Rate", "Lat Mean", "Spot Max Area"]:
        assert np.array_equal(new[col].values, old[col].values), col


def test_star_spot_count_from_coverage():
    s = make_star(**{"Spot Coverage pct": 1.0, "Spot Max Area": 10.0, "Period": 10.0, "Decay Time": 6.0})
    s.generate_spot_map(1000)
    exp = spots.nspots_for_coverage(0.01, 1000 + s.spot_burn_days, 10.0, 2.0, 60.0)
    assert s.spot_burn_days >= 120 and np.isclose(s.nspots_expected, exp)
    assert abs(len(s.spot_map) - exp) < 5 * np.sqrt(exp)
    assert s.spot_area_boost == 1.0
    # saturated star with small spots: capped count, bigger spots, same coverage
    s = make_star(**{"Spot Coverage pct": 20.0, "Spot Max Area": 3.0, "Period": 1.0})
    s.max_spots = 1000
    s.generate_spot_map(1000)
    assert s.spot_area_boost > 1 and abs(s.nspots_expected * 1000 / (1000 + s.spot_burn_days) - 1000) < 1
    assert abs(len(s.spot_map) * 1000 / (1000 + s.spot_burn_days) - 1000) < 150


def test_realized_coverage_is_stationary():
    """Burn-in: the spot map gives the target coverage from t = 0, also for short runs."""
    for ndays in (60, 600):
        s = make_star(**{"Spot Coverage pct": 2.0, "Spot Max Area": 10.0, "Period": 15.0,
                         "Decay Time": 6.0, "Seed": 7})
        s.generate_spot_map(ndays)
        t0, r0 = s.spot_map[:, 0], np.deg2rad(s.spot_map[:, 4])
        te, td = min(2, 15 * 6 / 10), max(15 * 6, 50)
        t = np.linspace(0, ndays, 400)
        dt = t[:, None] - t0[None, :]
        fac = np.where(dt < 0, np.exp(-dt ** 2 / te ** 2), np.exp(-dt ** 2 / td ** 2))
        area = 2 * np.pi * (1 - np.cos(fac * r0)) / (4 * np.pi)     # cap fraction of the sphere
        cov = area.sum(1)
        assert t0.min() < -td, ndays
        assert abs(cov.mean() / 0.02 - 1) < 0.25, (ndays, cov.mean())
        assert abs(cov[:40].mean() / cov.mean() - 1) < 0.3, (ndays, cov[:40].mean())


def test_legacy_star_without_coverage_column():
    s = make_star(**{"Activity Rate": 50.0})
    s.generate_spot_map(1000)
    assert len(s.spot_map) == 500 and np.isnan(s.nspots_expected)
