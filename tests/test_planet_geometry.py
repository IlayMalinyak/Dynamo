"""Spin-orbit geometry of planet hosts: stellar inclination consistent with the orbit."""
import logging
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import kstest

from helpers import make_star

_CACHE = {}


def _sims():
    if "s" not in _CACHE:
        from Dynamo.create_data import generate_simdata
        np.random.seed(5)
        with tempfile.TemporaryDirectory() as d:
            sims = generate_simdata(d, 6000, logging.getLogger("t"), sim_name="pg")
            back = pd.read_csv(f"{d}/simulation_properties.csv")
        _CACHE["s"], _CACHE["back"] = sims, back
    return _CACHE["s"], _CACHE["back"]


def test_obliquity_identity_every_planet():
    sims, _ = _sims()
    hosts = sims[sims.n_planets > 0]
    assert len(hosts) > 300
    for _, r in hosts.iterrows():
        istar = r["Inclination"]
        assert 0.0 <= istar <= np.pi / 2 + 1e-12
        for pl in r["planets"]:
            lhs = np.cos(pl["obliquity"])
            rhs = (np.cos(istar) * np.cos(pl["orbit_inclination"])
                   + np.sin(istar) * np.sin(pl["orbit_inclination"]) * np.cos(pl["spin_orbit_angle"]))
            assert abs(lhs - rhs) < 1e-9


def test_cool_hosts_edge_on_hot_hosts_broader():
    sims, _ = _sims()
    hosts = sims[sims.n_planets > 0]
    cool, hot = hosts[hosts.Teff < 6250], hosts[hosts.Teff > 6250]
    i_cool = np.rad2deg(cool["Inclination"])
    print(f"cool hosts n={len(cool)} median i*={i_cool.median():.1f} frac>60={np.mean(i_cool > 60):.2f}; "
          f"hot n={len(hot)} median i*={np.rad2deg(hot['Inclination']).median():.1f}")
    assert i_cool.median() > 75 and np.mean(i_cool > 60) > 0.8
    lam_c = np.abs(np.rad2deg(cool["planet_spin_orbit_angle"]))
    lam_h = np.abs(np.rad2deg(hot["planet_spin_orbit_angle"]))
    assert lam_h.median() > lam_c.median()
    assert np.mean(np.rad2deg(hot["Obliquity"]) > 40) > np.mean(np.rad2deg(cool["Obliquity"]) > 40)


def test_non_hosts_isotropic():
    sims, _ = _sims()
    cosi = np.cos(sims.loc[sims.n_planets == 0, "Inclination"])
    assert kstest(cosi, "uniform").pvalue > 0.01


def test_rv_uses_transiting_orbit():
    sims, _ = _sims()
    for pls in sims.loc[sims.n_planets > 0, "planets"]:
        for pl in pls:
            assert np.sin(pl["orbit_inclination"]) > 0.8


def test_planets_survive_csv_roundtrip():
    sims, back = _sims()
    k = int(np.where(sims.n_planets >= 2)[0][0])
    s = make_star()
    s.set_planet_parameters(back.iloc[k])
    assert s.n_planets == sims.n_planets.iloc[k]
    assert abs(s.planets[0]["spin_orbit_angle"] - sims.planets.iloc[k][0]["spin_orbit_angle"]) < 1e-12


def test_no_planet_inside_star():
    sims, _ = _sims()
    for pls in sims.loc[sims.n_planets > 0, "planets"]:
        for pl in pls:
            assert pl["semi_major_axis"] >= 3.0 - 1e-9
    assert (sims.n_planets == sims.planets.apply(len)).all()
