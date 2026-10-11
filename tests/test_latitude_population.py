"""Spot-latitude population (create_data latitude model 'gaussian')."""
import json
import logging
import tempfile

import numpy as np
import pytest

from helpers import make_star

_CACHE = {}


def _table(config, seed=5, n=3000):
    key = (json.dumps(config, sort_keys=True), seed, n)
    if key not in _CACHE:
        from Dynamo.create_data import generate_simdata
        np.random.seed(2)
        with tempfile.TemporaryDirectory() as d:
            sims = generate_simdata(d, n, logging.getLogger("t"), base_seed=seed, latitude_config=config)
            pop = json.load(open(f"{d}/population.json"))
        _CACHE[key] = (sims, pop)
    return _CACHE[key]


def test_population_json_and_ranges():
    sims, pop = _table({"mean": [10.0, 40.0]})
    lat = pop["latitude"]
    assert lat["model"] == "gaussian" and 10.0 <= lat["mean"] <= 40.0 and lat["mean_range"] == [10.0, 40.0]
    assert sims["Lat Mean"].between(0, lat["mean_max"]).all()
    assert sims["Lat Width"].between(lat["width_min"], lat["width_max"]).all()
    assert set(np.round(sims["Lat Drift"].unique(), 6)) <= {0.0, lat["drift"]}
    assert (sims.loc[~sims.Butterfly, "Lat Drift"] == 0).all()
    assert abs(sims["Lat Mean"].median() - lat["mean"]) < 2.0


def test_rossby_trend():
    sims, _ = _table({"rossby_slope": -15.0, "mean_scatter": 3.0})
    r = np.corrcoef(np.log10(sims.Rossby), sims["Lat Mean"])[0, 1]
    assert r < -0.5
    flat, _ = _table({"rossby_slope": 0.0})
    assert abs(np.corrcoef(np.log10(flat.Rossby), flat["Lat Mean"])[0, 1]) < 0.1


def test_other_draws_unchanged_by_latitude_config():
    a, _ = _table({"mean": 10.0})
    b, _ = _table({"mean": 50.0})
    for col in ["Teff", "Period", "Inclination", "Activity Rate", "Facula Q"]:
        assert np.array_equal(a[col].values, b[col].values), col


def test_band_model_restored():
    sims, _ = _table({"model": "band"})
    assert sims["Lat Mean"].isna().all()


def test_spot_latitudes_follow_star_parameters():
    s = make_star(**{"Lat Mean": 40.0, "Lat Width": 5.0, "Lat Drift": 0.0, "Butterfly": False,
                     "Activity Rate": 50.0})
    s.generate_spot_map(1000)
    lat = np.abs(90.0 - s.spot_map[:, 2])
    assert len(lat) > 100
    assert abs(lat.mean() - 40.0) < 1.5 and 4.0 < lat.std() < 7.0


def test_butterfly_drift_is_equatorward():
    s = make_star(**{"Lat Mean": 25.0, "Lat Width": 4.0, "Lat Drift": 20.0, "Butterfly": True,
                     "Activity Rate": 50.0, "Cycle Length": 2.0})
    s.generate_spot_map(1500)
    t = s.spot_map[:, 0]
    lat = np.abs(90.0 - s.spot_map[:, 2])
    assert np.corrcoef((t % 730) / 730, lat)[0, 1] < -0.3
