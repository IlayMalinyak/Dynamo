"""Part A tests: per-star seed, bounded priors, no-dynamo stars, spot contrast."""
import logging
import tempfile

import numpy as np
import pytest

from helpers import have_phoenix, make_star, flux_scale


def _spot_map(seed):
    s = make_star(Seed=seed, **{"Activity Rate": 2.0})
    s.generate_spot_map(ndays=300)
    return s.spot_map.copy()


def test_seed_controls_spot_map():
    a, b, c = _spot_map(1), _spot_map(1), _spot_map(2)
    assert np.array_equal(a, b), "same seed must give identical spot maps"
    assert a.shape != c.shape or not np.allclose(a, c), "different seeds must differ"


def test_simdata_bounds():
    from Dynamo.create_data import generate_simdata
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as d:
        sims = generate_simdata(d, 2000, logging.getLogger("t"), base_seed=7)
    for col in ["Seed", "Has Dynamo", "Spot Contrast Scale", "Spot Max Area"]:
        assert col in sims
    assert (sims["Cycle Length"] > 0).all() and (sims["Decay Time"] >= 1.5).all()
    assert sims["Seed"].is_unique and sims["Seed"].min() == 7
    assert not sims.loc[sims.Teff > 7000, "Has Dynamo"].any()
    assert not sims.loc[~sims.main_seq, "Has Dynamo"].any()


@pytest.mark.skipif(not have_phoenix(), reason="PHOENIX SpecInt models not installed")
def test_no_dynamo_is_flat():
    s = make_star(**{"Has Dynamo": False})
    s.generate_spot_map(ndays=50)
    assert s.spot_map.shape[0] == 0
    t = np.linspace(0, 50, 500)
    s.compute_forward(t=t)
    f = np.asarray(s.results["lc"]) / flux_scale(s)
    assert np.max(np.abs(f / np.median(f) - 1)) < 1e-12


def test_spot_contrast():
    from Dynamo.contrasts import spot_contrast_berdyugina
    assert abs(spot_contrast_berdyugina(5800) - 1840) < 50
    s = make_star(Teff=5800.0)
    assert abs(s.spot_T_contrast - spot_contrast_berdyugina(5800)) < 1e-6
    cool = make_star(Teff=3000.0, **{"Spot Contrast Scale": 1.5})
    assert cool.temperature_spot >= 2300.0
