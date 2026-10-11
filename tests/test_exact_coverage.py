"""Exact spot/facula area per grid cell (nbspectra.exact_spot_coverage)."""
import numpy as np
import pytest
from numba.typed import List

from Dynamo import nbspectra as nb
from helpers import have_phoenix

N = 30


def _grid():
    _, nr, *_ = nb.generate_grid_coordinates_nb(N)
    nr = np.array(nr)
    w = np.pi / (2 * N - 1)
    th1 = np.r_[0, (np.arange(1, N) - 0.5) * w]
    th2 = np.r_[0.5 * w, (np.arange(1, N) + 0.5) * w]
    return nr, w, np.repeat((np.cos(th1) - np.cos(th2)) * 2 * np.pi / nr, nr)


def _cov(thc, alc, r, rf=0.0):
    nr, _, _ = _grid()
    v = np.array([[np.cos(thc), np.sin(thc) * np.sin(alc), -np.sin(thc) * np.cos(alc)]])
    return nb.exact_spot_coverage(N, nr, List([0]), np.array([[0.0, 0.0, r, rf]]), v)


@pytest.mark.parametrize("r_deg", [1.0, 2.0, 5.0, 10.0])
def test_cap_area_conserved(r_deg):
    _, _, area = _grid()
    r = np.deg2rad(r_deg)
    cap = 2 * np.pi * (1 - np.cos(r))
    lon = [(_cov(np.deg2rad(40), a, r)[0] * area).sum() for a in np.linspace(0, 0.3, 40)]
    assert np.ptp(lon) / cap < 1e-12                  # longitude motion: exact
    colat = np.array([(_cov(t, 0.1, r)[0] * area).sum() for t in np.deg2rad(np.linspace(0, 80, 161))])
    assert np.abs(colat / cap - 1).max() < 1e-4        # colatitude: Gauss-Legendre


def test_facular_annulus_area():
    _, _, area = _grid()
    r, rf = np.deg2rad(4.0), np.deg2rad(8.0)
    _, afc = _cov(np.deg2rad(35), 1.0, r, rf)
    assert abs((afc * area).sum() / (2 * np.pi * (np.cos(r) - np.cos(rf))) - 1) < 1e-4


def test_cells_match_monte_carlo():
    nr, w, area = _grid()
    rng = np.random.default_rng(0)
    n = 3_000_000
    zmin = np.cos(np.deg2rad(60))
    z = rng.uniform(zmin, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - z ** 2)
    pts = np.c_[z, s * np.sin(ph), -s * np.cos(ph)]
    thc, alc, r = np.deg2rad(37), 0.7, np.deg2rad(6)
    c = np.array([np.cos(thc), np.sin(thc) * np.sin(alc), -np.sin(thc) * np.cos(alc)])
    ring = np.minimum(np.floor((np.arccos(z) + 0.5 * w) / w).astype(int), N - 1)
    wr = 2 * np.pi / nr[ring]
    cell = np.r_[0, np.cumsum(nr)[:-1]][ring] + (np.floor(((ph + 0.5 * wr) % (2 * np.pi)) / wr).astype(int) % nr[ring])
    dens = n / (2 * np.pi * (1 - zmin))
    mc = np.bincount(cell[pts @ c >= np.cos(r)], minlength=len(area)) / dens / area
    ex = _cov(thc, alc, r)[0]
    touched = (mc > 0) | (ex > 0)
    sigma = np.sqrt(np.maximum(ex[touched], 1e-3) / (dens * area[touched]))
    assert np.all(np.abs(ex - mc)[touched] < 5 * sigma + 1e-3)


@pytest.mark.skipif(not have_phoenix(), reason="PHOENIX SpecInt models not installed")
def test_pole_on_spot_is_flat_and_close_to_approx():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from grid_artifact import light_curve
    f, _ = light_curve(0.0, 30.0, 5.0, 30, exact=True)
    g, _ = light_curve(0.0, 30.0, 5.0, 30, exact=False)
    depth = 1 - f.mean()
    assert np.ptp(f) / depth < 1e-8                    # truly flat (was ~5% with approx)
    assert abs((1 - g.mean()) / depth - 1) < 0.05      # same spot depth to a few %


@pytest.mark.skipif(not have_phoenix(), reason="PHOENIX SpecInt models not installed")
def test_exact_matches_reference_scenario_shape():
    from make_reference_lc import run_scenario
    _, fe = run_scenario(exact=True)
    _, fa = run_scenario(exact=False)
    assert np.sqrt(np.mean((fe - fa) ** 2)) / np.ptp(fa) < 0.05
