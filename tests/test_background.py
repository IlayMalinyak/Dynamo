"""Stellar + instrumental background (Dynamo/background.py)."""
import numpy as np
from scipy.ndimage import uniform_filter1d

from Dynamo.background import (granulation_params, oscillation_modes, numax_scaling,
                               dnu_scaling, amax_scaling, point_noise_ppm, survey_mag,
                               star_background_params, synthesize_background, add_background)


def _giant_params():
    return dict(teff=4800., numax=100., dnu=9., gran=granulation_params(100., 1.2, 4800.),
                amax=40., gamma=0.3, mag=12., sigma_point=50.)


def test_solar_scalings():
    assert np.isclose(numax_scaling(5777, 4.438), 3090.0)
    assert np.isclose(dnu_scaling(1.0, 1.0), 135.1)
    assert abs(amax_scaling(1.0, 1.0, 5777) - 2.5) < 0.01
    a, _ = granulation_params(3090.0, 1.0, 5777)[0]
    assert 20 < a < 35                      # ~27 ppm per Harvey component


def test_power_conserved_without_integration():
    p = _giant_params()
    t = np.arange(0, 200, 1 / 1440)
    for comp, expect in [("granulation", np.sqrt(sum(a ** 2 for a, _ in p["gran"]))),
                         ("oscillations", np.sqrt((oscillation_modes(100., 9., 40., 4800.)[1] ** 2).sum()))]:
        rms = np.mean([synthesize_background(t, p, np.random.default_rng(s), t_int_s=1e-3,
                                             components=(comp,))[comp].std() for s in range(4)])
        assert abs(rms / expect - 1) < 0.05, (comp, rms, expect)


def test_integration_suppresses_dwarf_oscillations():
    row = dict(Teff=5777., logg=4.438, mass=1., R=1., L=1., Distance=300.)
    p = star_background_params(row)
    t = np.arange(0, 100, 1 / 48)
    osc = synthesize_background(t, p, np.random.default_rng(0), components=("oscillations",))
    total_intrinsic = np.sqrt((oscillation_modes(p["numax"], p["dnu"], p["amax"], 5777.)[1] ** 2).sum())
    assert osc["oscillations"].std() < 0.1 * total_intrinsic   # 30-min integration kills 3 mHz modes


def test_giant_oscillation_peak_at_numax():
    row = dict(Teff=4700., logg=2.7, mass=1.3, R=8., L=28., Distance=1500.)
    p = star_background_params(row)
    t = np.arange(0, 1000, 1 / 48)
    x = synthesize_background(t, p, np.random.default_rng(3), components=("oscillations",))["oscillations"]
    f = np.fft.rfftfreq(len(x), d=1800) * 1e6
    pw = uniform_filter1d(np.abs(np.fft.rfft(x)) ** 2, int(2 * p["dnu"] / f[1]))
    assert abs(f[np.argmax(pw)] / p["numax"] - 1) < 0.15


def test_flicker_matches_bastien_logg():
    """8-hr flicker of the synthetic background reproduces Bastien et al. 2016 logg(F8)."""
    def bastien_logg(f8_ppt):
        x = np.log10(f8_ppt)
        return 1.3724221 - 3.5002686 * x - 1.6838185 * x ** 2 - 0.37909094 * x ** 3
    t = np.arange(0, 90, 1 / 48)
    for logg, teff, m, r in [(2.8, 4700, 1.3, 7.5), (3.5, 5000, 1.2, 3.3), (4.44, 5777, 1., 1.)]:
        p = star_background_params(dict(Teff=teff, logg=logg, mass=m, R=r,
                                        L=r ** 2 * (teff / 5777) ** 4, Distance=500.))
        f8 = []
        for s in range(4):
            bg = synthesize_background(t, p, np.random.default_rng(s),
                                       components=("granulation", "oscillations"))
            x = bg["granulation"] + bg["oscillations"]
            f8.append(np.std(x - uniform_filter1d(x, 16, mode="nearest")) / 1e3)
        assert abs(bastien_logg(np.mean(f8)) - logg) < 0.25, (logg, bastien_logg(np.mean(f8)))


def test_white_noise_and_kepler_levels():
    assert 40 < point_noise_ppm(12.0, "Kepler") < 80          # ~16 ppm in 6.5 h
    assert point_noise_ppm(15.0, "Kepler") > point_noise_ppm(12.0, "Kepler")
    p = _giant_params()
    t = np.arange(0, 50, 1 / 48)
    w = synthesize_background(t, p, np.random.default_rng(0), components=("white",))["white"]
    assert abs(w.std() / 50 - 1) < 0.05


def test_survey_mag_range_and_determinism():
    m = [survey_mag(np.random.default_rng(s)) for s in range(2000)]
    assert 9.0 <= min(m) and max(m) <= 16.0 and 14.0 < np.median(m) < 15.5
    row = dict(Teff=5500., logg=4.4, mass=0.95, R=0.95, L=0.8, Distance=400.)
    t = np.arange(0, 30, 1 / 48)
    a = add_background(t, np.ones_like(t), row, seed=[1, 2, 3], mag=14.0)[0]
    b = add_background(t, np.ones_like(t), row, seed=[1, 2, 3], mag=14.0)[0]
    c = add_background(t, np.ones_like(t), row, seed=[1, 2, 4], mag=14.0)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_hot_star_has_no_convective_background():
    p = star_background_params(dict(Teff=8000., logg=4.1, mass=1.9, R=1.8, L=12., Distance=500.))
    assert p["gran"][0][0] < 0.1 and p["amax"] < 0.1     # vs ~27 ppm and 2.5 ppm for the Sun
