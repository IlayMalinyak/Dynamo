"""Part D test: facular pixels in the velocity-kernel spectral integrator."""
import numpy as np
import pytest

from helpers import have_phoenix, make_star

pytestmark = pytest.mark.skipif(not have_phoenix(), reason="PHOENIX SpecInt models not installed")


def test_D1_full_facular_coverage_is_between_photosphere_and_hot():
    from Dynamo import spectra
    from Dynamo.contrasts import DT_FAC_REF
    s = make_star(Inclination=np.pi / 2)
    s.n_grid_rings = 15
    wv = np.linspace(5000.0, 5100.0, 2000)
    mu, _, ph = spectra.interpolate_Phoenix_mu_lc_with_metallicity(s, 5800.0, 4.5, 0.0, wv_array=wv)
    _, _, hot = spectra.interpolate_Phoenix_mu_lc_with_metallicity(s, 5800.0 + DT_FAC_REF, 4.5, 0.0, wv_array=wv)
    npix = int(np.sum(s.get_theta_phi()[0]))
    cov = np.zeros((npix, 4))

    def run(cov, photo, hot_flux):
        f, _ = spectra.create_observed_spectra_kernel(
            s, wv, photo, photo, mu, cov, instrument_resolution=None,
            spectra_filter_name="Raw", resample=False, hot_flux=hot_flux)
        return f[50:-50]          # drop convolution edges

    cov[:, 0] = 1.0
    f_ph, f_hot = run(cov, ph, None), run(cov, hot, None)
    cov[:] = 0.0
    cov[:, 2] = 1.0
    f_fac = run(cov, ph, hot)
    lo, hi = np.minimum(f_ph, f_hot), np.maximum(f_ph, f_hot)
    tol = 1e-9 * np.abs(hi).max()
    assert ((f_fac >= lo - tol) & (f_fac <= hi + tol)).all()
    w = np.median((f_fac - f_ph) / (f_hot - f_ph))
    print("disk-averaged facular weight:", w)
    assert 0.15 < w < 0.45
