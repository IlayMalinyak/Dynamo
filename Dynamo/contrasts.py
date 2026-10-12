"""Temperature contrasts of spots and faculae.

Spots: Berdyugina (2005, Living Rev. Sol. Phys. 2, 8) polynomial fit of the
spot-minus-photosphere temperature difference against photospheric Teff.

Faculae: centre-to-limb temperature excess of solar faculae from Meunier et
al. (2010, A&A 512, A39), as used by Starsim and SOAP 2.0. The excess is
small at disk centre (~34 K) and large at the limb (~251 K), which makes
faculae limb-brightened features whose visibility depends on inclination.
"""
import numpy as np

# Tags: [paper] from a paper; [own assumption] our choice.
# Reference temperature excess (K) of the "hot" PHOENIX model that the facular
# intensity is blended toward. Facular intensity at each mu is a linear blend
# between the photosphere and T_ph + DT_FAC_REF with weight dT(mu)/DT_FAC_REF.
# [own assumption] the linear-in-temperature blend and the 300 K reference are our
# approximation (it avoids a PHOENIX model per mu); not from a paper.
DT_FAC_REF = 300.0

# [paper] Floor of the PHOENIX SpecInt grid (K), Husser et al. 2013.
T_PHOENIX_MIN = 2300.0


def spot_contrast_berdyugina(teff):
    """Spot temperature deficit (K, positive = cooler) for photospheric Teff (K).

    [paper] Berdyugina 2005 (Living Rev. Sol. Phys. 2, 8) fit to Doppler-imaging / TiO
    spot temperatures of mostly active stars; it gives 1822 K at 5772 K, ~3x the Sun's
    effective spot contrast (docs/sun_calibration.md).
    """
    teff = np.asarray(teff, dtype=float)
    return 3.58e-5 * teff ** 2 + 0.249 * teff - 808.0


def facula_dT_mu(mu, scale=1.0):
    """Facular temperature excess (K, positive = hotter) at mu = cos(angle).

    [paper] Solar fit dT = 250.9 - 407.7 mu + 190.9 mu^2 (Meunier et al. 2010, A&A 512, A39,
    arXiv:1001.1638, Sect. 2.2.2; the code had 407.4 until 2026-10-12, a 0.3 K difference).
    [own assumption] `scale` is a star-level
    multiplier standing in for the unknown Teff dependence.
    """
    mu = np.asarray(mu, dtype=float)
    return scale * (250.9 - 407.7 * mu + 190.9 * mu ** 2)


def facula_weight_mu(mu, scale=1.0):
    """Blend weight between photosphere (0) and the T_ph + DT_FAC_REF model (1)."""
    return facula_dT_mu(mu, scale) / DT_FAC_REF
