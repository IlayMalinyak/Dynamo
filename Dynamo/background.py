"""
Stellar + instrumental background for simulated light curves.

Every real star varies even without spots: surface granulation and stochastically
excited p-mode oscillations, observed through photon/read noise. This module adds
that background to a clean Dynamo light curve as a post-processing step (it is
additive and independent of the spot geometry, so one expensive simulation can
receive many background realisations / instruments).

Components (all in ppm of the mean flux):
  * granulation   two Harvey components, Kallinger et al. 2014 (A&A 570, A41):
                  b1 = 0.317 numax^0.970, b2 = 0.948 numax^0.992 [muHz],
                  a  = 3710 numax^-0.613 M^-0.26 [ppm rms per component]
  * oscillations  l = 0, 1, 2 p-mode comb (asymptotic relation, Mosser et al. 2011
                  epsilon), Gaussian envelope FWHM = 0.66 numax^0.88 (Mosser 2012),
                  radial-mode amplitude A_max = 2.5 L^0.838 M^-1.32 (Teff/Tsun)^-1
                  ppm (Huber et al. 2011), Lorentzian linewidth
                  0.95 (Teff/Tsun)^10 muHz (dwarfs ~1, red giants ~0.1)
  * white noise   photon + read/background noise from the apparent magnitude and
                  an instrument preset (Kepler LC, TESS FFI), plus a systematic floor.
                  The magnitude is either survey-like (survey_mag; default in
                  scripts/add_background.py) or from L, Teff and the table distance.

Phase-space realism:
  * finite integration: every signal is averaged over the cadence, which multiplies
    its power by sinc^2(nu * t_int);
  * aliasing: power above the Nyquist frequency is folded back (dwarf granulation
    and oscillations live at ~1000-3000 muHz, above the 283 muHz Kepler LC Nyquist).

Not modelled: delta Sct / gamma Dor pulsations of stars hotter than ~7000 K (their
convective background is tapered to zero instead), mixed l = 1 modes in red giants,
spot activity of giants (a separate change to the simulation itself).

numax / Dnu scaling: numax = 3090 (g/g_sun)(Teff/Tsun)^-0.5, Dnu = 135.1 (M/R^3)^0.5.
"""
import numpy as np

# Tags: [paper] from a paper; [own assumption] our choice; [numerical] numerical setting.
# [paper] solar reference values of the scaling relations, Huber et al. 2011 (ApJ 743,
# 143): Teff 5777 K (kept at 5777 to match that calibration; create_data uses the IAU
# 5772 K), numax 3090 muHz, Dnu 135.1 muHz, A_max 2.5 ppm. log g 4.438: IAU 2015 B3.
TEFF_SUN = 5777.0
LOGG_SUN = 4.438
NUMAX_SUN = 3090.0      # muHz
DNU_SUN = 135.1         # muHz
AMAX_SUN = 2.5          # ppm, radial-mode amplitude in the Kepler band
# [own assumption] no granulation/oscillation background for stars without a convective
# envelope: a logistic taper at 7000 K, width 150 K (both values our choice).
GRAN_TEFF_CUT = 7000.0  # convective background tapered off above this ...
GRAN_TEFF_WIDTH = 150.0  # ... with this logistic width [K]
VISIBILITY = {0: 1.0, 1: 1.54, 2: 0.58}   # Kepler-band mode visibilities (Mosser 2012)

# Instrument presets. n_e: electrons per cadence at mag_ref; read_e: read+background
# noise [e-] per cadence (scaled as sqrt(cadence)); floor_ppm: systematic floor per
# cadence point. Kepler: 4.5e9 e- in 6.5 h at Kp = 12 (Jenkins et al. 2010);
# TESS: 1.5e4 e-/s at T = 10 (Vanderspek et al. 2018), 80% duty cycle.
# [own assumption] read_e and floor_ppm (Kepler 6000 e-, 10 ppm; TESS 4000 e-, 85 ppm) are
# our settings for a plausible noise-vs-magnitude curve, not fitted to a published CDPP
# relation.
# mag_range: survey-like target magnitudes, p(m) ~ 10^(0.3 m) inside the range
# (number counts rise towards the faint limit; Kepler targets mostly Kp < 16).
# [own assumption] the 0.3 slope and the ranges (Kp 9-16, T 7-13) are approximate.
INSTRUMENTS = {
    "Kepler": dict(mag_ref=12.0, n_e=3.46e8, cadence_s=1765.5, read_e=6.0e3, floor_ppm=10.0,
                   mag_range=(9.0, 16.0)),
    "TESS": dict(mag_ref=10.0, n_e=2.16e7, cadence_s=1800.0, read_e=4.0e3, floor_ppm=85.0,
                 mag_range=(7.0, 13.0)),
}


# --------------------------------------------------------------------------- scalings
def numax_scaling(teff, logg):
    # [paper] numax ~ g Teff^-0.5 (Brown et al. 1991; Kjeldsen & Bedding 1995)
    return NUMAX_SUN * 10 ** (np.asarray(logg) - LOGG_SUN) * (np.asarray(teff) / TEFF_SUN) ** -0.5


def dnu_scaling(mass, radius):
    # [paper] Dnu ~ sqrt(mean density) (Ulrich 1986; Kjeldsen & Bedding 1995)
    return DNU_SUN * np.sqrt(np.asarray(mass) / np.asarray(radius) ** 3)


def convective_taper(teff):
    """1 for convective-envelope stars, -> 0 for stars hotter than ~7000 K."""
    return 1.0 / (1.0 + np.exp((np.asarray(teff) - GRAN_TEFF_CUT) / GRAN_TEFF_WIDTH))


def granulation_params(numax, mass, teff):
    """Two Harvey components [(a_ppm, b_muHz), ...] (Kallinger et al. 2014)."""
    a = 3710.0 * numax ** -0.613 * mass ** -0.26 * convective_taper(teff)
    return [(a, 0.317 * numax ** 0.970), (a, 0.948 * numax ** 0.992)]


def amax_scaling(lum, mass, teff):
    """Radial-mode amplitude [ppm] (Huber et al. 2011, s = 0.838, t = 1.32)."""
    return (AMAX_SUN * lum ** 0.838 * mass ** -1.32 * (teff / TEFF_SUN) ** -1.0
            * convective_taper(teff))


def mode_linewidth(teff):
    # [own assumption] 0.95 muHz at the solar Teff, steep ~Teff^10 rise, clipped to
    # 0.05-10 muHz. The steep Teff dependence follows the trend reported for Kepler
    # dwarfs (Appourchaux et al. 2012; citation not verified here); the normalisation,
    # exponent and clip are our choice.
    return float(np.clip(0.95 * (teff / TEFF_SUN) ** 10.0, 0.05, 10.0))


def oscillation_modes(numax, dnu, amax, teff):
    """Mode frequencies [muHz], rms amplitudes [ppm] and linewidths [muHz]."""
    # [paper] envelope FWHM 0.66 numax^0.88 and eps = 0.634 + 0.63 log Dnu (Mosser et al.
    # 2011, 2012). [own assumption] constant small separations d01 = 0.025 Dnu,
    # d02 = 0.12 Dnu (simplified from the Mosser 2011 universal pattern; not verified here).
    # [numerical] orders within +-4 sigma of numax; modes below 1e-3 A_max dropped.
    sigma = 0.66 * numax ** 0.88 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    eps = 0.634 + 0.63 * np.log10(dnu)
    n = np.arange(max(1, int((numax - 4 * sigma) / dnu) - 1), int((numax + 4 * sigma) / dnu) + 2)
    nu0 = dnu * (n + eps)
    freqs = {0: nu0, 1: nu0 + 0.5 * dnu - 0.025 * dnu, 2: nu0 - 0.12 * dnu}
    gamma = mode_linewidth(teff)
    f, a = [], []
    for ell, nu in freqs.items():
        env = np.exp(-(nu - numax) ** 2 / (2.0 * sigma ** 2))
        f.append(nu)
        a.append(amax * np.sqrt(VISIBILITY[ell] * env))
    f, a = np.concatenate(f), np.concatenate(a)
    keep = (f > 0) & (a > 1e-3 * max(amax, 1e-12))
    return f[keep], a[keep], np.full(keep.sum(), gamma)


# --------------------------------------------------------------------- magnitudes
def bc_v_torres(teff):
    """Bolometric correction BC_V (Torres 2010, AJ 140, 1158)."""
    lt = np.log10(np.asarray(teff, dtype=float))
    c1 = [-0.190537291496456e5, 0.155144866764412e5, -0.421278819301717e4, 0.381476328422343e3]
    c2 = [-0.370510203809015e5, 0.385672629965804e5, -0.150651486316025e5,
          0.261724637119416e4, -0.170623810323864e3]
    c3 = [-0.118115450538963e6, 0.137145973583929e6, -0.636233812100225e5,
          0.147412923562646e5, -0.170587278406872e4, 0.788731721804990e2]
    poly = lambda c: sum(ci * lt ** k for k, ci in enumerate(c))
    return np.where(lt < 3.70, poly(c1), np.where(lt < 3.90, poly(c2), poly(c3)))


def apparent_mag(lum, teff, distance_pc, instrument="Kepler"):
    """Approximate Kp / TESS magnitude from L, Teff and distance (no extinction).

    V from M_bol and the Torres BC_V; the band offsets are rough linear colour
    terms (redder bands are brighter for cool stars).
    [paper] M_bol,sun = 4.74 (IAU 2015 B2). [own assumption] the Kp and TESS colour offsets
    (-0.15 - 0.4 cool, -0.65 - 0.6 cool) are our approximation, not a published
    transformation; no extinction.
    """
    v = 4.74 - 2.5 * np.log10(lum) - bc_v_torres(teff) + 5.0 * np.log10(np.asarray(distance_pc) / 10.0)
    cool = np.clip((TEFF_SUN - np.asarray(teff)) / 1000.0, 0.0, None)
    if instrument == "Kepler":
        return v - 0.15 - 0.4 * cool
    if instrument == "TESS":
        return v - 0.65 - 0.6 * cool
    raise ValueError(f"unknown instrument {instrument}")


def survey_mag(rng, instrument="Kepler"):
    """Target magnitude drawn from p(m) ~ 10^(0.3 m) on the instrument's range."""
    lo, hi = INSTRUMENTS[instrument]["mag_range"]
    u = rng.uniform()
    return float(np.log10(10 ** (0.3 * lo) + u * (10 ** (0.3 * hi) - 10 ** (0.3 * lo))) / 0.3)


def point_noise_ppm(mag, instrument="Kepler", cadence_s=None):
    """Per-cadence white noise [ppm]: photon + read/background + systematic floor."""
    p = INSTRUMENTS[instrument]
    scale = (cadence_s or p["cadence_s"]) / p["cadence_s"]
    n_e = p["n_e"] * scale * 10 ** (-0.4 * (np.asarray(mag) - p["mag_ref"]))
    read = p["read_e"] * np.sqrt(scale)
    return np.sqrt((1e6 * np.sqrt(n_e + read ** 2) / n_e) ** 2 + p["floor_ppm"] ** 2)


# ----------------------------------------------------------------------- spectra
def _harvey(nu, a, b):
    zeta = 2.0 * np.sqrt(2.0) / np.pi            # integral over nu in [0, inf) = a^2
    return zeta * a ** 2 / b / (1.0 + (nu / b) ** 4)


def _fold(nu, fs):
    """Map frequencies onto [0, fs/2] (aliasing of a sampled signal)."""
    r = np.mod(nu, fs)
    return np.where(r > fs / 2, fs - r, r)


def granulation_psd(f, df, fs, t_int_s, gran, nu_max_grid=None):
    """Observed (integrated + aliased) granulation PSD on the grid f [muHz]."""
    nu_top = nu_max_grid or max(20000.0, 10 * max(b for _, b in gran))
    psd = np.zeros_like(f)
    kmax = int(np.ceil(nu_top / fs))
    for k in range(kmax + 1):
        for nu in (k * fs + f, (k + 1) * fs - f):
            att = np.sinc(nu * 1e-6 * t_int_s) ** 2
            psd += att * sum(_harvey(nu, a, b) for a, b in gran)
    # the two images of the k-th block overlap at f = 0 and f = fs/2; halve those
    psd[0] *= 0.5
    if len(f) and np.isclose(f[-1], fs / 2):
        psd[-1] *= 0.5
    return psd


def oscillation_psd(f, df, fs, t_int_s, freqs, amps, widths):
    """Observed oscillation PSD: one Lorentzian per mode at its aliased frequency.

    Each mode's power (amp^2, attenuated by the integration) is preserved exactly on
    the discrete grid, so modes narrower than the frequency resolution still carry
    the right power.
    """
    psd = np.zeros_like(f)
    for nu, a, g in zip(freqs, amps, widths):
        p = a ** 2 * np.sinc(nu * 1e-6 * t_int_s) ** 2
        if p <= 0:
            continue
        nu_obs = _fold(nu, fs)
        w = max(g, df)
        lo, hi = np.searchsorted(f, [nu_obs - 50 * w, nu_obs + 50 * w])
        seg = f[lo:hi]
        lor = 1.0 / (1.0 + ((seg - nu_obs) / (w / 2)) ** 2)
        s = lor.sum() * df
        if s > 0:
            psd[lo:hi] += p * lor / s
    return psd


# -------------------------------------------------------------------- synthesis
def _synth_from_psd(psd, n, df, rng):
    """Real Gaussian time series of length n whose one-sided PSD is psd [ppm^2/muHz]."""
    c = n * np.sqrt(psd * df) / 2.0
    spec = c * (rng.standard_normal(len(psd)) + 1j * rng.standard_normal(len(psd)))
    spec[0] = 0.0
    if n % 2 == 0:                       # Nyquist bin is real: keep its variance
        spec[-1] = np.sqrt(2.0) * c[-1] * rng.standard_normal()
    return np.fft.irfft(spec, n=n)


def star_background_params(row, instrument="Kepler", mag=None):
    """Derived background parameters for one simulation_properties row.

    mag: apparent magnitude in the instrument band; None -> from L, Teff, Distance.
    """
    teff, logg = float(row["Teff"]), float(row["logg"])
    mass, radius, lum = float(row["mass"]), float(row["R"]), float(row["L"])
    numax = float(numax_scaling(teff, logg))
    dnu = float(dnu_scaling(mass, radius))
    gran = granulation_params(numax, mass, teff)
    amax = float(amax_scaling(lum, mass, teff))
    if mag is None:
        mag = float(apparent_mag(lum, teff, float(row["Distance"]), instrument))
    return dict(teff=teff, numax=numax, dnu=dnu, gran=[(float(a), float(b)) for a, b in gran],
                amax=amax, gamma=mode_linewidth(teff), mag=mag,
                sigma_point=float(point_noise_ppm(mag, instrument)))


def synthesize_background(time_days, params, rng, t_int_s=None,
                          components=("granulation", "oscillations", "white")):
    """Background time series [ppm] sampled at time_days.

    Returns a dict with one array per requested component plus 'total'. The time
    grid may have gaps; the signal is generated on the uniform grid that spans it.
    """
    t = np.asarray(time_days, dtype=float)
    dt = float(np.median(np.diff(t)))
    idx = np.rint((t - t[0]) / dt).astype(int)
    n = int(idx[-1]) + 1
    n += n % 2
    t_int_s = t_int_s if t_int_s is not None else dt * 86400.0
    fs = 1e6 / (dt * 86400.0)                     # sampling frequency [muHz]
    df = fs / n
    f = np.arange(n // 2 + 1) * df
    out = {}
    if "granulation" in components:
        psd = granulation_psd(f, df, fs, t_int_s, params["gran"])
        out["granulation"] = _synth_from_psd(psd, n, df, rng)[idx]
    if "oscillations" in components and params["amax"] > 0:
        fr, am, gw = oscillation_modes(params["numax"], params["dnu"], params["amax"],
                                       params["teff"])
        psd = oscillation_psd(f, df, fs, t_int_s, fr, am, gw)
        out["oscillations"] = _synth_from_psd(psd, n, df, rng)[idx]
    if "white" in components:
        out["white"] = rng.normal(0.0, params["sigma_point"], len(t))
    out["total"] = sum(out.values()) if out else np.zeros_like(t)
    return out


def add_background(time_days, flux, row, instrument="Kepler", seed=0, mag=None, **kw):
    """Clean flux -> flux with stellar + instrumental background (multiplicative)."""
    params = star_background_params(row, instrument, mag=mag)
    bg = synthesize_background(time_days, params, np.random.default_rng(seed), **kw)
    return np.asarray(flux) * (1.0 + 1e-6 * bg["total"]), bg, params
