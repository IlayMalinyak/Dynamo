import numpy as np
from scipy import stats
import astropy.constants as const
import astropy.units as u


def generate_planet_parameters(stellar_mass, stellar_radius, planet_prob=0.1, 
                               max_planets=3, multi_planet_prob=0.3, seed=None):
    """
    Generate realistic planet parameters based on stellar properties.
    
    Supports multi-planet systems: each star can have 0-N planets.

    Parameters:
    -----------
    stellar_mass : array
        Stellar mass in solar masses
    stellar_radius : array
        Stellar radius in solar radii  
    planet_prob : float
        Probability that a star has at least one planet (default 0.1)
    max_planets : int
        Maximum number of planets per star (default 3)
    multi_planet_prob : float
        Probability of additional planets given at least one exists (default 0.3)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    result : dict
        Dictionary with:
        - 'planets': list of lists, where planets[i] is a list of planet dicts for sample i
        - 'n_planets': array of number of planets per sample
        
        Each planet dict contains:
        - 'period': Orbital period in days
        - 'transit_t0': Time of transit in days
        - 'radius': Radius in units of stellar radius
        - 'impact': Impact parameter (0-1+Rp)
        - 'esinw': e*sin(ω)
        - 'ecosw': e*cos(ω)
        - 'spin_orbit_angle': Spin-orbit angle in radians
        - 'semi_amplitude': RV semi-amplitude in m/s
        - 'semi_major_axis': Semi-major axis in stellar radii
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(stellar_mass)
    
    # Constants
    G = const.G.value  # G in m^3 kg^-1 s^-2
    M_sun = const.M_sun.value  # kg
    R_sun = const.R_sun.value  # m
    M_earth = const.M_earth.value  # kg
    R_earth = const.R_earth.value  # m
    AU = const.au.value  # m
    day_to_sec = 24 * 3600  # conversion from days to seconds

    M_star = stellar_mass * M_sun
    R_star = stellar_radius * R_sun

    # Determine number of planets per sample
    # First, decide which stars have at least one planet
    has_planet = np.random.random(n_samples) < planet_prob
    
    # For stars with planets, determine how many (1 to max_planets)
    n_planets = np.zeros(n_samples, dtype=int)
    n_planets[has_planet] = 1
    
    # Add additional planets with decreasing probability
    for _ in range(max_planets - 1):
        add_more = (n_planets > 0) & (np.random.random(n_samples) < multi_planet_prob)
        n_planets[add_more] += 1

    # Generate planet parameters for each sample
    planets = []
    
    for i in range(n_samples):
        sample_planets = []
        n_pl = n_planets[i]
        
        if n_pl == 0:
            planets.append(sample_planets)
            continue
            
        # Generate all planets for this sample (drop those with no allowed orbit)
        for _ in range(n_pl):
            planet = _generate_single_planet(
                M_star[i], R_star[i], stellar_mass[i],
                G, M_sun, R_sun, M_earth, R_earth, AU, day_to_sec
            )
            if planet is not None:
                sample_planets.append(planet)
        n_planets[i] = len(sample_planets)
        planets.append(sample_planets)

    # Also return backward-compatible flat arrays for samples with exactly 1 planet
    # This helps with gradual migration
    result = {
        'planets': planets,
        'n_planets': n_planets,
    }
    
    # Add flat arrays for backward compatibility
    result.update(_create_flat_arrays(planets, n_planets, n_samples))
    
    return result


# Innermost allowed orbit (stellar radii); the closest hot Jupiters sit at a/R* ~ 3.
# [own assumption] the cut value 3 (round number from the known closest hot Jupiters).
A_MIN_RSTAR = 3.0


def _generate_single_planet(M_star_kg, R_star_m, stellar_mass_msun,
                            G, M_sun, R_sun, M_earth, R_earth, AU, day_to_sec):
    """Generate parameters for a single planet."""
    
    # Period: log-uniform from 0.8 to 300 days, but never closer than
    # A_MIN_RSTAR stellar radii (otherwise large stars host planets inside
    # their own envelope). Returns None if no such orbit fits below 300 days.
    period_min, period_max = 0.8, 300
    p_inner = 2 * np.pi * np.sqrt((A_MIN_RSTAR * R_star_m) ** 3 / (G * M_star_kg)) / day_to_sec
    period_min = max(period_min, p_inner)
    if period_min >= period_max:
        return None
    period = np.exp(np.random.uniform(np.log(period_min), np.log(period_max)))
    
    # Semi-major axis from Kepler's law
    period_sec = period * day_to_sec
    a_m = (G * M_star_kg * period_sec ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    a_AU = a_m / AU
    a_Rstar = a_m / R_star_m  # in stellar radii
    
    # Mass distribution (from Malhotra 2015)
    if np.random.random() < 0.8:
        mass = np.random.lognormal(2.5, 0.7)  # Earth masses
    else:
        mass = np.exp(np.random.uniform(-2, 1))  # Earth masses
    
    mj = 318  # Jupiter mass in Earth masses
    
    # Radius from the mass-radius relation [paper] Baron, Mueller, Helled et al. 2024
    # (A&A 686, A296; arXiv:2311.12593, Eq. 1 / Table 3), Earth units. Until 2026-10-13
    # the exponents were inverted (M^(1/0.27) etc.) and the prefactors missing, which put
    # the median planet at ~0.24 R* (clip 0.3) instead of ~3 R_earth.
    if mass < 4.37:
        radius_rearth = 1.02 * mass ** 0.27
    elif mass < 127:
        radius_rearth = 0.56 * mass ** 0.67
    else:
        radius_rearth = 18.6 * mass ** (-0.06)
    
    radius_rstar = np.clip(radius_rearth * R_earth / R_star_m, 0, 0.3)
    
    # Impact parameter
    impact = np.random.uniform(0, 1.0 + radius_rstar)
    
    # Eccentricity and argument of periastron
    ecc = np.random.uniform(0, 0.3)  # Modest eccentricity for stability
    omega = np.random.uniform(0, 2 * np.pi)
    esinw = ecc * np.sin(omega)
    ecosw = ecc * np.cos(omega)
    
    # Transit time
    transit_t0 = np.random.uniform(0, 50)

    # Orbital inclination of the transiting orbit, from the impact parameter
    # (same relation as the transit geometry in nbspectra/rotator)
    i_orb = orbit_inclination(impact, a_Rstar, ecc, esinw)

    # RV semi-amplitude K (uses the orbital inclination of the transiting orbit)
    # [paper] standard Keplerian semi-amplitude in m/s (e.g. Cumming et al. 1999); this form
    # omits the (1 - e^2)^-1/2 factor (inherited from the original code).
    K = 203.25 * (mass * np.sin(i_orb) / mj) * \
        (stellar_mass_msun + mass * M_earth / M_sun) ** (-2/3) * period ** (-1/3)

    # Spin-orbit geometry (projected angle lambda, true obliquity psi, and the
    # stellar inclination) is assigned per SYSTEM by assign_spin_orbit_geometry,
    # which needs the stellar Teff. Placeholder: aligned.
    spin_orbit_angle = 0.0
    
    return {
        'period': float(period),
        'transit_t0': float(transit_t0),
        'radius': float(radius_rstar),
        'impact': float(impact),
        'esinw': float(esinw),
        'ecosw': float(ecosw),
        'spin_orbit_angle': float(spin_orbit_angle),
        'semi_amplitude': float(K),
        'semi_major_axis': float(a_Rstar),
        'orbit_inclination': float(i_orb),
        'obliquity': 0.0,
    }


def orbit_inclination(impact, a_rstar, ecc=0.0, esinw=0.0):
    """Orbital inclination (rad, 90 deg = edge-on) of a transiting orbit.

    Inverts the impact-parameter relation used by the transit geometry:
    cos i = (b / a) * (1 + e sin w) / (1 - e^2).
    [paper] b = (a cos i / R*) (1 - e^2) / (1 + e sin w), Winn 2010 (arXiv:1001.2010), eq. 7.
    """
    cosi = (impact / a_rstar) * (1 + esinw) / (1 - ecc ** 2)
    return float(np.arccos(np.clip(cosi, -1.0, 1.0)))


# Kraft break and obliquity distribution (Winn et al. 2010; Albrecht et al. 2012, 2022):
# cool hosts are mostly aligned, hot hosts have a broad obliquity distribution.
# [paper] the Kraft break at 6250 K and the cool-aligned / hot-misaligned trend.
# [own assumption] the numbers: aligned spread 10 deg, misaligned fractions 0.1 (cool)
# and 0.5 (hot), and the functional forms (isotropic misaligned, |gaussian| aligned,
# a step at the Kraft break). They are rounded values qualitatively matching the papers,
# not fitted.
KRAFT_BREAK_TEFF = 6250.0
ALIGNED_SIGMA_DEG = 10.0
MISALIGNED_FRAC_COOL = 0.1
MISALIGNED_FRAC_HOT = 0.5


def sample_obliquity(teff, rng=np.random):
    """True spin-orbit angle psi (rad, 0..pi) for a host of temperature teff."""
    frac = MISALIGNED_FRAC_HOT if teff > KRAFT_BREAK_TEFF else MISALIGNED_FRAC_COOL
    if rng.random() < frac:
        return float(np.arccos(rng.uniform(-1.0, 1.0)))       # isotropic
    return float(abs(np.deg2rad(rng.normal(0.0, ALIGNED_SIGMA_DEG))))


def assign_spin_orbit_geometry(teff, planets, rng=np.random):
    """Draw the stellar spin for a planetary system and make all angles consistent.

    Frame: Z = line of sight (toward the observer), sky plane = (X, Y). The
    first planet's orbit normal is n = (0, sin i_orb, cos i_orb), so its sky
    projection points along +Y. The spin s is drawn at true obliquity psi
    from n, at a uniform azimuth around n. Then

      stellar inclination  i_star = arccos(|s_Z|)   (0 = pole-on, 90 = equator-on)
      projected obliquity  lambda = atan2(s_X, s_Y) (sky angle, spin vs orbit normal)
      true obliquity       psi_k  = arccos(s . n_k) for every planet k.

    If s points away from the observer (s_Z < 0) it is mirrored through the sky
    plane: the sky projection, the Doppler map and the transit chord are
    unchanged, only the visible pole flips, and keeping i_star in [0, 90] deg
    matches the rest of the code. psi is recomputed for the mirrored spin, so
    cos psi = cos i_star cos i_orb + sin i_star sin i_orb cos lambda holds exactly.
    All planets are taken to share the first planet's sky-projected orbit normal
    (near-coplanar), each with its own orbital inclination.
    [own assumption] uniform spin azimuth around the orbit normal; coplanar systems.

    Updates each planet dict in place ('spin_orbit_angle' = lambda,
    'obliquity' = psi, 'orbit_inclination') and returns i_star in radians.
    """
    i1 = planets[0]['orbit_inclination']
    n1 = np.array([0.0, np.sin(i1), np.cos(i1)])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(n1, e1)
    psi = sample_obliquity(teff, rng)
    az = rng.uniform(0.0, 2 * np.pi)
    s = np.cos(psi) * n1 + np.sin(psi) * (np.cos(az) * e1 + np.sin(az) * e2)
    s[2] = abs(s[2])                                   # mirror through the sky plane if needed
    i_star = float(np.arccos(np.clip(s[2], 0.0, 1.0)))
    lam = float(np.arctan2(s[0], s[1]))
    for pl in planets:
        ik = pl['orbit_inclination']
        nk = np.array([0.0, np.sin(ik), np.cos(ik)])
        pl['spin_orbit_angle'] = lam
        pl['obliquity'] = float(np.arccos(np.clip(np.dot(s, nk), -1.0, 1.0)))
    return i_star


def _create_flat_arrays(planets, n_planets, n_samples):
    """Create backward-compatible flat arrays from planet lists."""
    
    # Initialize with zeros
    flat = {
        'simulate_planet': np.zeros(n_samples, dtype=int),
        'period': np.zeros(n_samples),
        'transit_t0': np.zeros(n_samples),
        'radius': np.zeros(n_samples),
        'impact': np.zeros(n_samples),
        'esinw': np.zeros(n_samples),
        'ecosw': np.zeros(n_samples),
        'spin_orbit_angle': np.zeros(n_samples),
        'semi_amplitude': np.zeros(n_samples),
        'obliquity': np.full(n_samples, np.nan),
        'orbit_inclination': np.full(n_samples, np.nan),
    }
    
    # Fill from first planet of each sample (for backward compat)
    for i in range(n_samples):
        if n_planets[i] > 0:
            flat['simulate_planet'][i] = 1
            p = planets[i][0]  # First planet
            flat['period'][i] = p['period']
            flat['transit_t0'][i] = p['transit_t0']
            flat['radius'][i] = p['radius']
            flat['impact'][i] = p['impact']
            flat['esinw'][i] = p['esinw']
            flat['ecosw'][i] = p['ecosw']
            flat['spin_orbit_angle'][i] = p['spin_orbit_angle']
            flat['semi_amplitude'][i] = p['semi_amplitude']
            flat['obliquity'][i] = p.get('obliquity', np.nan)
            flat['orbit_inclination'][i] = p.get('orbit_inclination', np.nan)
    
    return flat
