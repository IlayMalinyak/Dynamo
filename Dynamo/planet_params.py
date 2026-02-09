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
            
        # Generate all planets for this sample
        for _ in range(n_pl):
            planet = _generate_single_planet(
                M_star[i], R_star[i], stellar_mass[i],
                G, M_sun, R_sun, M_earth, R_earth, AU, day_to_sec
            )
            sample_planets.append(planet)
            
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


def _generate_single_planet(M_star_kg, R_star_m, stellar_mass_msun,
                            G, M_sun, R_sun, M_earth, R_earth, AU, day_to_sec):
    """Generate parameters for a single planet."""
    
    # Period: log-uniform from 0.8 to 300 days
    period_min, period_max = 0.8, 300
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
    
    # RV semi-amplitude K
    mj = 318  # Jupiter mass in Earth masses
    incl = np.random.uniform(0, np.pi/2)
    K = 203.25 * (mass * np.sin(incl) / mj) * \
        (stellar_mass_msun + mass * M_earth / M_sun) ** (-2/3) * period ** (-1/3)
    
    # Radius from mass-radius relation (Muller et al. 2023)
    if mass < 4.4:
        radius_rearth = mass ** (1/0.27)
    elif mass < 127:
        radius_rearth = mass ** (1/0.67)
    else:
        radius_rearth = mass ** (-1/0.06)
    
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
    
    # Spin-orbit angle
    # Hot Jupiters (large, short period) can have misaligned orbits
    is_hot_jupiter = (radius_rearth >= 8) and (period < 10)
    if is_hot_jupiter and np.random.random() < 0.3:
        spin_orbit_angle = np.deg2rad(np.random.normal(180, 20))  # Retrograde
    elif is_hot_jupiter:
        spin_orbit_angle = np.deg2rad(np.random.normal(0, 20))
    else:
        spin_orbit_angle = np.deg2rad(np.random.normal(0, 5))
    
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
    }


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
    
    return flat
