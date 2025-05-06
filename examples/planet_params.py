import numpy as np
from scipy import stats
import astropy.constants as const
import astropy.units as u


def generate_planet_parameters(stellar_mass, stellar_radius, planet_prob, seed=None):
    """
    Generate realistic planet parameters based on stellar properties.

    Parameters:
    -----------
    stellar_params : numpy structured array
        Array with fields:
        - 'mass': Stellar mass in solar masses
        - 'radius': Stellar radius in solar radii
        - 'teff': Effective temperature in K
        - 'logg': Surface gravity (log g)
        - 'rotation_period': Rotation period in days
        - 'age': Age in Gyr
        - 'metallicity': [Fe/H]

    n_planets : int, optional
        Number of planets to generate per star. Default is 1.

    include_transits_only : bool, optional
        If True, only include planets that transit. Default is False.

    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    planet_params : numpy structured array
        Array with fields:
        - 'simulate_planet': Boolean flag (1 for yes, 0 for no)
        - 'planet_period': Orbital period in days
        - 'planet_transit_t0': Time of transit in days
        - 'planet_radius': Radius in units of stellar radius
        - 'planet_impact_param': Impact parameter (0-1+Rp)
        - 'planet_esinw': e*sin(ω)
        - 'planet_ecosw': e*cos(ω)
        - 'planet_spin_orbit_angle': Spin-orbit angle in degrees
        - 'planet_semi_amplitude': RV semi-amplitude in m/s
        - 'planet_semi_major_axis': Semi-major axis in AU
        - 'planet_mass': Planet mass in Earth masses
        - 'planet_equilibrium_temp': Equilibrium temperature in K
        - 'transit_duration': Transit duration in hours
        - 'transit_depth': Transit depth in ppm
    """
    if seed is not None:
        np.random.seed(seed)

    # Set up output array
    dtype = [
        ('simulate_planet', np.int32),
        ('planet_period', np.float64),
        ('planet_transit_t0', np.float64),
        ('planet_radius', np.float64),
        ('planet_impact_param', np.float64),
        ('planet_esinw', np.float64),
        ('planet_ecosw', np.float64),
        ('planet_spin_orbit_angle', np.float64),
        ('planet_semi_amplitude', np.float64),
        ('planet_semi_major_axis', np.float64),
        ('planet_mass', np.float64),
        ('planet_equilibrium_temp', np.float64),
        ('transit_duration', np.float64),
        ('transit_depth', np.float64)
    ]

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

    simulate_planet = np.zeros_like(stellar_mass)
    n_planets = int(planet_prob * len(simulate_planet))
    indices = np.random.choice(len(simulate_planet), n_planets, replace=False).astype(np.int16)
    simulate_planet[indices] = 1

    all_transit_times = np.zeros_like(stellar_mass)
    transit_times = np.random.uniform(low=0, high=50, size=n_planets) # transit starts randomly between day 0 to 50
    all_transit_times[indices] = transit_times


    all_periods = np.zeros_like(simulate_planet)
    period_min, period_max = 0.8, 300
    period = np.exp(np.random.uniform(np.log(period_min), np.log(period_max), size=n_planets))
    all_periods[indices] = period

    all_a = np.zeros_like(simulate_planet)
    period_sec = period * day_to_sec
    a = (G * M_star[indices] * period_sec ** 2 / (4 * np.pi ** 2)) ** (1 / 3)  # in meters
    a_AU = a / AU  # in AU
    all_a[indices] = a_AU

    # THE MASS DISTRIBUTION FUNCTION OF PLANETS, R. Malhotra, 2015
    all_mass = np.zeros_like(simulate_planet)
    m1 = np.random.lognormal(2.5, 0.7, size=int(n_planets*0.8))
    m2 = np.exp(np.random.uniform(-2, 1, size=n_planets - int(n_planets*0.8)))
    mass = np.concatenate((m1, m2))
    np.random.shuffle(mass)
    all_mass[indices] = mass # earth

    # K = 203.244 * (Mp * sini / Mjup) * ((Ms + Mp) / Msun) ^ (-2 / 3) * (P / 1day) ^ (-1 / 3)
    mj = 318
    ms = stellar_mass[indices]
    i = np.random.uniform(0, np.pi/2, size=n_planets)
    all_semi_amplitude = np.zeros_like(simulate_planet)
    K = 203.25 * (mass * np.sin(i) / mj) * (ms + mass * M_earth / M_sun) ** (-2/3) * period ** (-1/3)
    all_semi_amplitude[indices] = K

    # The mass-radius relation of exoplanets revisited, S. Muller et al. 2023
    all_r = np.zeros_like(simulate_planet)
    all_r[all_mass < 4.4] = all_mass[all_mass < 4.4] ** (1/0.27)
    all_r[np.logical_and(all_mass > 4.4, all_mass < 127)] =\
        all_mass[np.logical_and(all_mass > 4.4, all_mass < 127)] ** (1/0.67)
    all_r[all_mass > 127] = all_mass[all_mass > 127] ** (-1/0.06)
    all_r = all_r * R_earth / R_sun
    all_r = np.clip(all_r, 0, 0.3) # no more than 3 Jupiter radii

    all_impacts = np.zeros_like(simulate_planet)
    impact = np.array([np.random.uniform(0, 1.0 + all_r[i]) for i in indices])
    all_impacts[indices] = impact

    all_esinw = np.zeros_like(simulate_planet)
    all_ecosw = np.zeros_like(simulate_planet)
    ecc = np.random.uniform(0, 1, size=n_planets)
    omega = np.random.uniform(0, 2 * np.pi)
    esinw = ecc * np.sin(omega)
    ecosw = ecc * np.cos(omega)
    all_esinw[indices] = esinw
    all_ecosw[indices] = ecosw

    all_spin_orbit_angle = np.zeros_like(simulate_planet)

    # Find hot Jupiters among ONLY the simulated planets
    # First, create arrays of properties for just the simulated planets
    r_simulated = all_r[indices]
    periods_simulated = all_periods[indices]

    # Now identify which of the simulated planets are hot Jupiters
    hj_mask = (r_simulated >= 8 * R_earth / R_star[indices]) & (periods_simulated < 10)
    hj_local_idx = np.where(hj_mask)[0]  # Local indices in the simulated planets array
    hj_global_idx = indices[hj_local_idx]  # Global indices in the original array

    # Create a Boolean mask for retrograde hot Jupiters
    retro_fraction = 0.3
    retro_count = int(len(hj_local_idx) * retro_fraction)
    retro_indices = np.random.choice(len(hj_local_idx), size=retro_count, replace=False) if len(
        hj_local_idx) > 0 else []

    # Generate spin-orbit angles for all planets
    spin_orbit_general = np.random.normal(0, 5, size=len(indices))

    # If we have any hot Jupiters
    if len(hj_local_idx) > 0:
        # Create arrays to track which indices are retrograde
        is_retro = np.zeros(len(hj_local_idx), dtype=bool)
        is_retro[retro_indices] = True

        # Generate the angles
        hj_retro_angles = np.random.normal(180, scale=20, size=retro_count)
        hj_normal_angles = np.random.normal(0, scale=20, size=len(hj_local_idx) - retro_count)

        # Assign the values directly to the local indices in spin_orbit_general
        spin_orbit_general[hj_local_idx[is_retro]] = hj_retro_angles
        spin_orbit_general[hj_local_idx[~is_retro]] = hj_normal_angles

    # Finally, assign all angles to the global array
    all_spin_orbit_angle[indices] = spin_orbit_general

    result = {'spin_orbit_angle': all_spin_orbit_angle,
              'esinw': all_esinw,
              'ecosw': all_ecosw,
              'radius': all_r,
              'impact': all_impacts,
              'semi_amplitude': all_semi_amplitude,
              'period': all_periods,
              'simulate_planet': simulate_planet,
              'transit_t0': all_transit_times
              }

    return result
