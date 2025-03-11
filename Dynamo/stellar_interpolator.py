import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator
import pandas as pd
import kiauhoku as kh
from tqdm import tqdm
import time


def get_yrec_params(grid_name, mass, feh, alpha, age):
    df = kh.load_full_grid(grid_name)
    # Parameters to interpolate
    params_to_interpolate = ['Log Teff(K)', 'logg', 'L/Lsun', 'R/Rsun', 'Prot(days)']

    # Convert inputs to arrays if they're not already
    mass_arr = np.atleast_1d(mass)
    feh_arr = np.atleast_1d(feh)
    alpha_arr = np.atleast_1d(alpha)
    age_arr = np.atleast_1d(age)

    results = {param: np.zeros_like(mass_arr, dtype=float) for param in params_to_interpolate}

    # Create the points array directly from the DataFrame index and Age column
    print("Preparing data points...")
    start_time = time.time()

    # Directly extract m, f, a values from MultiIndex
    m_values = df.index.get_level_values(0).values
    f_values = df.index.get_level_values(1).values
    a_values = df.index.get_level_values(2).values
    age_values = df['Age(Gyr)'].values

    # Combine coordinates into points array
    points = np.column_stack([m_values, f_values, a_values, age_values])
    query_points = np.column_stack([mass_arr, feh_arr, alpha_arr, age_arr])
    return points, query_points,params_to_interpolate, df


def get_mist_params(grid_name, mass, feh, age):
    df = kh.load_grid(grid_name)
    # Parameters to interpolate
    params_to_interpolate = ['log_Teff', 'log_g', 'log_L', 'log_R']

    # Convert inputs to arrays if they're not already
    mass_arr = np.atleast_1d(mass)
    feh_arr = np.atleast_1d(feh)
    age_arr = np.atleast_1d(age)

    # Create the points array directly from the DataFrame index and Age column
    print("Preparing data points...")
    start_time = time.time()

    # Directly extract m, f, a values from MultiIndex
    m_values = df.index.get_level_values(0).values
    f_values = df.index.get_level_values(1).values
    age_values = df['star_age'].values / 1e9

    # Combine coordinates into points array
    points = np.column_stack([m_values, f_values,age_values])
    query_points = np.column_stack([mass_arr, feh_arr, age_arr])
    return points, query_points, params_to_interpolate, df

def interpolate_stellar_parameters(mass, feh, alpha, age, grid_name='fastlaunch', method='nearest'):
    """
    Interpolate stellar parameters using values from the full grid.

    Parameters:
    -----------
    mass : float or array-like
        Mass value(s) to interpolate for
    feh : float or array-like
        [Fe/H] value(s) to interpolate for
    alpha : float or array-like
        [alpha/Fe] value(s) to interpolate for
    age : float or array-like
        Age value(s) to interpolate for
    method : str, optional
        Interpolation method: 'nearest', 'linear', 'rbf', or 'kd_tree'

    Returns:
    --------
    dict of interpolated values for Teff, logg, L/Lsun, and R/Rsun
    """
    start_time = time.time()
    # Load the grid directly
    print("Loading grid...")
    if grid_name != 'mist':
        points, query_points, params_to_interpolate, df = get_yrec_params(grid_name,mass, feh, alpha, age)
    else:
        points, query_points, params_to_interpolate, df = get_mist_params(grid_name, mass, feh, age)

    results = {param: np.zeros_like(mass, dtype=float) for param in params_to_interpolate}


    print(f"Points preparation took {time.time() - start_time:.2f} seconds")

    # Prepare query points

    # Select interpolation method
    interpolator_class = get_interpolator(method)

    # Use the selected interpolation method
    for param in params_to_interpolate:
        print(f"Interpolating {param}...")
        start_time = time.time()

        # Extract values directly
        values = df[param].values

        # Create interpolator
        if method == 'kd_tree':
            interp = interpolator_class(points, values[:, np.newaxis])
            results[param] = interp(query_points)[:, 0]
        else:
            interp = interpolator_class(points, values)
            results[param] = interp(query_points)

        print(f"Interpolation of {param} took {time.time() - start_time:.2f} seconds")

    if grid_name != 'mist':
        teff = 10 ** results['Log Teff(K)']
        logg = results['logg']
        L = results['L/Lsun']
        R = results['R/Rsun']
        Prot = results['Prot(days)']
    else:
        teff = 10 ** results['log_Teff']
        logg = results['log_g']
        L = 10 ** results['log_L']
        R = 10 ** results['log_R']
        Prot = None
    return {'Teff': teff, 'logg': logg, 'L': L, 'R': R, 'Prot': Prot}


def get_interpolator(method):
    if method == 'nearest':
        # Nearest neighbor interpolation (fastest)
        from scipy.interpolate import NearestNDInterpolator
        interpolator_class = NearestNDInterpolator
    elif method == 'linear':
        # Linear interpolation (slower but more accurate)
        from scipy.interpolate import LinearNDInterpolator
        interpolator_class = LinearNDInterpolator
    elif method == 'rbf':
        # Radial basis function interpolation (may be faster for large datasets)
        from scipy.interpolate import RBFInterpolator
        def create_interp(pts, vals):
            return RBFInterpolator(pts, vals, neighbors=10, kernel='thin_plate_spline')

        interpolator_class = create_interp
    elif method == 'kd_tree':
        # Custom KD-tree based interpolation
        from scipy.spatial import cKDTree

        def create_kdtree_interp(pts, vals):
            tree = cKDTree(pts)

            def interpolate_func(query_pts):
                # Find k nearest neighbors
                distances, indices = tree.query(query_pts, k=8)

                # Compute weights (inverse distance weighting)
                weights = 1.0 / (distances + 1e-10)  # Add small constant to avoid division by zero
                weights = weights / np.sum(weights, axis=1)[:, np.newaxis]

                # Compute weighted average
                result = np.sum(vals[indices] * weights[:, :, np.newaxis], axis=1)
                return result

            return interpolate_func

        interpolator_class = create_kdtree_interp
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    return interpolator_class

