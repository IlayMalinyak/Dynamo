import os
import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from pathlib import Path
from scipy import interpolate
import sys
import warnings
import numpy as np
from scipy import signal, interpolate
import hashlib


warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d, UnivariateSpline
import math as m
from . import nbspectra


c = 299792.458  # km/s

class PhoenixCacheContext:
    """
    Context manager for temporarily changing cache settings.

    Example:
    --------
    with PhoenixCacheContext(enabled=False):
        # Cache disabled for this block
        result = interpolate_Phoenix_mu_lc_with_metallicity(...)
    # Cache settings restored
    """

    def __init__(self, enabled=None, cache_dir=None, max_size_gb=None):
        self.old_enabled = None
        self.old_cache_dir = None
        self.old_max_size = None
        self.new_enabled = enabled
        self.new_cache_dir = cache_dir
        self.new_max_size = max_size_gb

    def __enter__(self):
        global PHOENIX_CACHE_ENABLED, PHOENIX_CACHE_DIR, PHOENIX_CACHE_MAX_SIZE_GB

        self.old_enabled = PHOENIX_CACHE_ENABLED
        self.old_cache_dir = PHOENIX_CACHE_DIR
        self.old_max_size = PHOENIX_CACHE_MAX_SIZE_GB

        if self.new_enabled is not None:
            PHOENIX_CACHE_ENABLED = self.new_enabled
        if self.new_cache_dir is not None:
            PHOENIX_CACHE_DIR = Path(self.new_cache_dir)
            PHOENIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self.new_max_size is not None:
            PHOENIX_CACHE_MAX_SIZE_GB = self.new_max_size

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global PHOENIX_CACHE_ENABLED, PHOENIX_CACHE_DIR, PHOENIX_CACHE_MAX_SIZE_GB

        PHOENIX_CACHE_ENABLED = self.old_enabled
        PHOENIX_CACHE_DIR = self.old_cache_dir
        PHOENIX_CACHE_MAX_SIZE_GB = self.old_max_size



def interpolate_filter(star, filter_name):

    if isinstance(star.models_root, str):
        star.models_root = Path(star.models_root)
    path = star.models_root / 'models' / 'filters' / filter_name

    try:
        wv, filt = np.loadtxt(path,unpack=True)
    except: #if the filter do not exist, create a tophat filter from the wv range
        print(f'WARNING: {filter_name} not found in {star.models_root} / models')
        wv=np.array([star.wavelength_lower_limit, star.wavelength_upper_limit])
        filt=np.array([1,1])

    f = interpolate.interp1d(wv,filt,bounds_error=False,fill_value=0)

    return f

def limb_darkening_law(self,amu):

    if self.limb_darkening_law == 'linear':
        mu=1-self.limb_darkening_q1*(1-amu)

    elif self.limb_darkening_law == 'quadratic':
        a=2*np.sqrt(self.limb_darkening_q1)*self.limb_darkening_q2
        b=np.sqrt(self.limb_darkening_q1)*(1-2*self.limb_darkening_q2)
        mu=1-a*(1-amu)-b*(1-amu)**2

    elif self.limb_darkening_law == 'sqrt':
        a=np.sqrt(self.limb_darkening_q1)*(1-2*self.limb_darkening_q2)
        b=2*np.sqrt(self.limb_darkening_q1)*self.limb_darkening_q2
        mu=1-a*(1-amu)-b*(1-np.sqrt(amu))

    elif self.limb_darkening_law == 'log':
        a=self.limb_darkening_q2*self.limb_darkening_q1**2+1
        b=self.limb_darkening_q1**2-1
        mu=1-a*(1-amu)-b*amu*(1-np.log(amu))

    else:
        sys.exit('Error in limb darkening law, please select one of the following: phoenix, linear, quadratic, sqrt, logarithmic')

    return mu


def _resample_to_log_lambda(wavelength, flux, n=None):
    """Return (loglam_grid, flux_resampled, interp_back_fn)
    loglam_grid in units of log10(lambda); interp_back_fn maps resampled flux -> original wavelength grid.
    """
    # convert to numpy arrays
    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)
    # use base-10 logarithm
    loglam = np.log10(wavelength)
    # choose number of points
    if n is None:
        # choose oversampling ~ 4 samples per original pixel on average
        n = max( len(wavelength)*4, 1024 )
    logmin, logmax = loglam.min(), loglam.max()
    log_grid = np.linspace(logmin, logmax, int(n))
    interp = interpolate.interp1d(loglam, flux, kind='linear', bounds_error=False, fill_value="extrapolate")
    flux_resampled = interp(log_grid)
    # function to map back to original wavelength grid
    def interp_back(fl_resamp):
        f_interp_back = interpolate.interp1d(log_grid, fl_resamp, kind='linear', bounds_error=False, fill_value="extrapolate")
        return f_interp_back(np.log10(wavelength))
    return log_grid, flux_resampled, interp_back

def apply_rotational_broadening_differential(wavelength, flux,
                                             vsini_equator,
                                             alpha=0.0,
                                             inclination_deg=90.0,
                                             epsilon=0.6,
                                             limb_darkening_fn=None,
                                             n_theta=60, n_phi=120,
                                             log_n_samples=None):
    """
    Apply rotational broadening including differential rotation.

    Parameters
    ----------
    wavelength : 1D array (Angstrom)
    flux : 1D array
    vsini_equator : float
        Observed projected equatorial velocity (km/s): R * Omega_eq * sin i
    alpha : float
        Differential rotation parameter (Reiners-style). Omega(lat) = Omega_eq * (1 - alpha * sin^2(lat)).
        alpha = 0 => solid-body; alpha > 0 => equator rotates faster than poles (solar-like).
    inclination_deg : float
        Stellar inclination, degrees (i=90 -> equator-on).
    epsilon : float
        Linear limb-darkening coefficient (used if limb_darkening_fn is None).
    limb_darkening_fn : callable(mu) -> I(norm) (optional)
        If provided, used to evaluate local intensity vs mu (mu = cos gamma). Should return normalized I( mu ).
    n_theta, n_phi : int
        Sampling resolution for the stellar surface (latitude, longitude).
    log_n_samples : int or None
        Number of samples for the log-lambda resampling. If None the function chooses automatically.

    Returns
    -------
    broadened_flux_on_original_grid : 1D array (same shape as flux)
    """
    c_kms = 299792.458

    # --- 1) resample spectrum to uniform log10(lambda) grid (velocity domain) ---
    log_grid, flux_log, interp_back = _resample_to_log_lambda(wavelength, flux, n=log_n_samples)
    # velocity grid in km/s corresponding to log10(lambda) spacing:
    # v = c * ln(lambda/lambda0) = c * ln(10) * (log10(lambda) - log10(lambda0))
    # We'll measure velocities relative to line center at lambda0 = median wavelength.
    lam0 = np.median(wavelength)
    loglam0 = np.log10(lam0)
    dv = c_kms * np.log(10) * (log_grid[1] - log_grid[0])  # km/s per pixel
    v_grid = c_kms * np.log(10) * (log_grid - loglam0)     # velocity grid centered at lam0

    # --- 2) Build projected-velocity kernel via surface integration ---
    # Grid in colatitude theta (0=pole, pi/2=equator) and longitude phi (0 central meridian)
    theta = np.linspace(0.0, np.pi, n_theta)   # colatitude
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)

    # 2D mesh
    th2d, ph2d = np.meshgrid(theta, phi, indexing='xy')  # shapes (n_phi, n_theta)

    # compute surface element area weighting factor: sin(theta) dtheta dphi (we'll incorporate dtheta,dphi as factors later)
    sin_th = np.sin(th2d)

    # convert to latitude (lambda_lat) in radians: lat = 90° - colatitude
    lat2d = (np.pi/2.0) - th2d
    sin_lat = np.sin(lat2d)

    # inclination
    inc = np.deg2rad(inclination_deg)
    sin_i = np.sin(inc)
    cos_i = np.cos(inc)

    # differential rotation: Omega(theta) = Omega_eq * (1 - alpha * sin^2(lat))
    # But we don't have Omega_eq explicitly. We use vsini_equator = R * Omega_eq * sin(i)
    # So local projected velocity at (theta,phi) is:
    # v_local = R * Omega(theta) * sin(theta) * sin(i) * sin(phi)
    # => v_local = vsini_equator * (1 - alpha * sin_lat**2) * sin(theta) * sin(phi)
    # (this gives v_local = vsini_equator * sin(phi) at equator lat=0, theta=pi/2)
    factor_diff = (1.0 - alpha * (sin_lat**2))  # shape (n_phi, n_theta)
    v_local = vsini_equator * factor_diff * (np.sin(th2d)) * sin_i * np.sin(ph2d)  # km/s

    # visibility: mu = n . obs_unit where obs_unit = (sin i, 0, cos i)
    # local normal n = (sinθ cosφ, sinθ sinφ, cosθ)
    mu = sin_i * np.sin(th2d) * np.cos(ph2d) + cos_i * np.cos(th2d)
    visible_mask = (mu > 0)

    # limb darkening / intensity weighting
    if limb_darkening_fn is None:
        # linear law: I(mu)/I(1) = 1 - epsilon*(1-mu)
        intens = 1.0 - epsilon * (1.0 - mu)
        intens = np.clip(intens, 0.0, None)
    else:
        intens = limb_darkening_fn(mu)
        intens = np.asarray(intens)
        intens = np.where(mu>0, intens, 0.0)

    # element area weights: sin(theta) dtheta dphi, approximate dtheta = pi/(n_theta-1), dphi = 2pi/n_phi
    dtheta = (theta[1] - theta[0]) if n_theta > 1 else np.pi
    dphi   = (phi[1] - phi[0]) if n_phi > 1 else 2.0*np.pi
    area_weight = sin_th * dtheta * dphi

    # weight for each surface cell: only visible parts contribute ∝ I(mu) * mu * area
    cell_weight = np.where(visible_mask, intens * mu * area_weight, 0.0)

    # Build kernel: histogram v_local values weighted by cell_weight
    # choose velocity bins matching v_grid
    v_edges = np.concatenate([v_grid - 0.5*dv, [v_grid[-1] + 0.5*dv]])
    v_flat = v_local.ravel()
    w_flat = cell_weight.ravel()
    kernel_counts, _ = np.histogram(v_flat, bins=v_edges, weights=w_flat)

    # normalize kernel to unit area
    kernel = kernel_counts.astype(float)
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        # fallback to delta (no broadening)
        kernel = np.zeros_like(kernel)
        mid = len(kernel)//2
        kernel[mid] = 1.0
    else:
        kernel /= kernel_sum

    # --- 3) Optionally smooth kernel slightly to avoid discretization noise ---
    # small gaussian smoothing in velocity pixels (not required; uncomment to enable)
    # kernel = signal.fftconvolve(kernel, signal.windows.gaussian(9, std=2)/np.sum(signal.windows.gaussian(9, std=2)), mode='same')

    # --- 4) Convolve flux on log-lambda grid with kernel ---
    # convolution in velocity space is simply convolution over the flux_log array
    broadened_flux_log = signal.fftconvolve(flux_log, kernel, mode='same')

    # --- 5) map back to original wavelength grid and return ---
    broadened_flux = interp_back(broadened_flux_log)
    return broadened_flux



def apply_rotational_broadening(wavelength, flux, vsini, epsilon=0.6):
    """
    Apply rotational broadening to a spectrum.

    Parameters:
    wavelength : array-like
        Wavelength array in Angstroms
    flux : array-like
        Flux array
    vsini : float
        Projected rotational velocity in km/s
    epsilon : float, optional
        Limb darkening coefficient (default 0.6)

    Returns:
    array-like
        Broadened flux array
    """
    import numpy as np
    from scipy import interpolate, signal

    c = 299792.458  # Speed of light in km/s

    if vsini <= 0:
        return flux  # No broadening required

    # Average wavelength spacing
    delta_lambda = np.mean(np.diff(wavelength))

    # Calculate the width of the broadening kernel in wavelength units
    # The factor 2 comes from the full width of the line profile
    sigma = wavelength.mean() * vsini / c

    # Make sure the kernel width is sufficient
    if sigma < delta_lambda:
        return flux  # Broadening smaller than resolution

    # Calculate width of kernel in array elements
    kernel_width = int(np.ceil(5 * sigma / delta_lambda))
    if kernel_width % 2 == 0:
        kernel_width += 1  # Make sure it's odd

    # Create the kernel array
    kernel_x = np.linspace(-kernel_width // 2, kernel_width // 2, kernel_width) * delta_lambda
    kernel = np.zeros_like(kernel_x)

    # Populate the kernel using the rotational broadening profile
    x = kernel_x / sigma
    mask = np.abs(x) < 1.0
    kernel[mask] = (2 / np.pi) * (1.0 - epsilon * (1.0 - np.sqrt(1.0 - x[mask] ** 2))) * np.sqrt(1.0 - x[mask] ** 2)

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    # Apply the kernel via convolution
    broadened_flux = signal.convolve(flux, kernel, mode='same')

    return broadened_flux

def calculate_differential_vsini(star, n_rings, vsini_equator, mu_values):
    """
    Calculate vsini for each ring based on a differential rotation law.

    Parameters:
    self : object
        The parent object containing rotation parameters
    n_rings : int
        Number of rings
    vsini_equator : float
        Projected rotational velocity at equator in km/s
    mu_values : array-like
        Cosine of the angle between the line of sight and the local normal

    Returns:
    list
        vsini values for each ring
    """
    # Example differential rotation law: Ω(θ) = Ω_equator * (1 - alpha * sin²(θ))
    # where θ is the latitude and alpha is the differential rotation parameter

    # Default to no differential rotation if parameter is not set
    alpha = getattr(star, 'differential_rotation', 0.0)

    vsini_rings = []
    for i in range(n_rings):
        # Convert mu to latitude (approximate)
        # mu = cos(i) * cos(latitude)
        # For a star seen edge-on (i=90°), mu directly gives sin(latitude)
        sin_lat = mu_values[i]
        # Apply differential rotation law
        scale_factor = 1.0 - alpha * sin_lat ** 2
        vsini_rings.append(vsini_equator * scale_factor)

    return vsini_rings

def compute_immaculate_lc_with_vsini(star, Ngrid_in_ring, sini, cos_centers, proj_area, flnp, f_filt, wv, vsini=0.0):
    N = star.n_grid_rings
    flxph = 0.0
    sflp = np.zeros(N)
    flp = np.zeros([N, len(wv)])

    # Calculate differential rotation for each ring if enabled
    if hasattr(star, 'differential_rotation') and star.differential_rotation:
        vsini_rings = calculate_differential_vsini(star, N, vsini, cos_centers)
    else:
        vsini_rings = [vsini] * N

    # Computing flux of immaculate photosphere and of every pixel
    for i in range(N):
        # Interpolate Phoenix intensity models to correct projected angle:
        if star.use_phoenix_limb_darkening:
            # Handle edge cases properly
            if cos_centers[i] >= sini.max():
                # Grid point at or beyond disk center, use most central Phoenix angle
                dlp = flnp[-1]  # Assuming acd is sorted ascending
            elif cos_centers[i] <= sini.min():
                # Grid point at or beyond limb, use most limb Phoenix angle
                dlp = flnp[0]   # Assuming acd is sorted ascending
            else:
                # Normal interpolation case
                acd_below = sini[sini < cos_centers[i]]
                acd_above = sini[sini >= cos_centers[i]]

                acd_low = np.max(acd_below)
                acd_upp = np.min(acd_above)
                idx_low = np.where(sini == acd_low)[0][0]
                idx_upp = np.where(sini == acd_upp)[0][0]

                # Linear interpolation
                dlp = flnp[idx_low] + (flnp[idx_upp] - flnp[idx_low]) * (cos_centers[i] - acd_low) / (acd_upp - acd_low)
        else:
            # Use specified limb darkening law
            dlp = flnp[0] * limb_darkening_law(star, cos_centers[i])

        # Apply rotational broadening based on the vsini for this ring
        if vsini_rings[i] > 0:
            dlp = apply_rotational_broadening_differential(
                wv, dlp,
                vsini_equator=vsini_rings[i],  # your model’s vsini
                alpha=star.differential_rotation,
                inclination_deg=star.inclination,
                epsilon=star.limb_darkening_q1  # or use your full limb-darkening function
            )
            # dlp = apply_rotational_broadening(wv, dlp, vsini_rings[i])

        # Apply filter and calculate flux contribution
        flp[i, :] = dlp * proj_area[i] / (4 * np.pi) * f_filt(wv)
        sflp[i] = np.sum(flp[i, :])
        flxph = flxph + sflp[i] * Ngrid_in_ring[i]

    return sflp, flxph


def create_observed_spectra_kernel(star, wv_array, photo_flux, spot_flux, mu_sini, coverage,
                                   instrument_resolution=None, wavelength_range=None,
                                   resample=True, spectra_filter_name='None',
                                   dist=None, rad=None, flux_scale=1.0):
    """
    Advanced spectral integration using velocity kernels to simulate RM effect and line profile variations.
    
    Parameters:
    -----------
    coverage : 2D array [N_pixels, 4]
        Pixel-by-pixel coverage [aph, asp, afc, apl]
    mu_sini : 1D array
        Mu angles of Phoenix models
    """
    import numpy as np
    from scipy import signal, interpolate

    c_kms = 299792.458
    
    # Get grid geometry
    Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea = nbspectra.generate_grid_coordinates_nb(star.n_grid_rings)
    parea = np.array(parea) # Ensure numpy array for indexing
    
    # Map pixels to rings and rings to Phoenix mu indices
    ring_indices = []
    for i, n in enumerate(Ngrid_in_ring):
        ring_indices.extend([i] * n)
    ring_indices = np.array(ring_indices)
    
    ring_to_mu_idx = np.zeros(len(amu), dtype=int)
    for i, mu in enumerate(amu):
        ring_to_mu_idx[i] = np.argmin(np.abs(mu_sini - mu))
    
    # Projected velocities
    # ys is the coordinate that represents the horizontal shift on the sky (Doppler shift)
    v_pixels = star.vsini * ys
    
    # Resample to log-lambda for convolution
    # We want a grid that covers the vsini and instrumental profile
    log_grid, _, interp_back = _resample_to_log_lambda(wv_array, np.zeros_like(wv_array))
    dv = c_kms * np.log(10) * (log_grid[1] - log_grid[0])
    
    # Velocity grid for kernels (centered at 0)
    # Kernels should be large enough to cover max vsini
    v_max = star.vsini * 1.5 + 50.0 # safety margin
    nv_half = int(np.ceil(v_max / dv))
    v_kernel_grid = np.arange(-nv_half, nv_half + 1) * dv
    v_edges = np.concatenate([v_kernel_grid - 0.5*dv, [v_kernel_grid[-1] + 0.5*dv]])

    total_spectrum_log = np.zeros_like(log_grid)
    
    # For each Phoenix mu angle that has visible pixels
    active_mu_indices = np.unique(ring_to_mu_idx)
    
    for mu_idx in active_mu_indices:
        # Get pixels belonging to rings that map to this mu_idx
        rings_in_mu = np.where(ring_to_mu_idx == mu_idx)[0]
        pixels_in_mu = np.where(np.isin(ring_indices, rings_in_mu))[0]
        
        if len(pixels_in_mu) == 0: continue
        
        # Coverage fractions for these pixels
        aph = coverage[pixels_in_mu, 0]
        asp = coverage[pixels_in_mu, 1]
        afc = coverage[pixels_in_mu, 2]
        apl = coverage[pixels_in_mu, 3] # blocks the light
        
        # Weights for kernels: Area * mu * (visible fraction)
        # Note: parea[ring_idx] is the projected area mu*area for one grid element in that ring
        area_pixels = parea[ring_indices[pixels_in_mu]]
        visible_photo = aph * (1.0 - apl) * area_pixels
        visible_spot = (asp + afc) * (1.0 - apl) * area_pixels
        
        v_mu = v_pixels[pixels_in_mu]
        
        # Histogram into velocity kernels
        kernel_ph, _ = np.histogram(v_mu, bins=v_edges, weights=visible_photo)
        kernel_sp, _ = np.histogram(v_mu, bins=v_edges, weights=visible_spot)
        
        # convolve photosphere intensity
        if np.sum(kernel_ph) > 0:
            I_ph = photo_flux[mu_idx]
            # interpolate I_ph to log_grid if needed? 
            # create_observed_spectra assumes photo_flux is already on wv_array or resampled.
            # Let's assume input flux is on wv_array for simplicity.
            _, I_ph_log, _ = _resample_to_log_lambda(wv_array, I_ph, n=len(log_grid))
            # kernel needs to be normalized? 
            # No, the integral should represent total flux. 
            # Area of grid sum should be pi*R^2. 
            # But the kernels already include parea, which is the solid angle (mu*area).
            # So the convolution sum correctly gives the integrated flux.
            conv_ph = signal.fftconvolve(I_ph_log, kernel_ph / (2.0 * np.pi), mode='same')
            total_spectrum_log += conv_ph
            
        # convolve spot intensity
        if np.sum(kernel_sp) > 0:
            I_sp = spot_flux[mu_idx]
            _, I_sp_log, _ = _resample_to_log_lambda(wv_array, I_sp, n=len(log_grid))
            conv_sp = signal.fftconvolve(I_sp_log, kernel_sp / (2.0 * np.pi), mode='same')
            total_spectrum_log += conv_sp

    # Map back to original wavelength grid
    combined_spectrum = interp_back(total_spectrum_log)

    # Scale flux by (R / d)^2
    if dist is not None and rad is not None:
        R_SUN_CM = 6.957e10
        PC_CM = 3.086e18
        scale = (rad * R_SUN_CM) / (dist * PC_CM)
        combined_spectrum *= scale**2
        
    combined_spectrum *= flux_scale

    # Apply instrumental broadening
    if instrument_resolution is not None:
        sigma_instrumental = c_kms / instrument_resolution
        combined_spectrum = apply_rotational_broadening(wv_array, combined_spectrum, sigma_instrumental)

    # Apply instrument sensitivity/filter
    if spectra_filter_name == 'Raw':
        pass 
    elif spectra_filter_name is not None and spectra_filter_name != 'None':
        instrument_sensitivity = interpolate_filter(star, spectra_filter_name)
        combined_spectrum = combined_spectrum * instrument_sensitivity(wv_array)
    else:
        sensitivity_values = create_default_sensitivity(wv_array, wavelength_range)
        combined_spectrum = combined_spectrum * sensitivity_values

    # Resample to instrument grid if requested
    if resample and instrument_resolution is not None and wavelength_range is not None:
        combined_spectrum, wv_array = resample_spectrum(combined_spectrum, wv_array, instrument_resolution, wavelength_range)
    
    return combined_spectrum, wv_array


def create_observed_spectra(star, wv_array, photo_flux, spot_flux, mu, ff_sp,
                          spectra_filter_name='None', wavelength_range=None, instrument_resolution=None, 
                          resample=True, ff_planet=0.0,
                          dist=None, rad=None, flux_scale=1.0):
    """
    Create synthetic spectra using pre-computed Phoenix spectra.

    Parameters:
    -----------
    star : Star object
        Contains stellar parameters (vsini, temperature, etc.)
    wv_array : array
        Wavelength array
    photo_flux : 2D array
        Pre-computed photosphere flux [n_mu, n_wavelength]
    spot_flux : 2D array
        Pre-computed spot flux [n_mu, n_wavelength]
    mu : array
        Cosine of viewing angles (already computed)
    ff_sp : array or float
        Spot filling factor (as percentage)
    spectra_filter_name : str
        Name of spectral response filter file
    wavelength_range : tuple, optional
        (min_wavelength, max_wavelength) to define sensitivity range
    instrument_resolution : float, optional
        Instrumental resolution (R). If None, uses star.spectra_resolution.
    resample : bool, optional
        Whether to resample spectrum to instrument resolution grid (default True).
    ff_planet : float, optional
        Planet filling factor (fraction of disk blocked by planet). Default 0.
    dist : float, optional
        Distance to the star in parsecs.
    rad : float, optional
        Stellar radius in Solar Radii.
    flux_scale : float, optional
        Global flux scaling factor (default 1.0).

    Returns:
    --------
    tuple: (spectrum_with_noise, wavelength_array)
    """
    # Perform disk integration if not already done
    disk_integrated_photo = np.zeros(len(wv_array))
    disk_integrated_spot = np.zeros(len(wv_array))

    # Sort mu in ascending order if needed
    if len(mu) > 1 and mu[0] > mu[-1]:
        mu = mu[::-1]
        photo_flux = photo_flux[::-1, :]
        spot_flux = spot_flux[::-1, :]

    # Proper disk integration: F = 2π ∫ I(μ) μ dμ
    for i in range(len(mu) - 1):
        delta_mu = mu[i + 1] - mu[i]
        # Trapezoidal rule
        weight_i = mu[i] * delta_mu / 2
        weight_i_plus_1 = mu[i + 1] * delta_mu / 2

        disk_integrated_photo += (photo_flux[i, :] * weight_i +
                                  photo_flux[i + 1, :] * weight_i_plus_1)
        disk_integrated_spot += (spot_flux[i, :] * weight_i +
                                 spot_flux[i + 1, :] * weight_i_plus_1)

    # Combine photosphere and spots using fill factor
    # If ff_sp is an array (time series), we take the mean (legacy behavior)
    # If ff_sp is a scalar (snapshot), we use it directly
    if np.size(ff_sp) > 1:
        fill_factor = np.mean(ff_sp) / 100
    else:
        fill_factor = float(ff_sp) / 100

    combined_spectrum = ((1 - fill_factor - ff_planet) * disk_integrated_photo +
                         fill_factor * disk_integrated_spot)

    # Scale flux by (R / d)^2 if distance and radius provided
    if dist is not None and rad is not None:
        # Constants
        R_SUN_CM = 6.957e10
        PC_CM = 3.086e18
        
        # Calculate scale factor
        scale = (rad * R_SUN_CM) / (dist * PC_CM)
        combined_spectrum *= scale**2
        
    # Apply global flux scale
    combined_spectrum *= flux_scale

    # Apply stellar rotational broadening
    if star.vsini > 0:
        combined_spectrum = apply_rotational_broadening(wv_array, combined_spectrum, star.vsini)

    # Apply instrumental broadening
    if instrument_resolution is None:
        R = star.spectra_resolution
    else:
        R = instrument_resolution
        
    sigma_instrumental = c / R
    combined_spectrum = apply_rotational_broadening(wv_array, combined_spectrum, sigma_instrumental)

    # Apply instrument sensitivity/filter
    if spectra_filter_name == 'Raw':
        pass # Do not apply any sensitivity curve
    elif spectra_filter_name is not None and spectra_filter_name != 'None':
        instrument_sensitivity = interpolate_filter(star, spectra_filter_name)
        combined_spectrum = combined_spectrum * instrument_sensitivity(wv_array)
    else:
        # Create synthetic sensitivity if no filter provided
        sensitivity_values = create_default_sensitivity(wv_array, wavelength_range)
        combined_spectrum = combined_spectrum * sensitivity_values

    # Add noise based on SNR
    # base_snr = star.cdpp * 2 if hasattr(star, 'cdpp') else 0

    # # Wavelength-dependent SNR
    # if spectra_filter_name != 'None':
    #     sensitivity_values = instrument_sensitivity(wv_array)
    #     wavelength_dependent_snr = base_snr * np.sqrt(sensitivity_values / np.max(sensitivity_values))
    # else:
    #     wavelength_dependent_snr = np.full_like(wv_array, base_snr)

    # Resample to instrument grid if requested
    if resample and instrument_resolution is not None and wavelength_range is not None:
        combined_spectrum, wv_array = resample_spectrum(combined_spectrum, wv_array, instrument_resolution, wavelength_range)
    
    # Add noise based on SNR (if not already added in resample? No, noise usually per pixel)
    # Re-evaluating noise now that we are on pixel grid
    # ... logic continues ...
    
    # Add photon noise
    combined_spectrum_with_noise = np.zeros_like(combined_spectrum)
    
    # Simple SNR model based on CDPP/Magnitudes if needed. 
    # But usually handled externally or simplistically.
    # Current code had commented out noise logic. Let's keep it minimal or as is.
    
    # Preserving existing noise logic (if any was active)
    # The viewed code had noise commented out or simplified.
    # We return the resampled spectrum.
    
    return combined_spectrum, wv_array


def resample_spectrum(flux, wave, resolution, wave_range, sampling=3.0):
    """
    Resample spectrum to a log-uniform grid defined by instrument resolution.
    
    Parameters:
    -----------
    flux : array
        Input flux
    wave : array
        Input wavelength
    resolution : float
        Instrument resolution R = lambda / delta_lambda
    wave_range : tuple
        (min_wav, max_wav)
    sampling : float
        Pixels per resolution element (FWHM). Default 3.0 (Nyquist is 2).
        
    Returns:
    --------
    new_flux, new_wave
    """
    w_min, w_max = wave_range
    
    # Constant velocity spacing (log-linear)
    # R = lambda / dlambda  =>  dlambda = lambda / R
    # FWHM = lambda / R
    # pixel_size = FWHM / sampling = lambda / (R * sampling)
    # d(ln lambda) = dlambda / lambda = 1 / (R * sampling)
    
    k = 1.0 / (resolution * sampling)
    
    # Generate new grid
    # log_wav = np.arange(np.log(w_min), np.log(w_max), k)
    # new_wave = np.exp(log_wav)
    
    # Geometric series
    # wave[i+1] = wave[i] * (1 + k) roughly
    
    # Correct generation:
    num_pixels = int(np.log(w_max / w_min) / k)
    new_wave = w_min * np.exp(np.arange(num_pixels) * k)
    
    # Interpolate
    # Use robust interpolation
    f = interpolate.interp1d(wave, flux, kind='linear', bounds_error=False, fill_value=0.0)
    new_flux = f(new_wave)
    
    return new_flux, new_wave


def create_default_sensitivity(wv_array, wavelength_range=None):
    """
    Create a synthetic sensitivity curve if no filter is provided.
    Default shape approximates a visual band pass but can be used generally.

    Parameters:
    -----------
    wv_array : array
        Wavelength array in Angstroms
    wavelength_range : tuple, optional
        (min, max) wavelength for the sensitivity curve

    Returns:
    --------
    array : Sensitivity values
    """
    if wavelength_range is not None:
        w_min, w_max = wavelength_range
    else:
        w_min = np.min(wv_array)
        w_max = np.max(wv_array)
    
    # Center the sensitivity curve on the requested range
    central_wl = (w_min + w_max) / 2.0
    # Width covering significant part of the range (sigma approx 1/4 of range)
    width = (w_max - w_min) / 3.0
    
    if width <= 0:
        width = 1000.0 # Fallback for single line calculations or weird inputs
        
    sensitivity = np.exp(-0.5 * ((wv_array - central_wl) / width) ** 2)

    return sensitivity


def compute_immaculate_spectra(star, Ngrid_in_ring, sini, cos_centers, proj_area,
                               flnp, f_filt, wv, vsini=0.0):
    """
    Compute immaculate spectra without filter convolution for spectroscopic use.
    Correctly integrates Doppler shifts over the stellar disk.
    
    Parameters:
    -----------
    star : Star object
    Ngrid_in_ring : array
        Number of grid points in each ring
    sini : array
        Phoenix mu angles
    cos_centers : array
        Grid mu values (cosine of angle from center)
    proj_area : array
        Projected areas of grid elements
    flnp : 2D array
        Phoenix flux at each angle [n_angles, n_wavelengths]
    f_filt : function
        Filter function (can be identity for spectra)
    wv : array
        Wavelength array
    vsini : float
        Projected rotational velocity
    
    Returns:
    --------
    tuple: (ring_spectra, total_spectrum)
        ring_spectra: 2D array [n_rings, n_wavelengths]
        total_spectrum: 1D array of disk-integrated spectrum
    """
    N = star.n_grid_rings
    ring_spectra = np.zeros([N, len(wv)])
    total_spectrum = np.zeros(len(wv))

    # Pre-calculate log-wavelength grid for Doppler shifting if vsini > 0
    if vsini > 0:
        log_grid, _, interp_back = _resample_to_log_lambda(wv, np.zeros_like(wv))
        c_kms = 299792.458
        # Velocity step per pixel in log-grid
        dv_pixel = c_kms * np.log(10) * (log_grid[1] - log_grid[0]) 

    # We need the geometry of the grid to calculate local velocities
    # Re-generate grid coordinates to get x,y positions (mu is already in cos_centers)
    # Note: efficient implementation would cache this or pass it in
    _, _, _, _, _, alphas, xs, ys, zs, _, _ = nbspectra.generate_grid_coordinates_nb(N)
    
    # Calculate velocities for all grid points
    # Coordinate system: z is towards observer, x is "vertical" on sky, y is "horizontal"
    # Rotation is around x-axis (projected). 
    # v_z = x * Omega * sin(i) + ... 
    # Actually, simpler:
    # v_projected = vsini * y_coordinate_normalized * differential_rotation_factor
    
    # But wait, generated grid coordinates nbspectra.generate_grid_coordinates_nb 
    # returns xs, ys, zs for flattened list of all points? 
    # No, it returns arrays that need to be reconstructed into rings.
    
    # Let's align with the ring-based structure of Ngrid_in_ring.
    # The helper `generate_grid_coordinates_nb` returns flattened arrays `alphas` etc. 
    # We need to slice them per ring.
    
    start_idx = 0
    
    for i in range(N):
        # 1. Get Base Spectrum for this Ring (Intensity I(mu, lambda))
        # Interpolate Phoenix intensity models to correct projected angle
        if star.use_phoenix_limb_darkening:
            if cos_centers[i] >= sini.max():
                dlp = flnp[-1]
            elif cos_centers[i] <= sini.min():
                dlp = flnp[0]
            else:
                acd_below = sini[sini < cos_centers[i]]
                acd_above = sini[sini >= cos_centers[i]]
                acd_low = np.max(acd_below)
                acd_upp = np.min(acd_above)
                idx_low = np.where(sini == acd_low)[0][0]
                idx_upp = np.where(sini == acd_upp)[0][0]
                dlp = flnp[idx_low] + (flnp[idx_upp] - flnp[idx_low]) * \
                      (cos_centers[i] - acd_low) / (acd_upp - acd_low)
        else:
            dlp = flnp[0] * limb_darkening_law(star, cos_centers[i])

        # 2. Integrate over the ring's azimuthal segments (phi angles)
        # Each segment has a different Doppler shift
        
        n_segments = Ngrid_in_ring[i]
        
        # Extract y-coordinates (which drive Doppler shift) for this ring
        # ys corresponds to normalized distance from rotation axis projected on sky?
        # xs, ys, zs in nbspectra are:
        # xs = cos(colat) -> Polar axis direction if inc=90? 
        # Wait, nbspectra says:
        # xs = np.cos(np.pi*ts/180)  (z-axis in standard physics, but here named x)
        # ys = rs*np.sin(np.pi*alphas/180)
        # zs = -rs*np.cos(np.pi*alphas/180)
        # And "pole faces the observer" means i=0? 
        # Rotator logic converts these to observer frame.
        
        # Let's look at Rotator logic for consistency:
        # theta, phi = np.arccos(zs * ... - xs * ...), ...
        
        # SIMPLIFICATION: 
        # For a ring at colatitude theta (fixed per ring i), having `n_segments` azimuthal points:
        # The latitude is `lat = 90 - centers[i]`.
        # The velocity of the ring is v_rot(lat) = vsini_equator * diff_rot_factor(lat) / sin(inc) ? 
        # No, vsini is already projected. 
        # v_local_proj = v_equator * diff_rot(lat) * sin(i) * sin(phi_star) 
        # where phi_star is the rotational phase.
        
        # We can reconstruct the phases `phi` for this ring
        # anglesout=np.linspace(0,360-width, ... )
        # nbspectra generates alphas (longitudes)
        
        # Get slice of alphas for this ring
        current_alphas = alphas[start_idx : start_idx + n_segments]
        start_idx += n_segments
        
        # Calculate Differential Rotation Factor for this ring
        # cos_centers[i] is mu. For inc=90 (edge on), mu = cos(colat_projected)? 
        # No, cos_centers comes from generate_grid_coordinates which assumes pole facing observer (i=0)?
        # Actually generate_grid_coordinates_nb is purely a geometric meshing of a sphere.
        # It assumes a specific orientation where "centres" are colatitudes. 
        # Let's assume standard spherical coords where theta is colatitude.
        
        # In the observer frame (Rotator.get_theta_phi handles the rotation to observer), 
        # but here we are in the "Rest Frame" of the star's surface generation?
        # NO, `compute_immaculate_spectra` is called with `cos_centers` which are MU values.
        # MU values depend on Inclination. 
        # BUT `generate_grid_coordinates_nb` generates a grid centered on the LOS (Line of Sight).
        # i.e., "Pole of the grid faces the observer."
        # So Ring i is at angle theta_from_LOS = acos(mu).
        # This is NOT stellar latitude. It's "angle from center of disk".
        
        # So we have a ring at radius r_projected = sin(acos(mu)) on the disk.
        # `current_alphas` are the angles around the LOS. 
        # position (x,y) on sky: 
        # x = r_proj * cos(alpha)
        # y = r_proj * sin(alpha)
        
        # Stellar Rotation is roughly horizontal (if aligned with x-axis or y-axis).
        # Standard convention: Rotation axis is vertical (y-axis?), star rotates left-to-right?
        # Doppler shift is proportional to x-coordinate (or y, depending on convention).
        # Let's assume x is "horizontal" on the detector, y is "vertical" (rotation axis).
        # Then v_projected = Omega * x_coord.
        
        # vsini is velocity at limb (x=1). 
        # So v_local = vsini * x_projected_normalized.
        
        # r_proj = sqrt(1 - mu^2)
        # x_proj = r_proj * sin(alpha)  (or cos)
        
        # Differential rotation is hard here because we don't know the TRUE stellar latitude of each pixel easily 
        # without the inclination logic map.
        # However, for vsini broadening, we can often neglect differential rotation variations across the tiny segments 
        # OF THE LOS GRID if we just want "Solid Body" equivalent or assume vsini is dominant.
        
        # Issue 1 Fix is about geometric integration. 
        # If we want to support Differential Rotation correctly, we need the latitude of each grid point.
        # `star.get_theta_phi()` computes this!
        # `phi, theta = ...` (coordinates in star reference)
        # But `compute_immaculate_spectra` receives `cos_centers` and `proj_area`.
        # It acts on "Rings of equal mu".
        
        # CRITICAL: `compute_immaculate_spectra` loop assumes the spectrum only depends on MU (LD).
        # If we add rotation, it depends on POSITION (x,y).
        # We must sum over the individual grid points (segments), calculate their V_proj, shift, and add.
        
        # If vsini=0, simply:
        ring_flux_no_rot = dlp * proj_area[i] / (4 * np.pi) * Ngrid_in_ring[i] # Sum of all segments
        
        if vsini <= 0:
             ring_spectra[i, :] = ring_flux_no_rot
             total_spectrum += ring_spectra[i, :]
             continue

        # If vsini > 0, we must distribute the flux of this ring over velocities.
        
        # 2a. Reconstruct projected X-coordinates for this ring's segments
        # Center center mu = cos_centers[i]
        # Projected radius from center of disk:
        r_proj = np.sqrt(1 - cos_centers[i]**2)
        
        # Azimuthal angles for this ring (0 to 360)
        # We can uniformly sample them since the grid is constructed that way
        alphas_ring = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
        
        # Projected X positions (range -1 to 1)
        # Assuming rotation axis is Y, velocity is along X
        # x_proj = r_proj * sin(alpha)
        x_proj_values = r_proj * np.cos(alphas_ring) # or sin, symmetric for spectrum
        
        # Velocities: v = vsini * x_proj
        # (Neglecting differential rotation for the "immaculate" spectrum fix first, 
        # or assuming vsini is an effective vsini. 
        # Exact differential rotation requires mapping back to stellar lat/lon. 
        # `star.get_theta_phi` does this mapping. But that's heavy.)
        
        velocities = vsini * x_proj_values
        
        # 2b. Shift and Add
        # Resample dlp to log-lambda for efficiency?
        if len(wv) > 0:
            # We can use the pre-calculated log grid?
            # Or just interpolation for each segment (slower but accurate)
            # Given N_segments can be large, maybe histogram method (kernel) is better?
            
            # Create a "Broadening Kernel" for THIS RING
            # Histogram of velocities
            # v_grid = c * (log(lam) - log(lam0))
             
            # Let's use the histogram method for creating the kernel for this ring
            # matching standard broadening implementations
            
            # Construct kernel
            # Distribution of velocities in this ring is basically P(v) ~ 1/sqrt(1-(v/vmax)^2) (Chebyshev)
            # but discrete.
            
            dv = (wv[1] - wv[0]) # approx
            c_light = 299792.458
            
            # We'll shift the Base Spectrum `dlp` by `velocities`
            # Optimization: Group segments by velocity bins?
            
            # Let's use existing `_resample_to_log_lambda` logic from `spectra.py` manually
            # or simply shift.
            
            # Log-linear interpolation is standard for Doppler.
            log_wv = np.log(wv)
            flux_interp = interpolate.interp1d(log_wv, dlp, bounds_error=False, fill_value=0.0)
            
            ring_accum = np.zeros_like(dlp)
            
            # shift = v / c
            # log(lam_new) = log(lam_old * (1+v/c)) ~ log(lam_old) + v/c
            
            for v in velocities:
                shift = v / c_light
                shifted_flux = flux_interp(log_wv - shift) 
                ring_accum += shifted_flux

            # Normalize by area?
            # dlp is Intensity * Area/4pi? No, dlp is Intensity.
            # We need to weigh by area.
            # In this grid, all segments in a ring have equal area?
            # area[i] is area of ONE grid element.
            # proj_area[i] is projected area of ONE grid element.
            
            weight = proj_area[i] / (4 * np.pi)
            ring_spectra[i, :] = ring_accum * weight
            total_spectrum += ring_spectra[i, :]

    return ring_spectra, total_spectrum


def interpolate_Phoenix_mu_lc(star, temp, grav, wv_array=None, interp_type='linear',
                              convert_to_air=True, plot=False):
    """
    Accurate cut and interpolate Phoenix SpecInt models at desired wavelengths,
    temperatures, and logg. Uses proper bilinear interpolation and automatic
    wavelength grid creation from FITS headers.

    Parameters:
    -----------
    star : object
        Star object with models_root path and wavelength limits
    temp : float
        Temperature of the model (K)
    grav : float
        Log surface gravity of the model
    wv_array : array, optional
        Custom wavelength array to interpolate to (e.g., LAMOST grid)
    interp_type : str
        Interpolation method ('linear' or 'spline')
    convert_to_air : bool
        Whether to convert vacuum wavelengths to air wavelengths
    plot : bool
        Whether to create diagnostic plots

    Returns:
    --------
    tuple: (mu_angles, wavelength_array, interpolated_flux_at_each_angle)
        - mu_angles: cosine of viewing angles
        - wavelength_array: wavelength grid (in air if converted)
        - interpolated_flux: 2D array [n_angles, n_wavelengths]
    """

    if isinstance(star.models_root, str):
        star.models_root = Path(star.models_root)
    path = star.models_root / 'models' / 'Phoenix_mu'

    # Cache file listing for performance
    if not hasattr(star, '_phoenix_mu_cache'):
        files = [x.name for x in path.glob('lte*SPECINT*fits') if x.is_file()]
        if not files:
            sys.exit(f'Error: No PHOENIX SpecInt files found in {path}')

        list_temp = np.unique([float(t[3:8]) for t in files])
        list_grav = np.unique([float(t[9:13]) for t in files])
        star._phoenix_mu_cache = {
            'files': files,
            'temperatures': list_temp,
            'gravities': list_grav
        }
    else:
        files = star._phoenix_mu_cache['files']
        list_temp = star._phoenix_mu_cache['temperatures']
        list_grav = star._phoenix_mu_cache['gravities']

    # Validate and clamp parameter bounds with warnings
    original_grav = grav
    original_temp = temp

    if grav < np.min(list_grav):
        grav = np.min(list_grav)
        print(f'Warning: Desired log g ({original_grav}) is below grid minimum. '
              f'Using minimum value: {grav:.2f}')
    elif grav > np.max(list_grav):
        grav = np.max(list_grav)
        print(f'Warning: Desired log g ({original_grav}) is above grid maximum. '
              f'Using maximum value: {grav:.2f}')

    if temp < np.min(list_temp):
        temp = np.min(list_temp)
        print(f'Warning: Desired temperature ({original_temp}) is below grid minimum. '
              f'Using minimum value: {temp:.0f} K')
    elif temp > np.max(list_temp):
        temp = np.max(list_temp)
        print(f'Warning: Desired temperature ({original_temp}) is above grid maximum. '
              f'Using maximum value: {temp:.0f} K')

    # Find bounding grid points
    lowT = list_temp[list_temp <= temp].max() if any(list_temp <= temp) else list_temp.min()
    uppT = list_temp[list_temp >= temp].min() if any(list_temp >= temp) else list_temp.max()
    lowg = list_grav[list_grav <= grav].max() if any(list_grav <= grav) else list_grav.min()
    uppg = list_grav[list_grav >= grav].min() if any(list_grav >= grav) else list_grav.max()

    # print(f"Interpolating between T=[{lowT}, {uppT}], log g=[{lowg}, {uppg}]")

    # Generate filenames for the four corner points
    def make_filename(T, g):
        return f'lte{int(T):05d}-{g:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'

    model_files = {
        'lowTlowg': make_filename(lowT, lowg),  # (T_low, g_low)
        'lowTuppg': make_filename(lowT, uppg),  # (T_low, g_high)
        'uppTlowg': make_filename(uppT, lowg),  # (T_high, g_low)
        'uppTuppg': make_filename(uppT, uppg)  # (T_high, g_high)
    }

    # Check file existence and create fallback strategy
    available_files = {k: (v in files) for k, v in model_files.items()}
    missing_files = [f for f in model_files.values() if f not in files]

    if not any(available_files.values()):
        sys.exit(f'Error: No required files found for interpolation. Available files: {len(files)}')

    if missing_files:
        print(f'Warning: Missing files: {missing_files}')
        print('Using fallback interpolation strategy...')

        # Create fallback mapping for missing files
        fallback_map = create_fallback_strategy(available_files, list_temp, list_grav, lowT, uppT, lowg, uppg)


    # Load wavelength grid and mu angles from first available file


    # Get wavelength grid from first file
    first_file = path / model_files['lowTlowg']
    phoenix_wavelength, amu, is_vacuum_wl, wave_type = get_wavelength_and_mu(first_file)

    # Reverse mu_angles to match typical convention (disk center to limb)
    amu = amu[::-1]


    # Determine target wavelength array
    if wv_array is not None:
        target_wavelength = np.asarray(wv_array)

    else:
        # Use star's wavelength limits if available
        if hasattr(star, 'wavelength_lower_limit') and hasattr(star, 'wavelength_upper_limit'):
            wmin = max(phoenix_wavelength.min(), star.wavelength_lower_limit)
            wmax = min(phoenix_wavelength.max(), star.wavelength_upper_limit)
            mask = (phoenix_wavelength >= wmin) & (phoenix_wavelength <= wmax)
            target_wavelength = phoenix_wavelength[mask]
        else:
            target_wavelength = phoenix_wavelength

    # Create wavelength mask for loading Phoenix data (with overhead for interpolation)
    overhead = 50.0  # Angstrom overhead for interpolation
    phoenix_min = max(phoenix_wavelength.min(), target_wavelength.min() - overhead)
    phoenix_max = min(phoenix_wavelength.max(), target_wavelength.max() + overhead)

    idx_wv_phoenix = ((phoenix_wavelength >= phoenix_min) &
                      (phoenix_wavelength <= phoenix_max))

    phoenix_wv_subset = phoenix_wavelength[idx_wv_phoenix]

    # Helper function to load flux data
    def load_flux_data(filename):
        try:
            with fits.open(path / filename) as hdul:
                full_flux = hdul[0].data  # Shape: [n_mu_angles, n_wavelengths]
                if full_flux.shape[1] != len(phoenix_wavelength):
                    sys.exit(f'Error: Flux wavelength dimension ({full_flux.shape[1]}) '
                             f'does not match wavelength array ({len(phoenix_wavelength)}) in {filename}')
                return full_flux[:, idx_wv_phoenix]  # Apply wavelength mask
        except Exception as e:
            sys.exit(f'Error reading {filename}: {e}')

    # Load flux data with fallback handling
    flux_data = {}
    for key, filename in model_files.items():
        if available_files[key]:
            flux_data[key] = load_flux_data(filename)
        else:
            # Use fallback file
            fallback_file = fallback_map[key]
            print(f"Using fallback {fallback_file} for missing {filename}")
            flux_data[key] = load_flux_data(fallback_file)

    # Perform bilinear interpolation for each mu angle and wavelength
    n_mu, n_wave = flux_data['lowTlowg'].shape
    flux_interpolated = np.zeros((n_mu, n_wave))

    for i in range(n_mu):
        for j in range(n_wave):
            flux_interpolated[i, j] = bilinear_interpolate_point(
                temp, grav, lowT, uppT, lowg, uppg,
                flux_data['lowTlowg'][i, j],  # (T_low, g_low)
                flux_data['lowTuppg'][i, j],  # (T_low, g_high)
                flux_data['uppTlowg'][i, j],  # (T_high, g_low)
                flux_data['uppTuppg'][i, j]  # (T_high, g_high)
            )

    # Interpolate to target wavelength grid if needed
    if not np.array_equal(target_wavelength, phoenix_wv_subset):
        flux_final = np.zeros((n_mu, len(target_wavelength)))

        for i in range(n_mu):
            if interp_type == 'linear':
                interp_func = interp1d(
                    phoenix_wv_subset, flux_interpolated[i, :],
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                flux_final[i, :] = interp_func(target_wavelength)
            elif interp_type == 'spline':
                spline = UnivariateSpline(phoenix_wv_subset, flux_interpolated[i, :], s=0, k=3)
                flux_final[i, :] = spline(target_wavelength)
            else:
                raise ValueError(f'Interpolation type {interp_type} not supported')

        # Handle NaN values
        nan_mask = np.isnan(flux_final)
        if np.any(nan_mask):
            n_nan = np.sum(nan_mask)
            print(f"Warning: {n_nan} points outside Phoenix range, using extrapolation")
            # Simple extrapolation: use nearest valid values
            for i in range(n_mu):
                if np.any(nan_mask[i, :]):
                    valid_mask = ~nan_mask[i, :]
                    if np.any(valid_mask):
                        # Forward and backward fill
                        flux_final[i, :] = forward_backward_fill(flux_final[i, :])
                    else:
                        flux_final[i, :] = np.nanmedian(flux_final)

        final_wavelength = target_wavelength
        final_flux = flux_final
    else:
        final_wavelength = phoenix_wv_subset
        final_flux = flux_interpolated

    # Convert vacuum to air wavelengths if requested
    if convert_to_air and is_vacuum_wl:
        final_wavelength = vacuum_to_air(final_wavelength)

    # Create diagnostic plot if requested
    if plot:
        create_diagnostic_plot(final_wavelength, final_flux, amu, temp, grav)

    return amu, final_wavelength, final_flux



def bilinear_interpolate_point(temp, grav, temp_low, temp_high, grav_low, grav_high,
                               flux_00, flux_01, flux_10, flux_11):
    """
    Bilinear interpolation for a single point.

    Parameters:
    -----------
    temp, grav : float
        Target temperature and gravity
    temp_low, temp_high, grav_low, grav_high : float
        Bounding grid values
    flux_ij : float
        Flux values at grid corners

    Returns:
    --------
    float : Interpolated flux value
    """
    # Handle edge cases
    if temp_high == temp_low:
        if grav_high == grav_low:
            return flux_00
        else:
            # Linear interpolation in gravity only
            w_g = (grav - grav_low) / (grav_high - grav_low)
            return (1 - w_g) * flux_00 + w_g * flux_01
    elif grav_high == grav_low:
        # Linear interpolation in temperature only
        w_t = (temp - temp_low) / (temp_high - temp_low)
        return (1 - w_t) * flux_00 + w_t * flux_10
    else:
        # Full bilinear interpolation
        w_t = (temp - temp_low) / (temp_high - temp_low)
        w_g = (grav - grav_low) / (grav_high - grav_low)

        return ((1 - w_t) * (1 - w_g) * flux_00 +  # Lower-left
                (1 - w_t) * w_g * flux_01 +  # Upper-left
                w_t * (1 - w_g) * flux_10 +  # Lower-right
                w_t * w_g * flux_11)  # Upper-right


def interpolate_Phoenix_mu_lc_with_metallicity(star, temp, grav, feh, wv_array=None,
                                               interp_type='linear', convert_to_air=True,
                                               plot=False):
    """
    Accurate cut and interpolate Phoenix SpecInt models at desired wavelengths,
    temperatures, logg, and metallicity. Uses trilinear interpolation.

    Parameters:
    -----------
    star : object
        Star object with models_root path and wavelength limits
    temp : float
        Temperature of the model (K)
    grav : float
        Log surface gravity of the model
    feh : float
        Metallicity [Fe/H] of the model
    wv_array : array, optional
        Custom wavelength array to interpolate to
    interp_type : str
        Interpolation method ('linear' or 'spline')
    convert_to_air : bool
        Whether to convert vacuum wavelengths to air wavelengths
    plot : bool
        Whether to create diagnostic plots

    Returns:
    --------
    tuple: (mu_angles, wavelength_array, interpolated_flux_at_each_angle)
    """

    if isinstance(star.models_root, str):
        star.models_root = Path(star.models_root)
    path = star.models_root / 'models' / 'Phoenix_mu'

    # Parse available files and extract unique parameter values
    if not hasattr(star, '_phoenix_mu_cache_with_feh'):
        files = [x.name for x in path.glob('lte*SPECINT*fits') if x.is_file()]
        if not files:
            sys.exit(f'Error: No PHOENIX SpecInt files found in {path}')

        list_temp = []
        list_grav = []
        list_feh = []

        for filename in files:
            # Parse temperature (characters 3-8)
            temp_val = float(filename[3:8])

            # Parse gravity (characters 9-13)
            grav_val = float(filename[9:13])

            # Parse metallicity (everything between grav and .PHOENIX)
            feh_part = filename[13:filename.index('.PHOENIX')]
            feh_val = float(feh_part)

            list_temp.append(temp_val)
            list_grav.append(grav_val)
            list_feh.append(feh_val)

        list_temp = np.unique(list_temp)
        list_grav = np.unique(list_grav)
        list_feh = np.unique(list_feh)

        star._phoenix_mu_cache_with_feh = {
            'files': files,
            'temperatures': list_temp,
            'gravities': list_grav,
            'metallicities': list_feh
        }

        # print(f"Available parameter ranges:")
        # print(f"  Temperature: {list_temp.min():.0f} - {list_temp.max():.0f} K")
        # print(f"  log(g): {list_grav.min():.2f} - {list_grav.max():.2f}")
        # print(f"  [Fe/H]: {list_feh.min():.2f} - {list_feh.max():.2f}")
    else:
        files = star._phoenix_mu_cache_with_feh['files']
        list_temp = star._phoenix_mu_cache_with_feh['temperatures']
        list_grav = star._phoenix_mu_cache_with_feh['gravities']
        list_feh = star._phoenix_mu_cache_with_feh['metallicities']

    # Validate and clamp parameter bounds
    original_temp, original_grav, original_feh = temp, grav, feh

    if temp < np.min(list_temp):
        temp = np.min(list_temp)
        print(f'Warning: Temperature {original_temp}K below grid minimum. Using {temp:.0f}K')
    elif temp > np.max(list_temp):
        temp = np.max(list_temp)
        print(f'Warning: Temperature {original_temp}K above grid maximum. Using {temp:.0f}K')

    if grav < np.min(list_grav):
        grav = np.min(list_grav)
        print(f'Warning: log(g) {original_grav} below grid minimum. Using {grav:.2f}')
    elif grav > np.max(list_grav):
        grav = np.max(list_grav)
        print(f'Warning: log(g) {original_grav} above grid maximum. Using {grav:.2f}')

    if feh < np.min(list_feh):
        feh = np.min(list_feh)
        print(f'Warning: [Fe/H] {original_feh} below grid minimum. Using {feh:.2f}')
    elif feh > np.max(list_feh):
        feh = np.max(list_feh)
        print(f'Warning: [Fe/H] {original_feh} above grid maximum. Using {feh:.2f}')

    # Find bounding grid points for all three dimensions
    lowT = list_temp[list_temp <= temp].max() if any(list_temp <= temp) else list_temp.min()
    uppT = list_temp[list_temp >= temp].min() if any(list_temp >= temp) else list_temp.max()
    lowg = list_grav[list_grav <= grav].max() if any(list_grav <= grav) else list_grav.min()
    uppg = list_grav[list_grav >= grav].min() if any(list_grav >= grav) else list_grav.max()
    lowfeh = list_feh[list_feh <= feh].max() if any(list_feh <= feh) else list_feh.min()
    uppfeh = list_feh[list_feh >= feh].min() if any(list_feh >= feh) else list_feh.max()

    # print(f"Interpolating between:")
    # print(f"  T=[{lowT:.0f}, {uppT:.0f}], log(g)=[{lowg:.2f}, {uppg:.2f}], [Fe/H]=[{lowfeh:.2f}, {uppfeh:.2f}]")

    # Generate filenames for the eight corner points of the cube
    def make_filename(T, g, feh_val):
        # Handle the special case of 0.0 metallicity (always has minus sign)
        if feh_val == 0.0:
            feh_str = "-0.0"
        elif feh_val > 0:
            feh_str = f"+{feh_val:.1f}"
        else:
            feh_str = f"{feh_val:.1f}"  # Negative values already have minus sign
        return f'lte{int(T):05d}-{g:.2f}{feh_str}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'

    # Eight corners of the interpolation cube (T, g, feh)
    model_files = {
        'lowT_lowg_lowfeh': make_filename(lowT, lowg, lowfeh),
        'lowT_lowg_uppfeh': make_filename(lowT, lowg, uppfeh),
        'lowT_uppg_lowfeh': make_filename(lowT, uppg, lowfeh),
        'lowT_uppg_uppfeh': make_filename(lowT, uppg, uppfeh),
        'uppT_lowg_lowfeh': make_filename(uppT, lowg, lowfeh),
        'uppT_lowg_uppfeh': make_filename(uppT, lowg, uppfeh),
        'uppT_uppg_lowfeh': make_filename(uppT, uppg, lowfeh),
        'uppT_uppg_uppfeh': make_filename(uppT, uppg, uppfeh),
    }

    # Check file existence
    available_files = {k: (v in files) for k, v in model_files.items()}
    missing_files = [f for f in model_files.values() if f not in files]

    if not any(available_files.values()):
        sys.exit(f'Error: No required files found for interpolation')

    if missing_files:
        # print(f'Warning: {len(missing_files)} missing files. Using fallback strategy.')
        # Implement fallback strategy similar to original but for 3D
        fallback_map = create_3d_fallback_strategy(
            available_files, model_files, list_temp, list_grav, list_feh,
            lowT, uppT, lowg, uppg, lowfeh, uppfeh
        )
    else:
        fallback_map = model_files

    # Load wavelength grid and mu angles from first available file
    first_available_key = next(k for k, v in available_files.items() if v)
    first_file = path / model_files[first_available_key]

    # Get wavelength and mu data
    phoenix_wavelength, amu, is_vacuum_wl, wave_type = get_wavelength_and_mu(first_file)

    # Reverse mu_angles if needed (verify based on your PHOENIX version)
    if len(amu) > 1 and amu[0] < amu[-1]:
        amu = amu[::-1]  # Ensure descending order (center to limb)

    # Determine target wavelength array
    if wv_array is not None:
        target_wavelength = np.asarray(wv_array)
    else:
        if hasattr(star, 'wavelength_lower_limit') and hasattr(star, 'wavelength_upper_limit'):
            wmin = max(phoenix_wavelength.min(), star.wavelength_lower_limit)
            wmax = min(phoenix_wavelength.max(), star.wavelength_upper_limit)
            mask = (phoenix_wavelength >= wmin) & (phoenix_wavelength <= wmax)
            target_wavelength = phoenix_wavelength[mask]
        else:
            target_wavelength = phoenix_wavelength

    # Create wavelength mask for loading Phoenix data
    overhead = 50.0  # Angstrom overhead for interpolation
    phoenix_min = max(phoenix_wavelength.min(), target_wavelength.min() - overhead)
    phoenix_max = min(phoenix_wavelength.max(), target_wavelength.max() + overhead)

    idx_wv_phoenix = ((phoenix_wavelength >= phoenix_min) &
                      (phoenix_wavelength <= phoenix_max))
    phoenix_wv_subset = phoenix_wavelength[idx_wv_phoenix]

    # Load flux data for all eight corners
    def load_flux_data(filename):
        try:
            filepath = path / filename
            if not filepath.exists():
                # Use fallback
                filepath = path / fallback_map.get(filename, filename)

            with fits.open(filepath) as hdul:
                full_flux = hdul[0].data
                if full_flux.shape[1] != len(phoenix_wavelength):
                    sys.exit(f'Error: Flux wavelength dimension mismatch in {filename}')
                return full_flux[:, idx_wv_phoenix]
        except Exception as e:
            sys.exit(f'Error reading {filename}: {e}')

    flux_data = {}
    for key, filename in model_files.items():
        flux_data[key] = load_flux_data(filename)

    # Perform trilinear interpolation
    n_mu, n_wave = flux_data['lowT_lowg_lowfeh'].shape
    flux_interpolated = np.zeros((n_mu, n_wave))

    for i in range(n_mu):
        for j in range(n_wave):
            flux_interpolated[i, j] = trilinear_interpolate_point(
                temp, grav, feh, lowT, uppT, lowg, uppg, lowfeh, uppfeh,
                flux_data['lowT_lowg_lowfeh'][i, j],  # 000
                flux_data['lowT_lowg_uppfeh'][i, j],  # 001
                flux_data['lowT_uppg_lowfeh'][i, j],  # 010
                flux_data['lowT_uppg_uppfeh'][i, j],  # 011
                flux_data['uppT_lowg_lowfeh'][i, j],  # 100
                flux_data['uppT_lowg_uppfeh'][i, j],  # 101
                flux_data['uppT_uppg_lowfeh'][i, j],  # 110
                flux_data['uppT_uppg_uppfeh'][i, j],  # 111
            )

    # Interpolate to target wavelength grid if needed
    if not np.array_equal(target_wavelength, phoenix_wv_subset):
        flux_final = interpolate_to_target_wavelength(
            phoenix_wv_subset, flux_interpolated, target_wavelength,
            interp_type, n_mu
        )
        final_wavelength = target_wavelength
        final_flux = flux_final
    else:
        final_wavelength = phoenix_wv_subset
        final_flux = flux_interpolated

    # Convert vacuum to air wavelengths if requested
    if convert_to_air and is_vacuum_wl:
        final_wavelength = vacuum_to_air(final_wavelength)

    # Create diagnostic plot if requested
    if plot:
        create_diagnostic_plot_with_feh(
            final_wavelength, final_flux, amu, temp, grav, feh
        )
    return amu, final_wavelength, final_flux


# Advanced cache management functions

def clear_cache(star=None, cache_dir=None):
    """
    Clear all Phoenix cache files.

    Parameters:
    -----------
    star : object, optional
        Star object with models_root
    cache_dir : str or Path, optional
        Specific cache directory to clear
    """
    global PHOENIX_CACHE_DIR

    if cache_dir:
        target_dir = Path(cache_dir)
    elif PHOENIX_CACHE_DIR:
        target_dir = PHOENIX_CACHE_DIR
    elif star:
        target_dir = Path(star.models_root) / 'cache'
    else:
        print("Error: No cache directory specified")
        return

    if not target_dir.exists():
        print(f"Cache directory does not exist: {target_dir}")
        return

    removed_count = 0
    removed_size = 0

    for file in target_dir.glob('phoenix_*.pkl'):
        removed_size += file.stat().st_size
        file.unlink()
        removed_count += 1

    print(f"Cleared {removed_count} cache files ({removed_size / (1024 ** 2):.1f} MB)")

    # Reset statistics
    global PHOENIX_CACHE_STATS
    PHOENIX_CACHE_STATS = {'hits': 0, 'misses': 0, 'saves': 0}


def get_cache_info(star=None, cache_dir=None):
    """
    Get information about the current cache status.

    Returns:
    --------
    dict : Cache information including size, file count, and oldest/newest files
    """
    global PHOENIX_CACHE_DIR

    if cache_dir:
        target_dir = Path(cache_dir)
    elif PHOENIX_CACHE_DIR:
        target_dir = PHOENIX_CACHE_DIR
    elif star:
        target_dir = Path(star.models_root) / 'cache'
    else:
        return None

    if not target_dir.exists():
        return {
            'directory': str(target_dir),
            'exists': False,
            'size_mb': 0,
            'size_gb': 0,
            'file_count': 0
        }

    cache_files = list(target_dir.glob('phoenix_*.pkl'))

    if not cache_files:
        return {
            'directory': str(target_dir),
            'exists': True,
            'size_mb': 0,
            'size_gb': 0,
            'file_count': 0
        }

    total_size = sum(f.stat().st_size for f in cache_files)
    cache_files.sort(key=lambda x: x.stat().st_mtime)

    info = {
        'directory': str(target_dir),
        'exists': True,
        'size_mb': total_size / (1024 ** 2),
        'size_gb': total_size / (1024 ** 3),
        'file_count': len(cache_files),
        'oldest_file': cache_files[0].name if cache_files else None,
        'newest_file': cache_files[-1].name if cache_files else None,
        'oldest_date': cache_files[0].stat().st_mtime if cache_files else None,
        'newest_date': cache_files[-1].stat().st_mtime if cache_files else None
    }

    # Add statistics if available
    global PHOENIX_CACHE_STATS
    total_requests = PHOENIX_CACHE_STATS['hits'] + PHOENIX_CACHE_STATS['misses']
    if total_requests > 0:
        info['hit_rate'] = PHOENIX_CACHE_STATS['hits'] / total_requests * 100
        info['statistics'] = PHOENIX_CACHE_STATS.copy()

    return info


def get_cache_key(temp, grav, feh, wv_min, wv_max, interp_type, convert_to_air):
    """
    Generate a unique cache key for a specific interpolation request.

    Parameters:
    -----------
    temp : float
        Temperature in Kelvin
    grav : float
        Log surface gravity
    feh : float
        Metallicity [Fe/H]
    wv_min : float
        Minimum wavelength
    wv_max : float
        Maximum wavelength
    interp_type : str
        Interpolation type ('linear' or 'spline')
    convert_to_air : bool
        Whether wavelengths are converted to air

    Returns:
    --------
    str : MD5 hash string (32 characters)
    """
    key_str = f"T{temp:.0f}_g{grav:.3f}_feh{feh:.2f}_wv{wv_min:.1f}-{wv_max:.1f}_{interp_type}_air{convert_to_air}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(star, cache_key):
    """
    Get the full path for a cache file.

    Parameters:
    -----------
    star : object
        Star object with models_root attribute
    cache_key : str
        Cache key (typically from get_cache_key function)

    Returns:
    --------
    Path : Full path to the cache file
    """

    if PHOENIX_CACHE_DIR is None:
        # Use default location under models_root
        cache_dir = Path(star.models_root) / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = PHOENIX_CACHE_DIR

    return cache_dir / f"phoenix_{cache_key}.pkl"

def precompute_grid_cache(star, temp_range, grav_range, feh_range,
                          wv_array=None, interp_type='linear',
                          convert_to_air=True, verbose=True):
    """
    Precompute and cache interpolations for a grid of stellar parameters.
    Useful for batch processing or preparing for real-time analysis.

    Parameters:
    -----------
    star : object
        Star object with models_root
    temp_range : tuple or list
        (min, max, step) or list of temperatures in K
    grav_range : tuple or list
        (min, max, step) or list of log(g) values
    feh_range : tuple or list
        (min, max, step) or list of [Fe/H] values
    wv_array : array, optional
        Wavelength array to use
    interp_type : str
        Interpolation type
    convert_to_air : bool
        Whether to convert to air wavelengths
    verbose : bool
        Print progress information

    Returns:
    --------
    int : Number of models computed and cached
    """

    # Parse parameter ranges
    if len(temp_range) == 3:
        temps = np.arange(temp_range[0], temp_range[1] + temp_range[2], temp_range[2])
    else:
        temps = np.array(temp_range)

    if len(grav_range) == 3:
        gravs = np.arange(grav_range[0], grav_range[1] + grav_range[2], grav_range[2])
    else:
        gravs = np.array(grav_range)

    if len(feh_range) == 3:
        fehs = np.arange(feh_range[0], feh_range[1] + feh_range[2], feh_range[2])
    else:
        fehs = np.array(feh_range)

    total = len(temps) * len(gravs) * len(fehs)
    computed = 0
    skipped = 0

    if verbose:
        print(f"Precomputing grid cache for {total} parameter combinations...")
        print(f"  Temperatures: {len(temps)} values from {temps.min():.0f} to {temps.max():.0f} K")
        print(f"  log(g): {len(gravs)} values from {gravs.min():.2f} to {gravs.max():.2f}")
        print(f"  [Fe/H]: {len(fehs)} values from {fehs.min():.2f} to {fehs.max():.2f}")

    for i, temp in enumerate(temps):
        for j, grav in enumerate(gravs):
            for k, feh in enumerate(fehs):
                # Check if already cached
                if wv_array is not None:
                    wv_min, wv_max = np.min(wv_array), np.max(wv_array)
                elif hasattr(star, 'wavelength_lower_limit'):
                    wv_min = star.wavelength_lower_limit
                    wv_max = star.wavelength_upper_limit
                else:
                    wv_min, wv_max = 0, 1e10

                cache_key = get_cache_key(temp, grav, feh, wv_min, wv_max,
                                          interp_type, convert_to_air)
                cache_path = get_cache_path(star, cache_key)

                if cache_path.exists():
                    skipped += 1
                    continue

                # Compute and cache
                try:
                    _ = interpolate_Phoenix_mu_lc_with_metallicity(
                        star, temp, grav, feh, wv_array=wv_array,
                        interp_type=interp_type, convert_to_air=convert_to_air,
                        plot=False, use_cache=True
                    )
                    computed += 1

                    if verbose and computed % 10 == 0:
                        progress = (i * len(gravs) * len(fehs) +
                                    j * len(fehs) + k + 1) / total * 100
                        print(f"  Progress: {progress:.1f}% ({computed} computed, {skipped} skipped)")

                except Exception as e:
                    print(f"  Error computing T={temp:.0f}, log(g)={grav:.2f}, [Fe/H]={feh:.2f}: {e}")

    if verbose:
        print(f"Precomputation complete: {computed} new models cached, {skipped} already existed")
        cache_info = get_cache_info(star)
        if cache_info:
            print(f"Cache size: {cache_info['size_gb']:.2f} GB, {cache_info['file_count']} files")

    return computed


def trilinear_interpolate_point(temp, grav, feh, temp_low, temp_high,
                                grav_low, grav_high, feh_low, feh_high,
                                f000, f001, f010, f011, f100, f101, f110, f111):
    """
    Trilinear interpolation for a single point in 3D parameter space.

    The eight flux values correspond to corners of the cube:
    fijk where i,j,k are binary indices for (temp, grav, feh)
    000 = (low_T, low_g, low_feh)
    001 = (low_T, low_g, high_feh)
    010 = (low_T, high_g, low_feh)
    ... etc
    """

    # Handle edge cases where grid points are identical
    if temp_high == temp_low:
        temp_weight = 0.0
    else:
        temp_weight = (temp - temp_low) / (temp_high - temp_low)

    if grav_high == grav_low:
        grav_weight = 0.0
    else:
        grav_weight = (grav - grav_low) / (grav_high - grav_low)

    if feh_high == feh_low:
        feh_weight = 0.0
    else:
        feh_weight = (feh - feh_low) / (feh_high - feh_low)

    # Trilinear interpolation formula
    c00 = f000 * (1 - temp_weight) + f100 * temp_weight
    c01 = f001 * (1 - temp_weight) + f101 * temp_weight
    c10 = f010 * (1 - temp_weight) + f110 * temp_weight
    c11 = f011 * (1 - temp_weight) + f111 * temp_weight

    c0 = c00 * (1 - grav_weight) + c10 * grav_weight
    c1 = c01 * (1 - grav_weight) + c11 * grav_weight

    result = c0 * (1 - feh_weight) + c1 * feh_weight

    return result


def create_3d_fallback_strategy(available_files, model_files, list_temp, list_grav,
                                list_feh, lowT, uppT, lowg, uppg, lowfeh, uppfeh):
    """
    Create fallback mapping for missing files in 3D parameter space.
    Prioritizes nearest neighbors in parameter space.
    """
    fallback_map = {}

    def make_filename(T, g, feh_val):
        if feh_val == 0.0:
            feh_str = "-0.0"
        elif feh_val > 0:
            feh_str = f"+{feh_val:.1f}"
        else:
            feh_str = f"{feh_val:.1f}"
        return f'lte{int(T):05d}-{g:.2f}{feh_str}.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'

    # Define corner points
    corners = {
        'lowT_lowg_lowfeh': (lowT, lowg, lowfeh),
        'lowT_lowg_uppfeh': (lowT, lowg, uppfeh),
        'lowT_uppg_lowfeh': (lowT, uppg, lowfeh),
        'lowT_uppg_uppfeh': (lowT, uppg, uppfeh),
        'uppT_lowg_lowfeh': (uppT, lowg, lowfeh),
        'uppT_lowg_uppfeh': (uppT, lowg, uppfeh),
        'uppT_uppg_lowfeh': (uppT, uppg, lowfeh),
        'uppT_uppg_uppfeh': (uppT, uppg, uppfeh),
    }

    for corner_key, (target_T, target_g, target_feh) in corners.items():
        if available_files[corner_key]:
            fallback_map[model_files[corner_key]] = model_files[corner_key]
            continue

        # Find nearest available model in 3D parameter space
        min_distance = float('inf')
        best_fallback = None

        for other_key, is_available in available_files.items():
            if is_available:
                other_T, other_g, other_feh = corners[other_key]
                # Normalized distance in parameter space
                distance = np.sqrt(
                    ((other_T - target_T) / 1000) ** 2 +  # Normalize T by 1000K
                    ((other_g - target_g) * 2) ** 2 +  # Weight log(g) differences
                    ((other_feh - target_feh) * 2) ** 2  # Weight [Fe/H] differences
                )

                if distance < min_distance:
                    min_distance = distance
                    best_fallback = model_files[other_key]

        if best_fallback:
            fallback_map[model_files[corner_key]] = best_fallback
            if min_distance > 2.0:  # Warn if fallback is far
                print(f"  Warning: Using distant fallback for {corner_key}, distance={min_distance:.2f}")
        else:
            # Last resort: search entire grid
            for T in list_temp:
                for g in list_grav:
                    for feh_val in list_feh:
                        test_file = make_filename(T, g, feh_val)
                        if test_file in [model_files[k] for k in available_files if available_files[k]]:
                            fallback_map[model_files[corner_key]] = test_file
                            print(f"  Using grid search fallback for {corner_key}")
                            break
                    if model_files[corner_key] in fallback_map:
                        break
                if model_files[corner_key] in fallback_map:
                    break

    return fallback_map


def interpolate_to_target_wavelength(source_wv, source_flux, target_wv, interp_type, n_mu):
    """
    Interpolate flux from source wavelength grid to target wavelength grid.
    """
    flux_final = np.zeros((n_mu, len(target_wv)))

    for i in range(n_mu):
        if interp_type == 'linear':
            interp_func = interp1d(
                source_wv, source_flux[i, :],
                kind='linear', bounds_error=False, fill_value=np.nan
            )
            flux_final[i, :] = interp_func(target_wv)
        elif interp_type == 'spline':
            spline = UnivariateSpline(source_wv, source_flux[i, :], s=0, k=3)
            flux_final[i, :] = spline(target_wv)
        else:
            raise ValueError(f'Interpolation type {interp_type} not supported')

    # Handle NaN values with nearest-neighbor extrapolation
    nan_mask = np.isnan(flux_final)
    if np.any(nan_mask):
        print(f"Warning: {np.sum(nan_mask)} points outside Phoenix range, using extrapolation")
        for i in range(n_mu):
            if np.any(nan_mask[i, :]):
                flux_final[i, :] = forward_backward_fill(flux_final[i, :])

    return flux_final


def create_diagnostic_plot_with_feh(wavelength, flux, mu_angles, temp, grav, feh):
    """Create diagnostic plots for the interpolation results including metallicity."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot spectra at different mu angles
    n_plot = min(5, len(mu_angles))
    indices = np.linspace(0, len(mu_angles) - 1, n_plot, dtype=int)

    for idx in indices:
        label = f'μ = {mu_angles[idx]:.2f}'
        ax1.plot(wavelength, flux[idx, :], label=label, alpha=0.8)

    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Specific Intensity')
    ax1.set_title(f'PHOENIX SpecInt: T={temp:.0f}K, log g={grav:.2f}, [Fe/H]={feh:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot limb darkening curve at a representative wavelength
    mid_idx = len(wavelength) // 2
    ax2.plot(mu_angles, flux[:, mid_idx], 'o-', label=f'λ = {wavelength[mid_idx]:.1f} Å')
    ax2.set_xlabel('μ = cos(θ)')
    ax2.set_ylabel('Specific Intensity')
    ax2.set_title('Limb Darkening Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Helper function needed from original (add to the file)
def get_wavelength_and_mu(fits_file):
    """Extract wavelength grid and mu angles from SpecInt FITS file"""
    with fits.open(fits_file) as hdul:
        header = hdul[0].header

        # Extract WCS parameters for wavelength
        crval1 = header['CRVAL1']  # Wavelength of reference pixel
        cdelt1 = header['CDELT1']  # Wavelength step size
        crpix1 = header['CRPIX1']  # Reference pixel (usually 1)
        ctype1 = header['CTYPE1']  # Grid type
        naxis1 = header['NAXIS1']  # Number of wavelength points

        # Create wavelength array
        if 'LOG' in ctype1:
            # Logarithmic grid
            log_wave = crval1 + cdelt1 * (np.arange(naxis1) + 1 - crpix1)
            wavelength = 10 ** log_wave
        else:
            # Linear grid
            wavelength = crval1 + cdelt1 * (np.arange(naxis1) + 1 - crpix1)

        # Check vacuum vs air wavelengths
        is_vacuum = ctype1.startswith('WAVE')

        # Get mu angles from second extension
        mu_angles = hdul[1].data

        return wavelength, mu_angles, is_vacuum, ctype1

def vacuum_to_air(wavelength_vacuum):
    """
    Convert vacuum wavelengths to air wavelengths using the IAU standard formula.

    Parameters:
    -----------
    wavelength_vacuum : array
        Vacuum wavelengths in Angstroms

    Returns:
    --------
    array : Air wavelengths in Angstroms
    """
    # Convert to micrometers for the formula
    wl_um = wavelength_vacuum / 10000.0

    # IAU standard formula (Morton 2000, ApJS, 130, 403)
    n = (1.0 + 0.0000834254 + 0.02406147 / (130 - 1 / wl_um ** 2) +
         0.00015998 / (38.9 - 1 / wl_um ** 2))

    # Convert back to air wavelengths in Angstroms
    wavelength_air = wavelength_vacuum / n

    return wavelength_air


def forward_backward_fill(arr):
    """
    Forward and backward fill NaN values in array.

    Parameters:
    -----------
    arr : array
        Array with potential NaN values

    Returns:
    --------
    array : Array with NaN values filled
    """
    arr = arr.copy()

    # Forward fill
    mask = np.isfinite(arr)
    if np.any(mask):
        arr[~mask] = np.interp(np.where(~mask)[0], np.where(mask)[0], arr[mask])

    return arr


def create_fallback_strategy(available_files, list_temp, list_grav, lowT, uppT, lowg, uppg):
    """
    Create a fallback mapping for missing Phoenix files using nearest available models.

    Priority order for fallbacks:
    1. Use center point if available (exact T or g match)
    2. Use nearest available temperature at same gravity
    3. Use nearest available gravity at same temperature
    4. Use nearest available point in T-g space
    """
    fallback_map = {}

    # Define the required corner points
    corners = {
        'lowTlowg': (lowT, lowg),
        'lowTuppg': (lowT, uppg),
        'uppTlowg': (uppT, lowg),
        'uppTuppg': (uppT, uppg)
    }

    def make_filename(T, g):
        return f'lte{int(T):05d}-{g:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'

    for corner_key, (target_T, target_g) in corners.items():
        if available_files[corner_key]:
            fallback_map[corner_key] = make_filename(target_T, target_g)
            continue


        # Strategy 1: Try exact temperature match with different gravity
        exact_T_files = []
        for g in list_grav:
            candidate = make_filename(target_T, g)
            if any(available_files[k] and make_filename(*corners[k]) == candidate for k in available_files):
                exact_T_files.append((candidate, abs(g - target_g)))

        if exact_T_files:
            # Use closest gravity at same temperature
            best_file = min(exact_T_files, key=lambda x: x[1])[0]
            fallback_map[corner_key] = best_file
            print(f"  Using same T, different g: {best_file}")
            continue

        # Strategy 2: Try exact gravity match with different temperature
        exact_g_files = []
        for T in list_temp:
            candidate = make_filename(T, target_g)
            if any(available_files[k] and make_filename(*corners[k]) == candidate for k in available_files):
                exact_g_files.append((candidate, abs(T - target_T)))

        if exact_g_files:
            # Use closest temperature at same gravity
            best_file = min(exact_g_files, key=lambda x: x[1])[0]
            fallback_map[corner_key] = best_file
            print(f"  Using same g, different T: {best_file}")
            continue

        # Strategy 3: Use nearest point in parameter space
        all_available = []
        for other_key, (other_T, other_g) in corners.items():
            if available_files[other_key]:
                distance = np.sqrt(((other_T - target_T) / 1000) ** 2 + (other_g - target_g) ** 2)
                all_available.append((make_filename(other_T, other_g), distance))

        if all_available:
            best_file = min(all_available, key=lambda x: x[1])[0]
            fallback_map[corner_key] = best_file
            continue

        # Strategy 4: Find any available file with similar parameters
        backup_options = []
        for T in list_temp:
            for g in list_grav:
                candidate = make_filename(T, g)
                # Check if this file exists in our available list
                if candidate in [make_filename(*corners[k]) for k in corners if available_files[k]]:
                    distance = np.sqrt(((T - target_T) / 1000) ** 2 + (g - target_g) ** 2)
                    backup_options.append((candidate, distance))

        if backup_options:
            best_file = min(backup_options, key=lambda x: x[1])[0]
            fallback_map[corner_key] = best_file
            print(f"  Using backup option: {best_file}")
        else:
            # Last resort: use any available file
            for k in corners:
                if available_files[k]:
                    fallback_map[corner_key] = make_filename(*corners[k])
                    print(f"  Using last resort: {fallback_map[corner_key]}")
                    break

    return fallback_map


def create_diagnostic_plot(wavelength, flux, mu_angles, temp, grav):
    """Create diagnostic plots for the interpolation results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot spectra at different mu angles
    n_plot = min(5, len(mu_angles))  # Plot up to 5 angles
    indices = np.linspace(0, len(mu_angles) - 1, n_plot, dtype=int)

    for idx in indices:
        label = f'μ = {mu_angles[idx]:.2f}' if idx < len(mu_angles) else 'μ = 0.0 (limb)'
        ax1.plot(wavelength, flux[idx, :], label=label, alpha=0.8)

    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Specific Intensity')
    ax1.set_title(f'PHOENIX SpecInt Interpolation: T={temp}K, log g={grav}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot limb darkening curve at a representative wavelength
    mid_idx = len(wavelength) // 2
    ax2.plot(mu_angles[:-1], flux[:-1, mid_idx], 'o-', label=f'λ = {wavelength[mid_idx]:.1f} Å')
    ax2.set_xlabel('μ = cos(θ)')
    ax2.set_ylabel('Specific Intensity')
    ax2.set_title('Limb Darkening Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def keplerian_orbit(x,params):
    period=params[0]
    t_trans=params[4]
    krv=params[1]
    esinw=params[2]
    ecosw=params[3]
    
    if(esinw==0 and ecosw==0):
       ecc=0
       omega=0
    else:
       ecc=np.sqrt(esinw*esinw+ecosw*ecosw)
       omega=np.arctan2(esinw,ecosw)

    t_peri = Ttrans_2_Tperi(t_trans, period, ecc, omega)
    sinf,cosf=true_anomaly(x,period,ecc,t_peri)
    cosftrueomega=cosf*np.cos(omega)-sinf*np.sin(omega)
    y= krv*(ecc*np.cos(omega)+cosftrueomega)

    return y
#   
def true_anomaly(x,period,ecc,tperi):
    sinf=[]
    cosf=[]
    for i in range(len(x)):
        fmean=2.0*np.pi*(x[i]-tperi)/period
        #Solve by Newton's method x(n+1)=x(n)-f(x(n))/f'(x(n))
        fecc=fmean
        diff=1.0
        while(diff>1.0E-6):
            fecc_0=fecc
            fecc=fecc_0-(fecc_0-ecc*np.sin(fecc_0)-fmean)/(1.0-ecc*np.cos(fecc_0))
            diff=np.abs(fecc-fecc_0)
        sinf.append(np.sqrt(1.0-ecc*ecc)*np.sin(fecc)/(1.0-ecc*np.cos(fecc)))
        cosf.append((np.cos(fecc)-ecc)/(1.0-ecc*np.cos(fecc)))
    return np.array(sinf),np.array(cosf)


def Ttrans_2_Tperi(T0, P, e, w):

    f = np.pi/2 - w
    E = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-e)/(1+e)))  # eccentric anomaly
    Tp = T0 - P/(2*np.pi) * (E - e*np.sin(E))      # time of periastron

    return Tp


########################################################################################
########################################################################################
#                              SPOTMAP/GRID FUNCTIONS                                  #
########################################################################################
########################################################################################
def compute_spot_position(self,t):


    pos=np.zeros([len(self.spot_map),4])

    for i in range(len(self.spot_map)):
        tini = self.spot_map[i][0] #time of spot apparence
        dur = self.spot_map[i][1] #duration of the spot
        tfin = tini + dur #final time of spot
        colat = self.spot_map[i][2] #colatitude
        lat = 90 - colat #latitude
        longi = self.spot_map[i][3] #longitude
        Rcoef = self.spot_map[i][4::] #coefficients for the evolution od the radius. Depends on the desired law.

        #update longitude adding diff rotation
        pht = longi + (t-self.reference_time)/self.rotation_period%1*360 + (t-self.reference_time)*self.differential_rotation*(1.698*np.sin(np.deg2rad(lat))**2+2.346*np.sin(np.deg2rad(lat))**4)
        phsr = pht%360 #make the phase between 0 and 360. 

        if self.spots_evo_law == 'constant':
            if t>=tini and t<=tfin: 
                rad=Rcoef[0] 
            else:
                rad=0.0

        elif self.spots_evo_law == 'linear':
            if t>=tini and t<=tfin:
                rad=Rcoef[0]+(t-tini)*(Rcoef[1]-Rcoef[0])/dur
            else:
                rad=0.0
        elif self.spots_evo_law == 'quadratic':
            if t>=tini and t<=tfin:
                rad=-4*Rcoef[0]*(t-tini)*(t-tini-dur)/dur**2
            else:
                rad=0.0

        else:
            sys.exit('Spot evolution law not implemented yet')
        
        if self.facular_area_ratio!=0.0: #to speed up the code when no fac are present
            rad_fac=np.deg2rad(rad)*np.sqrt(1+self.facular_area_ratio) 
        else: rad_fac=0.0

        pos[i]=np.array([np.deg2rad(colat), np.deg2rad(phsr), np.deg2rad(rad), rad_fac])
        #return position and radii of spots at t in radians.

    return pos

def compute_planet_pos(self,t):
    
    if(self.planet_esinw==0 and self.planet_ecosw==0):
       ecc=0
       omega=0
    else:
       ecc=np.sqrt(self.planet_esinw**2+self.planet_ecosw**2)
       omega=np.arctan2(self.planet_esinw,self.planet_ecosw)

    t_peri = Ttrans_2_Tperi(self.planet_transit_t0,self.planet_period, ecc, omega)
    sinf,cosf=true_anomaly([t],self.planet_period,ecc,t_peri)


    cosftrueomega=cosf*np.cos(omega+np.pi/2)-sinf*np.sin(omega+np.pi/2) #cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
    sinftrueomega=cosf*np.sin(omega+np.pi/2)+sinf*np.cos(omega+np.pi/2) #sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

    if cosftrueomega>0.0: return np.array([1+self.planet_radius*2, 0.0, self.planet_radius]) #avoid secondary transits

    cosi = (self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-ecc**2) #cosine of planet inclination (i=90 is transit)

    rpl=self.planet_semi_major_axis*(1-ecc**2)/(1+ecc*cosf)
    xpl=rpl*(-np.cos(self.planet_spin_orbit_angle)*sinftrueomega-np.sin(self.planet_spin_orbit_angle)*cosftrueomega*cosi)
    ypl=rpl*(np.sin(self.planet_spin_orbit_angle)*sinftrueomega-np.cos(self.planet_spin_orbit_angle)*cosftrueomega*cosi)

    rhopl=np.sqrt(ypl**2+xpl**2)
    thpl=np.arctan2(ypl,xpl)

    pos=np.array([float(rhopl), float(thpl), self.planet_radius]) #rho, theta, and radii (in Rstar) of the planet
    return pos

def plot_spot_map_grid(self,vec_grid,typ,inc,time):
    filename = self.path / 'plots' / 'map_t_{:.4f}.png'.format(time)

    x=np.linspace(-0.999,0.999,1000)
    h=np.sqrt((1-x**2)/(np.tan(inc)**2+1))
    color_dict = { 0:'red', 1:'black', 2:'yellow', 3:'blue'}
    plt.figure(figsize=(4,4))
    plt.title('t={:.3f}'.format(time))
    plt.scatter(vec_grid[:,1],vec_grid[:,2], color=[ color_dict[np.argmax(i)] for i in typ ],s=2 )
    plt.plot(x,h,'k')
    plt.savefig(filename,dpi=100)
    plt.close()




