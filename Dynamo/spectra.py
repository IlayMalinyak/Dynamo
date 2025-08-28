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

warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d, UnivariateSpline
import math as m
from . import nbspectra

def interpolate_filter(star, filter_name):

    if isinstance(star.models_root, str):
        star.models_root = Path(star.models_root)
    path = star.models_root / 'models' / 'filters' / filter_name

    try:
        wv, filt = np.loadtxt(path,unpack=True)
    except: #if the filter do not exist, create a tophat filter from the wv range
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
            dlp = apply_rotational_broadening(wv, dlp, vsini_rings[i])

        # Apply filter and calculate flux contribution
        flp[i, :] = dlp * proj_area[i] / (4 * np.pi) * f_filt(wv)
        sflp[i] = np.sum(flp[i, :])
        flxph = flxph + sflp[i] * Ngrid_in_ring[i]

    return sflp, flxph


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

    # Validate parameter bounds
    if grav < np.min(list_grav) or grav > np.max(list_grav):
        sys.exit(f'Error: Desired log g ({grav}) is outside the grid range '
                 f'[{np.min(list_grav):.2f}, {np.max(list_grav):.2f}]. '
                 f'Please download additional PHOENIX SpecInt models.')

    if temp < np.min(list_temp) or temp > np.max(list_temp):
        sys.exit(f'Error: Desired temperature ({temp}) is outside the grid range '
                 f'[{np.min(list_temp):.0f}, {np.max(list_temp):.0f}]. '
                 f'Please download additional PHOENIX SpecInt models.')

    # Find bounding grid points
    lowT = list_temp[list_temp <= temp].max() if any(list_temp <= temp) else list_temp.min()
    uppT = list_temp[list_temp >= temp].min() if any(list_temp >= temp) else list_temp.max()
    lowg = list_grav[list_grav <= grav].max() if any(list_grav <= grav) else list_grav.min()
    uppg = list_grav[list_grav >= grav].min() if any(list_grav >= grav) else list_grav.max()

    print(f"Interpolating between T=[{lowT}, {uppT}], log g=[{lowg}, {uppg}]")

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




